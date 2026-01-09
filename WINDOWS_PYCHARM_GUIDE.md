# Windows 10 + PyCharm에서 Azure Function App 배포 완벽 가이드
# 도메인: captcha-inference-app-f0ejcugkgqgvh4e9.koreacentral-01.azurewebsites.net

## 📋 목차

1. [환경 준비](#1-환경-준비)
2. [PyCharm 프로젝트 설정](#2-pycharm-프로젝트-설정)
3. [Azure 도구 설치](#3-azure-도구-설치)
4. [Function App 코드 작성](#4-function-app-코드-작성)
5. [로컬 테스트](#5-로컬-테스트)
6. [Azure에 배포](#6-azure에-배포)
7. [배포 후 검증](#7-배포-후-검증)
8. [API 사용 가이드](#8-api-사용-가이드)

---

## 1. 환경 준비

### 1.1 필수 소프트웨어 확인

**PyCharm 확인:**
- PyCharm 열기
- File → Settings (또는 Ctrl+Alt+S)
- Python Interpreter에서 Python 3.11+ 설치 확인

**Windows 커맨드 확인:**
- `Win + R` 입력 후 `cmd` 실행
- `python --version` 입력 → Python 3.11+ 표시 확인
- `pip --version` 입력 확인

### 1.2 필수 프로그램 설치 체크리스트

```
☐ Python 3.11+ 설치됨
☐ pip 최신 버전
☐ PyCharm 설치됨
☐ Git 설치됨 (선택사항)
```

---

## 2. PyCharm 프로젝트 설정

### 2.1 새 프로젝트 생성

**Step 1: PyCharm 메인 화면**
- File → New Project

**Step 2: 프로젝트 설정**
```
프로젝트명: captcha-inference-app
경로: C:\Users\{YourUsername}\captcha-inference-app
```

**Step 3: Python 인터프리터 선택**
- New environment using Virtualenv 선택
- Location: 자동 설정
- Python 3.11 선택
- Create 클릭

### 2.2 PyCharm 콘솔에서 확인

PyCharm 하단의 Terminal 탭 열기:

```bash
# 가상환경 활성화 확인 (prompt 앞에 (venv) 표시)
# Python 버전 확인
python --version
# 출력: Python 3.11.x
```

---

## 3. Azure 도구 설치

### 3.1 Azure CLI 설치

**방법 1: 직접 설치 (권장)**

1. https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows 방문
2. MSI 인스톨러 다운로드
3. 실행 및 설치
4. 완료 후 재부팅

**설치 확인:**

PyCharm Terminal에서:
```bash
az --version
# 출력 예:
# azure-cli                    2.56.0
# core                         2.56.0
```

### 3.2 Azure Functions Core Tools 설치

**PowerShell을 관리자로 실행:**

```powershell
choco install azure-functions-core-tools-4
```

(Chocolatey 미설치 시: https://chocolatey.org/install 참고)

**또는 npm 사용:**

```bash
npm install -g azure-functions-core-tools@4 --unsafe-perm true
```

**설치 확인:**

```bash
func --version
# 출력: 4.x.x
```

### 3.3 필요한 Python 패키지 설치

PyCharm Terminal에서:

```bash
pip install azure-functions azure-storage-blob torch pillow numpy requests
```

설치 진행률:
```
Collecting azure-functions
Downloading ... 
...
Successfully installed azure-functions-1.20.0
...
Successfully installed all packages
```

---

## 4. Function App 코드 작성

### 4.1 프로젝트 디렉토리 구조 생성

PyCharm에서 새 폴더 생성:

**Path: captcha-inference-app/**

```
captcha-inference-app/
├── function_app.py          (⭐ 메인 함수 코드)
├── requirements.txt         (의존성)
├── host.json               (설정)
├── local.settings.json     (로컬 설정)
├── model/                  (모델 파일 폴더)
│   └── model_best.pt      (학습된 모델)
├── test_client.py         (테스트 클라이언트)
└── .gitignore             (Git 무시 파일)
```

### 4.2 디렉토리 생성 (PyCharm)

**Right-click on project root:**
1. New → Directory
   - 이름: `model`
   - Create

### 4.3 파일 생성 및 작성

#### 파일 1: `requirements.txt`

**Right-click on project:**
1. New → File
2. 이름: `requirements.txt`
3. 다음 내용 입력:

```
torch==2.3.0
azure-functions==1.20.0
azure-storage-blob==12.20.0
azure-identity==1.15.0
pillow==10.1.0
numpy==1.24.3
requests==2.31.0
```

**저장: Ctrl+S**

#### 파일 2: `function_app.py`

**Right-click on project:**
1. New → File
2. 이름: `function_app.py`
3. 다음 전체 코드 입력:

```python
import azure.functions as func
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import string
import io
import time
from datetime import datetime

# ==================== 설정 ====================
IMAGE_WIDTH = 230
IMAGE_HEIGHT = 70
NUM_CLASSES = 26
BLANK_LABEL = 26
ALPHABETS = string.ascii_uppercase
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABETS)}

logger = logging.getLogger("InferenceFunction")
logger.setLevel(logging.INFO)

# ==================== 모델 ====================
class CRNN(nn.Module):
    """CNN-RNN-CTC 기반 캡차 인식 모델"""
    def __init__(self, img_h, num_classes, rnn_hidden_size=256, rnn_layers=2, rnn_dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
        )
        
        conv_output_h = img_h // 8
        self.rnn_input_size = 128 * conv_output_h
        
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            bidirectional=True,
            batch_first=True,
        )
        
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes + 1)
    
    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x

# ==================== 모델 관리 ====================
class ModelManager:
    """모델 로딩 및 관리"""
    _model = None
    _device = None
    _last_load_time = None
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls.load_model()
        return cls._model
    
    @classmethod
    def get_device(cls):
        if cls._device is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._device
    
    @classmethod
    def load_model(cls):
        try:
            model_path = Path(__file__).parent / "model" / "model_best.pt"
            
            if not model_path.exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
            device = cls.get_device()
            cls._model = CRNN(IMAGE_HEIGHT, NUM_CLASSES).to(device)
            cls._model.load_state_dict(torch.load(model_path, map_location=device))
            cls._model.eval()
            cls._last_load_time = datetime.now()
            
            logger.info(f"✓ 모델 로드 완료: {model_path}")
            logger.info(f"  디바이스: {device}")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise

# ==================== 이미지 처리 ====================
def preprocess_image(image_bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
        image_np = np.array(image, dtype=np.float32)
        image_np = image_np / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    except Exception as e:
        logger.error(f"❌ 이미지 전처리 실패: {e}")
        raise

# ==================== 추론 ====================
def ctc_decode(predictions):
    preds = predictions.argmax(dim=2)
    preds = preds.cpu().numpy().transpose(1, 0)
    
    decoded = []
    for p in preds:
        prev = -1
        seq = []
        for idx in p:
            if idx != prev and idx != BLANK_LABEL:
                seq.append(idx)
            prev = idx
        decoded.append("".join([IDX_TO_CHAR[i] for i in seq]))
    
    return decoded

def infer(image_tensor: torch.Tensor, model, device):
    try:
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            logits = model(image_tensor)
            predictions = ctc_decode(logits)
        
        return predictions[0] if predictions else ""
    except Exception as e:
        logger.error(f"❌ 추론 실패: {e}")
        raise

# ==================== HTTP Trigger ====================
app = func.FunctionApp()

@app.route(route="InferenceHttpTrigger", methods=["POST"])
def infer_captcha(req: func.HttpRequest) -> func.HttpResponse:
    """이미지를 받아 캡차 인식 수행"""
    start_time = time.time()
    
    try:
        logger.info("📥 추론 요청 수신")
        
        if "image" not in req.files:
            logger.warning("❌ 'image' 파일이 없습니다")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": "'image' 파일이 필요합니다 (multipart/form-data)"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        image_file = req.files["image"]
        image_bytes = image_file.read()
        
        if not image_bytes:
            logger.warning("❌ 이미지 파일이 비어있습니다")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": "이미지 파일이 비어있습니다"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        logger.info(f"📤 이미지 크기: {len(image_bytes)} bytes")
        
        # 이미지 전처리
        logger.info("🔄 이미지 전처리 중...")
        image_tensor = preprocess_image(image_bytes)
        logger.info(f"✓ 이미지 텐서 shape: {image_tensor.shape}")
        
        # 모델 로드
        logger.info("🔄 모델 로드 중...")
        model = ModelManager.get_model()
        device = ModelManager.get_device()
        logger.info(f"✓ 모델 준비 완료")
        
        # 추론 수행
        logger.info("🔄 추론 수행 중...")
        predicted_text = infer(image_tensor, model, device)
        logger.info(f"✓ 추론 결과: {predicted_text}")
        
        # 신뢰도 계산
        with torch.no_grad():
            image_tensor_device = image_tensor.to(device)
            logits = model(image_tensor_device)
            probs = torch.softmax(logits, dim=2)
            confidence = float(probs.max().item())
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"✅ 추론 완료 ({processing_time_ms:.1f}ms)")
        
        response = {
            "status": "success",
            "predicted_text": predicted_text,
            "confidence": round(confidence, 4),
            "processing_time_ms": round(processing_time_ms, 1)
        }
        
        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )
    
    except FileNotFoundError as e:
        logger.error(f"❌ 파일 오류: {e}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": f"모델 파일을 찾을 수 없습니다: {e}"
            }),
            status_code=500,
            mimetype="application/json"
        )
    
    except Exception as e:
        logger.error(f"❌ 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": f"추론 처리 중 오류 발생: {str(e)}"
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """헬스 체크 엔드포인트"""
    try:
        model = ModelManager.get_model()
        device = ModelManager.get_device()
        
        return func.HttpResponse(
            json.dumps({
                "status": "healthy",
                "model_loaded": model is not None,
                "device": str(device),
                "last_load_time": str(ModelManager._last_load_time)
            }),
            status_code=200,
            mimetype="application/json"
        )
    
    except Exception as e:
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e)
            }),
            status_code=500,
            mimetype="application/json"
        )
```

**저장: Ctrl+S**

#### 파일 3: `host.json`

```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true,
        "maxTelemetryItemsPerSecond": 20
      }
    }
  },
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.*, 5.0.0)"
  },
  "functionTimeout": "00:05:00"
}
```

#### 파일 4: `local.settings.json`

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "AzureWebJobsFeatureFlags": "EnableWorkerIndexing"
  }
}
```

### 4.4 모델 파일 복사

**Windows 탐색기 또는 PyCharm:**

1. `model/` 폴더에 오른쪽 클릭
2. Open in Explorer
3. 학습된 `model_best.pt` 파일 복사
4. 해당 폴더에 붙여넣기

결과:
```
captcha-inference-app/model/model_best.pt ✓
```

---

## 5. 로컬 테스트

### 5.1 PyCharm Terminal에서 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Azure Functions Core Tools로 실행
func start

# 출력:
# Azure Functions Core Tools
# Found Python version 3.11.x
# ...
# Now listening on: http://0.0.0.0:7071
# Application started. Press Ctrl+C to quit.
```

### 5.2 다른 Terminal에서 테스트

**새 Terminal 열기 (Ctrl+Shift+Alt+T 또는 Terminal → New)**

```bash
# 헬스 체크
curl http://localhost:7071/api/health

# 응답:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "device": "cpu",
#   "last_load_time": "2024-12-22 15:30:45.123456"
# }
```

**이미지 테스트:**

```bash
# test_image.png가 있다고 가정
curl -X POST http://localhost:7071/api/InferenceHttpTrigger ^
  -F "image=@test_image.png"

# 응답:
# {
#   "status": "success",
#   "predicted_text": "ABCDEF",
#   "confidence": 0.9523,
#   "processing_time_ms": 145.2
# }
```

---

## 6. Azure에 배포

### 6.1 Azure CLI 로그인

PyCharm Terminal:

```bash
az login
```

**브라우저 자동 열림:**
- Microsoft 계정으로 로그인
- 권한 허가
- Terminal에 성공 메시지 나타남

### 6.2 배포 준비

```bash
# Function App 이름 확인
az functionapp show ^
  --name captcha-inference-app ^
  --resource-group booster-ml ^
  --query "name" ^
  --output tsv

# 출력:
# captcha-inference-app
```

### 6.3 Azure에 배포

```bash
func azure functionapp publish captcha-inference-app

# 진행 상황:
# Getting site publishing info...
# Preparing archive...
# Uploading ... 100%
# Deployment successful
```

**배포 완료!**

---

## 7. 배포 후 검증

### 7.1 배포된 함수 URL 확인

```bash
az functionapp show ^
  --name captcha-inference-app ^
  --resource-group booster-ml ^
  --query "defaultHostName" ^
  --output tsv

# 출력:
# captcha-inference-app-f0ejcugkgqgvh4e9.koreacentral-01.azurewebsites.net
```

### 7.2 배포된 함수 헬스 체크

```bash
curl https://captcha-inference-app-f0ejcugkgqgvh4e9.koreacentral-01.azurewebsites.net/api/health

# 응답:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "device": "cpu"
# }
```

### 7.3 배포된 함수로 추론 테스트

```bash
curl -X POST https://captcha-inference-app-f0ejcugkgqgvh4e9.koreacentral-01.azurewebsites.net/api/InferenceHttpTrigger ^
  -F "image=@test_image.png"
```

---

## 8. API 사용 가이드

### 8.1 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/InferenceHttpTrigger` | POST | 캡차 인식 |
| `/api/health` | GET | 헬스 체크 |

### 8.2 요청 형식

**URL**: `https://captcha-inference-app-f0ejcugkgqgvh4e9.koreacentral-01.azurewebsites.net/api/InferenceHttpTrigger`

**Method**: POST

**Content-Type**: multipart/form-data

**파라미터**:
- Field name: `image`
- Field value: 이미지 파일 (PNG, JPG, BMP)

### 8.3 cURL 예시

```bash
curl -X POST ^
  "https://captcha-inference-app-f0ejcugkgqgvh4e9.koreacentral-01.azurewebsites.net/api/InferenceHttpTrigger" ^
  -F "image=@captcha.png"
```

### 8.4 Python 예시

```python
import requests

url = "https://captcha-inference-app-f0ejcugkgqgvh4e9.koreacentral-01.azurewebsites.net/api/InferenceHttpTrigger"

with open("captcha.png", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

result = response.json()
print(f"인식 결과: {result['predicted_text']}")
print(f"신뢰도: {result['confidence']:.2%}")
```

### 8.5 JavaScript 예시

```javascript
const formData = new FormData();
formData.append("image", imageFile);

const response = await fetch(
  "https://captcha-inference-app-f0ejcugkgqgvh4e9.koreacentral-01.azurewebsites.net/api/InferenceHttpTrigger",
  { method: "POST", body: formData }
);

const result = await response.json();
console.log(result);
```

### 8.6 응답 형식

**성공 (HTTP 200)**:
```json
{
  "status": "success",
  "predicted_text": "ABCDEF",
  "confidence": 0.9523,
  "processing_time_ms": 145.2
}
```

**실패 (HTTP 400/500)**:
```json
{
  "status": "error",
  "message": "오류 메시지"
}
```

---

## 📋 완료 체크리스트

```
[ ] Azure CLI 설치 및 로그인 완료
[ ] Azure Functions Core Tools 설치 완료
[ ] PyCharm 프로젝트 생성 완료
[ ] function_app.py 코드 작성 완료
[ ] requirements.txt 작성 완료
[ ] host.json 작성 완료
[ ] local.settings.json 작성 완료
[ ] model_best.pt 파일 복사 완료
[ ] 로컬 테스트 성공
[ ] Azure 배포 완료
[ ] 헬스 체크 성공
[ ] API 추론 테스트 성공
```

---

## 🆘 문제 해결

### "모델 파일을 찾을 수 없습니다"

**해결:**
1. PyCharm에서 `model/model_best.pt` 파일 확인
2. 파일이 없으면 복사
3. 재배포: `func azure functionapp publish captcha-inference-app`

### "Could not connect to the local Azure Function"

**해결:**
```bash
# 포트 충돌 확인
netstat -ano | findstr :7071

# 다른 포트로 실행
func start --port 7072
```

### "모듈을 찾을 수 없습니다 (ModuleNotFoundError)"

**해결:**
```bash
# 의존성 재설치
pip install --upgrade -r requirements.txt

# 캐시 삭제 후 재설치
pip cache purge
pip install -r requirements.txt
```

---

## 📞 지원 리소스

- [Azure Functions 문서](https://learn.microsoft.com/en-us/azure/azure-functions/)
- [Azure CLI 참고서](https://learn.microsoft.com/en-us/cli/azure/)
- [Azure Portal](https://portal.azure.com)

---

**축하합니다! 이제 Azure Function App에서 캡차 인식 API를 운영할 수 있습니다!** 🎉
