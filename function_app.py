import azure.functions as func

import json
import logging
import base64
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

NUM_CLASSES = 26
BLANK_LABEL = 26
ALPHABETS = string.ascii_uppercase
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABETS)}

# 모델별 이미지 크기 설정
MODEL_CONFIGS = {
    "melon": {"width": 230, "height": 70, "model_file": "model_melon.pt"},
    "nol": {"width": 210, "height": 70, "model_file": "model_nol.pt"}
}

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
    """모델 로딩 및 관리 (모델별 캐싱)"""

    _models = {}
    _device = None
    _last_load_time = {}

    @classmethod
    def get_model(cls, model_type: str):
        """모델 타입에 따라 모델 로드 또는 캐시에서 반환"""

        # 유효한 모델 타입 확인
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"유효하지 않은 모델 타입: {model_type}. 지원되는 타입: {list(MODEL_CONFIGS.keys())}")

        if model_type not in cls._models:
            cls.load_model(model_type)

        return cls._models[model_type]

    @classmethod
    def get_device(cls):
        """디바이스 반환 (CUDA 또는 CPU)"""
        if cls._device is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._device

    @classmethod
    def load_model(cls, model_type: str):
        """모델 로드"""
        try:
            config = MODEL_CONFIGS[model_type]
            model_path = Path(__file__).parent / config["model_file"]

            if not model_path.exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

            device = cls.get_device()

            # 이미지 높이를 기반으로 모델 생성
            model = CRNN(config["height"], NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            cls._models[model_type] = model
            cls._last_load_time[model_type] = datetime.now()

            logger.info(f"✓ 모델 로드 완료 [{model_type}]: {model_path}")
            logger.info(f"  - 이미지 크기: {config['width']}x{config['height']}")
            logger.info(f"  - 디바이스: {device}")

        except Exception as e:
            logger.error(f"❌ 모델 로드 실패 [{model_type}]: {e}")
            raise


# ==================== 이미지 처리 ====================

def preprocess_image(image_bytes, model_type: str) -> torch.Tensor:
    """이미지 전처리 (모델 타입에 따른 크기 조정)"""
    try:
        config = MODEL_CONFIGS[model_type]
        image_width = config["width"]
        image_height = config["height"]

        # Base64로 인코딩된 경우 디코딩
        if isinstance(image_bytes, str):
            try:
                image_bytes = base64.b64decode(image_bytes)
            except Exception:
                pass  # 이미 바이너리 데이터인 경우

        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((image_width, image_height), Image.BILINEAR)

        image_np = np.array(image, dtype=np.float32)
        image_np = image_np / 255.0

        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)

        logger.info(f"✓ 이미지 전처리 완료 [{model_type}]: tensor shape {image_tensor.shape}")

        return image_tensor

    except Exception as e:
        logger.error(f"❌ 이미지 전처리 실패: {e}")
        raise


# ==================== 추론 ====================

def ctc_decode(predictions):
    """CTC 디코딩"""
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
    """추론 수행"""
    try:
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            predictions = ctc_decode(logits)

        return predictions if predictions else [""]

    except Exception as e:
        logger.error(f"❌ 추론 실패: {e}")
        raise


# ==================== HTTP Trigger ====================

app = func.FunctionApp()


@app.route(route="InferenceHttpTrigger", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def infer_captcha(req: func.HttpRequest) -> func.HttpResponse:
    """
    이미지를 Base64 또는 Data URL 형식 JSON POST로 받아 캡차 인식 수행

    요청 형식:
    {
        "image": "data:image/png;base64,iVBORw0KGgo..." 또는 "iVBORw0KGgo...",
        "url": "melon" 또는 "nol"
    }
    """
    start_time = time.time()

    try:
        logger.info("📥 추론 요청 수신")

        # JSON 바디 파싱
        try:
            req_body = req.get_json()
        except ValueError:
            logger.warning("❌ JSON 파싱 실패")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": "유효한 JSON 형식이 아닙니다"
                }),
                status_code=400,
                mimetype="application/json"
            )

        # 필수 파라미터 확인
        if "image" not in req_body:
            logger.warning("❌ 'image' 필드가 없습니다")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": "'image' 필드가 필요합니다 (base64 형식 또는 Data URL)"
                }),
                status_code=400,
                mimetype="application/json"
            )

        if "url" not in req_body:
            logger.warning("❌ 'url' 필드가 없습니다")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": "'url' 필드가 필요합니다 ('melon' 또는 'nol')"
                }),
                status_code=400,
                mimetype="application/json"
            )

        image_data = req_body["image"]
        model_type = req_body["url"]

        # 모델 타입 유효성 확인
        if model_type not in MODEL_CONFIGS:
            logger.warning(f"❌ 유효하지 않은 모델 타입: {model_type}")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": f"유효하지 않은 'url' 값입니다. 지원되는 값: {list(MODEL_CONFIGS.keys())}"
                }),
                status_code=400,
                mimetype="application/json"
            )

        # Base64 이미지 데이터 검증
        if not image_data:
            logger.warning("❌ 이미지 데이터가 비어있습니다")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": "이미지 데이터가 비어있습니다"
                }),
                status_code=400,
                mimetype="application/json"
            )

        logger.info(f"📤 요청 정보 - 모델: {model_type}, 이미지 크기: {len(image_data)} chars")

        # Base64 디코딩 (Data URL 형식 지원)
        try:
            # Data URL 형식 처리 (예: data:image/png;base64,iVBORw0KGgo...)
            if isinstance(image_data, str) and image_data.startswith("data:"):
                # Data URL에서 base64 부분만 추출
                if ";base64," in image_data:
                    extracted_data = image_data.split(";base64,")[1]
                    logger.info(f"✓ Data URL 형식에서 base64 추출 완료")
                    image_data = extracted_data
                else:
                    logger.warning("❌ Data URL 형식이지만 ;base64, 구분자가 없습니다")
                    return func.HttpResponse(
                        json.dumps({
                            "status": "error",
                            "message": "Data URL 형식이 올바르지 않습니다. 'data:image/...;base64,...' 형식이어야 합니다"
                        }),
                        status_code=400,
                        mimetype="application/json"
                    )

            image_bytes = base64.b64decode(image_data)
            logger.info(f"✓ Base64 디코딩 완료: {len(image_bytes)} bytes")
        except Exception as e:
            logger.warning(f"❌ Base64 디코딩 실패: {e}")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": "Base64 디코딩 실패. 유효한 Base64 형식 또는 Data URL 형식(data:image/...;base64,...)인지 확인하세요"
                }),
                status_code=400,
                mimetype="application/json"
            )

        # 이미지 전처리
        logger.info(f"🔄 이미지 전처리 중... [{model_type}]")
        image_tensor = preprocess_image(image_bytes, model_type)
        logger.info(f"✓ 이미지 텐서 shape: {image_tensor.shape}")

        # 모델 로드
        logger.info(f"🔄 모델 로드 중... [{model_type}]")
        model = ModelManager.get_model(model_type)
        device = ModelManager.get_device()
        logger.info(f"✓ 모델 준비 완료")

        # 추론 수행
        logger.info("🔄 추론 수행 중...")
        predicted_texts = infer(image_tensor, model, device)
        predicted_text = predicted_texts[0] if predicted_texts else ""
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
            "model_type": model_type,
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


@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """헬스 체크 엔드포인트"""
    try:
        model_status = {}

        for model_type in MODEL_CONFIGS.keys():
            try:
                model = ModelManager.get_model(model_type)
                model_status[model_type] = {
                    "status": "loaded",
                    "last_load_time": str(ModelManager._last_load_time.get(model_type, "N/A"))
                }
            except Exception as e:
                model_status[model_type] = {
                    "status": "not_loaded",
                    "error": str(e)
                }

        device = ModelManager.get_device()

        return func.HttpResponse(
            json.dumps({
                "status": "healthy",
                "device": str(device),
                "models": model_status
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