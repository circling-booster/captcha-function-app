import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
import string

# ==================== 설정 ====================

NUM_CLASSES = 26
BLANK_LABEL = 26
ALPHABETS = string.ascii_uppercase
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABETS)}

# 모델별 설정
MODEL_CONFIGS = {
    "melon": {"width": 230, "height": 70, "model_file": "model_melon.pt"},
    "nol": {"width": 210, "height": 70, "model_file": "model_nol.pt"}
}

logger = logging.getLogger("ModelManager")

# ==================== 모델 정의 (CRNN) ====================

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

# ==================== 모델 관리자 ====================

class ModelManager:
    """모델 로딩 및 관리 (모델별 캐싱)"""

    _models = {}
    _device = None
    _last_load_time = {}

    @classmethod
    def get_model(cls, model_type: str):
        """모델 타입에 따라 모델 로드 또는 캐시에서 반환"""
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
            
            # [수정됨] 현재 파일(shared/model_manager.py)의 상위 폴더의 상위 폴더 -> models 폴더 참조
            base_path = Path(__file__).parent.parent 
            model_path = base_path / "models" / config["model_file"]

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