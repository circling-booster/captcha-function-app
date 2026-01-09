import azure.functions as func
import logging
import json
import base64
import io
import time
import torch
import numpy as np
from PIL import Image

# shared 패키지에서 임포트
from shared.model_manager import ModelManager, MODEL_CONFIGS, IDX_TO_CHAR, BLANK_LABEL

bp = func.Blueprint()
logger = logging.getLogger("InferenceBlueprint")

# ==================== 이미지 처리 유틸리티 ====================

def preprocess_image(image_bytes, model_type: str) -> torch.Tensor:
    """이미지 전처리"""
    try:
        config = MODEL_CONFIGS[model_type]
        image_width = config["width"]
        image_height = config["height"]

        if isinstance(image_bytes, str):
            try:
                image_bytes = base64.b64decode(image_bytes)
            except Exception:
                pass 

        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((image_width, image_height), Image.BILINEAR)

        image_np = np.array(image, dtype=np.float32)
        image_np = image_np / 255.0

        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    except Exception as e:
        logger.error(f"❌ 이미지 전처리 실패: {e}")
        raise

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

@bp.route(route="InferenceHttpTrigger", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def infer_captcha(req: func.HttpRequest) -> func.HttpResponse:
    """캡차 인식 수행 엔드포인트"""
    start_time = time.time()

    try:
        # JSON 파싱 및 유효성 검사
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(json.dumps({"status": "error", "message": "Invalid JSON"}), status_code=400, mimetype="application/json")

        if "image" not in req_body or "url" not in req_body:
            return func.HttpResponse(json.dumps({"status": "error", "message": "Missing 'image' or 'url'"}), status_code=400, mimetype="application/json")

        image_data = req_body["image"]
        model_type = req_body["url"]

        if model_type not in MODEL_CONFIGS:
             return func.HttpResponse(json.dumps({"status": "error", "message": f"Invalid 'url'. Supported: {list(MODEL_CONFIGS.keys())}"}), status_code=400, mimetype="application/json")

        # Base64 디코딩
        try:
            if isinstance(image_data, str) and image_data.startswith("data:"):
                if ";base64," in image_data:
                    image_data = image_data.split(";base64,")[1]
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return func.HttpResponse(json.dumps({"status": "error", "message": "Base64 decode failed"}), status_code=400, mimetype="application/json")

        # 추론 과정
        image_tensor = preprocess_image(image_bytes, model_type)
        model = ModelManager.get_model(model_type)
        device = ModelManager.get_device()

        predicted_texts = infer(image_tensor, model, device)
        predicted_text = predicted_texts[0] if predicted_texts else ""

        # 신뢰도 계산
        with torch.no_grad():
            image_tensor_device = image_tensor.to(device)
            logits = model(image_tensor_device)
            probs = torch.softmax(logits, dim=2)
            confidence = float(probs.max().item())

        processing_time_ms = (time.time() - start_time) * 1000

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "predicted_text": predicted_text,
                "model_type": model_type,
                "confidence": round(confidence, 4),
                "processing_time_ms": round(processing_time_ms, 1)
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )