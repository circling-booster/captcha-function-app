import azure.functions as func
import json
import logging
from shared.model_manager import ModelManager, MODEL_CONFIGS

bp = func.Blueprint()

@bp.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """헬스 체크 엔드포인트"""
    try:
        model_status = {}

        # 각 모델별 로드 상태 확인
        for model_type in MODEL_CONFIGS.keys():
            try:
                # get_model을 호출하여 로딩 시도 (캐시되어 있으면 바로 반환)
                ModelManager.get_model(model_type)
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
        logging.error(f"Health check failed: {e}")
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e)
            }),
            status_code=500,
            mimetype="application/json"
        )