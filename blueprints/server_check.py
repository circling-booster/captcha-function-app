import azure.functions as func
import logging
import json
from itertools import zip_longest

bp = func.Blueprint()

# ==================== 설정 ====================

# 서버에서 요구하는 최소 플랫폼 버전
MIN_REQUIRED_VERSION = "1.0.0"

# 사용자 목록 (검증할 userId 리스트)
USERS = {
    "user_01": {"status": "active"},
    "booster_admin": {"status": "active"},
    "tester": {"status": "active"}
}

# ==================== 도우미 함수 ====================

def compare_versions(v1: str, v2: str) -> int:
    """
    버전 문자열 비교 함수 (Semantic Versioning 지원)
    반환값: 0(같음), 1(v1이 큼), -1(v2가 큼)
    """
    if not v1 or not v2:
        return -1
    
    # '.'을 기준으로 나누고 정수형으로 변환
    v1_parts = [int(x) for x in v1.split('.')]
    v2_parts = [int(x) for x in v2.split('.')]
    
    # zip_longest를 사용하여 긴 쪽에 맞춰 0으로 채우며 비교
    for val1, val2 in zip_longest(v1_parts, v2_parts, fillvalue=0):
        if val1 > val2:
            return 1
        if val1 < val2:
            return -1
            
    return 0

# ==================== HTTP Trigger ====================

@bp.route(route="server_check", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def server_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    앱 버전 및 사용자 유효성 검증 함수
    Endpoint: /api/server_check
    """
    logging.info('Version verification request processed.')

    try:
        # 1. 방어 코드: Body 확인
        try:
            body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Request body is empty or invalid JSON"}),
                status_code=400,
                mimetype="application/json"
            )

        if not body:
            return func.HttpResponse(
                json.dumps({"error": "Request body is empty"}),
                status_code=400,
                mimetype="application/json"
            )

        user_id = body.get("userId")
        platform_version = body.get("platformVersion")

        # 2. 필수 파라미터 확인
        if not user_id or not platform_version:
            return func.HttpResponse(
                json.dumps({
                    "status": False,
                    "message": "Missing userId or platformVersion"
                }),
                status_code=200, # 클라이언트 로직 호환을 위해 200 유지
                mimetype="application/json"
            )

        # 3. User ID 존재 여부 확인
        if user_id not in USERS:
            return func.HttpResponse(
                json.dumps({
                    "status": False,
                    "message": "Invalid User ID"
                }),
                status_code=200,
                mimetype="application/json"
            )

        # 4. 버전 비교 로직 (Client Version >= Server Min Version)
        # compare_versions 결과가 0(같음) 또는 1(큼)이면 통과
        is_version_valid = compare_versions(platform_version, MIN_REQUIRED_VERSION) >= 0

        if is_version_valid:
            return func.HttpResponse(
                json.dumps({
                    "status": True,
                    "message": "Version check passed"
                }),
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({
                    "status": False,
                    "currentVersion": platform_version,
                    "minVersion": MIN_REQUIRED_VERSION,
                    "message": "Update required"
                }),
                status_code=200,
                mimetype="application/json"
            )

    except Exception as e:
        logging.error(f"Server check error: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )