import azure.functions as func
import logging

bp = func.Blueprint()

@bp.route(route="get_client_ip", auth_level=func.AuthLevel.ANONYMOUS)
def get_client_ip(req: func.HttpRequest) -> func.HttpResponse:
    """
    클라이언트 IP 확인 엔드포인트
    route="get_client_ip" -> /api/get_client_ip
    """
    logging.info('클라이언트 IP 확인 요청 처리')

    forwarded_for = req.headers.get('x-forwarded-for')
    client_ip = "Unknown"

    if forwarded_for:
        client_ip = forwarded_for.split(',')[0].strip()
        if ':' in client_ip:
            client_ip = client_ip.split(':')[0]
    else:
        logging.warning("x-forwarded-for 헤더 없음 (로컬 실행 가능성)")

    logging.info(f"Detected Client IP: {client_ip}")

    return func.HttpResponse(
        f"Your Client IP is: {client_ip}",
        status_code=200
    )