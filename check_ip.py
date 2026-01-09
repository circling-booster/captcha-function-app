import azure.functions as func
import logging

app = func.FunctionApp()

@app.route(route="get_client_ip", auth_level=func.AuthLevel.ANONYMOUS)
def get_client_ip(req: func.HttpRequest) -> func.HttpResponse:
    """
    클라이언트 IP를 확인하고 로깅하는 HTTP Trigger 함수
    
    최종 엔드포인트: https://<함수앱이름>.azurewebsites.net/api/get_client_ip
    route="get_client_ip" + host.json의 routePrefix="api" → /api/get_client_ip
    """
    logging.info('클라이언트 IP 확인 요청을 처리합니다.')

    # 1. x-forwarded-for 헤더에서 클라이언트 IP 추출
    forwarded_for = req.headers.get('x-forwarded-for')
    
    client_ip = "Unknown"

    if forwarded_for:
        # 2. 쉼표로 구분된 첫 번째 IP가 실제 클라이언트 IP
        client_ip = forwarded_for.split(',')[0].strip()
        
        # (선택) 포트 제거
        if ':' in client_ip:
            client_ip = client_ip.split(':')[0]
    else:
        logging.warning("x-forwarded-for 헤더를 찾을 수 없습니다. 로컬 실행 중일 수 있습니다.")

    # 3. IP 로깅
    logging.info(f"Detected Client IP: {client_ip}")

    # 4. 응답 반환
    return func.HttpResponse(
        f"Your Client IP is: {client_ip}",
        status_code=200
    )
