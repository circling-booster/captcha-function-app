import azure.functions as func

# 기존 Blueprint 임포트
from blueprints.inference import bp as inference_bp
from blueprints.health import bp as health_bp
from blueprints.server_info import bp as server_info_bp # (이전 단계의 IP 확인 기능)

# [NEW] 새로 만든 server_check Blueprint 임포트
from blueprints.server_check import bp as server_check_bp

app = func.FunctionApp()

# Blueprint 등록
app.register_functions(inference_bp)
app.register_functions(health_bp)
app.register_functions(server_info_bp)

# [NEW] 등록 추가
app.register_functions(server_check_bp)