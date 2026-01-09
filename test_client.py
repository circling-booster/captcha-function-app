#!/usr/bin/env python3
"""
Azure Function App 캡차 추론 테스트 클라이언트
"""

import requests
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

function_url = "https://captcha-inference-app-f0ejcugkgqgvh4e9.koreacentral-01.azurewebsites.net"

class CaptchaInferenceClient:
    """Azure Function 추론 클라이언트"""

    def __init__(self, function_url):
        self.function_url = "function_url"
        self.inference_endpoint = f"{function_url}/api/InferenceHttpTrigger"
        self.health_endpoint = f"{function_url}/api/health"

    def health_check(self):
        """헬스 체크"""
        try:
            print(f"🔍 헬스 체크: {self.health_endpoint}")
            response = requests.get(self.health_endpoint, timeout=10)

            if response.status_code == 200:
                result = response.json()
                print("✅ 헬스 체크 성공")
                print(json.dumps(result, indent=2))
                return True
            else:
                print(f"❌ 헬스 체크 실패: {response.status_code}")
                print(response.text)
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ 연결 오류: {e}")
            return False

    def infer(self, image_path):
        """캡차 인식 추론"""
        image_path = Path(image_path)

        if not image_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
            return None

        # 파일 크기 확인
        file_size_mb = image_path.stat().st_size / (1024 ** 2)
        if file_size_mb > 10:
            print(f"⚠️  파일 크기가 큽니다: {file_size_mb:.2f}MB (권장: <10MB)")

        try:
            print(f"📤 추론 요청: {image_path.name}")

            with open(image_path, "rb") as f:
                files = {"image": f}
                response = requests.post(
                    self.inference_endpoint,
                    files=files,
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                print("✅ 추론 성공")
                self._print_result(result)
                return result

            elif response.status_code == 400:
                print(f"❌ 요청 오류 (400): {response.text}")
                return None

            elif response.status_code == 500:
                print(f"❌ 서버 오류 (500): {response.text}")
                return None

            else:
                print(f"❌ 예기치 않은 응답: {response.status_code}")
                print(response.text)
                return None

        except requests.exceptions.Timeout:
            print("❌ 요청 시간 초과 (30초)")
            return None

        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 오류: {e}")
            return None

    def batch_infer(self, image_dir):
        """배치 추론"""
        image_dir = Path(image_dir)

        if not image_dir.exists():
            print(f"❌ 디렉토리를 찾을 수 없습니다: {image_dir}")
            return []

        # 이미지 파일 검색
        supported_formats = {".png", ".jpg", ".jpeg", ".bmp"}
        image_files = [
            f for f in image_dir.glob("*")
            if f.suffix.lower() in supported_formats
        ]

        if not image_files:
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_dir}")
            return []

        print(f"🔍 발견된 이미지: {len(image_files)}개")
        print("-" * 60)

        results = []
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] {image_file.name}")
            result = self.infer(image_file)
            if result:
                results.append({
                    "filename": image_file.name,
                    "result": result
                })

        return results

    def _print_result(self, result):
        """결과 출력"""
        if result.get("status") == "success":
            print(f"  📝 인식 결과: {result['predicted_text']}")
            print(f"  📊 신뢰도: {result['confidence']:.2%}")
            print(f"  ⏱️  처리시간: {result['processing_time_ms']:.1f}ms")
        else:
            print(f"  ❌ 오류: {result.get('message', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        description="Azure Function App 캡차 추론 클라이언트"
    )

    parser.add_argument(
        "function_url",
        help="Function App URL (예: https://captcha-inference-app.azurewebsites.net)"
    )

    parser.add_argument(
        "--health",
        action="store_true",
        help="헬스 체크만 수행"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="단일 이미지 파일 경로"
    )

    parser.add_argument(
        "--batch",
        type=str,
        help="이미지 디렉토리 경로 (배치 처리)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="결과 저장 파일 경로 (JSON)"
    )

    args = parser.parse_args()

    # URL 검증
    if not args.function_url.startswith(("http://", "https://")):
        print("❌ Function App URL은 http:// 또는 https://로 시작해야 합니다")
        sys.exit(1)

    client = CaptchaInferenceClient(args.function_url)

    # 헬스 체크
    if args.health:
        success = client.health_check()
        sys.exit(0 if success else 1)

    # 단일 이미지 처리
    if args.image:
        result = client.infer(args.image)
        if result and args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"💾 결과 저장: {args.output}")
        sys.exit(0 if result else 1)

    # 배치 처리
    if args.batch:
        results = client.batch_infer(args.batch)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 결과 저장: {args.output}")

        print(f"\n{'=' * 60}")
        print(f"처리 완료: {len(results)}/{len(list(Path(args.batch).glob('*')))}")
        sys.exit(0 if results else 1)

    # 기본: 헬스 체크
    if not args.image and not args.batch:
        client.health_check()


if __name__ == "__main__":
    main()