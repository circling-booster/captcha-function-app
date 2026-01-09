# test_new.py
import sys
import os
import base64
import json
import torch
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from function_app import (
    ModelManager,
    preprocess_image,
    infer,
    ctc_decode,
    MODEL_CONFIGS,
    logger
)

def load_test_image(image_path: str) -> bytes:
    """테스트 이미지 로드"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"테스트 이미지를 찾을 수 없습니다: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    logger.info(f"✓ 테스트 이미지 로드 완료: {image_path} ({len(image_bytes)} bytes)")
    return image_bytes

def test_inference_nol():
    """NOL 모델 추론 테스트"""
    
    print("\n" + "="*60)
    print("🧪 Azure Function App - 추론 테스트 시작 (NOL 모델)")
    print("="*60 + "\n")
    
    model_type = "melon"
    test_img_dir = project_root / "test_img"
    
    # test_img 디렉토리 확인
    if not test_img_dir.exists():
        print(f"❌ 테스트 이미지 디렉토리가 없습니다: {test_img_dir}")
        print(f"   생성하려면: mkdir {test_img_dir}")
        return
    
    # test_img 디렉토리에서 첫 번째 이미지 파일 찾기
    image_files = list(test_img_dir.glob("*.png")) + list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ test_img 디렉토리에 이미지 파일이 없습니다")
        print(f"   지원 형식: .png, .jpg, .jpeg")
        return
    
    print(f"📁 발견된 이미지 파일: {len(image_files)}개")
    
    for image_file in image_files:
        print(f"\n{'─'*60}")
        print(f"📝 테스트 이미지: {image_file.name}")
        print(f"{'─'*60}")
        
        try:
            # 1. 이미지 로드
            print(f"\n[1/5] 이미지 로드 중...")
            image_bytes = load_test_image(str(image_file))
            
            # 2. 이미지 전처리
            print(f"[2/5] 이미지 전처리 중... [{model_type}]")
            image_tensor = preprocess_image(image_bytes, model_type)
            print(f"✓ 텐서 shape: {image_tensor.shape}")
            print(f"  - 배치 크기: {image_tensor.shape[0]}")
            print(f"  - 채널: {image_tensor.shape[1]}")
            print(f"  - 높이: {image_tensor.shape[2]}")
            print(f"  - 너비: {image_tensor.shape[3]}")
            
            # 3. 모델 로드
            print(f"[3/5] 모델 로드 중... [{model_type}]")
            model = ModelManager.get_model(model_type)
            device = ModelManager.get_device()
            print(f"✓ 모델 로드 완료")
            print(f"  - 모델 타입: {model_type}")
            print(f"  - 디바이스: {device}")
            print(f"  - 이미지 크기: {MODEL_CONFIGS[model_type]['width']}x{MODEL_CONFIGS[model_type]['height']}")
            
            # 4. 추론 수행
            print(f"[4/5] 추론 수행 중...")
            predicted_texts = infer(image_tensor, model, device)
            predicted_text = predicted_texts[0] if predicted_texts else ""
            print(f"✓ 추론 결과: {predicted_text}")
            
            # 5. 신뢰도 계산
            print(f"[5/5] 신뢰도 계산 중...")
            with torch.no_grad():
                image_tensor_device = image_tensor.to(device)
                logits = model(image_tensor_device)
                probs = torch.softmax(logits, dim=2)
                confidence = float(probs.max().item())
                max_prob_idx = probs.max(dim=2)[1]
            
            print(f"✓ 신뢰도: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # 결과 출력
            print(f"\n{'='*60}")
            print(f"📊 최종 결과")
            print(f"{'='*60}")
            print(f"이미지: {image_file.name}")
            print(f"모델: {model_type}")
            print(f"인식 텍스트: {predicted_text}")
            print(f"신뢰도: {confidence:.4f}")
            print(f"{'='*60}\n")
            
        except FileNotFoundError as e:
            print(f"❌ 파일 오류: {e}")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()

def test_health_check():
    """헬스 체크 테스트"""
    print("\n" + "="*60)
    print("🏥 헬스 체크 테스트")
    print("="*60 + "\n")
    
    try:
        model_status = {}
        for model_type in MODEL_CONFIGS.keys():
            try:
                model = ModelManager.get_model(model_type)
                model_status[model_type] = {
                    "status": "loaded",
                    "last_load_time": str(ModelManager._last_load_time.get(model_type, "N/A"))
                }
                print(f"✓ {model_type}: 로드됨")
            except Exception as e:
                model_status[model_type] = {
                    "status": "not_loaded",
                    "error": str(e)
                }
                print(f"❌ {model_type}: {e}")
        
        device = ModelManager.get_device()
        print(f"\n디바이스: {device}")
        print(f"상태: healthy")
    except Exception as e:
        print(f"❌ 헬스 체크 실패: {e}")

if __name__ == "__main__":
    print("\n🚀 Azure Function App 로컬 테스트 스크립트")
    print(f"프로젝트 루트: {project_root}")
    
    # 헬스 체크 실행
    test_health_check()
    
    # NOL 모델 추론 테스트 실행
    test_inference_nol()
    
    print("\n✅ 테스트 완료!")
