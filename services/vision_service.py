import os
import torch
import timm
import cv2
from torchvision import transforms

from config import MODEL_NAME, NUM_CLASSES  # 예: MODEL_NAME='efficientnet_b0' 또는 'vit_base_patch16_224'

# -----------------------
# 유틸: 체크포인트 로딩/정제
# -----------------------
def _load_checkpoint(path, device):
    obj = torch.load(path, map_location=device)
    # 흔한 저장 형태들 호환
    if isinstance(obj, dict):
        # pytorch-lightning 형태
        for key in ('state_dict', 'model', 'model_state_dict'):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # 그냥 state_dict 자체가 저장된 경우
        return obj
    return obj  # 드물지만 state_dict가 아닌 경우 그대로 반환

def _strip_prefix(state_dict, prefixes=('module.', 'model.', 'models.')):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def _detect_arch_from_keys(state_keys):
    """키를 보고 대략적 아키텍처를 판별 (timm 기준)"""
    ks = " ".join(list(state_keys)[:300])
    # ViT 계열의 전형적인 키
    if ("cls_token" in ks and "pos_embed" in ks and "patch_embed.proj.weight" in ks) or ".attn.qkv." in ks:
        return "vit"
    # EfficientNet 전형 키
    if "conv_stem.weight" in ks and "bn1.weight" in ks:
        return "efficientnet"
    return "unknown"

def _create_model(arch_hint: str, fallback_name: str, num_classes: int):
    """
    arch_hint가 'vit' 또는 'efficientnet'이면 그 계열의 모델 생성.
    실패하면 fallback_name(MODEL_NAME)으로 생성, 그래도 실패 시 efficientnet_b0.
    """
    # 1) 힌트 기반 시도
    try:
        if arch_hint == 'vit':
            # 학습 당시 ViT 모델명과 동일해야 완전 일치. 모르면 가장 흔한 구성으로 시작.
            return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        if arch_hint == 'efficientnet':
            # EfficientNet 계열: MODEL_NAME이 efnet인 경우 그걸 우선 쓰고, 아니면 b0
            if fallback_name and 'efficientnet' in fallback_name:
                return timm.create_model(fallback_name, pretrained=False, num_classes=num_classes)
            return timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    except Exception:
        pass

    # 2) 설정상의 MODEL_NAME으로 시도
    try:
        if fallback_name:
            return timm.create_model(fallback_name, pretrained=False, num_classes=num_classes)
    except Exception:
        pass

    # 3) 최후: 사전학습 efficientnet_b0로라도 진행
    return timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

def _try_load_state(model, state, strict_first=True):
    """
    1) strict=True 로드시도 (완전 일치)
    2) 안 되면 classifier 헤드 무시하고 strict=False 로드시도
    """
    try:
        if strict_first:
            model.load_state_dict(state, strict=True)
            return True, "strict=True"
        else:
            raise RuntimeError("skip strict first")
    except Exception as e1:
        # 분류기 키 제거 후 완화 로드
        pruned = {}
        skip_tokens = ('classifier', 'head.fc', 'head.weight', 'head.bias', 'fc.weight', 'fc.bias')
        for k, v in state.items():
            if any(k.startswith(t) for t in skip_tokens):
                continue
            pruned[k] = v
        try:
            model.load_state_dict(pruned, strict=False)
            return True, f"strict=False (classifier ignored) - {str(e1)}"
        except Exception as e2:
            return False, f"strict=False also failed: {e1} / {e2}"

# -----------------------
# 전처리 (학습과 동일하게)
# -----------------------
def get_preprocess_transform():
    # ViT/EfficientNet 모두 224x224 기본
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# -----------------------
# 모델 로딩
# -----------------------
def load_parking_model(model_path='ml_models/best_parking_classifier.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Vision Service: 사용하는 디바이스 -> {device.type}")

    if not os.path.exists(model_path):
        print(f"Vision Service: 모델 파일 없음 ({model_path}) - 테스트 모드로 실행")
        return None, device

    try:
        # 1) 체크포인트 로딩 + 키 정리
        raw_state = _load_checkpoint(model_path, device)
        if not isinstance(raw_state, dict):
            print("Vision Service: 알 수 없는 체크포인트 포맷 - 사전학습 EfficientNet으로 폴백")
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
            model.to(device).eval()
            return model, device

        state = _strip_prefix(raw_state)
        arch = _detect_arch_from_keys(state.keys())
        print(f"Vision Service: 체크포인트 판별 -> {arch}")

        # 2) 힌트/설정 기반 모델 생성
        model = _create_model(arch_hint=arch, fallback_name=MODEL_NAME, num_classes=NUM_CLASSES)

        # 3) 로드 시도 (엄격→완화)
        ok, how = _try_load_state(model, state, strict_first=True)
        if not ok:
            ok, how = _try_load_state(model, state, strict_first=False)

        if ok:
            print(f"Vision Service: 모델 로딩 완료 ({how}).")
        else:
            print(f"Vision Service: 체크포인트 로딩 실패 -> 사전학습 EfficientNet 폴백")
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)

        model.to(device)
        model.eval()
        return model, device

    except Exception as e:
        print(f"Vision Service: 모델 로딩 실패 - {e}")
        # 최후 폴백
        try:
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
            model.to(device).eval()
            print("Vision Service: 사전학습 EfficientNet 폴백으로 가동")
            return model, device
        except Exception as e2:
            print(f"Vision Service: 폴백도 실패 - {e2}")
            return None, device

# -----------------------
# 분석 로직
# -----------------------
PARKING_MODEL, DEVICE = load_parking_model()
PREPROCESS_TRANSFORM = get_preprocess_transform()
CLASS_NAMES = ['empty', 'occupied']  # 학습 시 클래스 순서에 맞게 조정

def analyze_parking_slots(cropped_images):
    """
    잘라낸(cropped) 주차 공간 이미지 리스트 -> 각 슬롯의 'empty'/'occupied' 예측 반환
    """
    if not cropped_images:
        return []

    if PARKING_MODEL is None:
        print("Vision Service: 모델 없음 - 더미 결과 반환")
        return ['empty'] * len(cropped_images)

    batch = torch.stack([
        PREPROCESS_TRANSFORM(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for img in cropped_images
    ]).to(DEVICE)

    with torch.no_grad():
        outputs = PARKING_MODEL(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

    predictions = [CLASS_NAMES[p.item()] for p in preds]
    return predictions