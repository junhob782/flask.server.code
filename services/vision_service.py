#AI 두뇌의 핵심 로직 파일

# services/vision_service.py

import torch
import cv2
import timm
from torchvision import transforms

# --- 1. 모델과 전처리기 준비 ---
# 이 부분은 서버가 시작될 때 단 한 번만 실행되어 모델을 미리 메모리에 올려둡니다.
# 마치 컴퓨터 부팅 시 필요한 프로그램을 미리 켜두는 것과 같습니다.

# 설정 파일에서 모델 이름 가져오기
from config import MODEL_NAME, NUM_CLASSES

# 모델 로딩 함수
def load_parking_model(model_path='ml_models/best_parking_classifier.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Vision Service: 사용하는 디바이스 -> {device.type}")
    
    # 빈 모델 구조를 먼저 만듭니다. (학습 때와 동일한 구조)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    
    # 학습된 가중치(지식)를 불러와 모델에 덮어씌웁니다.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # 추론 모드로 설정 (매우 중요!)
    
    print("Vision Service: 주차 공간 분류 모델 로딩 완료.")
    return model, device

# 이미지 전처리 함수 (학습 때와 100% 동일해야 함)
def get_preprocess_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- 2. 실제 분석을 수행하는 함수 ---

# 모델과 전처리기를 전역 변수로 선언하여 미리 로드
PARKING_MODEL, DEVICE = load_parking_model()
PREPROCESS_TRANSFORM = get_preprocess_transform()
CLASS_NAMES = ['empty', 'occupied'] # 학습 시 ImageFolder가 정렬한 순서 (알파벳 순)

def analyze_parking_slots(cropped_images):
    """
    잘라낸(cropped) 여러 개의 주차 공간 이미지를 받아,
    각각의 점유 상태('occupied' 또는 'empty')를 분석하여 반환합니다.
    
    :param cropped_images: (list) OpenCV 이미지(Numpy array)들의 리스트
    :return: (list) 각 이미지에 대한 예측 결과 문자열('occupied' or 'empty') 리스트
    """
    if not cropped_images:
        return []

    # 1. 모든 이미지를 AI가 이해할 수 있는 텐서(Tensor)로 변환
    batch = torch.stack([PREPROCESS_TRANSFORM(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in cropped_images]).to(DEVICE)

    # 2. 모델에게 이미지 묶음을 보여주고 예측 요청 (추론)
    with torch.no_grad(): # 추론 시에는 그래디언트 계산이 필요 없으므로 성능 향상
        outputs = PARKING_MODEL(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

    # 3. 숫자 예측 결과(0 또는 1)를 사람이 이해할 수 있는 글자('empty' or 'occupied')로 변환
    predictions = [CLASS_NAMES[p.item()] for p in preds]
    
    return predictions