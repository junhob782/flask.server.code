import cv2
import numpy as np
from ultralytics import YOLO
import torch
from spot.config import MODEL_WEIGHTS, DEVICE, IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD, PARKING_SLOTS

# Step 1: IoU 계산 함수
def compute_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """
    두 박스의 교집합(overlap) 비율(IoU)을 계산합니다.
    boxA, boxB: (x1, y1, x2, y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea + 1e-6
    return interArea / denom if denom > 0 else 0.0

# Step 2: YOLOv12 모델 로드 및 전처리
def load_model() -> YOLO:
    """
    가중치 불러오고, 반절 정밀도(fp16)로 변환한 뒤 모델을 반환합니다.
    """
    model = YOLO(MODEL_WEIGHTS)
    model.fuse()  # conv + bn 합치기
    if DEVICE == "cuda":
        model.model.half()
    return model

MODEL = load_model()

# Step 3: 차량 검출 함수
def detect_vehicles(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    frame에 대해 YOLO 추론을 수행하고,
    car/motorcycle/bus/truck 클래스만 반환합니다.
    """
    results = MODEL.predict(
        source=frame,
        imgsz=IMGSZ,
        device=DEVICE,
        half=(DEVICE == "cuda"),
        conf=CONF_THRESHOLD,
        verbose=False
    )
    vehicle_boxes: list[tuple[int, int, int, int]] = []
    for res in results:
        for box in res.boxes:
            cls_id = int(box.cls.cpu().item())
            # COCO 클래스: car=2, motorcycle=3, bus=5, truck=7
            if cls_id in {2, 3, 5, 7}:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                vehicle_boxes.append((x1, y1, x2, y2))
    return vehicle_boxes

# Step 4: 슬롯 상태 판정
def get_slot_status(frame: np.ndarray) -> dict[int, bool]:
    """
    주차 프레임을 받아, 각 슬롯이 점유되었는지 여부를 반환합니다.
    반환값: {slot_id: occupied_flag}
    """
    vehicles = detect_vehicles(frame)
    status: dict[int, bool] = {}

    # PARKING_SLOTS: [(slot_id, x1, y1, x2, y2), ...] (spot.config에서 정의)
    for slot in PARKING_SLOTS:
        slot_id, x1, y1, x2, y2 = slot
        slot_box = (x1, y1, x2, y2)

        # 차량 박스 중 하나라도 IOU가 임계치 이상이면 점유로 간주
        occupied = any(
            compute_iou(slot_box, vb) >= IOU_THRESHOLD
            for vb in vehicles
        )
        status[slot_id] = occupied

    return status