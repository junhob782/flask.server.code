import torch

# YOLOv12 모델 가중치 파일 경로
MODEL_WEIGHTS = "yolo12n.pt"  # 또는 yolov12s.pt, yolov12m.pt 등

# 비디오 입력 소스 (웹캠 인덱스 또는 파일/스트림 경로)
VIDEO_SOURCE = r'C:\Users\hanhw\capstonedesign\lotbot_server\videos\2.mp4'  # 0: Default camera, "/path/to/video.mp4"

# 추론 및 상태 판정을 위한 임계값
CONF_THRESHOLD = 0.3  # 탐지 신뢰도 최소값
IOU_THRESHOLD = 0.1   # 슬롯 점유 판정 기준 IoU

# 주차 슬롯 ROI 정의: (slot_id, x1, y1, x2, y2)
PARKING_SLOTS = [
    (1, 448, 428, 84, 484),
    (2, 860, 479, 1113, 426),
    # 필요시 추가
]

SLOT_ROIS = [
    (448, 428, 84, 484),
    (860, 479, 1113, 426),

    # … 슬롯 개수만큼 추가
]

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLOv12 입력 크기(픽셀)
IMGSZ = 640