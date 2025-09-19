import cv2
import numpy as np
from ultralytics import YOLO
from spot.config import MODEL_WEIGHTS, DEVICE, IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD, SLOT_ROIS

# Load the YOLO model once at import time
def _load_model(weights: str, device: str) -> YOLO:
    model = YOLO(weights)
    model.to(device)
    model.fuse()  # Fuse Conv+BatchNorm for optimized inference
    return model

_model = _load_model(MODEL_WEIGHTS, DEVICE)


def compute_iou(boxA: tuple, boxB: tuple) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    boxA, boxB: (x1, y1, x2, y2)
    Returns IoU float in [0,1].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    unionArea = areaA + areaB - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


def get_slot_status(frame: np.ndarray) -> dict[int, bool]:
    """
    Given a BGR image frame, detect vehicles and determine
    the occupancy status of each predefined parking slot.

    Args:
        frame: np.ndarray, BGR image
    Returns:
        status: dict mapping slot index to occupied flag (True if occupied)
    """
    # Perform inference (first result only)
    results = _model(frame, device=DEVICE, imgsz=IMGSZ,
                     conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)[0]

    # Extract detection arrays
    xyxy = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, 'xyxy') else np.zeros((0, 4))
    confs = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, 'conf') else np.zeros(0)
    clss = results.boxes.cls.cpu().numpy() if hasattr(results.boxes, 'cls') else np.zeros(0)

    # Initialize all slots as empty
    status: dict[int, bool] = {i: False for i in range(len(SLOT_ROIS))}

    # Iterate detections
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
        if int(cls) != 2 or conf < CONF_THRESHOLD:
            continue
        det_box = (int(x1), int(y1), int(x2), int(y2))
        # Check IoU with each ROI slot
        for idx, roi in enumerate(SLOT_ROIS):
            if status[idx]:
                continue
            if compute_iou(det_box, roi) >= IOU_THRESHOLD:
                status[idx] = True
                break

    return status