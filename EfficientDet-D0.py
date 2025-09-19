
import os, glob, time, json
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
import torch

from effdet import create_model
from effdet.data import resolve_input_config

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
try:
    torch.set_num_threads(4)
except Exception:
    pass


# -----------------------------
# 0) 경로/설정
# -----------------------------
VIDEO_DIR = r"C:\Users\hanhw\capstonedesign\lotbot_server\videos"
VIDEO_BASENAME = "1"  # "1"이라는 이름의 파일(확장자 상관없이)
OUTPUT_DIR = os.path.join(VIDEO_DIR, "outputs_effdet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 슬롯 ROI: (x1,y1) (x2,y2)
SLOT_ROIS: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    1: ((445, 614), (533, 693)),
    2: ((576, 617), (705, 691)),
    3: ((817, 617), (995, 692)),
}

# 점유 판정 파라미터
VEHICLE_CATEGORIES = {2, 3, 5, 7}  # car(2/3), moto(3/4), bus(5/6), truck(7/8)
STRICT_VEHICLE_ONLY = True        # True면 위 카테고리만 통과
SCORE_THRES = 0.25                 # 감지 민감도
IOA_SLOT_THRES = 0.12              # (교차/슬롯면적)
IOA_BOX_THRES = 0.30               # (교차/박스면적)
CENTER_HIT = True                  # 박스 중심점이 슬롯 안이면 점유

IOU_THRES      = 0.15               # 새로 추가: IoU 기준
IOA_SLOT_THRES = 0.10               # 0.12 → 0.10
IOA_BOX_THRES  = 0.25               # 0.30 → 0.25
CENTER_HIT     = True
BOTTOM_HIT     = True               # 새로 추가: 박스 바닥 중심점이 ROI 안이면 점유
SLOT_INFLATE   = 8                  # 새로 추가: ROI를 바깥으로 픽셀만큼 키워서 관대하게
DRAW_DEBUG     = True

# 모델 이름
MODEL_NAME = "tf_efficientdet_d0"

# -----------------------------
# 1) 유틸
# -----------------------------
def find_video_file(video_dir: str, basename: str) -> str:
    matches = sorted(glob.glob(os.path.join(video_dir, basename + ".*")))
    if not matches:
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_dir}\\{basename}.*")
    return matches[0]

def _to_np(x):
    import numpy as np, torch
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def normalize_effdet_pred(pred):
    """
    EffDet DetBenchPredict 출력 다양성을 흡수 → (boxes[N,4], scores[N], labels[N]) 반환
    """
    import numpy as np, torch
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    labels = np.zeros((0,), dtype=np.int64)

    p0 = pred[0] if isinstance(pred, (list, tuple)) else pred

    if isinstance(p0, dict):
        if "detections" in p0:
            det = _to_np(p0["detections"])
            if det is not None and det.size:
                return det[:, :4].astype(np.float32), det[:, 4].astype(np.float32), det[:, 5].astype(np.int64)
        b = _to_np(p0.get("boxes"))
        s = _to_np(p0.get("scores"))
        l = _to_np(p0.get("labels"))
        if b is not None and s is not None and l is not None:
            return b.astype(np.float32), s.astype(np.float32), l.astype(np.int64)

    if torch.is_tensor(p0) or isinstance(p0, np.ndarray):
        arr = _to_np(p0)
        if arr is not None and arr.ndim == 2 and arr.size:
            if arr.shape[1] >= 6:  # x1 y1 x2 y2 score cls
                return arr[:, :4].astype(np.float32), arr[:, 4].astype(np.float32), arr[:, 5].astype(np.int64)
            elif arr.shape[1] == 4:  # 박스만 온 경우(예외)
                return arr.astype(np.float32), np.ones((arr.shape[0],), np.float32), np.zeros((arr.shape[0],), np.int64)

    return boxes, scores, labels

def inflate(rect, d=0):
    x1, y1, x2, y2 = rect
    return (x1 - d, y1 - d, x2 + d, y2 + d)

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    if inter == 0: return 0.0
    a_area = max(1.0, (ax2-ax1)*(ay2-ay1))
    b_area = max(1.0, (bx2-bx1)*(by2-by1))
    return inter / (a_area + b_area - inter)

def letterbox_resize_pad(img: np.ndarray, new_hw: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """비율 유지 리사이즈 + 패딩. 반환: (패딩 이미지, 스케일, (pad_left, pad_top))"""
    h, w = img.shape[:2]
    new_h, new_w = new_hw
    scale = min(new_w / w, new_h / h)
    scaled_w, scaled_h = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - scaled_w
    pad_h = new_h - scaled_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    out = np.zeros((new_h, new_w, 3), dtype=img.dtype)
    out[pad_top:pad_top + scaled_h, pad_left:pad_left + scaled_w] = img_resized
    return out, scale, (pad_left, pad_top)

def undo_letterbox_on_boxes(boxes_xyxy: np.ndarray, scale: float, pad_xy: Tuple[int, int]) -> np.ndarray:
    """모델 입력 좌표계 → 원본 프레임 좌표계"""
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    px, py = pad_xy
    boxes = boxes_xyxy.copy().astype(np.float32)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
    return boxes

def inter_metrics(slot_xyxy, box_xyxy):
    """교차면적, (교차/슬롯면적), (교차/박스면적)"""
    sx1, sy1, sx2, sy2 = slot_xyxy
    bx1, by1, bx2, by2 = box_xyxy
    x1 = max(sx1, bx1)
    y1 = max(sy1, by1)
    x2 = min(sx2, bx2)
    y2 = min(sy2, by2)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    slot_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
    box_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter, (inter / slot_area), (inter / box_area)

def inside(xy: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
    x, y = xy
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

def to_xyxy(tl: Tuple[int, int], br: Tuple[int, int]) -> Tuple[int, int, int, int]:
    return (int(tl[0]), int(tl[1]), int(br[0]), int(br[1]))

def draw_slots(frame: np.ndarray,
               statuses: Dict[int, bool],
               slot_rois_xyxy: Dict[int, Tuple[int, int, int, int]],
               thickness: int = 2) -> None:
    """
    statuses[sid] == True  → Occupied(빨강)
    statuses[sid] == False → Empty(초록)
    """
    for sid, (x1, y1, x2, y2) in slot_rois_xyxy.items():
        occ = bool(statuses.get(sid, False))
        color = (0, 0, 255) if occ else (0, 255, 0)
        label = "Occupied" if occ else "FREE"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"S{sid}:{label}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# -----------------------------
# 2) 모델 로드
# -----------------------------

def load_detector(device: torch.device):
    bench = create_model(
        MODEL_NAME,
        pretrained=True,
        bench_task='predict',  # DetBenchPredict
        num_classes=90         # COCO
    )
    bench = bench.to(device)
    bench.eval()

    # ★ 속도 스위치 (CPU에서 반드시!)
    if device.type == "cpu":
        bench.max_detection_points = 1500   # 5000 -> 1500
        bench.max_det_per_image    = 50     # 100  -> 50
    else:
        bench.max_detection_points = 3000   # GPU면 조금 넉넉히
        bench.max_det_per_image    = 100

    input_cfg = resolve_input_config({}, bench.config)
    if isinstance(input_cfg["input_size"], (tuple, list)):
        _, in_h, in_w = input_cfg["input_size"]
    else:
        in_h = in_w = int(input_cfg["input_size"])

    mean = np.array(input_cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(input_cfg["std"], dtype=np.float32).reshape(1, 1, 3)
    fill = input_cfg.get("fill_color", 0)

    return bench, (in_h, in_w), mean, std, fill


def load_detector(device: torch.device):
    bench = create_model(
        MODEL_NAME,
        pretrained=True,
        bench_task='predict',  # DetBenchPredict
        num_classes=90         # COCO
    )
    bench = bench.to(device)
    bench.eval()

    input_cfg = resolve_input_config({}, bench.config)
    if isinstance(input_cfg["input_size"], (tuple, list)):
        _, in_h, in_w = input_cfg["input_size"]
    else:
        in_h = in_w = int(input_cfg["input_size"])

    mean = np.array(input_cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(input_cfg["std"], dtype=np.float32).reshape(1, 1, 3)
    fill = input_cfg.get("fill_color", 0)

    return bench, (in_h, in_w), mean, std, fill

# -----------------------------
# 3) 추론 루프
# -----------------------------
def main():
    # 비디오 찾기
    video_path = find_video_file(VIDEO_DIR, VIDEO_BASENAME)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {video_path}")

    # 출력 비디오 준비
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path = os.path.join(OUTPUT_DIR, f"{Path(video_path).stem}_effdet.avi")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():  # 코덱 폴백
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(OUTPUT_DIR, f"{Path(video_path).stem}_effdet.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # 상태 로그(JSON Lines)
    status_path = os.path.join(OUTPUT_DIR, f"{Path(video_path).stem}_status.jsonl")
    status_f = open(status_path, "w", encoding="utf-8")

    # 디바이스/모델
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench, (IN_H, IN_W), mean, std, _ = load_detector(device)

    # ROI 사각형(원본 좌표, xyxy)
    slots_xyxy: Dict[int, Tuple[int, int, int, int]] = {
        i: to_xyxy(*tlbr) for i, tlbr in SLOT_ROIS.items()
    }

    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # 1) 전처리
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        lb_img, scale, (pad_left, pad_top) = letterbox_resize_pad(frame_rgb / 255.0, (IN_H, IN_W))
        norm = (lb_img - mean) / std
        inp = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # 2) 추론
        with torch.inference_mode():
            pred = bench(inp)

        # 2-1) 정규화
        boxes, scores, labels = normalize_effdet_pred(pred)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1,)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1,)

        # 2-2) 길이 맞추기 + 필터
        n = min(len(boxes), len(scores), len(labels))
        if n:
            boxes, scores, labels = boxes[:n], scores[:n], labels[:n]
            if STRICT_VEHICLE_ONLY:
                keep = (scores >= SCORE_THRES) & np.isin(labels, list(VEHICLE_CATEGORIES))
            else:
                keep = (scores >= SCORE_THRES)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # 2-3) 좌표 복원
            boxes = undo_letterbox_on_boxes(boxes, scale, (pad_left, pad_top))
            boxes = np.clip(boxes, [0, 0, 0, 0], [W, H, W, H])
        else:
            boxes = boxes[:0]
            scores = scores[:0]
            labels = labels[:0]

        # 3) 슬롯 점유 판정
        # 3) 슬롯 점유 판정 (중심/바닥중심/IoU/IoA 중 하나라도 TRUE면 점유)
        occ: Dict[int, bool] = {}
        for sid, sxyxy in slots_xyxy.items():
            sx1, sy1, sx2, sy2 = sxyxy
            slot_rect = inflate((sx1, sy1, sx2, sy2), SLOT_INFLATE)

            occupied = False
            best_dbg = None  # (iou, ioa_slot, ioa_box, score, label, center_hit, bottom_hit)

            for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
                # 중심점 / '바닥 중심점'(주차면 바닥과 맞닿는 지점이 ROI에 들어오면 강하게 인정)
                cx, cy = (0.5*(x1+x2), 0.5*(y1+y2))
                bx, by = (0.5*(x1+x2), y2)

                center_hit = CENTER_HIT and (slot_rect[0] <= cx <= slot_rect[2] and slot_rect[1] <= cy <= slot_rect[3])
                bottom_hit = BOTTOM_HIT and (slot_rect[0] <= bx <= slot_rect[2] and slot_rect[1] <= by <= slot_rect[3])

                iou_val = iou(slot_rect, (int(x1), int(y1), int(x2), int(y2)))
                _, ioa_slot, ioa_box = inter_metrics(slot_rect, (int(x1), int(y1), int(x2), int(y2)))
                io_hit = (iou_val >= IOU_THRES) or (ioa_slot >= IOA_SLOT_THRES) or (ioa_box >= IOA_BOX_THRES)

                if center_hit or bottom_hit or io_hit:
                    occupied = True
                    best_dbg = (iou_val, ioa_slot, ioa_box, float(sc), int(lb), center_hit, bottom_hit)
                    break

            occ[sid] = occupied

            # 디버그 숫자 오버레이(선택)
            if DRAW_DEBUG:
                dbg = best_dbg if best_dbg is not None else (0.0,0.0,0.0,0.0,-1,False,False)
                iou_v, ioas, ioab, scv, lbv, ch, bh = dbg
                txt = f"S{sid} {'OCC' if occupied else 'FREE'}  IoU={iou_v:.2f}  IoAs={ioas:.2f}  IoAb={ioab:.2f}  S={scv:.2f} L={lbv} CH={int(ch)} BH={int(bh)}"
                cv2.putText(frame_bgr, txt, (sx1, max(0, sy1-24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

        # 4) 시각화
        vis = frame_bgr.copy()

        # (선택) 탐지 박스: 주황색, 디버그용
        for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 80), 2)
            cv2.putText(
                vis,
                f"id{int(lb)}:{sc:.2f}",
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 80),
                1,
                cv2.LINE_AA,
            )

        # 슬롯 상태: parking_service.py처럼 ROI별로 초록/빨강
        draw_slots(vis, occ, slots_xyxy, thickness=3)

        # 상단 요약 텍스트
        free_cnt = sum(1 for v in occ.values() if not v)
        cv2.putText(
            vis,
            f"Free: {free_cnt}/{len(slots_xyxy)}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (80, 240, 80),
            2,
            cv2.LINE_AA,
        )

        for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 80), 2)
            cv2.putText(vis, f"id{int(lb)}:{sc:.2f}", (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 80), 1, cv2.LINE_AA)

            # 중심점(●), 바닥중심점(◆) 찍기
            cx, cy = int(0.5*(x1+x2)), int(0.5*(y1+y2))
            bx, by = int(0.5*(x1+x2)), int(y2)
            cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)   # 노란 점
            cv2.drawMarker(vis, (bx, by), (0, 255, 0), markerType=cv2.MARKER_DIAMOND, markerSize=8, thickness=2)  # 초록 마커

        # 5) 출력
        writer.write(vis)
        cv2.imshow("ROI: EfficientDet-D0 Parking", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 상태 로그 (한 줄당 한 프레임)
        status_line = {
            "frame": frame_idx,
            "free": free_cnt,
            "slots": {str(k): (not v) for k, v in occ.items()},  # True면 빈자리
        }
        status_f.write(json.dumps(status_line, ensure_ascii=False) + "\n")

        frame_idx += 1

    cap.release()
    writer.release()
    status_f.close()
    cv2.destroyAllWindows()

    dt = time.time() - t0
    print(f"완료: {out_path}")
    print(f"상태 로그(JSONL): {status_path}")
    print(f"총 프레임 {frame_idx} / 평균 FPS {frame_idx/max(1.0, dt):.2f}")

if __name__ == "__main__":
    main()
