# plate_crop.py
import cv2
import numpy as np
from typing import Optional, Tuple

def _largest_quad_contour(contours, min_area=800.0, max_area_ratio=0.9):
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area
    return best

def _imread_unicode(path: str):
    """윈도우 한글 경로 대응: np.fromfile + cv2.imdecode"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def crop_plate(
    img_path: str,
    out_path: Optional[str] = None,
    return_bbox: bool = False
) -> Optional[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    # ★ 절대 np.open 쓰지 마세요! (내장 open 또는 아래 방식 사용)
    img = _imread_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {img_path}")

    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate_quad = _largest_quad_contour(contours, min_area=0.0005 * (W*H))
    roi = None
    bbox = None

    def _pad_and_crop(x,y,w,h):
        pad_x = int(0.03 * W)
        pad_y = int(0.03 * H)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        return img[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

    if plate_quad is not None:
        x,y,w,h = cv2.boundingRect(plate_quad)
        roi, bbox = _pad_and_crop(x,y,w,h)
    else:
        # 사각형 실패 시, 비율/크기 기반 후보 선택
        candidates = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            area = w*h
            if area < 0.0005*(W*H):
                continue
            ratio = (w / float(h)) if h > 0 else 0
            if 2.5 <= ratio <= 6.5:  # 가로형 번호판 가정
                candidates.append((area, (x,y,w,h)))
        if candidates:
            candidates.sort(reverse=True)  # 넓은 후보 우선
            _, (x,y,w,h) = candidates[0]
            roi, bbox = _pad_and_crop(x,y,w,h)

    if roi is None:
        return None

    # 약간의 선명도 보정
    roi = cv2.GaussianBlur(roi, (3,3), 0)
    roi = cv2.bilateralFilter(roi, 9, 75, 75)

    if out_path:
        # imwrite도 한글 경로 이슈가 있을 수 있어 np/encode 방식 사용
        ok, enc = cv2.imencode('.png', roi)
        if ok:
            enc.tofile(out_path)  # 한글 경로 안전 저장
        else:
            raise RuntimeError("이미지 인코딩 실패")

    if return_bbox:
        return roi, bbox
    return roi
