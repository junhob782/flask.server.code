# crop_plate_with_vision.py
import os, sys, glob, argparse, pathlib, io, math
from typing import List, Tuple
from PIL import Image
import cv2
import numpy as np

# === 항상 이 경로로 저장되도록 기본값 고정 ===
DEFAULT_OUTDIR = r"C:\Users\hanhw\capstonedesign\lotbot_server\test_images\crops"

# ---------------- Vision Client ----------------
def get_vision_client():
    try:
        from google.cloud import vision
        from google.api_core.client_options import ClientOptions
    except Exception as e:
        print("[ERR] google-cloud-vision import 실패:", e, file=sys.stderr)
        print("      pip install google-cloud-vision", file=sys.stderr)
        sys.exit(3)
    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if api_key:
        return vision.ImageAnnotatorClient(client_options=ClientOptions(api_key=api_key))
    else:
        return vision.ImageAnnotatorClient()

CLIENT = None

def ensure_env():
    if not os.getenv("GOOGLE_VISION_API_KEY") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("[WARN] Vision 자격정보가 보이지 않습니다 (API_KEY 또는 ADC 필요)", file=sys.stderr)

# ---------------- I/O helpers ----------------
def load_image_cv(path):
    with open(path, "rb") as f:
        img_bytes = f.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), img_bytes

def save_crop_cv(img_bgr, out_path):
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    try:
        if cv2.imwrite(out_path, img_bgr):
            return
    except Exception:
        pass
    ext = os.path.splitext(out_path)[1].lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img_bgr)
    if ok:
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(out_path, quality=95)

def to_ascii_filename(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in "-_":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe) or "file"

# ---------------- Detection ----------------
def detect_plate_polys(img_bytes) -> List[List[Tuple[float,float]]]:
    """
    Vision Object Localization → 번호판 후보 다각형(정규화 좌표) 리스트 반환
    각 폴리: [(x,y), ...] in [0..1]
    """
    global CLIENT
    if CLIENT is None:
        CLIENT = get_vision_client()
    from google.cloud import vision
    image = vision.Image(content=img_bytes)
    resp = CLIENT.object_localization(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    polys = []
    for obj in resp.localized_object_annotations:
        label = (obj.name or "").lower()
        if any(k in label for k in ("license plate", "number plate", "registration plate", "license-plate")):
            pts = [(v.x, v.y) for v in obj.bounding_poly.normalized_vertices]
            # 보통 4점이지만, 혹시 3점/5점 이상도 들어올 수 있으니 정렬은 아래에서 처리
            polys.append(pts)
    return polys

# ---------------- Geometry helpers ----------------
def _order_quad(pts: np.ndarray) -> np.ndarray:
    """
    4점(임의 순서) → [tl, tr, br, bl] 로 정렬
    """
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _poly_to_quad(poly: List[Tuple[float,float]], w: int, h: int) -> np.ndarray:
    """
    Vision normalized poly → 이미지 픽셀 좌표 4점 근사
    - poly가 4점이 아니면, 컨벡스헐→최소사각형으로 보정
    """
    pts = np.array([(x*w, y*h) for x,y in poly], dtype=np.float32)
    if len(pts) < 4:
        # fallback: 바운딩 박스
        x1,y1 = np.min(pts, axis=0)
        x2,y2 = np.max(pts, axis=0)
        rect = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        return rect
    # 컨벡스 헐 → 최소 외접 사각형
    hull = cv2.convexHull(pts)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)  # 4x2
    box = _order_quad(box.astype(np.float32))
    return box

def warp_plate(img: np.ndarray, quad: np.ndarray, scale: float=1.0) -> np.ndarray:
    """
    퍼스펙티브로 번호판을 평평한 직사각형으로 펼침
    """
    quad = quad.astype(np.float32)
    tl,tr,br,bl = quad
    widthA = np.hypot(*(br - bl))
    widthB = np.hypot(*(tr - tl))
    heightA = np.hypot(*(tr - br))
    heightB = np.hypot(*(tl - bl))
    W = int(max(widthA, widthB) * scale)
    H = int(max(heightA, heightB) * scale)
    W = max(60, min(W, img.shape[1]*2))
    H = max(20, min(H, img.shape[0]*2))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (W,H), flags=cv2.INTER_CUBIC)
    return warped

# ---------------- Tight trim (no margins) ----------------
def _auto_threshold(gray: np.ndarray) -> np.ndarray:
    # 조명/색상 다양성 대응: 가우시안 블러 후 Otsu
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 문자 대비를 흰 배경/검은 글자 둘 다 커버하도록 반전 선택
    # 문자(어두움)가 배경의 다수라면 inverse가 유리
    white_ratio = (bw == 255).mean()
    if white_ratio < 0.5:
        bw = cv2.bitwise_not(bw)
    return bw

def _trim_box_from_binary(bw: np.ndarray, pad: int=2) -> Tuple[int,int,int,int]:
    """
    이진 영상에서 글자 연결성 기준으로 타이트한 바운딩 박스 추출
    """
    # morphology로 구멍 메우기
    h = max(1, bw.shape[0]//50)
    w = max(1, bw.shape[1]//50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w|1, h|1))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 가장 큰 컨투어 박스
    cnts,_ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0,0,bw.shape[1],bw.shape[0]
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    x = max(x-pad, 0)
    y = max(y-pad, 0)
    x2 = min(x+w+pad, bw.shape[1])
    y2 = min(y+h+pad, bw.shape[0])
    return x,y,x2,y2

def refine_tight_crop(warped: np.ndarray) -> np.ndarray:
    """
    퍼스펙티브 보정된 번호판에서 문자 영역만 타이트하게 트리밍
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    bw = _auto_threshold(gray)
    x1,y1,x2,y2 = _trim_box_from_binary(bw, pad=2)
    if x2-x1 <= 2 or y2-y1 <= 2:
        return warped
    tight = warped[y1:y2, x1:x2]
    return tight

# ---------------- Pipeline ----------------
def process_one(path, outdir, debug=False, ascii_name=False):
    img, img_bytes = load_image_cv(path)
    h, w = img.shape[:2]

    polys = detect_plate_polys(img_bytes)
    if debug:
        print(f"[DBG] {path} -> {len(polys)} polys")

    if not polys:
        print(f"[MISS] no plate: {path}")
        return False

    # 가장 큰 폴리곤(영역) 선택
    areas = []
    quads = []
    for poly in polys:
        quad = _poly_to_quad(poly, w, h)
        quads.append(quad)
        area = cv2.contourArea(quad.astype(np.float32))
        areas.append(area)
    quad = quads[int(np.argmax(areas))]

    # 1) 원근 보정으로 번호판 평탄화
    warped = warp_plate(img, quad, scale=1.1)

    # 2) 문자 기준 타이트 트림
    tight = refine_tight_crop(warped)

    # 저장 경로
    stem = pathlib.Path(path).stem
    if ascii_name:
        stem = to_ascii_filename(stem)
    out_path = os.path.join(outdir, f"{stem}_plate.jpg")
    save_crop_cv(tight, out_path)
    print(f"[OK] {path} -> {out_path}")
    return True

# ---------------- CLI ----------------
def iter_inputs(src):
    p = pathlib.Path(src)
    if any(ch in src for ch in ["*", "?", "["]):
        yield from glob.glob(src)
    elif p.is_dir():
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            yield from glob.glob(str(p / ext))
    else:
        yield src

def main():
    ap = argparse.ArgumentParser(description="Crop license plate (tight) with Google Vision")
    ap.add_argument("--src", required=True, help="이미지 경로(파일/폴더) 또는 와일드카드(*.jpg)")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help=f"크롭 저장 폴더 (기본: {DEFAULT_OUTDIR})")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ascii-name", action="store_true", help="출력 파일명을 ASCII로 치환")
    args = ap.parse_args()

    ensure_env()
    global CLIENT
    CLIENT = get_vision_client()

    outdir = os.path.normpath(os.path.expandvars(args.outdir))
    os.makedirs(outdir, exist_ok=True)

    src_list = list(iter_inputs(args.src))
    if not src_list:
        print("[ERR] no input matched for:", args.src, file=sys.stderr)
        sys.exit(1)

    ok = fail = 0
    for spath in src_list:
        try:
            if process_one(spath, outdir, debug=args.debug, ascii_name=args.ascii_name):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"[ERR] {spath} -> {e}")
            fail += 1
    print(f"done: ok {ok} / fail {fail}")

if __name__ == "__main__":
    main()
