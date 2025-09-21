# crop_plate_with_vision.py
import os, sys, glob, argparse, pathlib
import io
from PIL import Image
import cv2
import numpy as np

# === 항상 이 경로로 저장되도록 기본값 고정 ===
DEFAULT_OUTDIR = r"C:\Users\hanhw\capstonedesign\lotbot_server\test_images\crops"

# --- Google Vision 클라이언트 준비 (API_KEY 우선, 없으면 ADC 사용) ---
def get_vision_client():
    try:
        from google.cloud import vision
        from google.api_core.client_options import ClientOptions
    except Exception as e:
        print("[ERR] google-cloud-vision 미설치 또는 import 실패:", e, file=sys.stderr)
        print("      pip install google-cloud-vision", file=sys.stderr)
        sys.exit(3)

    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if api_key:
        return vision.ImageAnnotatorClient(client_options=ClientOptions(api_key=api_key))
    else:
        return vision.ImageAnnotatorClient()

# 모듈 전역에서 한 번만 생성
CLIENT = None

def ensure_env():
    """API 키가 없더라도 ADC가 있으면 동작 가능하므로 안내만."""
    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    adc_hint = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not api_key and not adc_hint:
        print("[WARN] GOOGLE_VISION_API_KEY 미설정. ADC(서비스계정) 사용 시 GOOGLE_APPLICATION_CREDENTIALS 필요.", file=sys.stderr)

def load_image_cv(path):
    # 한글 경로 대응: PIL 로드 → numpy → BGR
    with open(path, "rb") as f:
        img_bytes = f.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def save_crop_cv(img_bgr, out_path):
    """유니코드 경로에서도 확실히 저장되도록 다단계 폴백."""
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # 1) 일반 imwrite 시도
    try:
        ok = cv2.imwrite(out_path, img_bgr)
        if ok:
            return
    except Exception:
        pass

    # 2) imencode → 파일 직접쓰기
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        ext = ".jpg"
        out_path = os.path.splitext(out_path)[0] + ext

    ok, buf = cv2.imencode(ext, img_bgr)
    if ok:
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        return

    # 3) 최종 폴백: PIL로 저장
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(out_path, quality=95)

def detect_plates_with_vision(img_bytes):
    """Object Localization으로 번호판 후보 박스 반환: [(x1,y1,x2,y2)] normalized."""
    global CLIENT
    if CLIENT is None:
        CLIENT = get_vision_client()

    try:
        from google.cloud import vision
    except Exception:
        raise

    image = vision.Image(content=img_bytes)
    resp = CLIENT.object_localization(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)

    boxes = []
    for obj in resp.localized_object_annotations:
        label = (obj.name or "").lower()
        if any(k in label for k in ["license plate", "number plate", "registration plate", "license-plate"]):
            xs = [v.x for v in obj.bounding_poly.normalized_vertices]
            ys = [v.y for v in obj.bounding_poly.normalized_vertices]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes

def crop_by_norm_box(img_bgr, norm_box, pad=0.04):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = norm_box
    x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
    xi1 = max(int(x1 * w), 0)
    yi1 = max(int(y1 * h), 0)
    xi2 = min(int(x2 * w), w)
    yi2 = min(int(y2 * h), h)
    if xi2 - xi1 <= 1 or yi2 - yi1 <= 1:
        return None
    return img_bgr[yi1:yi2, xi1:xi2]

def to_ascii_filename(name: str) -> str:
    """한글/특수문자 제거하여 안전한 파일명으로 치환."""
    safe = []
    for ch in name:
        if ch.isalnum():
            safe.append(ch)
        elif ch in ('-', '_'):
            safe.append(ch)
        else:
            safe.append('_')
    return ''.join(safe) or 'file'

def process_one(path, outdir, debug=False, ascii_name=False):
    # 읽기
    with open(path, "rb") as f:
        img_bytes = f.read()
    img = load_image_cv(path)

    # Vision 박스 탐지
    boxes = detect_plates_with_vision(img_bytes)
    if debug:
        print(f"[DBG] {path} -> {len(boxes)} boxes")

    if not boxes:
        print(f"[MISS] no plate: {path}")
        return False

    # 가장 큰 박스 1개 선택
    boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    crop = crop_by_norm_box(img, boxes_sorted[0], pad=0.06)
    if crop is None:
        print(f"[MISS] bad box: {path}")
        return False

    # 저장 경로
    stem = pathlib.Path(path).stem
    if ascii_name:
        stem = to_ascii_filename(stem)

    out_path = os.path.join(outdir, f"{stem}_plate.jpg")
    save_crop_cv(crop, out_path)
    print(f"[OK] {path} -> {out_path}")
    return True

def iter_inputs(src):
    p = pathlib.Path(src)
    if any(ch in src for ch in ["*", "?", "["]):  # glob 패턴
        yield from glob.glob(src)
    elif p.is_dir():
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            yield from glob.glob(str(p / ext))
    else:
        yield src

def main():
    ap = argparse.ArgumentParser(description="Crop license plate with Google Vision")
    ap.add_argument("--src", required=True, help="이미지 경로(파일/폴더) 또는 와일드카드(*.jpg)")
    # --outdir를 선택 인자로 두되, 기본값을 고정 경로로.
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help=f"크롭 저장 폴더 (기본: {DEFAULT_OUTDIR})")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ascii-name", action="store_true", help="출력 파일명을 ASCII로 치환")
    args = ap.parse_args()

    ensure_env()

    # 전역 CLIENT를 미리 생성하여 인증 문제를 일찍 노출
    global CLIENT
    CLIENT = get_vision_client()

    # outdir 정규화 및 미리 생성
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
