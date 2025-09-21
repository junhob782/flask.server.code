# batch_crop.py
import os, glob
from plate_crop import crop_plate

IN_DIR  = r"C:\Users\hanhw\capstonedesign\lotbot_server\test_images"
OUT_DIR = os.path.join(IN_DIR, "cropped")

os.makedirs(OUT_DIR, exist_ok=True)

# 처리할 확장자들
exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

count_ok, count_fail = 0, 0

for pat in exts:
    for path in glob.glob(os.path.join(IN_DIR, pat)):
        name, ext = os.path.splitext(os.path.basename(path))
        out_path = os.path.join(OUT_DIR, f"{name}_plate.png")
        try:
            result = crop_plate(path, out_path, return_bbox=True)
            if result is None:
                print(f"[MISS] 번호판 없음: {path}")
                count_fail += 1
            else:
                _, (x,y,w,h) = result
                print(f"[OK] {path} -> {out_path}  ROI=({x},{y},{w},{h})")
                count_ok += 1
        except Exception as e:
            print(f"[ERR] {path} -> {e}")
            count_fail += 1

print(f"\n완료: 성공 {count_ok} / 실패 {count_fail} (출력: {OUT_DIR})")
