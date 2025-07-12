import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class OCRBase:
    def __init__(self):
        pass

class FlutterTesseractPlate(OCRBase):
    def __init__(self):
        super().__init__()

    def recognize_plate(self, image_bytes: bytes) -> str:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("[오류] 이미지 디코딩 실패")
            return ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        config = (
            "--oem 1 "
            "--psm 7 "
            "-l kor "
            "-c tessedit_char_whitelist=0123456789가나다라마바사아자차카타파하 "
        )

        try:
            raw_text = pytesseract.image_to_string(thresh, config=config)
            recognized = "".join(
                ch for ch in raw_text if ch.isalnum() or ('가' <= ch <= '힣')
            )
            return recognized.strip()
        except Exception as e:
            print(f"[오류] pytesseract OCR 실패: {e}")
            return ""

image_folder = r"E:\project\flask.server.code\test_images"

ocr = FlutterTesseractPlate()

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)
        print(f"\n[파일] {image_path}")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = ocr.recognize_plate(image_bytes)

        print("→ 인식 결과:", result)
