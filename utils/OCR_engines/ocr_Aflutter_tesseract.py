# utils/ocr_engines/ocr_flutter_tesseract.py

from .ocr_base import OCRBase
import cv2
import numpy as np
import pytesseract

class FlutterTesseractPlate(OCRBase):
    def __init__(self):
        # Flutter용 Tesseract는 보통 모바일 환경에서 쓰지만,
        # Python 테스트용으로 동일한 pytesseract를 사용할 수 있습니다.
        # 만약 실제 Flutter 환경에서 테스트하려면 안드로이드/iOS 쪽 샘플 코드가 필요합니다.
        pass

    def recognize_plate(self, image_bytes: bytes) -> str:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        plate_contour = None
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if len(approx) == 4:
                plate_contour = approx
                break
        if plate_contour is None:
            return ""

        def _order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        pts = plate_contour.reshape(4, 2)
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        plate_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789가-힣"
        raw_text = pytesseract.image_to_string(plate_img, config=config)
        recognized = "".join(ch for ch in raw_text if ch.isalnum() or ('가' <= ch <= '힣'))
        return recognized.strip()
