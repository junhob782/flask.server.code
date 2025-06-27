from .ocr_base import OCRBase
import cv2
import numpy as np
import pytesseract
import string

class FlutterTesseractPlate(OCRBase):
    def __init__(self):
        super().__init__()

    def recognize_plate(self, image_bytes: bytes) -> str:
        # ===== [1] 이미지 디코딩 및 유효성 검사 =====
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("[오류] 이미지 디코딩 실패")
            return ""

        # ===== [2] 전처리 (Grayscale + Blur + Canny) =====
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)  # 민감도 상향 조정

        # ===== [3] 윤곽선 검출 및 사각형 후보 선택 =====
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 사전 필터: 넓은 영역만 추출
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1500]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        plate_contour = None
        for i, cnt in enumerate(contours):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if 4 <= len(approx) <= 6:  # 유연한 사각형 판단
                plate_contour = approx
                break

        plate_img = None
        if plate_contour is not None:
            try:
                # ===== [4] 투시 변환 및 크기 최적화 =====
                def _order_points(pts):
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]
                    return rect

                pts = plate_contour.reshape(4, 2).astype("float32")
                rect = _order_points(pts)
                (tl, tr, br, bl) = rect

                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)

                maxWidth = max(int(widthA), int(widthB))
                maxHeight = max(int(heightA), int(heightB))

                if maxWidth == 0 or maxHeight == 0:
                    raise ValueError("잘못된 투시 변환 크기")

                # 이미지 크기 축소 (성능 향상)
                scale = 0.5
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                plate_img = cv2.warpPerspective(
                    img, M,
                    (int(maxWidth * scale), int(maxHeight * scale))
                )

            except Exception as e:
                print(f"[예외] 투시 변환 오류: {e}")
                plate_img = None

        # ===== [5] pytesseract 설정 최적화 =====
        whitelist = "0123456789" + "가-힣" + string.ascii_uppercase
        config = (
            "--oem 1 "  # LSTM 기반
            "--psm 7 "  # 단일 라인
            "-l kor+eng "  # 명시적 언어 설정
            "-c tessedit_char_whitelist=" + whitelist + " "
            "-c preserve_interword_spaces=0"  # 공백 제거로 일관성 확보
        )

        # ===== [6] OCR 처리 함수 정의 =====
        def ocr_image(target_img):
            try:
                raw_text = pytesseract.image_to_string(
                    target_img,
                    config=config
                )
                recognized = "".join(
                    ch for ch in raw_text if ch.isalnum() or ('가' <= ch <= '힣')
                )
                return recognized.strip()
            except Exception as e:
                print(f"[오류] pytesseract OCR 실패: {e}")
                return ""

        # ===== [7] 인식 우선 순위 및 Fallback 전략 =====
        if plate_img is not None:
            result = ocr_image(plate_img)
            if len(result) >= 4:  # 너무 짧은 경우 무시
                return result

        print("[경고] 번호판 영역 추출 실패 → 전체 이미지 OCR 시도")
        return ocr_image(img)
