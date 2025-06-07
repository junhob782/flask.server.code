from utils.OCR_engines.ocr_base import OCRBase
import easyocr
import cv2
import numpy as np
import re

class EasyOCRPlate(OCRBase):
    """
    EasyOCR 기반 번호판 인식 엔진 래퍼 클래스
    """
    def __init__(self, langs=['ko', 'en'], gpu=False):
        # 한글·영어·숫자 인식을 위한 Reader 초기화
        self.reader = easyocr.Reader(langs, gpu=gpu)

    def recognize_plate(self, image_bytes: bytes) -> str:
        # 바이트 스트림 → NumPy 배열 → OpenCV 이미지 디코딩
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""

        # OCR 수행
        results = self.reader.readtext(img)
        # 신뢰도 0.5 이상인 텍스트 중 번호판 패턴 매칭
        for _, text, conf in results:
            if conf >= 0.5:
                match = re.search(r"\d{2,3}[가-힣]\d{4}", text)
                if match:
                    return match.group(0)
        return ""