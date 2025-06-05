# utils/OCR_engines/ocr_easyocr.py

from .ocr_base import OCRBase
import easyocr
import numpy as np
import cv2
import re

class EasyOCRPlate(OCRBase):
    def __init__(self):
        # GPU 사용 여부
        # False -> CPU 사용, True -> GPU 사용
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False)

    def recognize_plate(self, image_bytes: bytes) -> str:
        """
        1) image_bytes -> OpenCV 이미지 디코딩
        2) EasyOCR로 번호판 인식 (전체 이미지 대상)
        3) 결과 텍스트 리스트를 순회하며, 한국식 번호판 패턴(숫자 2~3자리 + 한글 1글자 + 숫자 4자리) 추출
        4) 매칭되면 매칭 문자열 반환, 없으면 빈 문자열 반환
        """
        # (1) Bytes -> numpy array -> OpenCV BGR 이미지
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""

        # (2) EasyOCR로 텍스트 인식 (detail=0: 텍스트만 반환)
        result = self.reader.readtext(img, detail=0)

        # (3) 인식된 문자열들 중에서 한국식 번호판 패턴 검색
        for text in result:
            # \d{2,3}: 숫자 2~3자리  /  [가-힣]: 한글 1자  /  \d{4}: 숫자 4자리
            match = re.search(r'\d{2,3}[가-힣]\d{4}', text)
            if match:
                return match.group(0)

        # (4) 매칭 없으면 빈 문자열 리턴
        return ""
    
    
