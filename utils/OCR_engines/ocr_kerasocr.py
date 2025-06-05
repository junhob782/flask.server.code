# utils/ocr_engines/ocr_kerasocr.py

from .ocr_base import OCRBase
import keras_ocr
import numpy as np
import cv2

class KerasOCRPlate(OCRBase):
    def __init__(self):
        # 모델 로딩 시에는 약간의 시간이 걸리니, 싱글톤으로 한 번만 로드하도록 처리하는 게 좋습니다.
        pipeline = keras_ocr.pipeline.Pipeline()
        self.pipeline = pipeline

    def recognize_plate(self, image_bytes: bytes) -> str:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""

        # keras-ocr은 이미지를 RGB 형식으로 받길 원함
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 모델에 입력할 리스트 형태
        prediction_groups = self.pipeline.recognize([rgb])
        # prediction_groups는 List[List[(word, box_coords)]] 구조
        texts = [word for word, box in prediction_groups[0]]
        # 여러 텍스트 중 번호판 패턴을 찾아서 리턴
        import re
        for text in texts:
            match = re.search(r'\d{2,3}[가-힣]\d{4}', text)
            if match:
                return match.group(0)
        return ""
