# File: utils/OCR_engines/ocr_kerasocr.py

from .ocr_base import OCRBase
import tensorflow as tf
import keras_ocr
import numpy as np
import cv2
import re

# Monkey-patch Keras Dense to accept and safely ignore `weights` keyword
from tensorflow.keras.layers import Dense as _Dense
class Dense(_Dense):
    def __init__(self, *args, weights=None, **kwargs):
        # Pop weights argument if passed in kwargs
        w = weights if weights is not None else kwargs.pop('weights', None)
        # Initialize base Dense without weights arg
        super().__init__(*args, **kwargs)
        # Attempt to set weights only if provided and matching expected shape
        if w:
            try:
                self.set_weights(w)
            except Exception:
                # Ignore mismatches or layers without weights
                pass
# Override tf.keras.layers.Dense for compatibility
tf.keras.layers.Dense = Dense

class KerasOCRPlate(OCRBase):
    """
    Keras-OCR 기반 번호판 인식 엔진 래퍼 클래스
    싱글톤 패턴으로 Pipeline을 한 번만 로드하여 재사용합니다.
    """
    _pipeline = None

    def __init__(self):
        if KerasOCRPlate._pipeline is None:
            # Pipeline 초기화: detection + recognition 모델 로드
            KerasOCRPlate._pipeline = keras_ocr.pipeline.Pipeline()
        self.pipeline = KerasOCRPlate._pipeline

    def recognize_plate(self, image_bytes: bytes) -> str:
        # 1) 바이트 스트림 → OpenCV BGR 이미지
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            return ""

        # 2) 전처리: 그레이스케일 → 노이즈 제거 → 이진화 → RGB 변환
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        prep = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        # 3) OCR 수행
        prediction_groups = self.pipeline.recognize([prep])
        results = prediction_groups[0]  # List of (word, box_coords)

        # 4) 후보 필터링 & 번호판 패턴 매칭
        for word, box in results:
            # 4a) 박스 비율 필터링 (번호판 가로:세로 ≈ 2~6)
            coords = np.array(box)
            w = np.linalg.norm(coords[0] - coords[1])
            h = np.linalg.norm(coords[0] - coords[3])
            if h == 0:
                continue
            aspect = w / h
            if not (2.0 < aspect < 6.0):
                continue

            # 4b) 정교화된 정규식: 공백·하이픈 허용
            match = re.search(r"\b\d{2,3}[가-힣][\s\-]?\d{4}\b", word)
            if match:
                return match.group(0).replace(" ", "").replace("-", "")

        return ""
