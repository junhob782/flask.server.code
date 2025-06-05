# utils/ocr_engines/ocr_googlevision.py

from .ocr_base import OCRBase
from google.cloud import vision
import io

class GoogleVisionPlate(OCRBase):
    def __init__(self):
        # 환경변수 GOOGLE_APPLICATION_CREDENTIALS를 통해 서비스 계정 키 파일 경로를 지정하세요.
        self.client = vision.ImageAnnotatorClient()

    def recognize_plate(self, image_bytes: bytes) -> str:
        image = vision.Image(content=image_bytes)
        response = self.client.text_detection(image=image)
        if response.error.message:
            return ""
        texts = response.text_annotations  # texts[0]는 전체 텍스트, 이후 항목들은 단어별
        if not texts:
            return ""
        full_text = texts[0].description.replace("\n", " ")
        import re
        match = re.search(r'\d{2,3}[가-힣]\d{4}', full_text)
        return match.group(0) if match else ""
