# utils/OCR_engines/ocr_googlevision.py

from .ocr_base import OCRBase
from google.cloud import vision
import re

class GoogleVisionPlate(OCRBase):
    def __init__(self, api_key: str):
        # API key 기반 클라이언트 생성
        self.client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )

    def recognize_plate(self, image_bytes: bytes) -> str:
        # 1) 이미지 로드
        image = vision.Image(content=image_bytes)

        # 2) 한국어 텍스트 힌트 제공
        context = vision.ImageContext(
            language_hints=['ko']
        )

        # 3) 문서 텍스트 검출 호출 (더 많은 텍스트를 붙여서 리턴)
        response = self.client.document_text_detection(
            image=image,
            image_context=context
        )
        if response.error.message:
            # 오류나 힌트 부족 시 빈 문자열 리턴
            return ""

        # 4) 전체 텍스트 가져오기
        full_text = response.full_text_annotation.text or ""
        full_text = full_text.replace("\n", " ")

        # 5) 번호판 패턴 매칭 (예: "12가3456" 또는 "12 가 3456")
        match = re.search(r"\b\d{2,3}[가-힣]\s?\d{4}\b", full_text)
        if match:
            # 공백 제거하고 리턴
            return match.group(0).replace(" ", "")

        return ""
