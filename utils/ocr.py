# utils/ocr.py
import os
from typing import Optional
from utils.OCR_engines.ocr_googlevision import GoogleVisionPlate

# 지연 생성되는 단일 엔진 인스턴스
_ENGINE: Optional[GoogleVisionPlate] = None

def _get_engine() -> GoogleVisionPlate:
    global _ENGINE
    if _ENGINE is None:
        api_key = os.getenv("GOOGLE_VISION_API_KEY")
        if not api_key:
            # server.py에서도 체크하지만 여기서도 방어
            raise RuntimeError("GOOGLE_VISION_API_KEY is not set")
        _ENGINE = GoogleVisionPlate(api_key=api_key)
    return _ENGINE

def recognize_plate(image_bytes: bytes) -> str:
    """
    통일된 OCR 진입점.
    GoogleVisionPlate 엔진을 통해 번호판 텍스트를 반환.
    """
    engine = _get_engine()
    text = engine.recognize_plate(image_bytes)
    return (text or "").strip()
