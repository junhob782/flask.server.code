# utils/ocr.py

import cv2
import numpy as np
import pytesseract

def recognize_plate(image_bytes):
    # TODO: OpenCV, Tesseract 등 실제 OCR 호출
    plate = ocr_library.recognize(image_bytes) #OCR을 호출할 로직을 넣으세요라는 의미 ㅇㅇ
    return plate or None
