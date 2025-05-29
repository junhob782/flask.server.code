# utils/ocr.py
def recognize_plate(image_bytes):
    # TODO: OpenCV, Tesseract 등 실제 OCR 호출
    plate = ocr_library.recognize(image_bytes)
    return plate or None
