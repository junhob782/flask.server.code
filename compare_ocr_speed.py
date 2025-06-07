import time
import glob

from utils.OCR_engines.ocr_easyocr import EasyOCRPlate
from utils.OCR_engines.ocr_googlevision import GoogleVisionPlate
from utils.OCR_engines.ocr_kerasocr import KerasOCRPlate


def load_image_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


def measure_speed(ocr_engine, image_bytes_list):
    total = 0.0
    results = []
    for img_bytes in image_bytes_list:
        start = time.time()
        plate = ocr_engine.recognize_plate(img_bytes)
        elapsed = time.time() - start
        total += elapsed
        results.append((plate, elapsed))
    avg = total / len(image_bytes_list) if image_bytes_list else 0
    return avg, results


def main():
    # 1) 테스트할 이미지 리스트 로드
    image_paths = glob.glob(r'C:\Users\hanhw\capstonedesign\lotbot_server\test_images\*.jpg')
    image_bytes_list = [load_image_bytes(p) for p in image_paths]

    # 2) OCR 엔진 인스턴스 생성
    easyocr_engine = EasyOCRPlate()
    google_engine = GoogleVisionPlate(api_key="AIzaSyCxKXybbsSzNCESi3QKxVGRUuRR9Ir9n1c")
    kerasocr_engine = KerasOCRPlate()

    engines = [
        ("EasyOCR", easyocr_engine),
        ("GoogleVision", google_engine),
        ("KerasOCR", kerasocr_engine),
    ]

    # 3) 속도 측정 및 결과 출력
    for name, engine in engines:
        print(f"\n=== {name} Speed Test ===")
        avg_time, results = measure_speed(engine, image_bytes_list)
        print(f"평균 실행 시간(이미지 당): {avg_time:.3f}초")
        for idx, (plate, t) in enumerate(results, start=1):
            print(f"  [{idx}] Plate: {plate}  (Time: {t:.3f}s)")

if __name__ == "__main__":
    main()
