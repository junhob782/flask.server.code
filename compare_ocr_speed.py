# compare_ocr_speed.py

import time
import glob
from utils.OCR_engines.ocr_easyocr import EasyOCRPlate


def load_image_bytes(path):
    with open(path, 'rb') as f:
        return f.read() 

def measure_speed(ocr_engine, image_bytes_list):
    """
    ocr_engine: OCRBase를 상속한 객체 인스턴스
    image_bytes_list: bytes로 변환된 이미지 파일들의 리스트
    리턴: (average_time_per_image, results_list)
    """
    total_time = 0.0
    results = []
    for img_bytes in image_bytes_list:
        start = time.time()
        plate = ocr_engine.recognize_plate(img_bytes)
        elapsed = time.time() - start
        total_time += elapsed
        results.append((plate, elapsed))
    avg_time = total_time / len(image_bytes_list) if image_bytes_list else 0
    return avg_time, results

def main():
    # 테스트할 이미지 폴더 경로 (예: ./test_images/*.jpg)
    image_paths = glob.glob('C:/Users/hanhw/capstonedesign/lotbot_server/test_images/*.jpg')
    image_bytes_list = [load_image_bytes(p) for p in image_paths]

    # 각 엔진별 인스턴스 생성 (필요한 초기화 인자 전달)
    easyocr_engine = EasyOCRPlate()
    

    engines = [
        ("EasyOCR", easyocr_engine),
        
    ]

    for name, engine in engines:
        print(f"=== {name} Speed Test ===")
        avg_time, results = measure_speed(engine, image_bytes_list)
        print(f"평균 실행 시간(이미지 당): {avg_time:.3f} 초")
        
        for i, (plate, elapsed) in enumerate(results):
            print(f"  [{i+1}] Plate: {plate} (Time: {elapsed:.3f} s)")
        print()

if __name__ == "__main__":
    main()
