import time
import glob
import os
from flask import Flask, request, jsonify
#from utils.OCR_engines.ocr_easyocr import EasyOCRPlate
#from utils.OCR_engines.ocr_kerasocr import KerasOCRPlate
#from utils.OCR_engines.ocr_Aflutter_tesseract import FlutterTesseractPlate
from utils.OCR_engines.ocr_googlevision import GoogleVisionPlate
from services.car_services import upsert_car_and_get_id

# =======================
# 🔹 OCR 테스트용 함수들
# =======================

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

def run_ocr_speed_test():
    print("🔧 OCR 속도 측정 시작")
    image_paths = glob.glob(r'C:\Users\hanhw\capstonedesign\lotbot_server\test_images\*.jpg')
    image_bytes_list = [load_image_bytes(p) for p in image_paths]

    # OCR 엔진 인스턴스 생성
    #easyocr_engine      = EasyOCRPlate()
    #kerascrnn_engine    = KerasOCRPlate()
    #tesseract_engine    = FlutterTesseractPlate()
    

    engines = [
        #("EasyOCR",       easyocr_engine),
        #("KerasOCR",      kerascrnn_engine),
        #("TesseractOCR",  tesseract_engine),
        
    ]
    
    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if not api_key:
        print("[INFO] GOOGLE_VISION_API_KEY 미설정 → Googlevision 스킵")
    else:
        from utils.OCR_engines.ocr_googlevision import GoogleVisionPlate
    
    # ✅ 환경변수에서 읽은 api_key를 직접 전달
    googlevision_engine = GoogleVisionPlate(api_key=api_key)
    
    engines.append(("Googlevision", googlevision_engine))

    

    for name, engine in engines:
        print(f"\n=== {name} Speed Test ===")
        avg_time, results = measure_speed(engine, image_bytes_list)
        print(f"평균 실행 시간(이미지 당): {avg_time:.3f}초")
        for idx, (plate, t) in enumerate(results, start=1):
            print(f"  [{idx}] Plate: {plate}  (Time: {t:.3f}s)")

# =======================
# 🔹 Flask 서버 설정
# =======================

app = Flask(__name__)
#ocr_engine = EasyOCRPlate()  # 현재 OCR 엔진
ocr_engine = GoogleVisionPlate()

@app.route('/api/ocr/license_plate', methods=['POST'])
def ocr_license_plate():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is missing'}), 400

    file = request.files['image']
    image_bytes = file.read()
    plate_number = ocr_engine.recognize_plate(image_bytes)

    if plate_number:
        return jsonify({'plate_number': plate_number})
    else:
        return jsonify({'error': 'No license plate detected'}), 404

# =======================
# 🔹 메인 실행
# =======================

if __name__ == "__main__":
    run_ocr_speed_test()        # OCR 속도 테스트 실행
    engines.append(("Googlevision", googlevision_engine))
    app.run(host="0.0.0.0", port=5000, debug=True)