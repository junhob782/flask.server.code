# test/test_platenet_crnn.py

import os
import sys
import glob

# 프로젝트 루트(lotbot_server)를 모듈 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.OCR_engines.ocr_platenet_crnn import PlateNetCRNNPlate

def load_image_bytes(path):
    with open(path, 'rb') as f:
        return f.read()

if __name__ == "__main__":
    # 1) YOLOv8으로 학습된 .pt 모델 경로
    detector_weights = "C:/Users/hanhw/capstonedesign/lotbot_server/Automatic-License-Plate-Recognition-using-YOLOv8/license_plate_detector.pt"
    # 2) ocr_rcnn으로 학습된 CRNN 모델 가중치(.pth) 경로
    recognizer_weights = "C:/Users/hanhw/capstonedesign/lotbot_server/ocr_rcnn/ocr_crnn_best.pth"

    ocr_engine = PlateNetCRNNPlate(
        detector_model_path=detector_weights,
        recognizer_model_path=recognizer_weights
    )

    # test_images 폴더에 있는 모든 .jpg 파일을 처리
    image_paths = glob.glob("C:/Users/hanhw/capstonedesign/lotbot_server/test_images/*.jpg")

    for img_path in image_paths:
        img_bytes = load_image_bytes(img_path)
        plate_text = ocr_engine.recognize_plate(img_bytes)
    print(f"[{img_path}] → 인식된 번호판: {plate_text}")    
