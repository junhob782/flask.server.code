import os
from ocr_rcnn.OCR import PlateNetCRNNPlate

yolo_model_path = "yolov8n.pt"
crnn_model_path = "crnn.pth"

ocr = PlateNetCRNNPlate(yolo_model_path, crnn_model_path)

image_folder = "test_images"

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)
        print(f"\n[파일] {image_path}")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = ocr.recognize_plate(image_bytes)
        print("→ 인식 결과:", result)

#######OCR.py 실행하고 결과 출력하는 파일#######