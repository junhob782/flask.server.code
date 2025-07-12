import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from .ocr_base import OCRBase

# GPU 최적화 설정
torch.backends.cudnn.benchmark = True

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------
# 1) PlateNetDetector (YOLOv8 기반)
# ---------------------------------------

class PlateNetDetector:
    def __init__(self, model_path: str, device=DEVICE, img_size: int = 320, half: bool = True):
        self.device = device
        self.img_size = img_size
        try:
            self.model = YOLO(model_path).to(device)
            self.model.model.fuse()
            if half and device.type == 'cuda':
                self.model.model.half()
            print(f"[INFO][YOLO] Loaded {model_path} on {device}, img_size={img_size}, half={half}")
        except Exception as e:
            print(f"[ERROR][YOLO] Model init failed: {e}")
            self.model = None

    def detect(self, image: np.ndarray, conf_thres: float = 0.3):
        if self.model is None:
            return []

        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        if scale != 1.0:
            image_resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            image_resized = image

        try:
            results = self.model.predict(
                source=image_resized,
                imgsz=self.img_size,
                device=self.device,
                half=(self.device.type == 'cuda'),
                conf=conf_thres,
                verbose=False
            )
        except Exception as e:
            print(f"[ERROR][YOLO] Inference failed: {e}")
            return []

        boxes = []
        for res in results:
            for box in res.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = (coords / scale).astype(int)
                boxes.append((x1, y1, x2, y2))
        if not boxes:
            print("[WARN][YOLO] No plates detected")
        return boxes

# ---------------------------------------
# 2) CRNNRecognizer
# ---------------------------------------

class CRNNRecognizer:
    def __init__(self, model_path: str, device=DEVICE,
                 alphabet: str = "0123456789가-힣ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        from ocr_rcnn.CRNN import CRNN
        from ocr_rcnn.Common import LabelCodec

        self.device = device
        self.alphabet = alphabet
        self.converter = LabelCodec(alphabet)

        try:
            scripted_path = model_path.replace('.pth', '_scripted.pt')
            self.crnn = torch.jit.load(scripted_path, map_location=device)
            print(f"[INFO][CRNN] Loaded TorchScript model: {scripted_path}")
        except Exception:
            try:
                self.crnn = CRNN({'imgH': 32, 'n_classes': len(alphabet)})
                state = torch.load(model_path, map_location='cpu')
                self.crnn.load_state_dict(state)
                self.crnn = self.crnn.to(device).eval()
                if device.type == 'cuda':
                    self.crnn.half()
                print(f"[INFO][CRNN] Loaded PTH model on {device}, half={device.type=='cuda'}")
            except Exception as e:
                print(f"[ERROR][CRNN] Model load failed: {e}")
                self.crnn = None

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def recognize(self, plate_img: np.ndarray) -> str:
        if self.crnn is None:
            return ""

        img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda':
            tensor = tensor.half()

        with torch.no_grad():
            preds = self.crnn(tensor)
            if preds is None or preds.numel() == 0:
                print("[WARN][CRNN] Empty prediction")
                return ""
            _, idx = preds.max(2)
            idx = idx.view(-1)

        try:
            decoded = self.converter.decode(idx.cpu(), torch.LongTensor([preds.size(0)]))
        except Exception as e:
            print(f"[ERROR][LabelCodec] Decode failed: {e}")
            decoded = "".join(
                self.alphabet[i] for i in idx.cpu().numpy() if i < len(self.alphabet)
            )
        return decoded

# ---------------------------------------
# 3) PlateNetCRNNPlate (전체 통합)
# ---------------------------------------

class PlateNetCRNNPlate(OCRBase):
    def __init__(self, yolo_model_path: str, crnn_model_path: str):
        self.device = DEVICE
        self.detector = PlateNetDetector(yolo_model_path, device=self.device, img_size=320, half=True)
        self.recognizer = CRNNRecognizer(crnn_model_path, device=self.device)

    def recognize_plate(self, image_bytes: bytes) -> str:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("[ERROR] Failed to decode image bytes")
            return ""

        boxes = self.detector.detect(img, conf_thres=0.3)
        plate = None
        if boxes:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = boxes[0]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 - x1 >= 16 and y2 - y1 >= 16:
                crop = img[y1:y2, x1:x2]
                plate = cv2.resize(crop, (100, 32))
            else:
                print("[WARN] Invalid box size, skipping crop")

        if plate is not None:
            text = self.recognizer.recognize(plate)
            if len(text) >= 4:
                return text

        print("[WARN] Region OCR failed, performing full-image OCR")
        full = cv2.resize(img, (100, 32))
        return self.recognizer.recognize(full)
