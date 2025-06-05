# utils/ocr_engines/ocr_platenet_crnn.py

from .ocr_base import OCRBase
import cv2
import numpy as np
import re
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO  # YOLOv8 클래스 임포트

# ---------------------------------------
# 1) PlateNetDetector (YOLOv8 기반 검출기)
# ---------------------------------------
class PlateNetDetector:
    """
    YOLOv8 기반으로 학습된 번호판 검출 모델을 로드하고,
    이미지를 입력하면 [(x1,y1,x2,y2), ...] 형태의 바운딩 박스 리스트를 반환합니다.
    """
    def __init__(self, model_path: str):
        # model_path: YOLOv8로 학습해서 얻은 .pt 가중치 파일 경로 (예: 'yolov8_lp.pt')
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray):
        """
        image: OpenCV BGR 형태의 numpy array (H x W x 3)
        반환: [(x1, y1, x2, y2), ...] 형태의 리스트 (픽셀 좌표, 정수형)
        """
        results = self.model(image)  # YOLOv8 추론 결과 객체
        boxes = []
        for res in results:  # 배치 단위로 결과
            if hasattr(res, "boxes"):
                for box in res.boxes:
                    coords = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = coords.astype(int)
                    boxes.append((x1, y1, x2, y2))
        return boxes


# ----------------------------------------------------------
# 2) CRNNRecognizer (ocr_rcnn 폴더 아래의 CRNN.py, Common.py 기준)
# ----------------------------------------------------------
class CRNNRecognizer:
    """
    ocr_rcnn/CRNN.py에 정의된 CRNN 모델 구조와 학습된 가중치를 불러와,
    plate 이미지를 넣으면 문자열을 반환합니다.
    """
    def __init__(
        self,
        model_path: str,
        alphabet: str = "0123456789가-힣ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ):
        """
        model_path: 학습된 CRNN 가중치(.pth) 파일 경로
        alphabet: CRNN이 예측하는 문자의 집합 (예: '0'~'9', 한글, 영문 등)
        """
        # (1) CRNN 아키텍처(ocr_rcnn/CRNN.py)를 import
        from ocr_rcnn.CRNN import CRNN

        # (2) CTC 디코더로 사용할 LabelCodec(ocr_rcnn/Common.py) import
        from ocr_rcnn.Common import LabelCodec

        # (3) 디바이스 설정 (GPU가 있으면 cuda, 없으면 cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (4) CRNN 모델 생성 및 학습된 가중치 불러오기
        #     ocr_rcnn/CRNN.py의 CRNN 클래스는 __init__(param) 형태인데,
        #     우리가 직접 필요한 파라미터를 dict 형태로 넘겨줘야 합니다.
        #     예시로 imgH=32, n_classes=len(alphabet)일 경우:
        param = {
            'imgH': 32,
            'n_classes': len(alphabet),
        }
        self.crnn = CRNN(param)
        self.crnn.load_state_dict(torch.load(model_path, map_location=self.device))
        self.crnn.to(self.device).eval()

        # (5) LabelCodec을 이용하여 CTC 디코더 준비
        self.converter = LabelCodec(alphabet)
        self.alphabet = alphabet

        # (6) 이미지 전처리 파이프라인 (Grayscale -> Resize -> ToTensor -> Normalize)
        #     ocr_rcnn/DatasetLoader.py의 transform과 맞춰서 사용합니다.
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 흑백
            transforms.Resize((32, 100)),                 # (H=32, W=100) 예시
            transforms.ToTensor(),                        # Tensor로 변환 ([0,1] float)
            transforms.Normalize((0.5,), (0.5,))          # 정규화
        ])

    def recognize(self, plate_img: np.ndarray) -> str:
        """
        plate_img: OpenCV BGR numpy array (번호판 영역만 잘라낸 이미지)
        반환: 인식된 문자열 (예: "12가3456") 또는 빈 문자열
        """
        # (1) OpenCV BGR -> PIL Image (RGB)
        img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

        # (2) 전처리(transform) 후 배치 차원 추가 -> (1, C, H, W)
        processed = self.transform(img).unsqueeze(0).to(self.device)

        # (3) CRNN 모델 추론
        with torch.no_grad():
            # ocr_rcnn/CRNN.py의 forward는 (T x B x num_classes) 형태로 반환
            preds = self.crnn(processed)  # 예: (T, B, H)
            _, preds_index = preds.max(2)      # (T, B)
            preds_index = preds_index.view(-1) # (T * B,)

        # (4) CTC 디코딩: LabelCodec.decode(text, length, raw=False) 사용
        #     여기서 length는 (batch별 sequence 길이) 정보를 담아줘야 하는데,
        #     preds.size(0) = T, preds.size(1) = B (=1)
        batch_size = 1
        T, B, _ = preds.size()
        # 각 배치의 sequence 길이를 담은 텐서
        pred_sizes = torch.LongTensor([T] * batch_size)
        # preds_index는 (T*B,) 형태이므로, LabelCodec.decode에 맞춰서 reshape 필요
        decoded = self.converter.decode(preds_index, pred_sizes)

        return decoded  # 이미 decode()가 문자열을 반환

# ----------------------------------------------------------
# 3) PlateNetCRNNPlate (OCRBase를 상속받아 두 모델을 통합)
# ----------------------------------------------------------
class PlateNetCRNNPlate(OCRBase):
    """
    PlateNet(YOLOv8)으로 번호판 영역을 검출하고,
    CRNNRecognizer로 텍스트를 뽑아 리턴하는 클래스입니다.
    """
    def __init__(self, detector_model_path: str, recognizer_model_path: str):
        # PlateNetDetector 초기화(YOLOv8으로 번호판 검출)
        self.detector = PlateNetDetector(model_path=detector_model_path)
        # CRNNRecognizer 초기화(ocr_rcnn 폴더의 CRNN 모델 + CTC Decoder)
        self.recognizer = CRNNRecognizer(model_path=recognizer_model_path)

    def recognize_plate(self, image_bytes: bytes) -> str:
        """
        image_bytes: 이미지 파일을 bytes 형태로 전달받음.
        반환: 인식된 번호판 문자열 (예: "12가3456") 또는 빈 문자열.
        """
        # (1) bytes → numpy array → OpenCV BGR 이미지
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""

        # (2) PlateNetDetector를 이용해 번호판 영역(box) 검출
        boxes = self.detector.detect(img)
        if not boxes:
            return ""

        # (3) 첫 번째 검출된 박스를 이용해서 plate_img 크롭
        x1, y1, x2, y2 = boxes[0]
        plate_img = img[y1:y2, x1:x2]

        # (4) CRNNRecognizer로 plate_img를 인식
        text = self.recognizer.recognize(plate_img)

        # (5) 숫자+한글+영문만 남기도록 정규식으로 필터(필요 시)
        cleaned = "".join(re.findall(r"[0-9가-힣A-Za-z]+", text))
        return cleaned
