# OCR.py

import os
import torch
import cv2
import numpy as np
from PIL import Image
from CRNN import CRNN        # CRNN 모델 정의
from Common import LabelCodec  # CTC 디코더
from DatasetLoader import PlateDataset  # 데이터셋 로더
from CTCLoss import CTCLoss  # CTC 손실 함수

# ------------------------------------------------------------
# OCR 파이프라인 래퍼 클래스(예시)
# ------------------------------------------------------------
class OCR:
    """
    CRNN을 이용해 단일 이미지(번호판 크롭)에서 문자열을 읽어오는 클래스.
    모델 로드, 전처리, 추론, 디코딩 과정을 모두 감싸서 간편하게 사용할 수 있도록 함.
    """
    def __init__(self, model_path, alphabet, imgH=32, imgW=100, device=None):
        """
        Arguments:
          model_path (str): 학습된 CRNN 가중치(.pth) 파일 경로
          alphabet (str): 허용되는 문자 집합
          imgH, imgW (int): CRNN 입력 이미지 크기
          device (str or torch.device): 'cpu' 혹은 'cuda'
        """
        # 디바이스 설정
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # CRNN 모델 생성 및 가중치 로드
        param = {
            'imgH': imgH,
            'n_classes': len(alphabet) + 1  # CTC blank 포함
        }
        self.model = CRNN(param).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 디코더(LabelCodec) 초기화
        self.codec = LabelCodec(alphabet)

        # 전처리 파이프라인 (흑백+리사이즈+Tensor+정규화)
        self.transform = torch.nn.Sequential(
            # PIL Image를 Tensor로 변환하는 것이므로,
            # 나중에 cv2 이미지를 PIL로 변환 후 이 파이프라인 사용
        )

    def predict(self, plate_img):
        """
        하나의 넘파이 배열(plate_img: OpenCV BGR)에서 문자열을 예측.

        Steps:
          1) OpenCV BGR → PIL Image(RGB) 변환
          2) 전처리(transform) → Tensor (1,1,H,W)
          3) 모델 추론 → (T,1,num_classes)
          4) 디코딩(CTC 룰) → 문자열 반환
        """
        # (1) BGR → RGB → PIL Image
        img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

        # (2) 전처리: 그레이스케일, 리사이즈, Tensor, 정규화
        img = img.convert('L')  # 흑백
        img = img.resize((self.model.imgW, self.model.imgH), Image.BICUBIC)
        tensor = torch.FloatTensor(np.array(img) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
        # (1,1,H,W)

        # (3) 모델 추론
        with torch.no_grad():
            preds = self.model(tensor)  # (T,1,num_classes)
            _, preds_index = preds.max(2)
            preds_size = torch.LongTensor([preds.size(0)])
        # (4) 디코딩
        text = self.codec.decode(preds_index.view(-1), preds_size)[0]
        return text
