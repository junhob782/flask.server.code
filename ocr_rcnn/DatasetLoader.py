# DatasetLoader.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# ------------------------------------------------------------
# CRNN 학습용 데이터셋 로더
# ------------------------------------------------------------
class PlateDataset(Dataset):
    """
    번호판 이미지와 해당 레이블(문자열)을 불러와,
    모델 학습에 필요한 전처리 후 이미지 텐서와 정수 인덱스 시퀀스를 반환하는 커스텀 Dataset.

    각 데이터 항목은 다음과 같은 구조를 가정:
      image_path,label_string
    이렇게 콤마(,)로 구분된 텍스트 파일을 읽어들여서 파싱.
    """
    def __init__(self, data_list, alphabet, imgH=32, imgW=100):
        """
        Arguments:
          data_list (list[str]): 'image_path,label' 형식의 문자열 리스트
          alphabet (str): 문자를 인덱스로 변환하기 위한 문자 집합
          imgH (int): 모델 입력 높이 (예: 32)
          imgW (int): 모델 입력 너비 (예: 100)
        """
        super(PlateDataset, self).__init__()
        self.data_list = data_list
        self.alphabet = alphabet
        self.imgH = imgH
        self.imgW = imgW

        # 레이블 인덱스 변환: 예를 들어 LabelCodec과 같은 역할
        self.char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}
        # 0은 CTC blank 토큰을 위해 남겨둠

        # 이미지 전처리 파이프라인 (흑백, 크기 조정, Tensor 변환, 정규화)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),       # 흑백
            transforms.Resize((imgH, imgW)),                  # (높이, 너비) 고정 크기로 리사이즈
            transforms.ToTensor(),                            # [0,1] 범위 FloatTensor
            transforms.Normalize((0.5,), (0.5,))              # 평균 0.5, 표준편차 0.5 정규화
        ])

    def __len__(self):
        """
        전체 데이터셋 크기 반환.
        """
        return len(self.data_list)

    def __getitem__(self, index):
        """
        인덱스에 해당하는 이미지 + 레이블 로드 및 전처리 후 반환.
        Returns:
          image_tensor (Tensor): (1, imgH, imgW)
          label_indices (LongTensor): 라벨 문자열의 인덱스 시퀀스
        """
        # 'image_path,label' 구조 파싱
        line = self.data_list[index].strip().split(',')
        image_path, label = line[0], line[1]

        # 이미지 열기 및 전처리
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  # (1, imgH, imgW)

        # 레이블 문자열을 인덱스 리스트로 변환
        label_indices = []
        for char in label:
            if char in self.char_to_idx:
                label_indices.append(self.char_to_idx[char])
        # LongTensor 형태로 변환
        label_indices = torch.LongTensor(label_indices)

        return image, label_indices
