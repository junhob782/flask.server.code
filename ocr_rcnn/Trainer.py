# Trainer.py

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from CRNN import CRNN              # CRNN 모델
from Common import LabelCodec      # 디코더 및 정확도 측정
from CTCLoss import CTCLoss        # CTC 손실 함수
from DatasetLoader import PlateDataset  # 데이터 로더

# ------------------------------------------------------------
# 학습용 Trainer 클래스
# ------------------------------------------------------------
class Trainer:
    """
    CRNN 모델 학습 루프를 관리하는 클래스.
    학습-검증 절차, 손실 함수, 옵티마이저, 학습률 스케줄러 등을 포함한다.
    """
    def __init__(self, train_list, val_list, alphabet, model_save_dir, batch_size=32, num_workers=4, lr=0.001):
        """
        Arguments:
          train_list (list[str]): 학습 데이터 리스트 (image_path,label)
          val_list (list[str]): 검증 데이터 리스트
          alphabet (str): 문자 집합
          model_save_dir (str): 체크포인트(.pth) 저장 디렉터리
          batch_size (int): 배치 크기
          num_workers (int): 데이터 로드 워커 수
          lr (float): 초기 학습률
        """
        self.train_list = train_list
        self.val_list = val_list
        self.alphabet = alphabet
        self.model_save_dir = model_save_dir

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 생성
        param = {
            'imgH': 32,
            'n_classes': len(alphabet) + 1  # CTC blank 포함
        }
        self.model = CRNN(param).to(self.device)

        # 손실 함수(CTC) 및 옵티마이저, 스케줄러 설정
        self.criterion = CTCLoss(blank=len(alphabet))  # 마지막 인덱스를 blank로 사용
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

        # 데이터로더 설정
        self.train_dataset = PlateDataset(train_list, alphabet, imgH=32, imgW=100)
        self.val_dataset = PlateDataset(val_list, alphabet, imgH=32, imgW=100)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, collate_fn=self.collate_fn)

        # 디코더 및 정확도 측정 클래스 초기화
        self.codec = LabelCodec(alphabet)

    def collate_fn(self, batch):
        """
        커스텀 collate_fn: 배치 내에서 가변 길이의 레이블을 처리하기 위한 함수.
        이미지 텐서는 (B,1,H,W) 형태로 텐서화하고, 레이블 시퀀스는 1차원 LongTensor로 합침.
        """
        images, labels = zip(*batch)
        # 이미지 배치 텐서 생성
        images = torch.stack(images, 0)

        # 레이블 인덱스는 가변 길이이므로 길이 정보를 함께 저장
        label_lengths = [len(l) for l in labels]
        concatenated = torch.cat(labels)  # (sum(label_lengths),)
        return images, concatenated, torch.IntTensor(label_lengths)

    def train(self, num_epochs):
        """
        전체 학습 루프. 매 epoch마다 학습 및 검증을 수행.
        """
        best_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            # 1) 학습 모드
            self.model.train()
            total_loss = 0.0
            for i, (images, targets, target_lengths) in enumerate(self.train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                # 예측 시퀀스 길이는 이미지 너비(특징 차원)로 계산
                preds = self.model(images)     # (T, B, num_classes)
                preds_size = torch.IntTensor([preds.size(0)] * images.size(0))

                # 손실 계산
                loss = self.criterion(preds, targets, preds_size, target_lengths)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)  # gradient clipping
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)

            # 2) 검증 모드
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets, target_lengths in self.val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    preds = self.model(images)
                    preds_size = torch.IntTensor([preds.size(0)] * images.size(0))
                    loss = self.criterion(preds, targets, preds_size, target_lengths)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(self.val_loader)

            # 학습률 스케줄러 업데이트 (검증 손실 기준 감소)
            self.scheduler.step(avg_val_loss)

            print(f"Epoch [{epoch}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

            # 가장 낮은 검증 손실일 때 모델 저장
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_path = os.path.join(self.model_save_dir, f"crnn_epoch{epoch}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  ★ 모델 저장: {save_path}")

        print("Training complete.")
