# CTCLoss.py

import torch
import torch.nn as nn

# ------------------------------------------------------------
# CTC 손실 함수 래퍼
# ------------------------------------------------------------
class CTCLoss(nn.Module):
    """
    PyTorch 내장 CTCLoss (Connectionist Temporal Classification) 래퍼 클래스
    CRNN과 같은 시퀀스 인식 네트워크 학습 시 사용.
    """
    def __init__(self, blank=0):
        """
        Arguments:
          blank (int): CTC blank 토큰 인덱스 (보통 0)
        """
        super(CTCLoss, self).__init__()
        # reduction='mean'는 배치 내 손실 평균 계산
        # zero_infinity=True는 무한대 손실을 0으로 처리
        self.loss_func = nn.CTCLoss(blank=blank, reduction='mean', zero_infinity=True)

    def forward(self, preds, targets, pred_lengths, target_lengths):
        """
        CTC 손실 계산

        Arguments:
          preds          : (T, N, C) — 모델 예측 점수 (T=시퀀스 길이, N=배치 크기, C=클래스 수)
          targets        : 실제 레이블 인덱스 시퀀스 (1차원 텐서)
          pred_lengths   : 각 배치별 예측 시퀀스 길이 벡터 (1차원 LongTensor)
          target_lengths : 각 배치별 실제 시퀀스 길이 벡터 (1차원 LongTensor)

        Returns:
          ctc_loss (Tensor): 스칼라 손실 값
        """
        # CTCLoss 입력 형식에 맞추어 차원 변경
        # preds는 이미 (seq_len, batch, num_classes) 형태여야 함
        # 입력이 (T, N, C), targets는 (sum(target_lengths),)
        ctc_loss = self.loss_func(preds.log_softmax(2), targets, pred_lengths, target_lengths)
        return ctc_loss
