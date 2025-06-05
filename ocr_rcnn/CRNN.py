# CRNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# CRNN(Convolutional Recurrent Neural Network) 모델 정의
# ------------------------------------------------------------
class CRNN(nn.Module):
    """
    이미지 기반 시퀀스 인식(Sequence Recognition)을 위해,
    Convolution (CNN) + Recurrent (RNN: LSTM) + Linear 레이어를 결합한 구조.
    주로 OCR(문자 인식)에서 글자 순서를 예측할 때 사용.
    """
    def __init__(self, param):
        """
        Arguments:
          param (dict): 하이퍼파라미터 딕셔너리
            - imgH    : 입력 이미지 높이(가로 크기는 데이터 로더에서 조정)
            - n_classes: 예측할 클래스(문자 집합 크기 + 1(CTC blank))
            - nh      : LSTM 히든 레이어 크기
            - nc      : 입력 채널 수(흑백=1, RGB=3 등)
        """
        super(CRNN, self).__init__()
        assert param['imgH'] % 16 == 0, "imgH는 16의 배수여야 합니다!"
        self.imgH = param['imgH']
        self.n_classes = param['n_classes']
        self.nc = param.get('nc', 1)  # 디폴트 흑백
        self.nh = param.get('nh', 256)

        # --------------------------------------------------------
        # CNN 부분: 여러 개의 Convolution + BatchNorm + ReLU + Pooling
        # --------------------------------------------------------
        ks = [3, 3, 3, 3, 3, 3, 2]  # 커널 크기
        ps = [1, 1, 1, 1, 1, 1, 0]  # 패딩
        ss = [1, 1, 1, 1, 1, 1, 1]  # 스트라이드
        nm = [64, 128, 256, 256, 512, 512, 512]  # 출력 채널 수

        cnn = nn.Sequential()
        def conv_relu(i, batch_norm=False):
            """
            i번째 레이어 구성 함수:
              Conv2d → (BatchNorm2d) → ReLU → (MaxPool2d)
              필요한 추가 파라미터는 nm, ks, ps, ss 리스트에서 가져옴.
            """
            nIn = self.nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            # Convolution 2D
            cnn.add_module(f"conv{i}", nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            # Batch Normalization (선택적)
            if batch_norm:
                cnn.add_module(f"batchnorm{i}", nn.BatchNorm2d(nOut))
            # 활성화 함수 ReLU
            cnn.add_module(f"relu{i}", nn.ReLU(True))
            # 다섯 번째, 여섯 번째 레이어 이후 풀링 형태가 달라짐
            if i in [0, 1, 3, 5]:
                cnn.add_module(f"pool{i}", nn.MaxPool2d(2, 2))
            elif i == 2:
                # 풀링 크기 (2,1)
                cnn.add_module(f"pool{i}", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
            elif i == 4:
                cnn.add_module(f"pool{i}", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        # 7개의 conv_relu 레이어 반복하여 네트워크 생성
        conv_relu(0)
        conv_relu(1)
        conv_relu(2, batch_norm=True)
        conv_relu(3)
        conv_relu(4, batch_norm=True)
        conv_relu(5)
        conv_relu(6)

        self.cnn = cnn

        # --------------------------------------------------------
        # RNN 부분: 양방향 LSTM 2개 층으로 구성
        # --------------------------------------------------------
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, self.nh, self.nh),
            BidirectionalLSTM(self.nh, self.nh, self.n_classes),
        )

    def forward(self, input):
        """
        Forward pass (추론) 함수.
        입력: (batch, channel=1, imgH, imgW)
        1) CNN을 통과시켜 feature map 획득 → (batch, c, h, w)
        2) 전치(transpose) + reshape를 통해 시퀀스로 변환: (w, batch, c*h)
        3) RNN을 통과시켜 텍스트 시퀀스 예측: (w, batch, n_classes)
        """
        # (batch, nc, imgH, imgW) → CNN
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # h가 1이어야 RNN 입력 시 편리 (CNN 아키텍처 상 h이 1이어야 함)
        assert h == 1, "이미지 높이 imgH 내부 구조가 잘못되어 h != 1 발생"
        conv = conv.squeeze(2)  # (batch, c, w)
        # (batch, c, w) → (w, batch, c)
        conv = conv.permute(2, 0, 1)
        # RNN을 통해 시퀀스 인코딩(양방향 LSTM)
        output = self.rnn(conv)
        return output  # (seq_len=w, batch, n_classes)


# ------------------------------------------------------------
# Bidirectional LSTM 유닛 정의
# ------------------------------------------------------------
class BidirectionalLSTM(nn.Module):
    """
    양방향 LSTM 레이어 + Fully-Connected 레이어로 구성.
    입력: (seq_len, batch, input_size)
    출력: (seq_len, batch, output_size)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        # LSTM 레이어 정의
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        # LSTM 출력 두 배(양방향) → FC 레이어로 차원 축소
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # input: (seq_len, batch, input_size)
        recurrent, _ = self.rnn(input)  # recurrent: (seq_len, batch, hidden_size*2)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)  # (seq_len*batch, hidden_size*2)
        output = self.fc(t_rec)           # (seq_len*batch, output_size)
        output = output.view(T, b, -1)    # (seq_len, batch, output_size)
        return output
