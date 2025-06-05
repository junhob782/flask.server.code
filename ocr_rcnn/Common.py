# Common.py

import torch
import numpy as np
import os, math, re
from textdistance import levenshtein as lev  # 문자열 거리 계산(Levenshtein 거리) 함수 임포트

# ------------------------------------------------------------
# 유틸리티 함수: 디렉터리 생성
# ------------------------------------------------------------
def wrap_dir(path):
    """
    주어진 경로(path)가 존재하지 않으면 디렉터리를 생성하고,
    항상 해당 경로 문자열을 반환합니다.
    """
    if not os.path.exists(path):
        os.makedirs(path)  # 디렉터리가 없다면 생성
    return path

# ------------------------------------------------------------
# 데이터 분리 함수: 학습/검증/테스트 데이터셋 분할
# ------------------------------------------------------------
def split(samples, **kwargs):
    """
    입력된 샘플 리스트(samples)를 train/val/test로 분리하는 함수.

    Arguments:
      samples (list): 샘플 데이터(파일 경로 혹은 (이미지, 라벨) 튜플) 리스트
      kwargs: 'train_ratio', 'val_ratio', 'test_ratio' 등을 키워드 인자로 받을 수 있음.

    Returns:
      data_train (list), data_val (list), data_test (list)
    """
    total = len(samples)
    # kwargs에서 비율이 주어지지 않으면 기본으로 0.8/0.1/0.1 사용
    train_ratio = kwargs.get('train_ratio', 0.8)
    val_ratio = kwargs.get('val_ratio', 0.1)
    test_ratio = kwargs.get('test_ratio', 0.1)
    # 샘플 개수에 비율을 곱해서 인덱스 계산
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    # 테스트는 나머지 전부
    data_train = samples[:n_train]
    data_val = samples[n_train:n_train + n_val]
    data_test = samples[n_train + n_val:]
    return data_train, data_val, data_test

# ------------------------------------------------------------
# 정확도 측정 클래스
# ------------------------------------------------------------
class AccuracyMeasure:
    """
    CRNN 모델 예측 결과와 정답 레이블을 비교하여 정확도를 계산.
    Levenshtein 거리(편집 거리)를 활용해 문자열 유사도 계산.
    """
    def __init__(self, alphabet, target_transform):
        """
        Arguments:
          alphabet (str): 허용되는 문자 집합(예: '0123456789가-힣ABCDEFGHIJKLMNOPQRSTUVWXYZ')
          target_transform: 정답 라벨(문자열)을 인덱스 시퀀스로 변환하는 객체
        """
        self.alphabet = alphabet
        self.target_transform = target_transform
        self.reset()

    def reset(self):
        """
        누적 정확도 기록을 초기화.
        """
        self.n_correct = 0  # 정확하게 예측된 샘플 수
        self.n_total = 0    # 전체 샘플 수
        self.total_distance = 0  # 누적 편집 거리(Levenshtein)

    def update(self, preds, targets):
        """
        한 배치(batch)의 예측(preds)과 정답(targets)을 비교 후 내부 통계 업데이트.

        Arguments:
          preds (list[str]): 모델이 예측한 문자열 리스트
          targets (list[str]): 실제 정답 문자열 리스트
        """
        for p, t in zip(preds, targets):
            self.n_total += 1
            if p == t:
                self.n_correct += 1
            # 문자열 길이 기준 편집 거리 계산
            dist = lev(p, t)
            self.total_distance += dist

    def get_accuracy(self):
        """
        전체 정확도(정확히 일치한 비율)와 평균 편집 거리를 반환.

        Returns: (accuracy, avg_distance)
        """
        if self.n_total == 0:
            return 0, 0
        accuracy = self.n_correct / self.n_total
        avg_distance = self.total_distance / self.n_total
        return accuracy, avg_distance

# ------------------------------------------------------------
# 모델 평가 함수
# ------------------------------------------------------------
def Eval(model, data_loader, device, target_transform):
    """
    검증(validation) 혹은 테스트(test) 데이터에 대해 모델을 평가하고,
    AccuracyMeasure를 통해 정확도와 평균 편집 거리(Levenshtein)를 계산.

    Arguments:
      model: CRNN 모델 객체 (torch.nn.Module)
      data_loader: DataLoader 객체(검증/테스트용)
      device: 'cpu' 혹은 'cuda'
      target_transform: 레이블 변환 객체
    Returns:
      accuracy (float), avg_distance (float)
    """
    model.eval()  # 평가 모드로 전환
    acc_measure = AccuracyMeasure(alphabet=target_transform.alphabet,
                                  target_transform=target_transform)
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            # 모델 추론: (T, B, num_classes) 형태 예측 결과
            preds = model(images)
            # 예측된 인덱스(가장 높은 확률)를 가져옴
            _, preds_index = preds.max(2)
            # 인덱스를 문자열로 변환
            pred_strings = target_transform.decode(preds_index)
            # 레이블(문자열)
            true_strings = [target_transform.decode_label(l) for l in labels]
            # 정확도 측정 클래스 업데이트
            acc_measure.update(pred_strings, true_strings)
    return acc_measure.get_accuracy()

# ------------------------------------------------------------
# 문자열 ↔ 인덱스 변환 클래스 (CTC 디코더)
# ------------------------------------------------------------
class LabelCodec:
    """
    CTC(연속 시간 정렬) 디코더를 구현하기 위해,
    문자 집합(alphabet)과 같은 인덱스 변환 메서드를 제공합니다.

    encode(text) : 문자열 → 인덱스 자릿수 리스트
    decode(indices) : 인덱스 리스트 → 문자열
    """
    def __init__(self, alphabet):
        """
        Arguments:
          alphabet (str): 허용되는 문자 집합(예: '0123456789가-힣ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        """
        self.alphabet = alphabet
        # CTC에서 blank 토큰 인덱스로 사용하기 위해 마지막에 추가
        self.mapping = {c: i for i, c in enumerate(alphabet)}
        self.rev_mapping = {i: c for i, c in enumerate(alphabet)}
        self.blank_index = len(alphabet)  # blank 토큰 인덱스

    def encode(self, text):
        """
        주어진 문자열(text)을 인덱스 리스트로 변환.
        e.g., '12가3456' → [‘1’->’0’, ‘2’->’1’, ‘가’->’10’, …]
        """
        indices = []
        for char in text:
            if char in self.mapping:
                indices.append(self.mapping[char])
            else:
                # 알파벳에 없는 문자는 무시하거나 처리
                pass
        return torch.LongTensor(indices)

    def decode(self, preds_index, preds_size):
        """
        CTC 디코딩: 연속된 동일 인덱스를 하나로 합치고, blank 토큰 제거.
        arguments:
          preds_index: (T*B,) 형태의 인덱스 시퀀스 텐서
          preds_size: 각 배치별 시간 축 길이가 담긴 텐서
        Returns:
          decoded_strs (list[str])
        """
        decoded_strs = []
        index_start = 0
        for size in preds_size:
            # size 만큼 잘라서 하나의 시퀀스로 처리
            seq = preds_index[index_start:index_start + size]
            index_start += size
            # CTC 룰에 따라 연속 중복 인덱스 제거, blank 토큰 제거
            prev = -1
            decoded = ""
            for idx in seq:
                idx = idx.item()
                if idx != prev and idx != self.blank_index:
                    decoded += self.rev_mapping.get(idx, "")
                prev = idx
            decoded_strs.append(decoded)
        return decoded_strs
