"""CTG-net (Ogasawara et al., 2021) 구조를 한국 CTG의 5분 창에 맞춘 경량 CNN.

출처: project/experiment-13/2-google-reproduction/model.py 의 CTGNet을 그대로 옮기고
입력 길이만 바꿨다. 그 파일은 Chiou et al. (2025, npj Women's Health)이 기반 모델로
채택한 CTG-net을 논문 기술로부터 재구현한 것이다.

원 설계의 요지:
  conv1  30초 길이 시간 커널로 두 채널(FHR·TOCO)을 함께 합성곱
  conv2  depthwise 합성곱 — 채널(필터)별 독립 처리
  conv3  separable 합성곱 (depthwise + pointwise)
  flatten → 완전연결 1개 → 로짓

구글은 552건이라는 작은 데이터에 맞춰 이 구조를 약 2,100 파라미터로 유지했다.
experiment-8의 SE-ResNet50Local은 19,010,993 파라미터를 7,902개 학습 세그먼트에
썼다(표본당 약 2,400 파라미터). 이 파일은 그 대비를 실험으로 확인하기 위한 것이다.

입력 규약은 experiment-8과 동일하다: [B, 2, 157]
  = 채널별 z-정규화 150샘플 파형 + 논문 명시 통계량 7개
시간 커널 k1은 0.5 Hz에서의 30초 = 15샘플로 잡았다(원 설계의 '30초 커널'에 대응).
"""

from __future__ import annotations

import torch
import torch.nn as nn

INPUT_LEN = 157   # 150 파형 샘플 + 7 통계량
K_30S = 15        # 0.5 Hz에서 30초


class CTGNetMini(nn.Module):
    def __init__(self, width: int = 16, in_channels: int = 2, k1: int = K_30S,
                 k2: int = 15, k3: int = 15, pool: int = 2, dropout: float = 0.25,
                 input_len: int = INPUT_LEN) -> None:
        """width가 세 합성곱 층의 필터 수(f1=f2=f3)다 — 용량 사다리의 유일한 손잡이."""
        super().__init__()
        f1 = f2 = f3 = width
        self.conv1 = nn.Conv1d(in_channels, f1, k1)
        self.bn1 = nn.BatchNorm1d(f1)
        self.conv2 = nn.Conv1d(f1, f2, k2, groups=f1)          # depthwise
        self.bn2 = nn.BatchNorm1d(f2)
        self.conv3_dw = nn.Conv1d(f2, f2, k3, groups=f2)       # separable: depthwise
        self.conv3_pw = nn.Conv1d(f2, f3, 1)                   #            + pointwise
        self.bn3 = nn.BatchNorm1d(f3)
        self.act = nn.ELU()
        self.pool = nn.AvgPool1d(pool)
        self.drop = nn.Dropout(dropout)

        length = input_len - k1 + 1
        length = (length - k2 + 1) // pool
        length = (length - k3 + 1) // pool
        if length <= 0:
            raise ValueError(f"입력 {input_len}에 커널이 너무 큼 (남은 길이 {length})")
        self.fc = nn.Linear(f3 * length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.drop(self.pool(self.act(x)))
        x = self.bn3(self.conv3_pw(self.conv3_dw(x)))
        x = self.drop(self.pool(self.act(x)))
        return self.fc(torch.flatten(x, 1)).squeeze(1)   # 로짓


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for w in (4, 8, 16, 32, 64, 128, 256):
        m = CTGNetMini(width=w)
        out = m(torch.zeros(2, 2, INPUT_LEN))
        print(f"width {w:3d} → 파라미터 {count_parameters(m):>9,}개, 출력 {tuple(out.shape)}")
