"""Extracted unchanged from experiment-15/preprocess.py."""
import numpy as np

def gap_runs(mask):
    """True 구간의 (start, end) 런 목록. end는 exclusive."""
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))

def impute_short_gaps(x, max_gap_samples):
    """3단계: max_gap_samples 이하의 0-런을 선형 보간. 더 긴 런은 0 유지.

    [재구현 선택] 논문은 "shorter than 15 seconds"를 보간한다고 쓴다 — 정확히
    15초(4 Hz에서 60표본)인 런의 귀속은 이산화 관례의 문제로, 여기서는 보간에
    포함한다(≤).
    """
    x = x.copy()
    for s, e in gap_runs(x == 0):
        if e - s > max_gap_samples:
            continue  # 긴 결측: 0 유지 (논문: set them to zero)
        left = x[s - 1] if s > 0 else np.nan
        right = x[e] if e < len(x) else np.nan
        if np.isnan(left) and np.isnan(right):
            continue
        if np.isnan(left):
            x[s:e] = right
        elif np.isnan(right):
            x[s:e] = left
        else:
            x[s:e] = np.linspace(left, right, e - s + 2)[1:-1]
    return x

def smooth_masked(x, window, missing, kernel=None):
    """결측 표지를 존중하는 rolling (가중)평균 — 창 안의 **유효 표본만** 평균한다.

    논문의 결측 0은 심박수가 아니라 "여기는 신호가 없었다"는 표지다(set them to
    zero to maintain temporal dependencies). 뒤이은 평활(smoothed to reduce the
    effect of noise / smoothed with a rolling window)은 신호의 노이즈를 누르는
    단계이지 표지를 값으로 평균하는 단계가 아니므로, 분자·분모를 함께 합성곱해
    유효 표본만으로 평균을 내고 결측 위치는 0으로 되돌린다. 결측을 NaN으로 둔
    pandas의 rolling(window, min_periods=1).mean()과 동치이며(boxcar 기준),
    0을 값으로 취급하는 단순 이동평균이 만드는 두 인공물(경계 오염, 긴 결측의
    계곡화)이 없다.

    kernel: 가중 창 (기본 None = boxcar). 신경망 파이프라인은 A.1 명시대로
    np.hamming(15)를 넘긴다.
    """
    if window <= 1:
        return x
    valid = (~missing).astype(np.float64)
    kernel = np.ones(window) if kernel is None else np.asarray(kernel, dtype=np.float64)
    num = np.convolve(np.where(missing, 0.0, x), kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.divide(num, den, out=np.zeros(len(x), dtype=np.float64), where=den > 0)
    out[missing] = 0.0
    return out
