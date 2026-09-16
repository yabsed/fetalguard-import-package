"""규칙 기반(rules-based) 특성 추출 + XGBoost 기준선.

Chiou et al. (2025)의 특성 추출 파이프라인 재구현. 정의의 준거는 medRxiv
프리프린트(doi:10.1101/2024.03.05.24303805) Appendix A.2다 — 출판본(npj Women's
Health)에는 특성 이름만 나열되어 있으나, 프리프린트에는 검출·분류 규칙이
수치까지 명시되어 있다.

추출 특성 17개 = Methods 목록 16개 + 조기감속(본문 목록에는 없으나 Fig. 2B
상관행렬에 등장):
  자궁수축 수(#UC), FHR 기저선, 기저선 변이도, 가속 수,
  감속 수(전체/조기/후기/변동/중증/장기), FHR 히스토그램의
  폭·최소·최대·중앙값·평균·최빈값·표준편차

Appendix A.2가 명시하는 규칙 (전부 프리프린트 원문 기준):
  - 전처리: 짧은 결측(≤15초) 보간 → 120포인트(30초) 롤링 평활 → 긴 결측도
    선형 보간
  - 자궁수축: "bell-shaped gradual increase, 총 지속 45~120초"인 UC 피크
  - 기저선: 10분 창의 **평균**으로 초기화, 가속·감속을 제외하며 반복 갱신,
    변화량 < 0.5에서 수렴
  - 기저선 변이도: 기저선 요동을 이루는 소규모 피크·트로프 사이 진폭 변화의 평균
  - 가속: 기저선 +15 bpm 초과·15초 초과 지속·개시→정점 30초 미만·총 지속 10분 미만
  - 감속: 기저선 −15 bpm 초과 하강·15초 초과 지속
  - 후기감속: 감속 **개시**가 대응(최근접) UC **개시** 20초 후 ~ UC 종료 전
  - 후기가 아닌 감속: 지속 <3분이면 조기, 3~5분이면 장기(prolonged),
    >5분이면 중증(severe) — 중증이 심박 수치가 아니라 지속시간 기준임에 주의
  - 변동감속: 개시→최저점 30초 미만 (다른 유형과 독립으로 계수)

논문이 공개하지 않은 세부(UC 피크 탐지 파라미터, 기저선 초기 창의 위치,
변이도 극점 산정, '최근접 UC'의 거리 척도)만 [재구현 선택]으로 표기한다.

[재구현 선택] 지속시간 경계의 이산화: "more than 15 seconds" 등의 임계값에서
정확히 경계값(4 Hz에서 60표본)인 런의 귀속은 관례의 문제다 — 여기서는 경계값을
포함(≥)한다. 아래 임계 상수들의 비교 연산이 그 관례를 따른다.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths

from .preprocessing import impute_short_gaps, smooth_masked
FS = 4

ACC_MIN_BPM = 15          # 가속: 기저선 대비 상승 폭 (A.2 명시)
ACC_MIN_S = 15            # 가속: 최소 지속 (A.2 명시)
ACC_ONSET_PEAK_MAX_S = 30  # 가속: 개시→정점 상한 (A.2 명시)
ACC_MAX_S = 600           # 가속: 총 지속 상한 10분 (A.2 명시)
DEC_MIN_BPM = 15          # 감속: 기저선 대비 하강 폭 (A.2 명시)
DEC_MIN_S = 15            # 감속: 최소 지속 (A.2 명시)
LATE_UC_ONSET_DELAY_S = 20  # 후기감속: UC 개시 후 지연 하한 (A.2 명시)
EARLY_MAX_S = 180         # 비후기 감속 중 조기: 지속 < 3분 (A.2 명시)
PROLONGED_MAX_S = 300     # 비후기 감속 중 장기: 3~5분 (A.2 명시); 초과는 중증
VAR_ONSET_NADIR_MAX_S = 30  # 변동감속: 개시→최저점 < 30초 (A.2 명시)
UC_DUR_MIN_S = 45         # 자궁수축 총 지속 하한 (A.2 명시)
UC_DUR_MAX_S = 120        # 자궁수축 총 지속 상한 (A.2 명시)
BASELINE_TOL = 0.5        # 기저선 반복 수렴 조건 (A.2 명시)
BASELINE_WIN_S = 600      # 기저선 초기 추정 창 10분 (A.2 명시; 창 위치는 아래 참조)

FEATURE_NAMES = [
    "n_uc", "baseline", "baseline_variability",
    "n_accelerations", "n_decelerations", "n_early_decelerations",
    "n_variable_decelerations", "n_severe_decelerations",
    "n_late_decelerations", "n_prolonged_decelerations",
    "hist_width", "hist_min", "hist_max", "hist_median",
    "hist_mean", "hist_mode", "hist_std",
]


def _runs(mask, min_len=1):
    """True 런의 (start, end) 목록 (end exclusive, 길이 min_len 이상만)."""
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def detect_contractions(uc, fs=FS):
    """자궁수축 검출 → (onset, peak, end) 표본 인덱스 목록.

    A.2 명시: "bell-shaped gradual increase in the UC signal between 45 and
    120 seconds in total duration"인 피크. 총 지속 45~120초 필터는 명시된
    값이고, 피크 후보 탐지(prominence = 신호 표준편차의 0.5배, 최소 간격
    60초)와 수축 경계 산정(피크 높이의 90% 하강 지점, peak_widths
    rel_height=0.9)은 [재구현 선택]이다.
    """
    if uc.std() == 0:
        return []
    peaks, _ = find_peaks(uc, distance=int(60 * fs), prominence=uc.std() * 0.5)
    if len(peaks) == 0:
        return []
    _, _, lefts, rights = peak_widths(uc, peaks, rel_height=0.9)
    out = []
    for p, l, r in zip(peaks, lefts, rights):
        dur_s = (r - l) / fs
        if UC_DUR_MIN_S <= dur_s <= UC_DUR_MAX_S:
            out.append((int(round(l)), int(p), int(round(r))))
    return out


def estimate_baseline(fhr, fs=FS, tol=BASELINE_TOL):
    """기저선: 10분 창 평균으로 초기화, 가속·감속 후보를 제외하며 반복 갱신.

    A.2 명시: "The initial FHR baseline was the average value of the FHR
    signal within the [10-minute] window and the baseline was iteratively
    updated as accelerations and decelerations were identified and removed
    from the baseline calculation. This process was repeated until the
    magnitude of the FHR baseline change was less than 0.5."

    [재구현 선택] 30분 크롭의 어느 10분을 초기 창으로 쓰는지는 명시가 없다 —
    유효(>0) 표본의 앞 10분을 쓴다. 갱신 단계의 대푯값도 초기값과 같은
    평균으로 통일한다.
    """
    valid = fhr[fhr > 0]
    if len(valid) == 0:
        return np.nan, np.zeros(len(fhr), dtype=bool)
    win = int(BASELINE_WIN_S * fs)
    baseline = float(np.mean(valid[:min(win, len(valid))]))
    keep = np.ones(len(fhr), dtype=bool)
    for _ in range(50):
        dev = fhr - baseline
        events = (dev >= ACC_MIN_BPM) | (dev <= -DEC_MIN_BPM)
        keep = ~(events | (fhr == 0))
        new = float(np.mean(fhr[keep])) if keep.any() else baseline
        if abs(new - baseline) < tol:
            baseline = new
            break
        baseline = new
    return baseline, keep


def baseline_variability(fhr, keep, fs=FS):
    """기저선 변이도: 소규모 피크·트로프 사이 진폭 변화의 평균.

    A.2 명시: "identifying the average amplitude change between successive
    small-scale peaks and troughs that represent baseline fluctuations."

    [재구현 선택] '소규모 피크·트로프'는 이벤트(가속·감속)와 결측을 제외한
    연속 구간 안에서 1계 차분의 부호가 바뀌는 국소 극점으로 정의한다 —
    입력이 이미 30초 롤링 평활을 거쳤으므로 남은 극점이 곧 기저선 요동이다.
    연속 구간마다 이웃 극점 간 |진폭 차|를 모아 전체 평균한다.
    """
    amps = []
    for s, e in _runs(keep, 3):
        seg = fhr[s:e]
        ext_vals = []
        direction = 0.0
        for i in range(1, len(seg)):
            step = seg[i] - seg[i - 1]
            if step == 0:
                continue
            s_dir = 1.0 if step > 0 else -1.0
            if direction != 0.0 and s_dir != direction:
                ext_vals.append(seg[i - 1])
            direction = s_dir
        if len(ext_vals) >= 2:
            amps.extend(np.abs(np.diff(np.asarray(ext_vals))))
    return float(np.mean(amps)) if amps else 0.0


def detect_accelerations(fhr, baseline, fs=FS):
    """가속 (start, end) 목록 (A.2: >15 bpm 상승, >15초, 개시→정점 <30초,
    총 지속 <10분)."""
    dev = fhr - baseline
    out = []
    for s, e in _runs((dev >= ACC_MIN_BPM) & (fhr > 0), int(np.ceil(ACC_MIN_S * fs))):
        if (e - s) >= ACC_MAX_S * fs:
            continue
        peak = s + int(np.argmax(fhr[s:e]))
        if (peak - s) < ACC_ONSET_PEAK_MAX_S * fs:
            out.append((s, e))
    return out


def detect_decelerations(fhr, baseline, fs=FS):
    """감속 (start, end) 목록 (A.2: >15 bpm 하강, >15초)."""
    dev = fhr - baseline
    return _runs((dev <= -DEC_MIN_BPM) & (fhr > 0), int(np.ceil(DEC_MIN_S * fs)))


def classify_deceleration(s, e, nadir, contractions, fs=FS):
    """감속 하나 → (주 유형, 변동 여부). A.2 명시 규칙:

    1) 후기: 감속 개시가 최근접 UC의 개시 20초 후 ~ UC 종료 전
    2) 후기가 아니면 지속시간으로: <3분 조기 / 3~5분 장기(prolonged) /
       >5분 중증(severe)
    3) 변동: 개시→최저점 <30초 — 주 유형과 독립으로 계수
    [재구현 선택] '최근접 UC'는 감속 개시와 UC 개시의 거리로 고른다.
    """
    dur_s = (e - s) / fs
    is_late = False
    if contractions:
        uc_on, _, uc_end = min(contractions, key=lambda c: abs(s - c[0]))
        is_late = (s > uc_on + LATE_UC_ONSET_DELAY_S * fs) and (s < uc_end)
    if is_late:
        main = "late"
    elif dur_s < EARLY_MAX_S:
        main = "early"
    elif dur_s <= PROLONGED_MAX_S:
        main = "prolonged"
    else:
        main = "severe"
    is_variable = (nadir - s) < VAR_ONSET_NADIR_MAX_S * fs
    return main, is_variable


def extract_features(fhr, uc, fs=FS):
    """정제된 30분 크롭(4 Hz)에서 특성 17개 추출. 입력 전처리는 호출부 책임."""
    feats = {}

    # --- 자궁수축 (A.2: 종형 상승, 총 지속 45~120초)
    contractions = detect_contractions(uc, fs)
    feats["n_uc"] = len(contractions)

    # --- 기저선·변이도
    baseline, keep = estimate_baseline(fhr, fs)
    feats["baseline"] = baseline
    feats["baseline_variability"] = baseline_variability(fhr, keep, fs)

    # --- 가속·감속 (A.2 명시 조건)
    accs = detect_accelerations(fhr, baseline, fs)
    decs = detect_decelerations(fhr, baseline, fs)
    feats["n_accelerations"] = len(accs)
    feats["n_decelerations"] = len(decs)

    # --- 감속 유형 분류 (A.2 명시 규칙 — classify_deceleration 주석 참조)
    counts = {"early": 0, "late": 0, "prolonged": 0, "severe": 0}
    n_variable = 0
    for s, e in decs:
        nadir = s + int(np.argmin(fhr[s:e]))
        main, is_var = classify_deceleration(s, e, nadir, contractions, fs)
        counts[main] += 1
        n_variable += is_var
    feats["n_early_decelerations"] = counts["early"]
    feats["n_late_decelerations"] = counts["late"]
    feats["n_variable_decelerations"] = n_variable
    feats["n_severe_decelerations"] = counts["severe"]
    feats["n_prolonged_decelerations"] = counts["prolonged"]

    # --- FHR 히스토그램 통계
    valid = fhr[fhr > 0]
    if len(valid):
        feats["hist_min"] = float(valid.min())
        feats["hist_max"] = float(valid.max())
        feats["hist_width"] = feats["hist_max"] - feats["hist_min"]
        feats["hist_median"] = float(np.median(valid))
        feats["hist_mean"] = float(valid.mean())
        # [재구현 선택] 히스토그램 구간 수 24는 논문에 없다 (최빈값 산출에만 영향)
        hist, edges = np.histogram(valid, bins=24)
        feats["hist_mode"] = float((edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2)
        feats["hist_std"] = float(valid.std())
    else:
        for k in ("hist_min", "hist_max", "hist_width", "hist_median",
                  "hist_mean", "hist_mode", "hist_std"):
            feats[k] = 0.0
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)


def preprocess_for_features(fhr, uc, fs=FS, smooth_seconds=30):
    """특성추출 전용 전처리 (A.2 명시 순서).

    짧은 결측 보간 → 30초 rolling 평활 → 긴 결측도 선형 보간.
    smooth_seconds=0 팔은 같은 결측 정책을 유지한 채 평활만 생략한다.

    평활은 논문의 "smoothed with a rolling window of 120 time points"를 결측 표지가
    있는 신호의 rolling mean으로 구현한다(`preprocess.smooth_masked`) — 유효 표본만
    평균하므로, 아직 0으로 남아 있는 긴 결측이 평균에 섞여 경계를 오염시키거나
    계곡으로 바뀌어 마지막 보간 단계가 결측을 놓치는 일이 없다.

    [재구현 선택] A.1의 품질 평가(5분 창 결측 ≥50% → 0 재지정)는 적용하지 않는다.
    A.2는 "up to and including the imputation of missing values"의 단계를 따른다고
    쓰지만, 곧이어 남은 긴 결측까지 전부 선형 보간하므로 0 재지정과 양립하지
    않는다 — 최종 신호에 0이 남지 않는 A.2의 서술을 우선했다.
    """
    fhr = impute_short_gaps(fhr, int(15 * fs))
    uc = impute_short_gaps(uc, int(15 * fs))
    if smooth_seconds not in (0, 30):
        raise ValueError("Predeclared feature smoothing arms are 0 and 30 seconds")
    if smooth_seconds:
        fhr = smooth_masked(fhr, int(smooth_seconds * fs), fhr == 0)
        uc = smooth_masked(uc, int(smooth_seconds * fs), uc == 0)
    fhr = impute_short_gaps(fhr, len(fhr))   # 남은 결측 전부 선형 보간
    uc = impute_short_gaps(uc, len(uc))
    return fhr, uc
