"""Report §4.1 / §5.1: exp15 Cat28, parameterized by actual sampling rate.

Korean CTG has 2-second samples. A ±60s lag is tested at the native 2s
spacing; no fictitious 1s resolution is introduced. Event durations use
ceil(seconds * fs), fixing int(7.5) truncation in the 4Hz source code.
"""
import numpy as np
from . import feature_rules as rules

CAT28 = [
    "fhr_mean", "fhr_min", "fhr_max", "fhr_sd", "stv", "brady_frac", "tachy_frac",
    "n_decel", "decel_max_depth", "decel_time_frac", "n_accel",
    "toco_mean", "toco_max", "toco_sd", "n_contractions",
    "ft_corr0", "ft_corr_min", "ft_corr_min_lag_s",
    "figo_baseline", "figo_baseline_var", "n_early_decel", "n_late_decel",
    "n_variable_decel", "n_severe_decel", "n_prolonged_decel",
    "hist_width", "hist_median", "hist_mode",
]
# Original 18-feature baseline, evaluated with exactly the same preprocessing
# and grouped splits as the 28-feature extension (not a historical score).
CAT18 = CAT28[:18]
GROUPS = {
    "signal": ["fhr_mean", "fhr_min", "fhr_max", "fhr_sd"],
    "baseline": ["figo_baseline", "hist_median", "hist_mode", "hist_width"],
    "variability": ["stv", "figo_baseline_var"],
    "range": ["brady_frac", "tachy_frac"],
    "events": ["n_decel", "decel_max_depth", "decel_time_frac", "n_accel",
               "n_early_decel", "n_late_decel", "n_variable_decel", "n_severe_decel", "n_prolonged_decel"],
    "contractions_coupling": ["toco_mean", "toco_max", "toco_sd", "n_contractions",
                              "ft_corr0", "ft_corr_min", "ft_corr_min_lag_s"],
}
ROBUST = [k for k in CAT28 if k not in GROUPS["events"] + ["n_contractions", "figo_baseline", "figo_baseline_var"]]


def corr(a, b):
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_segment(fhr, toco, fs):
    fhr, toco = np.asarray(fhr, float), np.asarray(toco, float)
    if len(fhr) != len(toco) or len(fhr) < 2 or not np.isfinite([fhr, toco]).all():
        raise ValueError("Invalid numeric segment")
    base, keep = rules.estimate_baseline(fhr, fs)
    decs = rules.detect_decelerations(fhr, base, fs)
    accs = rules.detect_accelerations(fhr, base, fs)
    contractions = rules.detect_contractions(toco, fs)
    f = dict(fhr_mean=fhr.mean(), fhr_min=fhr.min(), fhr_max=fhr.max(), fhr_sd=fhr.std(),
             stv=np.abs(np.diff(fhr)).mean(), brady_frac=(fhr < 110).mean(), tachy_frac=(fhr > 160).mean(),
             n_decel=len(decs), n_accel=len(accs),
             decel_max_depth=max((base - fhr[s:e].min() for s, e in decs), default=0.0),
             decel_time_frac=sum(e - s for s, e in decs) / len(fhr),
             toco_mean=toco.mean(), toco_max=toco.max(), toco_sd=toco.std(), n_contractions=len(contractions),
             figo_baseline=base, figo_baseline_var=rules.baseline_variability(fhr, keep, fs))
    counts = dict(early=0, late=0, variable=0, prolonged=0, severe=0)
    for s, e in decs:
        kind, var = rules.classify_deceleration(s, e, s + int(np.argmin(fhr[s:e])), contractions, fs)
        counts[kind] += 1
        counts["variable"] += int(var)
    f.update({f"n_{k}_decel": v for k, v in counts.items()})
    hist, edges = np.histogram(fhr, bins=24)
    k = int(hist.argmax())
    f.update(hist_width=fhr.max() - fhr.min(), hist_median=np.median(fhr), hist_mode=(edges[k] + edges[k + 1]) / 2)
    best_r, best_lag = corr(fhr, toco), 0
    f["ft_corr0"] = best_r
    limit = min(int(60 * fs), len(fhr) - 2)
    step = max(1, int(round(fs)))
    for lag in range(-limit, limit + 1, step):
        if lag == 0:
            continue
        a, b = (fhr[lag:], toco[:-lag]) if lag > 0 else (fhr[:lag], toco[-lag:])
        r = corr(a, b)
        if r < best_r:
            best_r, best_lag = r, lag
    f.update(ft_corr_min=best_r, ft_corr_min_lag_s=best_lag / fs)
    return {k: float(f[k]) for k in CAT28}


def interpolate_signal(fhr, toco):
    """CNN preserves raw shape; only FHR zero gaps are interpolated.

    TOCO zero is a possible real measurement, not automatically missing.
    The feature arm retains exp15 A.2's separate TOCO-zero preprocessing.
    """
    fhr = np.asarray(fhr, float).copy()
    valid = fhr > 0
    if not valid.any():
        raise ValueError("Entire FHR segment is missing")
    fhr[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), fhr[valid])
    return np.stack([fhr, toco]).astype(np.float32)
