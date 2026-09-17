"""Discover Training/Validation/Test or evaluation directories, never hardcode N.

The original source split is inventoried; all research splits are re-created
by mother. Identical duplicate JSONs are deduplicated, disagreements fail.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from PIL import Image

from .common import numeric, note, read_json, sha256, write_json
from .features import extract_segment, interpolate_signal, CAT28
from .feature_rules import preprocess_for_features
from .signal_io import load_case_signal, parse_window_abnormality
from .resilience import Issues, Unavailable

PRENATAL = ["Mother.Height", "Mother.Weight", "Mother.Gravida", "Mother.Para", "Mother.SBP", "Mother.DBP",
            "Mother.GHTN", "Mother.Hypertension", "Mother.GDM", "Mother.DM", "Mother.pre-eclampsia"]
EMR_FEATURES = ["emr_" + name for name in PRENATAL] + ["maternal_age", "gestational_age"]
OFFICIAL_COLUMNS = PRENATAL + ["Delivery", "GA.wks", "GA.day", "FetalDistress", "FGR", "Placenta.Complication",
    "Sex", "Weight", "Height", "HC", "Jaundice", "prematurity", "LBW", "Anomaly"] + [f"Anomaly{i}" for i in range(1, 9)] + [
    "twins", "min_fhr", "max_fhr", "median_fhr", "mean_fhr", "min_toco", "max_toco", "median_toco", "mean_toco",
    "Mother.age", "prop_abnormal"]


def discover(root):
    root = Path(root)
    catalog = {k: {} for k in ("annotation_person", "labels", "emr", "images")}
    fingerprints, duplicate_count = [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".json" and path.parent.name in catalog and path.parent.name != "images":
            kind = path.parent.name
        elif path.suffix.lower() == ".png" and path.parent.name == "refine_images":
            kind = "images"
        else:
            continue
        digest = sha256(path)
        fingerprints.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": digest})
        previous = catalog[kind].get(path.stem)
        if previous:
            duplicate_count += 1
            if previous["sha256"] != digest:
                if kind == "images":
                    raise ValueError(f"Conflicting duplicate image: {path} / {previous['path']}")
                a, b = read_json(previous["path"]), read_json(path)
                # The classification distribution omits boxes present in detection distribution.
                if kind == "labels":
                    a_box, b_box = a.pop("Bbox", None), b.pop("Bbox", None)
                    if a == b and (a_box == b_box or not a_box or not b_box):
                        if b_box and not a_box:
                            catalog[kind][path.stem] = {"path": str(path), "sha256": digest}
                        continue
                    raise ValueError(f"Conflicting duplicate labels/boxes: {path} / {previous['path']}")
                if a != b:
                    raise ValueError(f"Conflicting duplicate {kind}: {path} / {previous['path']}")
            continue
        catalog[kind][path.stem] = {"path": str(path), "sha256": digest}
    if not catalog["annotation_person"] or not catalog["labels"]:
        raise ValueError("annotation_person/*.json 및 labels/*.json을 찾지 못했습니다. 압축 해제한 데이터 상위 폴더를 선택하세요.")
    return catalog, fingerprints, duplicate_count


def bbox_check(rid, label, image_path):
    flags = parse_window_abnormality(label, len(str(label.get("Abnormality", "")).split("_")))
    boxes = label.get("Bbox") or []
    row = dict(record_id=rid, n_boxes=len(boxes), n_segments=len(flags), n_abnormal=int(flags.sum()),
               image_present=bool(image_path), checked=False, valid=None, reason="no_boxes")
    if not boxes:
        # No-box records in the classification-only subset cannot establish annotation equivalence.
        row["reason"] = "positive_without_boxes" if flags.any() else "normal_without_boxes"
        return row
    if not image_path:
        row["reason"] = "image_missing"
        return row
    with Image.open(image_path) as image:
        width, height = image.size
    positions, errors = [], []
    for box in boxes:
        if len(box) != 6:
            errors.append("box_shape")
            continue
        x1, y1, x2, y2, w, h = map(float, box)
        step = width / len(flags)
        pos = int(round(x1 / step))
        positions.append(pos)
        if abs(x1 - pos * step) > 0.5 or abs(x2 - (pos + 1) * step) > 0.5:
            errors.append("segment_boundary")
        if abs(y1) > 0.5 or abs(y2 - height) > 0.5 or w != width or h != height:
            errors.append("image_dimensions")
    if sorted(positions) != list(np.flatnonzero(flags)):
        errors.append("flag_positions")
    row.update(checked=True, valid=not errors, reason=",".join(sorted(set(errors))) or "ok")
    return row


def official_row(rid, emr, label, annotation, case, mode):
    row = {k: numeric(emr.get(k, label.get(k))) for k in OFFICIAL_COLUMNS}
    birth, mother_birth = numeric(emr.get("Birth Date")), numeric(emr.get("Mother.Birth Date"))
    row["Mother.age"] = birth // 100 - mother_birth // 100 + 1
    row["prop_abnormal"] = round(str(label["Abnormality"]).split("_").count("1") / case.selected_segment_count, 1)
    for channel in ("fhr", "toco"):
        if mode == "original":
            values = np.asarray([int(s) for s in re.findall(r"\d+", "".join(str(s[channel]) for s in annotation["data"]))], float)
        else:
            values = getattr(case, channel)
        row.update({f"min_{channel}": values.min(), f"max_{channel}": values.max(),
                    f"median_{channel}": int(np.median(values)), f"mean_{channel}": int(values.mean())})
    return dict(record_id=rid, target=numeric(label.get("Emergency")), **row)


def longest_source_run(flags, names):
    """Do not bridge recording prefixes, skipped indices, or unknown ordering.

    Consecutive source image indices are only a proxy for continuity, not timestamps.
    Unknown naming makes this covariate missing rather than fabricating a long run.
    """
    if len(flags) != len(names):
        raise ValueError("Run summary needs one source name per label")
    previous, current, longest = None, 0, 0
    for flag, name in zip(flags, names):
        match = re.fullmatch(r"(.+)_p(\d+)(?:\.png)?", str(name), re.IGNORECASE)
        if match is None:
            return np.nan, "unknown_source_order"
        item = (match.group(1), int(match.group(2)))
        contiguous = previous is not None and item[0] == previous[0] and item[1] == previous[1] + 1
        current = (current + 1 if contiguous else 1) if flag else 0
        longest = max(longest, current)
        previous = item
    return longest, "source_index_proxy"


def feature_quality(frame):
    """Descriptive measurement audit; no feature selection or threshold tuning."""
    rows = []
    for feature in CAT28:
        x = frame[feature]
        note_code = ("sample_difference_not_clinical_stv" if feature == "stv" else
                     "duration_exceeds_window" if feature == "n_severe_decel" else "operational_definition")
        rows.append(dict(feature=feature, n=len(frame), mothers=frame.mother_id.nunique(),
            positive_mothers=frame.loc[frame.target == 1, "mother_id"].nunique(),
            negative_mothers=frame.loc[frame.target == 0, "mother_id"].nunique(),
            n_unique=x.nunique(), zero_fraction=float(x.eq(0).mean()),
            zero_mothers=frame.loc[x.eq(0), "mother_id"].nunique(),
            nonzero_mothers=frame.loc[x.ne(0), "mother_id"].nunique(),
            min=float(x.min()), max=float(x.max()), mean=float(x.mean()), std=float(x.std()),
            status="structurally_unobservable" if feature == "n_severe_decel" else "constant" if x.nunique() <= 1 else "ok",
            definition_note=note_code))
    return pd.DataFrame(rows)


def prepare(catalog, out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    issues = Issues(out)
    bbox = []
    for rid, entry in catalog["labels"].items():
        with issues.guard("bbox:" + rid):
            bbox.append(bbox_check(rid, read_json(entry["path"]), catalog["images"].get(rid, {}).get("path")))
    bad_bbox = {r["record_id"] for r in bbox if r["checked"] and not r["valid"]}
    # A failed box parser cannot establish alignment either.
    bad_bbox.update(r["job"][5:] for r in issues.rows if r["job"].startswith("bbox:"))
    pd.DataFrame(bbox).to_csv(out / "bbox_audit.csv", index=False)
    failed = [r for r in bbox if r["checked"] and not r["valid"]]
    records, segments, signals, exclusions, original, corrected, placeholders = [], [], [], [], [], [], Counter()
    unsmoothed, extraction_seconds = [], {"smooth30": 0.0, "smooth0": 0.0}
    emr_fields = Counter()
    emr_missing = Counter()
    annotations = catalog["annotation_person"]
    for number, (rid, entry) in enumerate(annotations.items(), 1):
        # Transactional record buffers: no half-record can misalign CSV and NPY.
        buffers = (records, segments, signals, exclusions, original, corrected, unsmoothed)
        lengths = [len(values) for values in buffers]
        counters = (placeholders, emr_fields, emr_missing)
        snapshots = [counter.copy() for counter in counters]
        def rollback():
            for values, length in zip(buffers, lengths):
                del values[length:]
            for counter, snapshot in zip(counters, snapshots):
                counter.clear()
                counter.update(snapshot)
        with issues.guard("record:" + rid, rollback=rollback):
            if cfg["strict_bbox"] and rid in bad_bbox:
                raise ValueError("Bbox validation failed or unavailable due to invalid input")
            if rid not in catalog["labels"]:
                raise ValueError(f"Annotation has no label: {rid}")
            annotation = read_json(entry["path"])
            label = read_json(catalog["labels"][rid]["path"])
            if str(label.get("ID")) != rid:
                raise ValueError(f"Label ID mismatch: {rid}")
            emr = read_json(catalog["emr"][rid]["path"]) if rid in catalog["emr"] else {}
            if emr and str(emr.get("ID")) != rid:
                raise ValueError(f"EMR ID mismatch: {rid}")
            case = load_case_signal(rid, Path(entry["path"]).parent)
            flags = parse_window_abnormality(label, case)
            mid = str(emr.get("Mother.de-identification_ID", "")).strip()
            if not mid or mid in {"9999", "nan", "None"}:
                raise ValueError(f"Missing mother ID for {rid}; refusing unsafe record-based split")
            longest, continuity = longest_source_run(flags, case.selected_segments)
            rec = dict(record_id=rid, mother_id=mid, site=case.site, is_twin=case.is_twin, n_segments=len(flags),
                       abnormal_fraction=float(flags.mean()), any_abnormal=int(flags.any()),
                       longest_abnormal_run=longest, continuity_status=continuity,
                       image_path=catalog["images"].get(rid, {}).get("path", ""))
            rec.update({"emr_" + k: numeric(v) for k, v in emr.items() if k != "ID"})
            rec.update({"label_" + k: numeric(v) for k, v in label.items() if k not in {"ID", "Bbox", "Abnormality"}})
            birth, mbirth = numeric(emr.get("Birth Date")), numeric(emr.get("Mother.Birth Date"))
            rec["maternal_age"] = birth // 100 - mbirth // 100 + 1
            rec["gestational_age"] = numeric(emr.get("GA.wks")) + numeric(emr.get("GA.day", 0)) / 7
            for field in emr:
                emr_fields[field] += 1
                emr_missing[field] += int(not np.isfinite(numeric(emr[field])))
            for part in annotation.get("data", []):
                for key in ("baseline", "baseline_var", "accel", "decel", "cervix"):
                    if key in part:
                        placeholders[(key, str(part[key]))] += 1
            kept, missing_values = 0, []
            for index, (fhr, toco, target) in enumerate(zip(case.fhr_windows, case.toco_windows, flags)):
                if not np.isclose(case.dt * len(fhr), 300):
                    raise ValueError(f"{rid}/{index}: expected 300s window, got {case.dt * len(fhr)}s")
                if not np.isclose(case.dt, 2) or len(fhr) != 150:
                    raise ValueError(f"{rid}: this approved CNN contract needs 0.5Hz / 150 samples, got dt={case.dt}, N={len(fhr)}")
                if (fhr < 0).any():
                    raise ValueError(f"{rid}: negative FHR")
                missing = float((fhr == 0).mean())
                missing_values.append(missing)
                if missing > cfg["max_missing_fraction"]:
                    exclusions.append(dict(record_id=rid, seg_idx=index, reason="fhr_missing_gt_threshold", missing_fraction=missing))
                    continue
                started = time.perf_counter()
                clean_f, clean_t = preprocess_for_features(fhr, toco, 1 / case.dt)
                factors = extract_segment(clean_f, clean_t, 1 / case.dt)
                extraction_seconds["smooth30"] += time.perf_counter() - started
                started = time.perf_counter()
                raw_f, raw_t = preprocess_for_features(fhr, toco, 1 / case.dt, smooth_seconds=0)
                raw_factors = extract_segment(raw_f, raw_t, 1 / case.dt)
                extraction_seconds["smooth0"] += time.perf_counter() - started
                if not np.isfinite(list(factors.values())).all():
                    raise ValueError(f"Nonfinite Cat28: {rid}/{index}")
                if not np.isfinite(list(raw_factors.values())).all():
                    raise ValueError(f"Nonfinite unsmoothed Cat28: {rid}/{index}")
                unsmoothed.append(dict(record_id=rid, seg_idx=index, **raw_factors))
                segments.append(dict(record_id=rid, mother_id=mid, site=case.site, seg_idx=index, target=int(target),
                                     missing_fraction=missing, record_all_normal=int(not flags.any()), **factors))
                signal = interpolate_signal(fhr, toco)
                if not np.isfinite(signal).all():
                    raise ValueError(f"Nonfinite CNN signal: {rid}/{index}")
                signals.append(signal)
                kept += 1
            rec.update(n_kept=kept, missing_fraction=float(np.mean(missing_values)))
            records.append(rec)
            if emr:
                for mode, destination in (("original", original), ("corrected", corrected)):
                    with issues.guard(f"official_input:{mode}:{rid}"):
                        destination.append(official_row(rid, emr, label, annotation, case, mode))
            if number % 250 == 0:
                note(f"인자 추출 {number:,}/{len(annotations):,} records; {len(segments):,} segments")
    if not signals:
        issues.record("usable_signals", Unavailable("No usable signal segments; descriptive outputs only"))
    excluded_records = [row for row in issues.rows if row["job"].startswith("record:")]
    pd.DataFrame([dict(record_id=row["job"][7:], type=row["type"], reason=row["error"]) for row in excluded_records],
                 columns=["record_id", "type", "reason"]).to_csv(out / "record_exclusions.csv", index=False)
    record_frame = (pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "record_id", "mother_id", "site", "n_segments", "n_kept", "any_abnormal",
        "is_twin", "abnormal_fraction", "longest_abnormal_run", "continuity_status",
        "missing_fraction", "maternal_age", "gestational_age"]))
    record_frame.to_csv(out / "records.csv", index=False)
    segment_frame = pd.DataFrame(segments, columns=[
        "record_id", "mother_id", "site", "seg_idx", "target", "missing_fraction", "record_all_normal"] + CAT28)
    raw_frame = pd.DataFrame(unsmoothed, columns=["record_id", "seg_idx"] + CAT28)
    segment_frame.to_csv(out / "segments.csv", index=False)
    raw_frame.to_csv(out / "segments_unsmoothed.csv", index=False)
    feature_quality(segment_frame).to_csv(out / "feature_quality.csv", index=False)
    pd.DataFrame([dict(feature=f, mean_abs_change=float(np.abs(segment_frame[f] - raw_frame[f]).mean()),
        zero_fraction_smooth30=float(segment_frame[f].eq(0).mean()), zero_fraction_smooth0=float(raw_frame[f].eq(0).mean()))
        for f in CAT28]).to_csv(out / "preprocessing_sensitivity.csv", index=False)
    np.save(out / "signals.npy", np.stack(signals) if signals else np.empty((0, 2, 150), dtype=np.float32))
    pd.DataFrame(exclusions, columns=["record_id", "seg_idx", "reason", "missing_fraction"]).to_csv(out / "exclusions.csv", index=False)
    pd.DataFrame(original, columns=["record_id", "target"] + OFFICIAL_COLUMNS).to_csv(out / "official_original.csv", index=False)
    pd.DataFrame(corrected, columns=["record_id", "target"] + OFFICIAL_COLUMNS).to_csv(out / "official_corrected.csv", index=False)
    pd.DataFrame([dict(field=k, value=v, count=n) for (k, v), n in placeholders.items()]).to_csv(out / "annotation_fields.csv", index=False)
    pd.DataFrame([dict(field=k, present=emr_fields[k], missing=emr_missing[k], missing_or_absent=len(records) - emr_fields[k] + emr_missing[k])
                  for k in sorted(emr_fields)]).to_csv(out / "emr_missingness.csv", index=False)
    record_frame.groupby("site").agg(records=("record_id", "size"), mothers=("mother_id", "nunique"),
        positive_records=("any_abnormal", "sum"), twins=("is_twin", "sum")).to_csv(out / "site_inventory.csv")
    summary = dict(records=len(records), mothers=len({r["mother_id"] for r in records}), segments=len(segments),
                   positive_segments=sum(s["target"] for s in segments), excluded_segments=len(exclusions),
                   records_without_usable_segments=sum(r["n_kept"] == 0 for r in records),
                   label_only_records=sorted(set(catalog["labels"]) - set(annotations)),
                   bbox_checked=sum(r["checked"] for r in bbox), bbox_failed=len(failed),
                   bbox_unverified=sum(not r["checked"] and r["n_abnormal"] > 0 for r in bbox),
                   channel_contract=["FHR", "TOCO"], sample_interval_seconds=2,
                   feature_extraction_seconds=extraction_seconds,
                   feature_extraction_ms_per_segment={k: v * 1000 / len(segments) if segments else None for k, v in extraction_seconds.items()},
                   continuity_unknown_records=sum(r["continuity_status"] != "source_index_proxy" for r in records),
                   features=CAT28)
    summary.update(input_records=len(annotations), excluded_records=len(excluded_records),
                   status="complete_with_issues" if issues.rows or not signals else "complete",
                   usable_for_training=bool(signals), exclusion_policy="whole_invalid_record_no_label_repair")
    write_json(out / "summary.json", summary)
    write_json(out / "status.json", issues.finish(usable_for_training=bool(signals)))
    note(f"데이터 준비 완료: {len(records):,} records / {len(segments):,} segments")
