"""Export-review projections of internal diagnostics, never copies of private reports.

The independent manifest lets completed runs acquire this supplement without
rewriting their experiment files or their original screened export manifest.
Only fixed labels, aggregate numbers and generated prose cross this boundary.
"""
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import re
import tempfile

from .survey import write_csv
from .telemetry import save_json
from .visit_audit import Evidence, training_rows, QUESTIONS

DIRECTORY = re.compile(r"visit_audit(?:_[A-Za-z0-9_-]+)?\Z")
SCREENED = "screened_aggregate_pending_review"
SUPPRESSED = "suppressed_requires_review"
STAGES = ("package_verification", "source_survey", "preflight", "input_hash_and_discovery",
          "data", "splits", "experiment_a", "experiment_b", "supplementary", "official",
          "report", "export_review", "onsite_figures")
TITLES = {
    "coverage": "자료 수록 현황", "analysis_status": "분석 수행·미수행 상태", "configuration": "실행 설정",
    "source_inventory": "원천 파일 구성·연결", "source_fields": "필드 구성·결측 조사",
    "shape_variants": "표본 간격·채널 길이", "data_quality": "분석 대상·제외·Bbox 검증",
    "site_composition": "기관별 코호트 구성", "emr_missingness": "EMR 관측·결측",
    "signal_quality": "원천 FHR·TOCO 품질", "annotation_placeholders": "수치 JSON 주석 필드 분포",
    "confusion": "모델별 혼동행렬", "score_distribution": "모델 점수 분포", "calibration": "확률 보정",
    "roc_pr": "고정 임계값 ROC·정밀도-재현율", "subgroups": "하위군 성능", "site_performance": "기관별 홀드아웃 성능",
    "shap_direction": "인자 값 순위별 평균 SHAP 방향", "training_diagnostics": "학습 종료·예산 진단",
    "training_history": "모든 학습 후보의 반복 이력", "stage_timings": "단계별 소요 시간",
    "resources": "자원 사용량 요약", "execution_events": "실행·중단·재개 이벤트 집계", "next_visit_plan": "다음 방문 준비",
}
NOTES = {
    "source_inventory": "원천 파일/ID 수는 중복 배포와 두 태아를 포함할 수 있으며 최종 분석 산모 수와 다릅니다.",
    "source_fields": "고정된 필드 이름만 수록합니다. 파일 단위 조사이며 임상 결측의 의미를 확정하지 않습니다.",
    "shape_variants": "원천 구간 분모입니다. 작은 형식 집단이 있으면 전체 분포를 가립니다.",
    "signal_quality": "원천 표본의 0·음수·비수치 비율입니다. TOCO 0을 임상 결측으로 단정하지 않습니다.",
    "roc_pr": "0~1의 고정 임계값 21개에서 계산한 기술통계입니다. 개인별 점수나 고유 ROC 임계값을 내보내지 않습니다.",
    "shap_direction": "인자별 순위 4분위의 평균 기여입니다. 개별 SHAP 점·인자 원값·구간 식별자는 포함하지 않습니다.",
    "training_diagnostics": "상한 도달과 patience 충족은 별개입니다. 둘 다 수렴/데이터 포화의 증명은 아닙니다.",
    "training_history": "학습/validation만 사용합니다. 분모 확인이 안 되는 후보의 점수와 loss는 공란입니다. 학습곡선은 PNG에 전 후보를 수록합니다.",
    "resources": "주기 표본의 요약이며 순간 피크가 아닙니다. GPU 값은 장치 전체 사용량입니다.",
}
NUMERIC_TRAINING = ("seed", "iterations_run", "best_iteration", "stale_iterations", "cap", "patience",
                    "recent_window", "recent_best_gain", "last_validation_score", "seconds")
SAFE_FIELDS = set("ID code twins category original_image refine_image data Abnormality Bbox BaseLine Baseline_Variability Acceleration Early_deceleration Late_deceleration Variable_deceleration Prolonged_deceleration CA Emergency Sinosoical_pattern".split()) | {
    "Mother.de-identification_ID", "de-identification_ID", "Birth Date", "Mother.Birth Date", "Father.Birth Date",
    "Mother.Height", "Mother.Weight", "Mother.Gravida", "Mother.Para", "Mother.SBP", "Mother.DBP", "Mother.GHTN",
    "Mother.Hypertension", "Mother.GDM", "Mother.DM", "Mother.pre-eclampsia", "Mother.ABO type", "Mother.RH type",
    "Mother.MEASURE_DATE", "GA.wks", "GA.day", "UA.pH", "UA.pO2", "UA.pCO2", "UA.BE", "APGAR.1min", "APGAR.5min",
    "Delivery", "FetalDistress", "FGR", "Placenta.Complication", "Placenta.Weight", "Sex", "Weight", "Height", "HC",
    "Jaundice", "prematurity", "LBW", "Anomaly", "Cervix", "Fetal_Monitor", "Intubation", "NICU.Adm",
} | {f"Anomaly{i}" for i in range(1, 9)}


def numeric(value):
    if isinstance(value, bool):
        return int(value)
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def safe_path(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Symbolic-link diagnostic source/destination is not accepted")
    return path


class Reader(Evidence):
    def read(self, path, default=None):
        return super().read(safe_path(path), default)


def find_attempt(run, reader):
    refs = sorted((run / "visit_audit/attempts").glob("*.json")) if run else []
    if not refs:
        return None
    reference = reader.read(refs[-1], {})
    value = reference.get("path")
    if not isinstance(value, str):
        return None
    # Journal references are deliberately outside the run, but confined to visits/.
    candidate = safe_path(run / "visit_audit" / value).resolve()
    visits = (run.parent.parent / "visits").resolve()
    if not candidate.is_relative_to(visits):
        raise ValueError("Visit reference escapes the visits directory")
    return candidate


def reviewed(labels, values, allowed=True):
    return dict(labels, **{key: numeric(value) if allowed else None for key, value in values.items()},
                review_status=SCREENED if allowed else SUPPRESSED)


def partition_ok(counts, minimum):
    return all(numeric(value) is not None and (float(value) == 0 or float(value) >= minimum) for value in counts)


def collect_operational(run, attempt, cfg, reader):
    tables = {name: [] for name in TITLES}
    minimum = max(2, int(cfg.get("export_min_mothers", 10)))
    for key in ("seed", "threads", "test_folds", "validation_folds", "max_missing_fraction", "cnn", "official_models", "strict_bbox", "export_min_mothers"):
        if key in cfg:
            tables["configuration"].append(reviewed(dict(setting=key), {"value": cfg[key]}))
    for key in ("cv_folds", "tree_iterations", "tree_patience", "cnn_epochs", "cnn_patience", "batch_size", "bootstrap", "shap_samples"):
        if key in cfg.get("budget", {}):
            tables["configuration"].append(reviewed(dict(setting=key), {"value": cfg["budget"][key]}))
    for key in ("seeds", "cnn_widths"):
        for index, value in enumerate(cfg.get("budget", {}).get(key, [])):
            tables["configuration"].append(reviewed(dict(setting=key + "_" + str(index)), {"value": value}))
    statuses = {"complete", "ok", "failed", "interrupted", "running", "disabled_by_config", "missing_field",
                "insufficient_classes", "insufficient_outcome_groups", "insufficient_classes_in_grouped_fold", "no_complete_records", "no_images"}
    if run:
        for stage in ("experiment_a", "experiment_b", "official"):
            state = reader.read(run / stage / "status.json", {})
            complete = (run / ".state" / (stage + ".json")).is_file()
            status = state.get("status", "complete" if complete else "uncollected")
            tables["analysis_status"].append(dict(analysis=stage, status=status if status in statuses else "uncollected", review_status=SCREENED))
        outcomes = reader.read(run / "supplementary/outcomes.json", {})
        for name in ("ph_lt_7_20", "apgar1_lt_7", "apgar5_lt_7"):
            state = outcomes.get(name, {}).get("status")
            tables["analysis_status"].append(dict(analysis=name, status=state if state in statuses else "uncollected", review_status=SCREENED))
    summary = reader.read(attempt / "survey/summary.json", {}) if attempt else {}
    for category in ("inventory", "parsed_json_files", "unique_ids", "duplicate_id_files", "links"):
        keys = ("annotation_person", "labels", "emr") if category != "links" else (
            "annotation_ids", "annotation_with_label", "annotation_with_label_and_emr", "annotation_without_label",
            "annotation_without_emr", "labels_without_annotation", "emr_without_annotation")
        values = summary.get(category, {})
        selected = [key for key in keys if key in values]
        allowed = partition_ok([values[key] for key in selected], minimum)
        tables["source_inventory"] += [reviewed(dict(category=category, kind=key), {"count": values[key]}, allowed) for key in selected]
    if attempt:
        fields = reader.read(attempt / "survey/fields.csv", [])
        for row in fields:
            if row.get("field") not in SAFE_FIELDS or row.get("kind") not in ("annotation_person", "labels", "emr"):
                continue
            keys = ("parsed_file_denominator", "present", "absent", "null_or_blank", "sentinel_9999", "zero")
            allowed = partition_ok([row.get(key) for key in keys], minimum)
            tables["source_fields"].append(reviewed({k: row[k] for k in ("kind", "field")}, {k: row.get(k) for k in keys}, allowed))
        shapes = reader.read(attempt / "survey/shape_variants.csv", [])
        allowed = partition_ok([r.get("source_windows") for r in shapes], minimum)
        for row in shapes:
            labels = {k: row.get(k) if row.get(k) in ("str", "list", "NoneType") else "other" for k in ("fhr_type", "toco_type")}
            tables["shape_variants"].append(reviewed(labels, {k: row.get(k) for k in ("interval", "fhr_samples", "toco_samples", "source_windows")}, allowed))
    if run:
        for parent in (run / ".state", run.parent / ".state"):
            for stage in STAGES:
                state = reader.read(parent / (stage + ".json"), {})
                if state:
                    tables["stage_timings"].append(reviewed(dict(stage=stage, scope="original_completed_stage"), {"seconds": state.get("seconds")}))
    resources = reader.read(attempt / "resources.jsonl", []) if attempt else []
    events = reader.read(attempt / "events.jsonl", []) if attempt else []
    kinds = {"visit_started", "visit_finished", "stage_started", "stage_finished", "stage_failed", "stage_reused",
             "fit_started", "fit_finished", "fit_resumed", "fit_reused", "epoch_finished", "monitor_unavailable", "stage_artifacts_hashed"}
    counts = Counter((row.get("event"), row.get("stage") if row.get("stage") in STAGES else "other") for row in events if row.get("event") in kinds)
    for (event, stage), count in sorted(counts.items()):
        tables["execution_events"].append(reviewed(dict(event=event, stage=stage), {"count": count}))
    for metric in ("process_rss_bytes", "process_cpu_percent_one_core", "disk_free_bytes"):
        values = [numeric(r.get(metric)) for r in resources]
        values = [v for v in values if v is not None]
        if values:
            tables["resources"].append(reviewed(dict(metric=metric, scope="sampled_process"),
                dict(samples=len(values), mean=sum(values) / len(values), minimum=min(values), maximum=max(values))))
    for metric in ("utilization_percent", "used_mib", "total_mib"):
        values = [numeric(g.get(metric)) for r in resources for g in r.get("gpus", []) if isinstance(g, dict)]
        values = [v for v in values if v is not None]
        if values:
            tables["resources"].append(reviewed(dict(metric=metric, scope="sampled_device_wide"),
                dict(samples=len(values), mean=sum(values) / len(values), minimum=min(values), maximum=max(values))))
    return tables


def collect_clinical(run, attempt, cfg, reader, tables):
    """Recompute denominators from private rows; never trust a global N for small cells."""
    import pandas as pd
    from .export_review import MODELS, cohort_counts, adequate, METRICS
    from .features import CAT28
    from .evaluation import evaluate, group_folds

    minimum = max(2, int(cfg.get("export_min_mothers", 10)))
    def frame(relative):
        return pd.DataFrame(reader.read(run / relative, []))
    records, segments = frame("data/records.csv"), frame("data/segments.csv")
    assignment = frame("splits/segments.csv")
    # Match the training loader's left-hand row order when reconstructing folds.
    if "split" not in segments and {"record_id", "seg_idx", "split"}.issubset(assignment) and not segments.empty:
        segments = segments.merge(assignment[["record_id", "seg_idx", "split"]], on=["record_id", "seg_idx"], validate="one_to_one", sort=False)
    def num(frame, keys):
        for key in keys:
            if key in frame:
                frame[key] = pd.to_numeric(frame[key], errors="coerce")
        return frame
    num(segments, ["target", "seg_idx"])
    if "is_twin" in records:
        records["is_twin"] = records.is_twin.replace({"True": "1", "False": "0"})
    num(records, ["n_segments", "n_kept", "any_abnormal", "missing_fraction", "is_twin"])
    def mothers(sub):
        return sub.mother_id.nunique() if "mother_id" in sub else 0
    def enough(sub):
        return adequate(cohort_counts(sub), minimum)
    def cells_ok(parts):
        return all(len(part) == 0 or mothers(part) >= minimum for part in parts)
    sites = sorted(set(records.get("site", [])))
    aliases = {site: f"S{i:02d}" for i, site in enumerate(sites, 1)}
    if {"mother_id", "record_id", "n_kept", "n_segments", "site", "any_abnormal"}.issubset(records):
        usable, excluded = records[records.n_kept.gt(0)], records[records.n_kept.eq(0)]
        allowed = cells_ok([usable, excluded]) and mothers(records) >= minimum
        tables["data_quality"].append(reviewed(dict(item="prepared_cohort"), dict(records=len(records), mothers=mothers(records),
            segments=records.n_segments.sum(), kept_segments=records.n_kept.sum(), records_without_windows=len(excluded)), allowed))
        for site, sub in records.groupby("site", sort=True):
            parts = [sub[sub.any_abnormal.eq(k)] for k in (0, 1)]
            tables["site_composition"].append(reviewed(dict(site=aliases[site]), dict(records=len(sub), mothers=mothers(sub),
                segments=sub.n_kept.sum(), positive_records=len(parts[1])), mothers(sub) >= minimum and cells_ok(parts)))
        exclusions = frame("data/exclusions.csv")
        if {"record_id", "reason"}.issubset(exclusions):
            ex = exclusions.merge(records[["record_id", "mother_id"]], on="record_id", validate="many_to_one")
            if set(ex.reason) <= {"fhr_missing_gt_threshold"}:
                tables["data_quality"].append(reviewed(dict(item="fhr_missing_gt_threshold"),
                    dict(segments=len(ex), records=ex.record_id.nunique(), mothers=mothers(ex)), len(ex) == 0 or mothers(ex) >= minimum))
        bbox = frame("data/bbox_audit.csv")
        if {"record_id", "reason"}.issubset(bbox):
            bbox = bbox.merge(records[["record_id", "mother_id"]], on="record_id", how="inner", validate="one_to_one")
            known = {"ok", "normal_without_boxes", "positive_without_boxes", "image_missing", "box_shape", "segment_boundary", "image_dimensions", "flag_positions"}
            parts = [sub for _, sub in bbox.groupby("reason")]
            for reason, sub in bbox.groupby("reason"):
                if set(reason.split(",")) <= known:
                    tables["data_quality"].append(reviewed(dict(item="bbox_" + reason.replace(",", "_")), dict(records=len(sub), mothers=mothers(sub)), cells_ok(parts)))
        for field in sorted(SAFE_FIELDS - {"ID", "Mother.de-identification_ID", "de-identification_ID"}):
            key = "emr_" + field
            if key not in records:
                continue
            values = pd.to_numeric(records[key], errors="coerce")
            present = records[values.notna()]
            missing = records[values.isna()]
            tables["emr_missingness"].append(reviewed(dict(field=field), dict(observed_records=len(present),
                observed_mothers=mothers(present), missing_records=len(missing), missing_mothers=mothers(missing)), cells_ok([present, missing])))
    if attempt and {"record_id", "mother_id"}.issubset(records):
        raw = pd.DataFrame(reader.read(attempt / "survey/raw_signal_quality.csv", []))
        if {"record_id", "fhr_samples", "toco_samples"}.issubset(raw):
            raw = raw.merge(records[["record_id", "mother_id"]], on="record_id", how="inner", validate="many_to_one")
            for channel in ("fhr", "toco"):
                for stat in ("zeros", "negative", "invalid"):
                    key = channel + "_" + stat
                    if key not in raw:
                        continue
                    values = pd.to_numeric(raw[key], errors="coerce")
                    parts = [raw[values.eq(0)], raw[values.gt(0)]]
                    samples = pd.to_numeric(raw[channel + "_samples"], errors="coerce").sum()
                    tables["signal_quality"].append(reviewed(dict(channel=channel, issue=stat), dict(samples=samples,
                        affected_samples=values.sum(), fraction=values.sum() / samples if samples else None), mothers(raw) >= minimum and cells_ok(parts)))
    placeholders = frame("data/annotation_fields.csv")
    if {"field", "value", "count"}.issubset(placeholders):
        # Placeholder values can contain arbitrary source text; export only fixed numeric sentinels.
        for field, sub in placeholders.groupby("field"):
            if field not in {"baseline", "baseline_var", "accel", "decel", "cervix"}:
                continue
            allowed = partition_ok(sub["count"], minimum)
            for row in sub.to_dict("records"):
                value = numeric(row["value"])
                if value in (0, 9999, -1):
                    tables["annotation_placeholders"].append(reviewed(dict(field=field, category="sentinel_" + str(int(value))), {"count": row["count"]}, allowed))

    predictions = pd.concat([frame("experiment_a/test_predictions.csv"), frame("experiment_b/test_predictions.csv")], ignore_index=True)
    num(predictions, ["target", "score", "threshold"])
    if {"model", "mother_id", "target", "score", "threshold"}.issubset(predictions):
        for model, sub in predictions.groupby("model", sort=True):
            if model not in MODELS:
                continue
            sub = sub[sub.target.isin([0, 1]) & sub.score.between(0, 1) & sub.threshold.notna()]
            parts = [sub[sub.target.eq(actual) & sub.score.ge(sub.threshold).eq(bool(predicted))] for actual in (0, 1) for predicted in (0, 1)]
            allowed = enough(sub) and cells_ok(parts)
            for index, part in enumerate(parts):
                tables["confusion"].append(reviewed(dict(model=model, actual=index // 2, predicted=index % 2), dict(segments=len(part), mothers=mothers(part)), allowed))
            bins = [sub[sub.score.ge(k / 10) & (sub.score.lt((k + 1) / 10) if k < 9 else sub.score.le(1))] for k in range(10)]
            hist_parts = [part[part.target.eq(label)] for part in bins for label in (0, 1)]
            allowed = enough(sub) and cells_ok(hist_parts)
            for k, part in enumerate(hist_parts):
                tables["score_distribution"].append(reviewed(dict(model=model, bin=k // 2, actual=k % 2), dict(segments=len(part), mothers=mothers(part)), allowed))
            for k, part in enumerate(bins):
                tables["calibration"].append(reviewed(dict(model=model, bin=k), dict(segments=len(part), mothers=mothers(part),
                    mean_prediction=part.score.mean(), observed_fraction=part.target.mean()), allowed))
            for k in range(21):
                threshold = k / 20
                parts = [sub[sub.target.eq(a) & sub.score.ge(threshold).eq(bool(p))] for a in (0, 1) for p in (0, 1)]
                tn, fp, fn, tp = map(len, parts)
                tables["roc_pr"].append(reviewed(dict(model=model, threshold=threshold), dict(
                    fpr=fp / (tn + fp) if tn + fp else None, recall=tp / (tp + fn) if tp + fn else None,
                    precision=tp / (tp + fp) if tp + fp else None), enough(sub) and cells_ok(parts)))
        # Recompute subgroup denominators instead of trusting raw subgroups.json counts.
        base = predictions[predictions.model.eq("cat28")]
        if {"record_id", "site"}.issubset(records) and "record_id" in base:
            by_site = base.drop(columns=["site"], errors="ignore").merge(records[["record_id", "site"]], on="record_id", validate="many_to_one")
            for site, sub in by_site.groupby("site"):
                result = evaluate(sub, sub.score, sub.threshold, 0).get("point", {})
                tables["site_performance"].append(reviewed(dict(site=aliases[site], model="cat28"),
                    {key: result.get(key) for key in METRICS if key not in ("normal_record_alarm_rate", "false_alarms_per_normal_hour")}, enough(sub)))
        for field, groups in (("is_twin", (("singleton", 0, 0), ("twin", 1, 1))),
                              ("gestational_age", (("preterm", 0, 36.999999), ("term", 37, 60)))):
            if field not in records or "record_id" not in base:
                continue
            joined = base.merge(records[["record_id", field]], on="record_id", validate="many_to_one")
            values = pd.to_numeric(joined[field], errors="coerce")
            for group, lower, upper in groups:
                sub = joined[values.between(lower, upper)]
                result = evaluate(sub, sub.score, sub.threshold, 0).get("point", {}) if len(sub) else {}
                tables["subgroups"].append(reviewed(dict(group=group), {key: result.get(key) for key in METRICS if key not in ("normal_record_alarm_rate", "false_alarms_per_normal_hour")}, enough(sub)))
    explanations = frame("experiment_a/local_explanations.csv")
    if {"record_id", "seg_idx", "mother_id"}.issubset(segments) and {"record_id", "seg_idx"}.issubset(explanations):
        num(explanations, ["seg_idx"])
        if "mother_id" not in explanations:
            explanations = explanations.merge(segments[["record_id", "seg_idx", "mother_id"]], on=["record_id", "seg_idx"], validate="one_to_one")
        for feature in CAT28:
            if not {"value_" + feature, "shap_" + feature}.issubset(explanations):
                continue
            values = pd.to_numeric(explanations["value_" + feature], errors="coerce")
            contributions = pd.to_numeric(explanations["shap_" + feature], errors="coerce")
            bins = pd.qcut(values, 4, labels=False, duplicates="drop") if values.nunique() > 1 else pd.Series(0, index=values.index)
            parts = [explanations[bins.eq(k) & contributions.notna()] for k in sorted(bins.dropna().unique())]
            allowed = bool(parts) and all(mothers(p) >= minimum for p in parts)
            for k, part in enumerate(parts):
                tables["shap_direction"].append(reviewed(dict(feature=feature, quantile=k), dict(mothers=mothers(part),
                    segments=len(part), mean_shap=contributions.loc[part.index].mean()), allowed))

    # Reconstruct only the deterministic development splits; never refit a model.
    cohorts = {}
    if {"split", "target", "mother_id"}.issubset(segments):
        cohorts["holdout"] = [segments[segments.split.eq(s)] for s in ("train", "val")]
        development = segments[segments.split.ne("test")].reset_index(drop=True)
        if len(development):
            for seed in cfg.get("budget", {}).get("seeds", []):
                try:
                    for fold, (tr, _) in enumerate(group_folds(development, cfg["budget"]["cv_folds"], seed)):
                        pool = development.iloc[tr].reset_index(drop=True)
                        it, iv = next(group_folds(pool, 4, seed + fold))
                        cohorts[f"cv_{seed}_{fold}"] = [pool.iloc[it], pool.iloc[iv]]
                except ValueError:
                    pass
        for site in sites:
            held = records.loc[records.site.eq(site), "mother_id"]
            pool = segments[~segments.mother_id.isin(held)].reset_index(drop=True)
            try:
                it, iv = next(group_folds(pool, 5, cfg["seed"]))
                cohorts["loso_" + aliases[site]] = [pool.iloc[it], pool.iloc[iv]]
            except (ValueError, KeyError):
                pass
    for outcome, column, cutoff, lower, upper in (
        ("ph_lt_7_20", "emr_UA.pH", 7.2, 6, 8),
        ("apgar1_lt_7", "emr_APGAR.1min", 7, 0, 10),
        ("apgar5_lt_7", "emr_APGAR.5min", 7, 0, 10),
    ):
        if column not in records or "mother_id" not in records:
            continue
        values = pd.to_numeric(records[column], errors="coerce")
        valid = records[values.between(lower, upper)].copy()
        valid["target"] = values.loc[valid.index].lt(cutoff).astype(int)
        valid = valid.reset_index(drop=True)
        try:
            for fold, (tr, _) in enumerate(group_folds(valid, cfg["budget"]["cv_folds"], cfg["seed"])):
                pool = valid.iloc[tr].reset_index(drop=True)
                it, iv = next(group_folds(pool, 3, cfg["seed"] + fold))
                cohorts[outcome + "_fold" + str(fold)] = [pool.iloc[it], pool.iloc[iv]]
        except (ValueError, KeyError):
            pass
    return aliases, {key: all(enough(part) for part in parts) for key, parts in cohorts.items()}


def collect_training(run, cfg, reader, tables, aliases, cohorts):
    from .features import GROUPS
    models = {"cat28", "cat18", "xgb28", "robust", "logistic28", "cat28_unsmoothed"} | {"without_" + g for g in GROUPS}
    rows, curves = training_rows(run, cfg, reader)
    for index, row in enumerate(rows, 1):
        stem = Path(row["job"]).stem
        match = re.fullmatch(r"cv_(\d+)_(\d+)_(.+)", stem)
        held = re.fullmatch(r"holdout_(.+)_(\d+)", stem)
        cnn = re.fullmatch(r"(channel_maxabs|per_segment_z)_w(\d+)_s(\d+)", Path(row["job"]).name)
        emr = re.fullmatch(r"emr_added_(\d+)", stem)
        outcome = re.fullmatch(r"(ph_lt_7_20|apgar1_lt_7|apgar5_lt_7)_(\d+)_(clinical|clinical_plus_reading)", stem)
        model, cohort = "unclassified", "unverified"
        if match and match[3] in models:
            cohort, model = f"cv_{match[1]}_{match[2]}", match[3]
        elif held and held[1] in models:
            cohort, model = "holdout", held[1]
        elif cnn:
            cohort, model = "holdout", f"cnn_{cnn[1]}_w{cnn[2]}"
        elif emr:
            cohort, model = "holdout", "cat28_emr"
        elif stem.startswith("loso_") and stem[5:] in aliases:
            cohort, model = "loso_" + aliases[stem[5:]], "cat28"
        elif outcome:
            cohort, model = outcome[1] + "_fold" + outcome[2], outcome[3]
        permitted = cohorts.get(cohort, False)
        label = dict(candidate=f"J{index:04d}", model=model, cohort=cohort,
                     family=row.get("family") if row.get("family") in ("cnn", "cat", "xgb", "tree") else "unknown",
                     stop_reason=row["stop_reason"], history_status=row["history_status"],
                     score_review_status=SCREENED if permitted else SUPPRESSED)
        numbers = {key: row.get(key) for key in NUMERIC_TRAINING}
        for key in ("recent_best_gain", "last_validation_score"):
            if not permitted:
                numbers[key] = None
        numbers.update(hit_cap=row.get("hit_cap"), patience_met=row.get("patience_met"))
        tables["training_diagnostics"].append(reviewed(label, numbers))
        row["export_label"], row["scores_permitted"] = label, permitted
    for row, history, score, loss, lr in curves:
        for index, item in enumerate(history, 1):
            tables["training_history"].append(reviewed(row["export_label"], dict(iteration=index,
                validation_score=item.get(score) if row["scores_permitted"] else None,
                train_loss=item.get(loss) if row["scores_permitted"] else None,
                learning_rate=item.get(lr) if lr else .05, epoch_seconds=item.get("epoch_seconds"))))
    return rows


def collect(run, attempt, cfg, reader):
    tables = collect_operational(run, attempt, cfg, reader)
    aliases, cohorts = {}, {}
    clinical_status = "unavailable"
    if run:
        try:
            aliases, cohorts = collect_clinical(run, attempt, cfg, reader, tables)
            clinical_status = "complete"
        except ImportError:
            clinical_status = "dependencies_unavailable"
        training = collect_training(run, cfg, reader, tables, aliases, cohorts)
    else:
        training = []
    capped = sum(r.get("hit_cap") is True and r.get("patience_met") is False for r in training)
    for topic, observation, action in (
        ("input", "입력 파일·연결·형식·결측 집계 확인", "누락 원인과 데이터 사전을 확인하고 필요한 입력 변환을 준비"),
        ("training", f"상한 도달·patience 미충족 후보 {capped}개", "validation 곡선과 비용으로 다음 예산 비교를 설계"),
        ("data_size", "산모 수 learning curve는 미실행", "고정 validation에서 train 산모 25/50/75/100% 비교 설계"),
        ("definitions", "측정 시각·장비·0/9999 의미·판독 과정 확인 필요", "기관 질문 목록에 대한 답변 확보"),
        ("evaluation", "관측한 holdout을 독립 확증으로 재사용할 수 없음", "다음 최종 평가용 미관측 산모·기관·기간 확보"),
    ):
        tables["next_visit_plan"].append(dict(topic=topic, observation=observation, next_action=action, review_status=SCREENED))
    for name in TITLES:
        if name == "coverage":
            continue
        rows = tables[name]
        tables["coverage"].append(dict(section=name, rows=len(rows), suppressed_rows=sum(r["review_status"] == SUPPRESSED for r in rows),
            status="available" if rows else "uncollected", review_status=SCREENED))
    summary = dict(format=1, purpose="screened_visit_diagnostics", review_status="pending_institution_review",
        export_min_mothers=max(2, int(cfg.get("export_min_mothers", 10))), clinical_projection=clinical_status,
        read_errors=len(reader.issues), profile=cfg.get("profile") if cfg.get("profile") in ("mock", "full") else "unspecified",
        excluded_categories=["identifiers", "raw_paths", "raw_signal", "individual_predictions", "individual_shap", "weights", "free_text_logs", "user_answers"],
        learning_curve_status="not_run", screening_is_approval=False)
    state = reader.read(attempt / "status.json", {}) if attempt else {}
    summary["invocation_status"] = state.get("status") if state.get("status") in ("complete", "failed", "interrupted", "running") else "uncollected"
    return tables, summary


def render(out, tables, summary, images=True):
    """Render only the projected tables, with no links back into internal/."""
    (out / "csv").mkdir()
    (out / "images").mkdir()
    for name, rows in tables.items():
        columns = list(dict.fromkeys(key for row in rows for key in row)) or ["review_status"]
        write_csv(out / "csv" / (name + ".csv"), rows, columns)
    # Export the fixed questionnaire, never the operator's edited answers.
    (out / "questions.md").write_text(QUESTIONS.replace("(내부용)", "(반출 검토용 빈 양식)").replace(
        "재생성 시 이 파일은 덮어쓰지 않습니다.", "답변은 이 자동 생성본에 포함되지 않습니다."), encoding="utf-8")
    image_files = []
    if images:
        try:
            image_files = render_images(out / "images", tables)
        except ImportError:
            summary["image_status"] = "dependencies_unavailable"
    summary.setdefault("image_status", "complete" if images else "not_requested")
    save_json(out / "summary.json", summary)
    page = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>반출 검토 · 진단 및 추가 집계</title>",
        "<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#203749}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}section{margin:30px 0}.scroll{overflow:auto}img{max-width:100%}</style>",
        "<h1>반출 검토 · 방문 진단과 추가 집계</h1><p>internal 자료에서 반출 심사용으로 다시 만든 집계입니다. CSV·PNG·HTML을 이 폴더에서 확인할 수 있습니다.</p>",
        "<p>공란은 미수집 또는 작은 집단 억제입니다. 0으로 해석하지 않습니다. 실제 반출은 기관 심사 대상입니다.</p>",
        "<p><a href='questions.md'>기관 확인 질문 · 빈 양식</a></p><ul>"]
    page += [f"<li><a href='#{name}'>{title}</a></li>" for name, title in TITLES.items()]
    page.append("</ul>")
    from .visit_audit import tabulate
    for name, title in TITLES.items():
        rows = tables[name]
        page += [f"<section id='{name}'><h2>{title}</h2><p>{html.escape(NOTES.get(name, ''))}</p>",
                 f"<p><a href='csv/{name}.csv'>전체 CSV ({len(rows):,}행)</a></p>"]
        columns = list(rows[0]) if rows else []
        page.append(tabulate(rows[:30], [(c, c) for c in columns]))
        if len(rows) > 30:
            page.append("<p>첫 30행 미리보기입니다. 전체 후보·반복은 CSV와 아래 PNG에 포함됩니다.</p>")
        for filename in image_files:
            if filename.startswith(name + "_"):
                page.append(f"<a href='images/{filename}'><img loading='lazy' src='images/{filename}'></a>")
        page.append("</section>")
    page.append("</html>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")


def render_images(out, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from .onsite_figures import choose_font
    import io
    font, korean = choose_font()
    files = []
    def save(fig, name):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160, facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        with Image.open(buffer) as image:
            image.convert("RGB").save(out / name, format="PNG")
        files.append(name)
    with plt.rc_context({"font.family": [font, "DejaVu Sans"], "font.size": 10, "axes.unicode_minus": False}):
        for name, rows in tables.items():
            if not rows or name == "training_history":
                continue
            # Every projected row is included; pages are bounded for readability.
            keys = [key for key in rows[0] if key not in ("review_status", "score_review_status")]
            column_pages = [keys] if len(keys) <= 10 else [keys[:3] + keys[start:start + 7] for start in range(3, len(keys), 7)]
            pages = [(start, columns) for start in range(0, len(rows), 18) for columns in column_pages]
            for page_number, (start, columns) in enumerate(pages, 1):
                chunk = rows[start:start + 18]
                fig, ax = plt.subplots(figsize=(20, 11))
                ax.axis("off")
                ax.set_title(TITLES[name] if korean else name.replace("_", " "), pad=20, fontsize=19)
                cells = [[("—" if row.get(key) is None else f"{row[key]:.4g}" if isinstance(row.get(key), float) else str(row.get(key))) for key in columns] for row in chunk]
                import textwrap
                cells = [["\n".join(textwrap.wrap(cell, max(12, 130 // max(len(columns), 1)))) for cell in row] for row in cells]
                table = ax.table(cellText=cells, colLabels=[k.replace("_", "\n") for k in columns], cellLoc="left", bbox=[0, .06, 1, .86])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                for (r, _), cell in table.get_celld().items():
                    cell.set_edgecolor("#d9e3e8")
                    cell.set_facecolor("#dfeef1" if r == 0 else "white")
                ax.text(0, .01, f"Review candidate | rows {start + 1}-{start + len(chunk)} / {len(rows)} | blank = suppressed / unavailable", transform=ax.transAxes)
                save(fig, f"{name}_p{page_number:03d}.png")
        grouped = {}
        for row in tables["training_history"]:
            grouped.setdefault(row["candidate"], []).append(row)
        items = list(grouped.items())
        for start in range(0, len(items), 4):
            fig, axes = plt.subplots(2, 2, figsize=(20, 12), constrained_layout=True)
            for ax, (candidate, rows) in zip(axes.flat, items[start:start + 4]):
                ax.set_title(candidate + " / " + rows[0]["model"] + " / " + rows[0]["cohort"])
                usable = [r for r in rows if r["validation_score"] is not None]
                if usable:
                    ax.plot([r["iteration"] for r in usable], [r["validation_score"] for r in usable], label="validation score", color="#247b91")
                    ax.legend(loc="upper left")
                else:
                    ax.text(.5, .5, "Score unavailable / suppressed", transform=ax.transAxes, ha="center")
                loss = [r for r in rows if r["train_loss"] is not None]
                if loss:
                    other = ax.twinx()
                    other.plot([r["iteration"] for r in loss], [r["train_loss"] for r in loss], color="#d26742", alpha=.6, label="train loss")
                    other.set_ylabel("train loss")
                ax.set_xlabel("iteration / epoch")
                ax.set_ylabel("validation score")
            for ax in list(axes.flat)[len(items[start:start + 4]):]:
                ax.axis("off")
            save(fig, f"training_history_p{start // 4 + 1:03d}.png")
    return files


def validate_diagnostics(out):
    out = safe_path(out)
    paths = list(out.rglob("*"))
    for path in paths:
        safe_path(path)
    actual = {p.relative_to(out).as_posix() for p in paths if p.is_file()}
    allowed = {"index.html", "questions.md", "summary.json", "EXPORT_MANIFEST.json"} | {"csv/" + n + ".csv" for n in TITLES}
    images = {name for name in actual if re.fullmatch(r"images/(?:" + "|".join(TITLES) + r")_p\d{3,}\.png", name)}
    if actual - allowed - images:
        raise ValueError("Unexpected diagnostic review file")
    manifest = json.loads((out / "EXPORT_MANIFEST.json").read_text())
    if manifest.get("purpose") != "screened_visit_diagnostics" or manifest.get("screening_is_approval") is not False:
        raise ValueError("Invalid diagnostic review scope")
    if set(manifest["files"]) != actual - {"EXPORT_MANIFEST.json"}:
        raise ValueError("Diagnostic inventory mismatch")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Diagnostic checksum mismatch")
    if not {"index.html", "questions.md", "summary.json"}.issubset(actual):
        raise ValueError("Incomplete diagnostic review")
    for name in images:
        # This projection writes RGB PNG without text/EXIF or trailing payloads.
        import struct
        data = (out / name).read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Invalid diagnostic PNG")
        offset, ended = 8, False
        while offset + 12 <= len(data):
            size = struct.unpack(">I", data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            if kind not in (b"IHDR", b"IDAT", b"IEND", b"pHYs"):
                raise ValueError("Unexpected diagnostic PNG metadata")
            offset += size + 12
            if kind == b"IEND":
                ended = True
                break
        if not ended or offset != len(data):
            raise ValueError("Diagnostic PNG trailing payload")
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag in {"script", "iframe", "object", "embed", "form"}:
                raise ValueError("Active diagnostic HTML content")
            for key, value in attrs:
                if key.startswith("on") or (key in {"href", "src"} and value not in actual and value not in {"#" + n for n in TITLES}):
                    raise ValueError("Diagnostic link escapes review supplement")
    Links().feed((out / "index.html").read_text())
    return dict(status="validated_pending_institution_review", files=len(actual), png_files=len(images))


def build_export_diagnostics(run, out, cfg=None, *, attempt=None, images=True):
    run = safe_path(run).absolute() if run is not None else None
    out = safe_path(out).absolute()
    if out.parent.name != "export_review" or not DIRECTORY.fullmatch(out.name):
        raise ValueError("Choose export_review/visit_audit[_suffix]")
    if out.exists():
        validate_diagnostics(out)
        return out / "index.html"
    reader = Reader(out)
    if cfg is None:
        cfg = reader.read(run / "run_manifest.json", {}).get("config", {}) if run else {}
    if attempt is None:
        attempt = find_attempt(run, reader)
    attempt = safe_path(attempt) if attempt else None
    tables, summary = collect(run, attempt, cfg, reader)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-diagnostics-", dir=out.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        render(staging, tables, summary, images=images)
        manifest = dict(summary, files={p.relative_to(staging).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(staging.rglob("*")) if p.is_file()})
        save_json(staging / "EXPORT_MANIFEST.json", manifest)
        validate_diagnostics(staging)
        staging.rename(out)
    return out / "index.html"
