from __future__ import annotations

import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .common import note, table, write_json, require_two_classes
from .features import CAT18, CAT28, GROUPS, ROBUST
from .evaluation import group_folds, load_segments, threshold90, evaluate, paired
from .telemetry import emit
from .survey import write_csv


def logistic():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=42))


class SingleFeatureSpline(TransformerMixin, BaseEstimator):
    """Fixed-complexity univariate basis; bounds/knots use fitting data only.

    Constant and entirely missing predictors become an intercept-only model.
    Uniform knots avoid failures from repeated quantiles in sparse event counts.
    Values beyond the training bounds have constant extrapolation.
    """
    def __init__(self, n_knots=5, degree=3):
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X, y=None):
        x = np.asarray(X, float)
        if x.ndim != 2 or x.shape[1] != 1 or not np.isfinite(x).all():
            raise ValueError("Spline expects one finite, imputed predictor")
        self.n_features_in_ = 1
        self.constant_ = bool(np.ptp(x[:, 0]) == 0)
        self.spline_ = None if self.constant_ else SplineTransformer(
            n_knots=self.n_knots, degree=self.degree, knots="uniform",
            extrapolation="constant", include_bias=False).fit(x)
        return self

    def transform(self, X):
        x = np.asarray(X, float)
        if x.ndim != 2 or x.shape[1] != 1 or not np.isfinite(x).all():
            raise ValueError("Spline expects one finite, imputed predictor")
        return np.zeros((len(x), 1)) if self.constant_ else self.spline_.transform(x)


def nonlinear_logistic():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         SingleFeatureSpline(), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=1000, random_state=42))


def feature_response(frame, bins=8):
    """Descriptive training-only quantile bins, not fitted causal effects."""
    if "split" in frame and not frame.split.eq("train").all():
        raise ValueError("Feature-response summaries must use training rows only")
    rows = []
    for feature in CAT28:
        values = frame[feature].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite feature-response input: {feature}")
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        labels = np.searchsorted(edges[1:-1], values, side="left")
        for bin_id in np.unique(labels):
            selected = frame.iloc[np.flatnonzero(labels == bin_id)]
            positive = selected.target.eq(1)
            rows.append(dict(feature=feature, bin=int(bin_id),
                lower=float(edges[bin_id]), upper=float(edges[min(bin_id + 1, len(edges) - 1)]),
                n_segments=len(selected), n_mothers=int(selected.mother_id.nunique()),
                n_positive=int(positive.sum()), positive_mothers=int(selected.loc[positive, "mother_id"].nunique()),
                negative_mothers=int(selected.loc[~positive, "mother_id"].nunique()),
                positive_fraction=float(positive.mean())))
    return pd.DataFrame(rows)


def align_unsmoothed(frame, candidate):
    """Replace only feature values, preserving all labels, splits and row order."""
    keys = ["record_id", "seg_idx"]
    if frame.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError("Duplicate preprocessing-sensitivity segment keys")
    base_keys = pd.MultiIndex.from_frame(frame[keys])
    other_keys = pd.MultiIndex.from_frame(candidate[keys])
    if len(base_keys) != len(other_keys) or len(base_keys.difference(other_keys)):
        raise ValueError("Unsmoothed cohort must exactly match the primary cohort")
    if "target" in candidate:
        target = candidate.set_index(keys).target.reindex(base_keys).to_numpy()
        if not np.array_equal(frame.target.to_numpy(), target):
            raise ValueError("Unsmoothed target mismatch")
    result = frame.drop(columns=CAT28).merge(candidate[keys + CAT28], on=keys,
                                            validate="one_to_one", how="left", sort=False)
    if not np.isfinite(result[CAT28].to_numpy(float)).all():
        raise ValueError("Unsmoothed feature values must be finite")
    return result


def timed_predict(model, frame, columns, repeats=3):
    """Warm batch inference; extraction/IO and tuning are deliberately excluded."""
    scores = predict(model, frame, columns)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(model, frame, columns)
        elapsed.append(time.perf_counter() - start)
    duration = float(np.median(elapsed))
    return scores, dict(median_batch_seconds=duration, segments=len(frame), repeats=repeats,
                       median_microseconds_per_segment=duration * 1e6 / len(frame),
                       scope="feature_matrix_only_excludes_extraction_and_io")


def fit_tree(train, val, columns, cfg, seed, kind="cat", path=None):
    require_two_classes(train.target, "training")
    require_two_classes(val.target, "validation")
    opts = cfg["budget"]
    if kind == "cat":
        model = CatBoostClassifier(iterations=opts["tree_iterations"], depth=6, learning_rate=0.05,
            loss_function="Logloss", eval_metric="PRAUC", random_seed=seed, auto_class_weights="SqrtBalanced",
            thread_count=cfg["threads"], allow_writing_files=False, verbose=False)
    else:
        pos = int(train.target.sum())
        model = XGBClassifier(n_estimators=opts["tree_iterations"], max_depth=4, learning_rate=0.05,
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=cfg["threads"],
            random_state=seed, scale_pos_weight=np.sqrt((len(train) - pos) / max(pos, 1)),
            early_stopping_rounds=opts["tree_patience"])
    if path is not None and path.exists():
        model.load_model(str(path))
        emit("fit_reused", family=kind, job=str(path), seed=seed,
             history_available=path.with_name(path.name + ".training.json").is_file())
        return model
    started = time.monotonic()
    emit("fit_started", family=kind, job=str(path), seed=seed, train_rows=len(train), validation_rows=len(val))
    try:
        if kind == "cat":
            model.fit(train[columns], train.target, eval_set=(val[columns], val.target),
                      early_stopping_rounds=opts["tree_patience"])
        else:
            model.fit(train[columns], train.target, eval_set=[(val[columns], val.target)], verbose=False)
    except BaseException as exc:
        emit("fit_failed", family=kind, job=str(path), seed=seed,
             error_type=type(exc).__name__, seconds=time.monotonic() - started)
        raise
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        history = model.get_evals_result() if kind == "cat" else model.evals_result()
        history_columns = {f"{split}/{metric}": values for split, metrics in history.items() for metric, values in metrics.items()}
        count = max((len(values) for values in history_columns.values()), default=0)
        best = int(model.get_best_iteration() if kind == "cat" else model.best_iteration) + 1
        score_column = "validation/PRAUC" if kind == "cat" else "validation_0/aucpr"
        stopped = count < opts["tree_iterations"] and count - best >= opts["tree_patience"]
        # Save diagnostics before the model: a crash cannot leave a reusable model without its history.
        write_csv(path.with_name(path.name + ".history.csv"),
                  [dict(iteration=i + 1, **{key: values[i] for key, values in history_columns.items() if i < len(values)})
                   for i in range(count)], ["iteration"] + list(history_columns))
        write_json(path.with_name(path.name + ".training.json"), dict(family=kind, seed=seed,
                   iterations_run=count, best_iteration=best, stale_iterations=count - best,
                   cap=opts["tree_iterations"], patience=opts["tree_patience"],
                   hit_cap=count >= opts["tree_iterations"], patience_met=count - best >= opts["tree_patience"],
                   stop_reason="early_stopping" if stopped else "iteration_cap" if count >= opts["tree_iterations"] else "stopped_before_cap",
                   score_column=score_column, score_metric="CatBoost PRAUC" if kind == "cat" else "XGBoost aucpr",
                   seconds=time.monotonic() - started, train_rows=len(train), validation_rows=len(val),
                   scope="training_and_validation_only", learning_rate=0.05))
        temp = path.with_name("temporary_" + path.name)
        model.save_model(str(temp))
        temp.replace(path)
    emit("fit_finished", family=kind, job=str(path), seed=seed, seconds=time.monotonic() - started)
    return model


def predict(model, frame, columns):
    return np.asarray(model.predict_proba(frame[columns])[:, 1], float)


def fit_variant(train, val, columns, cfg, seed, kind, path):
    if kind != "logistic":
        return fit_tree(train, val, columns, cfg, seed, kind, path)
    if path.exists():
        emit("fit_reused", family="logistic", job=str(path), seed=seed)
        return load(path)
    require_two_classes(train.target, "logistic training")
    started = time.monotonic()
    emit("fit_started", family="logistic", job=str(path), seed=seed, train_rows=len(train))
    model = logistic().fit(train[columns], train.target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("temporary_" + path.name)
    dump(model, temp)
    temp.replace(path)
    iterations = int(model[-1].n_iter_.max())
    emit("fit_finished", family="logistic", job=str(path), seed=seed, iterations=iterations,
         hit_cap=iterations >= model[-1].max_iter, seconds=time.monotonic() - started)
    return model


def with_metadata(run):
    frame = load_segments(run)
    records = table(run / "data/records.csv")
    from .data import EMR_FEATURES
    for column in EMR_FEATURES:
        if column not in records:
            records[column] = np.nan
    cols = ["record_id", "maternal_age", "gestational_age"] + [c for c in EMR_FEATURES if c not in {"maternal_age", "gestational_age"}]
    return frame.merge(records[cols], on="record_id", validate="many_to_one", how="left", sort=False)


def run_a(run, out, cfg):
    frame = load_segments(run)
    dev = frame[frame.split != "test"].reset_index(drop=True)
    train, val, test = [frame[frame.split == s].reset_index(drop=True) for s in ("train", "val", "test")]
    out.mkdir(parents=True, exist_ok=True)
    opts = cfg["budget"]
    variants = {"cat28": ("cat", CAT28, "primary"), "cat18": ("cat", CAT18, "primary"),
                "xgb28": ("xgb", CAT28, "primary"), "robust": ("cat", ROBUST, "primary"),
                "logistic28": ("logistic", CAT28, "primary")}
    variants.update({"without_" + name: ("cat", [c for c in CAT28 if c not in columns], "primary")
                     for name, columns in GROUPS.items()})
    frames = {"primary": frame}
    sensitivity_path = run / "data/segments_unsmoothed.csv"
    if sensitivity_path.exists():
        frames["unsmoothed"] = align_unsmoothed(frame, table(sensitivity_path))
        variants["cat28_unsmoothed"] = ("cat", CAT28, "unsmoothed")
    write_json(out / "preprocessing_sensitivity.json", dict(
        status="ok" if "unsmoothed" in frames else "unavailable",
        comparison="cat28_minus_cat28_unsmoothed", same_cohort=True,
        changed_factor="feature smoothing only; gap imputation, splits and tuning budgets fixed",
        interpretation="Sensitivity analysis; not used to select the primary model."))
    dev_frames = {name: value[value.split != "test"].reset_index(drop=True) for name, value in frames.items()}
    holdout_frames = {name: tuple(value[value.split == split].reset_index(drop=True)
                                 for split in ("train", "val", "test")) for name, value in frames.items()}
    feature_response(train).to_csv(out / "feature_response_train.csv", index=False)
    univariate_families = {"linear": ("single_", logistic), "nonlinear_spline": ("nonlinear_single_", nonlinear_logistic)}
    extensions = {"cat": ".cbm", "xgb": ".json", "logistic": ".joblib"}
    all_cv_names = list(variants) + [prefix + c for prefix, _ in univariate_families.values() for c in CAT28]
    cv_metrics, cv_summary = [], []
    for seed in opts["seeds"]:
        oof = dev[["record_id", "mother_id", "seg_idx", "target", "record_all_normal"]].copy()
        prediction_columns = [prefix + name for name in all_cv_names for prefix in ("score__", "threshold__")]
        oof = pd.concat([oof, pd.DataFrame(np.nan, index=oof.index, columns=prediction_columns + ["fold"])], axis=1)
        for fold, (tr, te) in enumerate(group_folds(dev, opts["cv_folds"], seed)):
            pool = dev.iloc[tr].reset_index(drop=True)
            it, iv = next(group_folds(pool, 4, seed + fold))
            fitting, tuning, hold = pool.iloc[it], pool.iloc[iv], dev.iloc[te]
            for name, (kind, columns, source) in variants.items():
                note(f"A CV seed={seed} fold={fold + 1}/{opts['cv_folds']} {name}")
                view = dev_frames[source]
                fit_view, tune_view, hold_view = view.iloc[tr[it]], view.iloc[tr[iv]], view.iloc[te]
                path = out / "models" / f"cv_{seed}_{fold}_{name}{extensions[kind]}"
                model = fit_variant(fit_view, tune_view, columns, cfg, seed, kind, path)
                scores = predict(model, hold_view, columns)
                threshold = threshold90(tune_view.target, predict(model, tune_view, columns))
                oof.loc[te, "score__" + name] = scores
                oof.loc[te, "threshold__" + name] = threshold
                oof.loc[te, "fold"] = fold
                cv_metrics.append(dict(seed=seed, fold=fold, model=name, **evaluate(hold, scores, threshold)["point"]))
            # H2: preprocessing is fitted only inside the training fold.
            for prefix, factory in univariate_families.values():
                for feature in CAT28:
                    model = factory().fit(fitting[[feature]], fitting.target)
                    scores = predict(model, hold, [feature])
                    threshold = threshold90(tuning.target, predict(model, tuning, [feature]))
                    oof.loc[te, "score__" + prefix + feature] = scores
                    oof.loc[te, "threshold__" + prefix + feature] = threshold
        if oof.isna().any().any():
            raise ValueError("OOF coverage is incomplete")
        oof.to_csv(out / f"cv_predictions_seed{seed}.csv", index=False)
        for name in all_cv_names:
            result = evaluate(oof, oof["score__" + name], oof["threshold__" + name].to_numpy(), opts["bootstrap"], seed)
            cv_summary.append(dict(seed=int(seed), model=name, result=result))
    pd.DataFrame(cv_metrics).to_csv(out / "cv_fold_metrics.csv", index=False)
    write_json(out / "cv_summary.json", cv_summary)
    pd.DataFrame([dict(seed=r["seed"], model=r["model"], **r["result"]["point"]) for r in cv_summary]).to_csv(out / "cv_summary.csv", index=False)
    holdout_results, prediction_rows, selection, seed_metrics = {}, [], [], []
    chosen, seed_predictions = {}, {}
    for name, (kind, columns, source) in variants.items():
        train_view, val_view, test_view = holdout_frames[source]
        candidates = []
        for seed in opts["seeds"]:
            path = out / "models" / f"holdout_{name}_{seed}{extensions[kind]}"
            model = fit_variant(train_view, val_view, columns, cfg, seed, kind, path)
            score = float(average_precision_score(val_view.target, predict(model, val_view, columns)))
            candidates.append((score, seed, model, path))
            selection.append(dict(model=name, seed=seed, validation_auprc=score))
        _, seed, model, path = max(candidates, key=lambda item: item[0])
        # Selection is finished before any test predictions are inspected.
        vp = predict(model, val_view, columns)
        tp, timing = timed_predict(model, test_view, columns)
        threshold = threshold90(val.target, vp)
        holdout_results[name] = evaluate(test, tp, threshold, opts["bootstrap"], cfg["seed"])
        holdout_results[name].update(threshold=threshold, selected_seed=seed, model_file=str(path.relative_to(run)),
                                    features=columns, selection="validation_AP", inference_timing=timing,
                                    model_bytes=path.stat().st_size, feature_source=source)
        chosen[name] = (model, tp, vp, columns)
        prediction_rows.append(test[["record_id", "mother_id", "seg_idx", "target", "site", "record_all_normal"]].assign(model=name, score=tp, threshold=threshold))
        for score, candidate_seed, candidate, _ in candidates:
            cp = tp if candidate_seed == seed else predict(candidate, test_view, columns)
            ct = threshold90(val.target, predict(candidate, val_view, columns))
            seed_predictions[(name, candidate_seed)] = cp
            point = evaluate(test, cp, ct)["point"]
            seed_metrics.append(dict(seed=candidate_seed, model=name, validation_auprc=score,
                                     selected=candidate_seed == seed, **point))
    single_rows = []
    for family, (_, factory) in univariate_families.items():
        singles = []
        for feature in CAT28:
            model = factory().fit(train[[feature]], train.target)
            vp = predict(model, val, [feature])
            score = float(roc_auc_score(val.target, vp))
            singles.append((score, feature, model, vp))
            single_rows.append(dict(family=family, feature=feature, validation_auroc=score))
        _, feature, single, vp = max(singles, key=lambda item: item[0])
        name = "best_single" if family == "linear" else "best_single_nonlinear"
        sp, timing = timed_predict(single, test, [feature])
        st = threshold90(val.target, vp)
        holdout_results[name] = evaluate(test, sp, st, opts["bootstrap"], cfg["seed"])
        path = out / "models" / (name + ".joblib")
        dump(single, path)
        holdout_results[name].update(feature=feature, family=family, threshold=st, selection="validation_AUROC",
                                    inference_timing=timing, model_file=str(path.relative_to(run)), model_bytes=path.stat().st_size)
        chosen[name] = (single, sp, vp, [feature])
        prediction_rows.append(test[["record_id", "mother_id", "seg_idx", "target", "site", "record_all_normal"]].assign(model=name, score=sp, threshold=st))
    pd.DataFrame(single_rows).to_csv(out / "single_feature_selection.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(out / "seed_holdout_metrics.csv", index=False)
    seed_deltas = []
    for seed in opts["seeds"]:
        for name in variants:
            if name != "cat28":
                delta = paired(test, seed_predictions[("cat28", seed)], seed_predictions[(name, seed)], 0, cfg["seed"])
                seed_deltas.append(dict(seed=seed, comparator=name,
                    **{k + "_difference": v["difference"] for k, v in delta["metrics"].items()}))
    pd.DataFrame(seed_deltas).to_csv(out / "seed_paired_cat28_minus_comparator.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(selection).to_csv(out / "validation_selection.csv", index=False)
    write_json(out / "holdout_metrics.json", holdout_results)
    comparisons = {name: paired(test, chosen["cat28"][1], values[1], opts["bootstrap"], cfg["seed"])
                   for name, values in chosen.items() if name != "cat28"}
    write_json(out / "paired_cat28_minus_comparator.json", comparisons)
    model = chosen["cat28"][0]
    importance = pd.DataFrame({"feature": CAT28, "prediction_values_change": model.feature_importances_})
    sample = test.sample(n=min(len(test), opts["shap_samples"]), random_state=cfg["seed"])
    shap = model.get_feature_importance(Pool(sample[CAT28], sample.target), type="ShapValues", thread_count=cfg["threads"])
    importance["mean_abs_shap"] = np.abs(shap[:, :-1]).mean(axis=0)
    importance.sort_values("mean_abs_shap", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    explanation = sample[["record_id", "seg_idx", "target"]].reset_index(drop=True)
    explanation["expected_logit"] = shap[:, -1]
    for i, c in enumerate(CAT28):
        explanation["value_" + c] = sample[c].to_numpy()
        explanation["shap_" + c] = shap[:, i]
    explanation.to_csv(out / "local_explanations.csv", index=False)
    note("실험 A 완료: 동일 분할 CV·Cat18·선형/비선형 단일 인자·평활 민감도·seed 안정성·SHAP 저장")
