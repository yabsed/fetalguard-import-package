from __future__ import annotations

import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .common import note, table, write_json, require_two_classes
from .features import CAT28, GROUPS, ROBUST
from .evaluation import group_folds, load_segments, threshold90, evaluate, paired


def logistic():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=42))


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
        return model
    if kind == "cat":
        model.fit(train[columns], train.target, eval_set=(val[columns], val.target),
                  early_stopping_rounds=opts["tree_patience"])
    else:
        model.fit(train[columns], train.target, eval_set=[(val[columns], val.target)], verbose=False)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name("temporary_" + path.name)
        model.save_model(str(temp))
        temp.replace(path)
    return model


def predict(model, frame, columns):
    return np.asarray(model.predict_proba(frame[columns])[:, 1], float)


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
    variants = {"cat28": ("cat", CAT28), "xgb28": ("xgb", CAT28), "robust": ("cat", ROBUST)}
    variants.update({"without_" + name: ("cat", [c for c in CAT28 if c not in columns]) for name, columns in GROUPS.items()})
    cv_metrics, cv_summary = [], []
    for seed in opts["seeds"]:
        oof = dev[["record_id", "mother_id", "seg_idx", "target", "record_all_normal"]].copy()
        for fold, (tr, te) in enumerate(group_folds(dev, opts["cv_folds"], seed)):
            pool = dev.iloc[tr].reset_index(drop=True)
            it, iv = next(group_folds(pool, 4, seed + fold))
            fitting, tuning, hold = pool.iloc[it], pool.iloc[iv], dev.iloc[te]
            for name, (kind, columns) in variants.items():
                note(f"A CV seed={seed} fold={fold + 1}/{opts['cv_folds']} {name}")
                ext = ".cbm" if kind == "cat" else ".json"
                model = fit_tree(fitting, tuning, columns, cfg, seed, kind, out / "models" / f"cv_{seed}_{fold}_{name}{ext}")
                scores = predict(model, hold, columns)
                threshold = threshold90(tuning.target, predict(model, tuning, columns))
                oof.loc[te, "score__" + name] = scores
                oof.loc[te, "threshold__" + name] = threshold
                oof.loc[te, "fold"] = fold
                cv_metrics.append(dict(seed=seed, fold=fold, model=name, **evaluate(hold, scores, threshold)["point"]))
            # H2: preprocessing is fitted only inside the training fold.
            for feature in CAT28:
                model = logistic().fit(fitting[[feature]], fitting.target)
                scores = predict(model, hold, [feature])
                threshold = threshold90(tuning.target, predict(model, tuning, [feature]))
                oof.loc[te, "score__single_" + feature] = scores
                oof.loc[te, "threshold__single_" + feature] = threshold
        if oof.isna().any().any():
            raise ValueError("OOF coverage is incomplete")
        oof.to_csv(out / f"cv_predictions_seed{seed}.csv", index=False)
        for name in list(variants) + ["single_" + c for c in CAT28]:
            result = evaluate(oof, oof["score__" + name], oof["threshold__" + name].to_numpy(), opts["bootstrap"], seed)
            cv_summary.append(dict(seed=int(seed), model=name, result=result))
    pd.DataFrame(cv_metrics).to_csv(out / "cv_fold_metrics.csv", index=False)
    write_json(out / "cv_summary.json", cv_summary)
    pd.DataFrame([dict(seed=r["seed"], model=r["model"], **r["result"]["point"]) for r in cv_summary]).to_csv(out / "cv_summary.csv", index=False)
    holdout_results, prediction_rows, selection = {}, [], []
    chosen = {}
    for name, (kind, columns) in variants.items():
        candidates = []
        for seed in opts["seeds"]:
            ext = ".cbm" if kind == "cat" else ".json"
            path = out / "models" / f"holdout_{name}_{seed}{ext}"
            model = fit_tree(train, val, columns, cfg, seed, kind, path)
            score = float(average_precision_score(val.target, predict(model, val, columns)))
            candidates.append((score, seed, model, path))
            selection.append(dict(model=name, seed=seed, validation_auprc=score))
        _, seed, model, path = max(candidates, key=lambda item: item[0])
        vp, tp = predict(model, val, columns), predict(model, test, columns)
        threshold = threshold90(val.target, vp)
        holdout_results[name] = evaluate(test, tp, threshold, opts["bootstrap"], cfg["seed"])
        holdout_results[name].update(threshold=threshold, selected_seed=seed, model_file=str(path.relative_to(run)), features=columns)
        chosen[name] = (model, tp, vp, columns)
        prediction_rows.append(test[["record_id", "mother_id", "seg_idx", "target", "site", "record_all_normal"]].assign(model=name, score=tp, threshold=threshold))
    singles = []
    for feature in CAT28:
        model = logistic().fit(train[[feature]], train.target)
        vp = predict(model, val, [feature])
        singles.append((roc_auc_score(val.target, vp), feature, model, vp))
    _, feature, single, vp = max(singles, key=lambda item: item[0])
    sp = predict(single, test, [feature])
    st = threshold90(val.target, vp)
    holdout_results["best_single"] = evaluate(test, sp, st, opts["bootstrap"], cfg["seed"])
    holdout_results["best_single"].update(feature=feature, threshold=st, selection="validation_AUROC")
    prediction_rows.append(test[["record_id", "mother_id", "seg_idx", "target", "site", "record_all_normal"]].assign(model="best_single", score=sp, threshold=st))
    pd.DataFrame([dict(feature=f, validation_auroc=s) for s, f, _, _ in singles]).to_csv(out / "single_feature_selection.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(selection).to_csv(out / "validation_selection.csv", index=False)
    write_json(out / "holdout_metrics.json", holdout_results)
    comparisons = {name: paired(test, chosen["cat28"][1], values[1], opts["bootstrap"], cfg["seed"])
                   for name, values in chosen.items() if name != "cat28"}
    comparisons["best_single"] = paired(test, chosen["cat28"][1], sp, opts["bootstrap"], cfg["seed"])
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
    note("실험 A 완료: CV·홀드아웃·절제·단일 인자·SHAP 저장")
