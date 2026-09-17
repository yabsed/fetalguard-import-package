"""Experiment B: report-aligned raw 2x150 input, no appended hand features.

CTGNetMini is the repository's Ogasawara/Chiou architecture reimplementation,
not a Google-distributed pretrained model. All CNN arms train from scratch.
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from .common import note, table, write_json, read_json
from .ctgnet import CTGNetMini, count_parameters
from .evaluation import load_segments, threshold90, evaluate, paired
from .telemetry import emit
from .survey import write_csv


CAPACITY_ARMS = {"small": lambda width: width <= 32, "medium": lambda width: width >= 64}
WIDTH_POLICY = "maximum mean validation AP across configured seeds; ties choose smaller width"
SEED_POLICY = "maximum validation AP within chosen width; ties choose smaller seed"


def capacity_summary(candidates, expected_seeds):
    """Reject incomplete seed searches before selecting a width, without test data."""
    rows = []
    frame = pd.DataFrame(candidates)
    expected = sorted(expected_seeds)
    for (mode, width), group in frame.groupby(["normalization", "width"], sort=True):
        if sorted(group.seed.tolist()) != expected:
            raise ValueError(f"Incomplete/duplicate CNN seed grid: {mode} width={width}")
        values = group.validation_auprc.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite CNN validation AP")
        rows.append(dict(normalization=mode, width=int(width), seeds=len(values),
                         mean_validation_auprc=float(values.mean()),
                         std_validation_auprc=float(values.std(ddof=0)),
                         min_validation_auprc=float(values.min()),
                         max_validation_auprc=float(values.max())))
    return rows


def select_capacity(candidates, summary, mode, arm):
    eligible = [row for row in summary if row["normalization"] == mode and CAPACITY_ARMS[arm](row["width"])]
    if not eligible:
        raise ValueError(f"CNN profile needs at least one width in {arm} group for {mode}")
    selected_width = max(eligible, key=lambda row: (row["mean_validation_auprc"], -row["width"]))
    chosen = max((row for row in candidates if row["normalization"] == mode and row["width"] == selected_width["width"]),
                 key=lambda row: (row["validation_auprc"], -row["seed"]))
    return dict(chosen, width_mean_validation_auprc=selected_width["mean_validation_auprc"],
                width_selection=WIDTH_POLICY, seed_selection=SEED_POLICY)


def matched_normalization_plan(candidates, summary, seeds):
    """Fix width using only the reference normalization's validation search."""
    indexed = {(row["normalization"], row["width"], row["seed"]): row for row in candidates}
    plan = []
    for arm in CAPACITY_ARMS:
        chosen = select_capacity(candidates, summary, "channel_maxabs", arm)
        for seed in seeds:
            left = ("channel_maxabs", chosen["width"], seed)
            right = ("per_segment_z", chosen["width"], seed)
            if left not in indexed or right not in indexed:
                raise ValueError(f"Matched normalization pair is missing: width={chosen['width']} seed={seed}")
            plan.append(dict(arm=arm, width=chosen["width"], seed=seed,
                             left=indexed[left], right=indexed[right],
                             reference_width_mean_validation_auprc=chosen["width_mean_validation_auprc"]))
    return plan


def normalize_holdout(raw, mode, normalization):
    if mode == "channel_maxabs":
        scale = np.asarray(normalization["scale"])
        return (raw / scale[None, :, None]).astype(np.float32)
    if mode == "per_segment_z":
        return ((raw - raw.mean(axis=2, keepdims=True)) /
                np.maximum(raw.std(axis=2, keepdims=True), 1e-6)).astype(np.float32)
    raise ValueError(mode)


def normalize(raw, training_mask, mode):
    if mode == "channel_maxabs":
        scale = np.maximum(np.abs(raw[training_mask]).max(axis=(0, 2)), 1e-6)
        return (raw / scale[None, :, None]).astype(np.float32), {"scale": scale.tolist(), "centering": False}
    if mode == "per_segment_z":
        return ((raw - raw.mean(axis=2, keepdims=True)) / np.maximum(raw.std(axis=2, keepdims=True), 1e-6)).astype(np.float32), {"per_segment": True}
    raise ValueError(mode)


def scores(model, x, device, batch=1024):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunks.append(torch.sigmoid(model(torch.from_numpy(x[start:start + batch]).to(device))).cpu().numpy())
    return np.concatenate(chunks)


def save_state(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_one(x, frame, width, mode, seed, out, cfg):
    opts = cfg["budget"]
    train, val = frame.split.eq("train").to_numpy(), frame.split.eq("val").to_numpy()
    device = torch.device(cfg["resolved_device"])
    model = CTGNetMini(width=width, input_len=150).to(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    # Reset after setting seeds; construction above is not used for training.
    model = CTGNetMini(width=width, input_len=150).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts["cnn_epochs"])
    y = frame.target.to_numpy(np.float32)
    pos = float(y[train].sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt((train.sum() - pos) / max(pos, 1)), device=device))
    out.mkdir(parents=True, exist_ok=True)
    completed = out / "complete.json"
    if completed.exists():
        model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
        emit("fit_reused", family="cnn", job=str(out), seed=seed)
        return model, read_json(completed)
    last = out / "last.pt"
    begin, best, best_epoch, stale, history = 0, -1.0, -1, 0, []
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        begin, best, best_epoch, stale = (checkpoint[k] for k in ("next_epoch", "best", "best_epoch", "stale"))
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["rng"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint["cuda_rng"]])
    started = time.monotonic()
    emit("fit_resumed" if begin else "fit_started", family="cnn", job=str(out), seed=seed,
         next_epoch=begin + 1, cap=opts["cnn_epochs"], patience=opts["cnn_patience"])
    history_columns = ["epoch", "train_loss", "validation_auprc", "learning_rate", "epoch_seconds"]
    if history:
        write_csv(out / "history.csv", history, history_columns)
    train_indices = np.flatnonzero(train)
    for epoch in range(begin, opts["cnn_epochs"]):
        if stale >= opts["cnn_patience"]:
            break
        epoch_started = time.monotonic()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        order = np.random.default_rng(seed + epoch * 10007).permutation(train_indices)
        total_loss = 0.0
        for start in range(0, len(order), opts["batch_size"]):
            idx = order[start:start + opts["batch_size"]]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("CNN loss became nonfinite")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
        scheduler.step()
        metric = float(average_precision_score(y[val], scores(model, x[val], device)))
        history.append(dict(epoch=epoch + 1, train_loss=total_loss / len(order), validation_auprc=metric,
                            learning_rate=learning_rate, epoch_seconds=time.monotonic() - epoch_started))
        if metric > best + 1e-8:
            best, best_epoch, stale = metric, epoch + 1, 0
            save_state(out / "best.pt", {k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        save_state(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                   next_epoch=epoch + 1, best=best, best_epoch=best_epoch, stale=stale, history=history,
                   rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
        write_csv(out / "history.csv", history, history_columns)
        write_json(out / "training_state.json", dict(epoch=epoch + 1, best_epoch=best_epoch, stale_epochs=stale,
                   cap=opts["cnn_epochs"], patience=opts["cnn_patience"], status="running_or_interrupted"))
        emit("epoch_finished", family="cnn", job=str(out), best_epoch=best_epoch, stale_epochs=stale,
             **history[-1])
        if epoch == 0 or (epoch + 1) % 10 == 0:
            note(f"CNN {mode} w={width} seed={seed} epoch={epoch + 1}: val AP={metric:.4f}, best={best:.4f}")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    result = dict(width=width, normalization=mode, seed=seed, parameters=count_parameters(model),
                  best_epoch=best_epoch, epochs_run=len(history), hit_cap=len(history) >= opts["cnn_epochs"],
                  validation_auprc=best, seconds_this_invocation=time.monotonic() - started,
                  stale_epochs=stale, patience=opts["cnn_patience"], cap=opts["cnn_epochs"],
                  patience_met=stale >= opts["cnn_patience"],
                  stop_reason="epoch_cap" if len(history) >= opts["cnn_epochs"] else "early_stopping")
    write_csv(out / "history.csv", history, history_columns)
    write_json(out / "training_state.json", dict(result, status="complete"))
    write_json(completed, result)
    emit("fit_finished", family="cnn", job=str(out), **result)
    return model, result


def run_b(run, out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    if not cfg["cnn"]:
        write_json(out / "status.json", {"status": "disabled_by_config", "H4": "not_tested"})
        return
    frame = load_segments(run)
    raw = np.load(run / "data/signals.npy", mmap_mode="r")
    masks = {s: frame.split.eq(s).to_numpy() for s in ("train", "val", "test")}
    opts = cfg["budget"]
    candidates = []
    for mode in opts["cnn_norms"]:
        x, norm = normalize(raw, masks["train"], mode)
        write_json(out / f"normalization_{mode}.json", norm)
        for width in opts["cnn_widths"]:
            for seed in opts["seeds"]:
                job = out / "models" / f"{mode}_w{width}_s{seed}"
                note(f"B training {job.name}")
                trained, result = train_one(x, frame, width, mode, seed, job, cfg)
                candidates.append(dict(**result, job=str(job.relative_to(run))))
                del trained
    pd.DataFrame(candidates).to_csv(out / "capacity_validation.csv", index=False)
    summary = capacity_summary(candidates, opts["seeds"])
    pd.DataFrame(summary).to_csv(out / "capacity_summary.csv", index=False)
    # Width and seed policies are frozen before evaluating any test predictions.
    selections = [(f"cnn_{arm}_{mode}", select_capacity(candidates, summary, mode, arm))
                  for mode in opts["cnn_norms"] for arm in CAPACITY_ARMS]
    matched_plan = (matched_normalization_plan(candidates, summary, opts["seeds"])
                    if {"channel_maxabs", "per_segment_z"}.issubset(opts["cnn_norms"]) else [])
    write_json(out / "protocol.json", {
        "width_selection": WIDTH_POLICY, "seed_selection": SEED_POLICY,
        "normalization_contrast": "channel_maxabs minus per_segment_z at the same width and seed",
        "normalization_width_selection": "channel_maxabs mean validation AP within each capacity arm",
        "inference_scope": "Selected width depends on reference-normalization validation performance; this is not an average causal effect over all widths.",
        "interpretation": "Paired intervals quantify test-cohort sampling uncertainty conditional on fitted models; no equivalence/noninferiority margin was specified.",
    })
    selected, predictions, results, costs = [], [], {}, []
    test, val = frame.loc[masks["test"]].reset_index(drop=True), frame.loc[masks["val"]].reset_index(drop=True)
    inference_cache = {}

    def inference(chosen):
        key = (chosen["normalization"], chosen["width"], chosen["seed"])
        if key in inference_cache:
            return inference_cache[key]
        mode = chosen["normalization"]
        norm = read_json(out / f"normalization_{mode}.json")
        device = torch.device(cfg["resolved_device"])
        model = CTGNetMini(width=chosen["width"], input_len=150).to(device)
        checkpoint = run / chosen["job"] / "best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        vp = scores(model, normalize_holdout(raw[masks["val"]], mode, norm), device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        normalized = normalize_holdout(raw[masks["test"]], mode, norm)
        tp = scores(model, normalized, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        cost = dict(normalization=mode, width=chosen["width"], seed=chosen["seed"],
                    parameters=chosen["parameters"], model_file_bytes=checkpoint.stat().st_size,
                    device=str(device), windows=len(test), seconds=seconds,
                    milliseconds_per_window=seconds * 1000 / len(test),
                    normalized_input_bytes=normalized.nbytes,
                    cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                    scope="in-memory raw windows: normalization and batched inference, after validation warm-up; excludes model/data loading and signal interpolation",
                    timing_repeats=1)
        inference_cache[key] = (vp, tp, cost)
        return inference_cache[key]

    for name, chosen in selections:
        vp, tp, cost = inference(chosen)
        threshold = threshold90(val.target, vp)
        results[name] = evaluate(test, tp, threshold, opts["bootstrap"], cfg["seed"])
        results[name].update(selection=chosen, threshold=threshold)
        selected.append(dict(model=name, **chosen))
        costs.append(dict(model=name, **cost))
        predictions.append(test[["record_id", "mother_id", "seg_idx", "target", "site", "record_all_normal"]].assign(model=name, score=tp, threshold=threshold))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(out / "test_predictions.csv", index=False)
    write_json(out / "metrics.json", results)
    write_json(out / "selection.json", selected)
    pd.DataFrame(costs).to_csv(out / "inference_cost.csv", index=False)
    matched, matched_predictions = [], []
    for job in matched_plan:
        lv, lp, _ = inference(job["left"])
        rv, rp, _ = inference(job["right"])
        lt, rt = threshold90(val.target, lv), threshold90(val.target, rv)
        matched.append(dict(arm=job["arm"], width=job["width"], seed=job["seed"],
                            reference_width_mean_validation_auprc=job["reference_width_mean_validation_auprc"],
                            left_normalization="channel_maxabs", right_normalization="per_segment_z",
                            left_validation_auprc=job["left"]["validation_auprc"],
                            right_validation_auprc=job["right"]["validation_auprc"],
                            left_metrics=evaluate(test, lp, lt), right_metrics=evaluate(test, rp, rt),
                            paired_difference=paired(test, lp, rp, opts["bootstrap"], cfg["seed"])))
        for mode, prediction, threshold in (("channel_maxabs", lp, lt), ("per_segment_z", rp, rt)):
            matched_predictions.append(test[["record_id", "mother_id", "seg_idx", "target", "site", "record_all_normal"]]
                .assign(arm=job["arm"], width=job["width"], seed=job["seed"], normalization=mode, score=prediction, threshold=threshold))
    write_json(out / "normalization_matched.json", {
        "status": "ok" if matched else "requires_both_normalizations",
        "contrast": "channel_maxabs minus per_segment_z",
        "selection_policy": "same width chosen by channel_maxabs seed-mean validation AP, compared at every configured seed",
        "note": "Each comparison holds architecture and seed fixed. Seeds share test mothers and are not independent cohorts; intervals do not include training uncertainty.",
        "comparisons": matched})
    if matched_predictions:
        pd.concat(matched_predictions, ignore_index=True).to_csv(out / "normalization_matched_predictions.csv", index=False)
    baseline = table(run / "experiment_a/test_predictions.csv").query("model == 'cat28'")
    comparisons = {}
    for name, rows in pred.groupby("model"):
        match = rows.merge(baseline[["record_id", "seg_idx", "score"]], on=["record_id", "seg_idx"], suffixes=("", "_cat28"), validate="one_to_one")
        if len(match) != len(test):
            raise ValueError("CNN and Cat28 have different test cohorts")
        comparisons[name] = paired(match, match.score_cat28, match.score, opts["bootstrap"], cfg["seed"])
    write_json(out / "paired_cat28_minus_cnn.json", comparisons)
    write_json(out / "status.json", {"status": "complete", "runs": len(candidates), "hit_cap": sum(r["hit_cap"] for r in candidates),
                                    "note": "Mock training cap is intentional. In full mode inspect hit_cap before claiming convergence."})
