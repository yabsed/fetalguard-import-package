"""Experiment B: report-aligned raw 2x150 input, no appended hand features.

CTGNetMini is the repository's Ogasawara/Chiou architecture reimplementation,
not a Google-distributed pretrained model. All CNN arms train from scratch.
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from .common import note, table, write_json, read_json
from .ctgnet import CTGNetMini, count_parameters
from .evaluation import load_segments, threshold90, evaluate, paired


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
    train_indices = np.flatnonzero(train)
    for epoch in range(begin, opts["cnn_epochs"]):
        if stale >= opts["cnn_patience"]:
            break
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
        history.append(dict(epoch=epoch + 1, train_loss=total_loss / len(order), validation_auprc=metric))
        if metric > best + 1e-8:
            best, best_epoch, stale = metric, epoch + 1, 0
            save_state(out / "best.pt", {k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        save_state(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                   next_epoch=epoch + 1, best=best, best_epoch=best_epoch, stale=stale, history=history,
                   rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            note(f"CNN {mode} w={width} seed={seed} epoch={epoch + 1}: val AP={metric:.4f}, best={best:.4f}")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    result = dict(width=width, normalization=mode, seed=seed, parameters=count_parameters(model),
                  best_epoch=best_epoch, epochs_run=len(history), hit_cap=len(history) >= opts["cnn_epochs"],
                  validation_auprc=best, seconds_this_invocation=time.monotonic() - started)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    write_json(completed, result)
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
                _, result = train_one(x, frame, width, mode, seed, job, cfg)
                candidates.append(dict(**result, job=str(job.relative_to(run))))
    pd.DataFrame(candidates).to_csv(out / "capacity_validation.csv", index=False)
    # B2 vs B3 capacity groups fixed before observing test outcomes.
    selected, predictions, results = [], [], {}
    test, val = frame.loc[masks["test"]].reset_index(drop=True), frame.loc[masks["val"]].reset_index(drop=True)
    for mode in opts["cnn_norms"]:
        x, _ = normalize(raw, masks["train"], mode)
        for arm, predicate in (("small", lambda r: r["width"] <= 32), ("medium", lambda r: r["width"] >= 64)):
            subset = [r for r in candidates if r["normalization"] == mode and predicate(r)]
            if not subset:
                raise ValueError(f"CNN profile needs at least one width in {arm} group")
            chosen = max(subset, key=lambda r: r["validation_auprc"])
            model = CTGNetMini(width=chosen["width"], input_len=150).to(cfg["resolved_device"])
            model.load_state_dict(torch.load(run / chosen["job"] / "best.pt", map_location=cfg["resolved_device"], weights_only=True))
            vp = scores(model, x[masks["val"]], cfg["resolved_device"])
            tp = scores(model, x[masks["test"]], cfg["resolved_device"])
            threshold = threshold90(val.target, vp)
            name = f"cnn_{arm}_{mode}"
            results[name] = evaluate(test, tp, threshold, opts["bootstrap"], cfg["seed"])
            results[name].update(selection=chosen, threshold=threshold)
            selected.append(dict(model=name, **chosen))
            predictions.append(test[["record_id", "mother_id", "seg_idx", "target", "site", "record_all_normal"]].assign(model=name, score=tp, threshold=threshold))
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(out / "test_predictions.csv", index=False)
    write_json(out / "metrics.json", results)
    write_json(out / "selection.json", selected)
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
