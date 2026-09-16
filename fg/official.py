"""Offline inference of bundled AI-Hub weights, with explicit task boundaries."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from .common import note, read_json, table, write_json
from .data import OFFICIAL_COLUMNS
from .evaluation import threshold90, evaluate, paired
from .signal_io import parse_window_abnormality


def letterbox(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = image.shape[:2]
    ratio = min(640 / h, 640 / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    dw, dh = (640 - new_w) / 2, (640 - new_h) / 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, int(round(dh - 0.1)), int(round(dh + 0.1)),
        int(round(dw - 0.1)), int(round(dw + 0.1)), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return tensor, ratio, dw, dh, w, h


def nms_single_class(prediction, confidence=0.001, iou=0.6, max_det=300):
    """YOLOv5 single-class objectness*class score, class-agnostic greedy NMS."""
    p = np.asarray(prediction)
    if p.ndim != 2 or p.shape[1] != 6:
        raise ValueError(f"Expected official single-class Nx6 predictions, got {p.shape}")
    score = p[:, 4] * p[:, 5]
    keep = score > confidence
    p, score = p[keep], score[keep]
    if not len(p):
        return np.empty((0, 5))
    boxes = np.column_stack([p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2])
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    order = np.argsort(-score, kind="stable")[:30000]
    selected = []
    while len(order) and len(selected) < max_det:
        index = order[0]
        selected.append(index)
        others = order[1:]
        lo, hi = np.maximum(boxes[index, :2], boxes[others, :2]), np.minimum(boxes[index, 2:], boxes[others, 2:])
        intersection = np.maximum(hi - lo, 0).prod(axis=1)
        overlap = intersection / np.maximum(area[index] + area[others] - intersection, 1e-12)
        order = others[overlap <= iou]
    return np.column_stack([boxes[selected], score[selected]])


def segment_scores(boxes, width, n_segments):
    scores = np.zeros(n_segments, float)
    for x1, y1, x2, y2, confidence in boxes:
        center = np.clip((x1 + x2) / 2, 0, width)
        index = min(int(center / width * n_segments), n_segments - 1)
        scores[index] = max(scores[index], confidence)
    return scores


def preflight_models(package, cfg):
    booster = xgb.Booster()
    booster.load_model(str(package / "models/official_xgboost/xgb_clf.json"))
    if booster.num_features() != len(OFFICIAL_COLUMNS):
        raise ValueError("Official XGBoost feature schema mismatch")
    model = torch.jit.load(str(package / "models/official_yolo/inference.torchscript"), map_location=cfg["resolved_device"])
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 640, 640, device=cfg["resolved_device"]))[0]
    if tuple(output.shape) != (1, 25200, 6) or not torch.isfinite(output).all():
        raise ValueError("Official YOLO runtime probe failed")


def run_official(package, run, out, cfg, catalog):
    out.mkdir(parents=True, exist_ok=True)
    if not cfg["official_models"]:
        write_json(out / "status.json", {"status": "disabled_by_config"})
        return
    booster = xgb.Booster()
    booster.load_model(str(package / "models/official_xgboost/xgb_clf.json"))
    records = table(run / "data/records.csv")
    xgb_metrics, predictions = {}, []
    for mode in ("original", "corrected"):
        frame = table(run / f"data/official_{mode}.csv")
        valid = frame[OFFICIAL_COLUMNS + ["target"]].notna().all(axis=1) & frame.target.isin([0, 1])
        usable = frame.loc[valid].merge(records[["record_id", "mother_id"]], on="record_id", validate="one_to_one")
        if usable.empty:
            xgb_metrics[mode] = {"status": "no_complete_records", "excluded": len(frame)}
            continue
        p = booster.predict(xgb.DMatrix(usable[OFFICIAL_COLUMNS].to_numpy(np.float32)))
        xgb_metrics[mode] = evaluate(usable, p, 0.5, cfg["budget"]["bootstrap"], cfg["seed"])
        xgb_metrics[mode].update(excluded=int((~valid).sum()), task="record-level Emergency", threshold=0.5,
                                 scope="Replay of distributed weights; training overlap unknown; not Abnormality accuracy")
        xgb_metrics[mode]["accuracy"] = float(((p >= 0.5) == usable.target.to_numpy()).mean())
        predictions.append(usable[["record_id", "mother_id", "target"]].assign(parser=mode, score=p))
    write_json(out / "xgboost_emergency_metrics.json", xgb_metrics)
    if predictions:
        pd.concat(predictions).to_csv(out / "xgboost_emergency_predictions.csv", index=False)
    model = torch.jit.load(str(package / "models/official_yolo/inference.torchscript"), map_location=cfg["resolved_device"]).eval()
    images = [(rid, item) for rid, item in catalog["images"].items() if rid in catalog["labels"]]
    rows = []
    by_record = records.set_index("record_id")
    cache = out / "yolo_record_cache"
    cache.mkdir(exist_ok=True)
    for number, (rid, item) in enumerate(images, 1):
        label = read_json(catalog["labels"][rid]["path"])
        flags = parse_window_abnormality(label, len(label["Abnormality"].split("_")))
        cached = cache / f"{rid}.json"
        if cached.exists():
            p = np.asarray(read_json(cached)["scores"], float)
        else:
            x, ratio, dw, dh, width, height = letterbox(item["path"])
            with torch.no_grad():
                raw = model(torch.from_numpy(x).to(cfg["resolved_device"]))[0][0].cpu().numpy()
            boxes = nms_single_class(raw)
            if len(boxes):
                boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / ratio, 0, width)
                boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / ratio, 0, height)
            p = segment_scores(boxes, width, len(flags))
            write_json(cached, {"scores": p, "boxes_xyxy_conf": boxes})
        if len(p) != len(flags):
            raise ValueError(f"YOLO cache/label mismatch: {rid}")
        mid = str(by_record.loc[rid, "mother_id"]) if rid in by_record.index else "unmatched_" + rid
        for index, (target, score) in enumerate(zip(flags, p)):
            rows.append(dict(record_id=rid, mother_id=mid, seg_idx=index, target=int(target), score=score, record_all_normal=int(not flags.any())))
        if number % 50 == 0:
            note(f"공식 YOLO {number}/{len(images)} images")
    if not rows:
        write_json(out / "yolo_metrics.json", {"status": "no_images", "reason": "refine_images PNG required"})
    else:
        pred = pd.DataFrame(rows)
        pred.to_csv(out / "yolo_predictions.csv", index=False)
        # Only records with known mother IDs enter intervals; labels-only images remain in raw replay.
        splits = table(run / "splits/segments.csv")
        matched = pred.merge(splits[["record_id", "seg_idx", "split"]], on=["record_id", "seg_idx"], validate="one_to_one")
        val, test = matched[matched.split == "val"], matched[matched.split == "test"]
        info = dict(status="ok", all_images=len(images), matched_segments=len(matched),
                    preprocessing="640x640 square letterbox, RGB float/255, conf=0.001, IoU NMS=0.6, max_det=300; center-to-segment max confidence",
                    scope="Image-available subset only; do not impute missing PNG as zero detections. Official training overlap unknown.",
                    unmatched_images=len(set(pred.record_id) - set(records.record_id)))
        if val.target.nunique() == 2 and test.target.nunique() == 2:
            threshold = threshold90(val.target, val.score)
            info.update(threshold=threshold, metrics=evaluate(test, test.score, threshold, cfg["budget"]["bootstrap"], cfg["seed"]))
            base = table(run / "experiment_a/test_predictions.csv").query("model == 'cat28'")
            join = test.merge(base[["record_id", "seg_idx", "score", "threshold"]], on=["record_id", "seg_idx"], suffixes=("", "_cat28"), validate="one_to_one")
            info["cat28_same_image_subset"] = evaluate(join, join.score_cat28, join.threshold, cfg["budget"]["bootstrap"], cfg["seed"])
            info["paired_cat28_minus_yolo"] = paired(join, join.score_cat28, join.score, cfg["budget"]["bootstrap"], cfg["seed"])
        else:
            info["comparison_status"] = "insufficient_image_subset_classes"
        write_json(out / "yolo_metrics.json", info)
    write_json(out / "status.json", {"status": "complete", "models": ["AI-Hub XGBoost", "AI-Hub YOLOv5s"],
        "original_parser": "Intentionally reproduces official string-boundary/signed/twin parsing errors for baseline replay only.",
        "corrected_parser": "Sensitivity analysis on the SAME fixed official weights, not a retrained model."})
