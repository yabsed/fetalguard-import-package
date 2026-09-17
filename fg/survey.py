"""Tolerant read-only reconnaissance, independent of ML imports and training.

Counts use source files / source windows, NOT the selected-fetus training cohort.
No imputation, inferred units, ID repair, or clinical interpretation is performed.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from .telemetry import emit, save_json

KINDS = ("annotation_person", "labels", "emr")


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def longest_run(values, predicate):
    longest = current = 0
    for value in values:
        current = current + 1 if predicate(value) else 0
        longest = max(longest, current)
    return longest


def signal_stats(raw):
    parts = raw.split(",") if isinstance(raw, str) else raw if isinstance(raw, list) else []
    values, invalid = [], 0
    for part in parts:
        try:
            value = float(part) if not isinstance(part, bool) else float("nan")
        except (ValueError, TypeError, OverflowError):
            value = float("nan")
        invalid += int(not math.isfinite(value))
        values.append(value)
    finite = [value for value in values if math.isfinite(value)]
    flat = best_flat = 0
    previous = None
    for value in values:
        # Zero-valued runs are already counted separately; TOCO zero meaning is unknown.
        flat = flat + 1 if value == previous and math.isfinite(value) and value != 0 else int(math.isfinite(value) and value != 0)
        best_flat = max(best_flat, flat)
        previous = value
    return dict(samples=len(parts), invalid=invalid, zeros=sum(value == 0 for value in finite),
                negative=sum(value < 0 for value in finite),
                longest_zero_samples=longest_run(values, lambda v: v == 0),
                longest_flat_nonzero_samples=best_flat,
                minimum=min(finite) if finite else None, maximum=max(finite) if finite else None,
                representation=type(raw).__name__)


def run_survey(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    if root == out or root in out.parents:
        raise ValueError("Survey output must be outside source data")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Survey output exists: use a new visit directory")
    out.mkdir(parents=True, exist_ok=True)
    inventory, bytes_by_kind, parsed = Counter(), Counter(), Counter()
    catalog = {kind: defaultdict(list) for kind in KINDS}
    schemas = defaultdict(Counter)
    variant = Counter()
    raw_rows, sites, errors, source_rows = [], defaultdict(Counter), [], []
    duplicate_signals = defaultdict(set)

    def problem(path, category, detail):
        row = dict(path=str(path.relative_to(root)), category=category, detail=detail)
        errors.append(row)
        # Persist malformed inputs immediately, even if the visit is later interrupted.
        with (out / "issues.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    save_json(out / "status.json", dict(status="running", scope="all_source_files_not_training_cohort"))
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            problem(path, "symlink_skipped", "Resolve source ownership before analysis")
            continue
        kind = path.parent.name if path.suffix.lower() == ".json" and path.parent.name in KINDS else "other:" + path.suffix.lower()
        inventory[kind] += 1
        try:
            stat = path.stat()
            bytes_by_kind[kind] += stat.st_size
            source = dict(path=str(path.relative_to(root)), kind=kind, bytes=stat.st_size,
                          mtime_ns=stat.st_mtime_ns, sha256=None)
            source_rows.append(source)
            if kind not in KINDS:
                continue
            catalog[kind][path.stem].append(str(path.relative_to(root)))
            content = path.read_bytes()
            source["sha256"] = hashlib.sha256(content).hexdigest()
            obj = json.loads(content.decode("utf-8-sig"))
            if not isinstance(obj, dict):
                raise ValueError("JSON root is not an object")
        except (OSError, ValueError) as exc:
            problem(path, "unreadable_json", str(exc))
            continue
        parsed[kind] += 1
        for field, value in obj.items():
            stat = schemas[kind, field]
            stat["present"] += 1
            stat["type:" + type(value).__name__] += 1
            stat["null_or_blank"] += int(value is None or isinstance(value, str) and not value.strip())
            stat["sentinel_9999"] += int(str(value).strip() == "9999")
            stat["zero"] += int(value == 0 or value == "0")
        if kind != "annotation_person":
            continue
        segments = obj.get("data")
        if not isinstance(segments, list) or not segments:
            problem(path, "unsupported_data", "Expected a nonempty data list")
            continue
        site = str(obj.get("code", path.stem)).split("_")[0]
        sites[site]["annotation_files"] += 1
        canonical = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                problem(path, "unsupported_segment", f"data[{index}] is not an object")
                continue
            row = dict(path=str(path.relative_to(root)), record_id=path.stem, site=site, source_window=index)
            interval = segment.get("interval_sec")
            try:
                dt = float(interval) if not isinstance(interval, bool) else None
                dt = dt if dt is not None and math.isfinite(dt) and dt > 0 else None
            except (ValueError, TypeError, OverflowError):
                dt = None
            row["interval_sec"] = dt
            stats = {channel: signal_stats(segment.get(channel)) for channel in ("fhr", "toco")}
            for channel, info in stats.items():
                row.update({channel + "_" + key: value for key, value in info.items()})
                row[channel + "_longest_zero_seconds"] = info["longest_zero_samples"] * dt if dt else None
            contract = (dt == 2 and all(info["samples"] == 150 and info["invalid"] == 0 for info in stats.values())
                        and stats["fhr"]["negative"] == 0)
            row["shape_contract_candidate"] = contract
            row["equal_channel_lengths"] = stats["fhr"]["samples"] == stats["toco"]["samples"]
            variant[(str(interval), stats["fhr"]["samples"], stats["toco"]["samples"],
                     stats["fhr"]["representation"], stats["toco"]["representation"])] += 1
            raw_rows.append(row)
            sites[site]["source_windows"] += 1
            sites[site]["shape_contract_candidates"] += int(contract)
            for channel, info in stats.items():
                for name in ("samples", "zeros", "invalid", "negative"):
                    sites[site][channel + "_" + name] += info[name]
            # Exact representation fingerprint, not approximate physiological similarity.
            canonical.append([segment.get("fhr"), segment.get("toco"), interval])
        if canonical:
            digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            duplicate_signals[digest].add(path.stem)
        if parsed[kind] % 250 == 0:
            emit("survey_progress", annotation_files=parsed[kind], source_windows=len(raw_rows))

    ids = {kind: set(items) for kind, items in catalog.items()}
    for kind in KINDS:
        if not ids[kind]:
            problem(root, "missing_expected_kind", f"No {kind}/*.json files found")
    anno, labels, emr = (ids[kind] for kind in KINDS)
    duplicate_rows = [dict(sha256=digest, distinct_record_ids=len(records), record_ids="|".join(sorted(records)))
                      for digest, records in duplicate_signals.items() if len(records) > 1]
    field_rows = [dict(kind=kind, field=field, parsed_file_denominator=parsed[kind],
                      absent=parsed[kind] - stat["present"],
                      types=json.dumps({key[5:]: val for key, val in stat.items() if key.startswith("type:")}),
                      **{key: stat[key] for key in ("present", "null_or_blank", "sentinel_9999", "zero")})
                  for (kind, field), stat in sorted(schemas.items())]
    summary = dict(status="complete_with_issues" if errors else "complete", scope="all_source_files_not_training_cohort",
        inventory=dict(inventory), bytes=dict(bytes_by_kind), parsed_json_files=dict(parsed),
        unique_ids={kind: len(values) for kind, values in ids.items()},
        links=dict(annotation_ids=len(anno), annotation_with_label=len(anno & labels),
                   annotation_with_label_and_emr=len(anno & labels & emr),
                   annotation_without_label=len(anno - labels), annotation_without_emr=len(anno - emr),
                   labels_without_annotation=len(labels - anno), emr_without_annotation=len(emr - anno)),
        duplicate_id_files={kind: sum(len(paths) - 1 for paths in items.values()) for kind, items in catalog.items()},
        source_windows=len(raw_rows), shape_contract_candidates=sum(row["shape_contract_candidate"] for row in raw_rows),
        issues=len(errors), exact_source_signal_duplicate_groups=len(duplicate_rows),
        limits=["File-link counts do not imply valid JSON, label alignment, selected twin, mother ID or usable training windows.",
                "Shape candidates have not passed labels, bbox, mother linkage, missingness or feature extraction checks.",
                "All source windows/files are counted, including duplicate distributions and both fetuses.",
                "TOCO zero, flat signals and extrema are descriptive, not automatic exclusion/clipping rules.",
                "Exact source representations only; duplicated twin source documents can be intentional.",
                "Acquisition timestamps, devices, labeler agreement, units and 0/9999 meaning remain unverified."])
    write_csv(out / "fields.csv", field_rows, ["kind", "field", "parsed_file_denominator", "present", "absent", "types", "null_or_blank", "sentinel_9999", "zero"])
    write_csv(out / "source_inventory.csv", source_rows, ["path", "kind", "bytes", "mtime_ns", "sha256"])
    write_csv(out / "raw_signal_quality.csv", raw_rows, list(raw_rows[0]) if raw_rows else ["path", "record_id", "source_window"])
    site_rows = [dict(site=site, **stats) for site, stats in sorted(sites.items())]
    write_csv(out / "site_quality.csv", site_rows, list(site_rows[0]) if site_rows else ["site", "source_windows"])
    write_csv(out / "shape_variants.csv", [dict(interval=key[0], fhr_samples=key[1], toco_samples=key[2],
              fhr_type=key[3], toco_type=key[4], source_windows=count) for key, count in sorted(variant.items())],
              ["interval", "fhr_samples", "toco_samples", "fhr_type", "toco_type", "source_windows"])
    write_csv(out / "duplicate_signals.csv", duplicate_rows, ["sha256", "distinct_record_ids", "record_ids"])
    write_csv(out / "duplicate_ids.csv", [dict(kind=kind, record_id=rid, files=len(paths), paths="|".join(paths))
              for kind, items in catalog.items() for rid, paths in items.items() if len(paths) > 1], ["kind", "record_id", "files", "paths"])
    save_json(out / "summary.json", summary)
    save_json(out / "status.json", dict(status=summary["status"]))
    emit("survey_complete", source_windows=len(raw_rows), issues=len(errors))
    return summary
