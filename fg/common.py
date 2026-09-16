from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def table(path):
    return pd.read_csv(path, dtype={"record_id": str, "mother_id": str, "site": str}, low_memory=False)


def numeric(value):
    try:
        result = float(value)
        return result if np.isfinite(result) and result != 9999 else np.nan
    except (TypeError, ValueError):
        return np.nan


def note(message):
    from datetime import datetime
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def require_two_classes(y, context):
    if set(np.asarray(y).astype(int)) != {0, 1}:
        raise ValueError(f"{context}: 두 클래스(0/1)가 모두 필요합니다.")
