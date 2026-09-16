"""Validated, twin-safe access to the Korean-CTG signal data.

shared/src/fetalguard — 공용 신호 IO. experiment-4/supervised/data_io.py에서 이전.

Only numeric FHR/TOCO samples and structural metadata are exposed from
``annotation_person``.  The per-segment baseline/deceleration fields in those
files are dataset placeholders and are intentionally never returned here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, Path]

CLASSIFICATION_DATASET_DIR = Path("__EXPLICIT_DATA_PATH_REQUIRED__")
DEFAULT_SIGNAL_DIR = CLASSIFICATION_DATASET_DIR / "annotation_person"
DEFAULT_LABEL_DIR = CLASSIFICATION_DATASET_DIR / "labels"


class DataValidationError(ValueError):
    """Raised when a source JSON violates the expected CTG data contract."""


@dataclass(frozen=True)
class CaseSignal:
    """Twin-safe numeric signal for one labelled fetus.

    ``selected_segments`` contains only segment identifiers, never the raw
    annotation dictionaries.  ``fhr_windows`` and ``toco_windows`` preserve
    the selected 5-minute windows, while ``fhr`` and ``toco`` concatenate them
    in source order.
    """

    case_id: str
    site: str
    is_twin: bool
    selected_segments: Tuple[str, ...]
    fhr_windows: Tuple[np.ndarray, ...]
    toco_windows: Tuple[np.ndarray, ...]
    fhr: np.ndarray
    toco: np.ndarray
    dt: float
    source_segment_count: int
    selected_segment_count: int
    source_path: Path

    @property
    def id(self) -> str:
        """Alias for integrations that use a lowercase ``id`` field."""

        return self.case_id

    @property
    def ID(self) -> str:  # noqa: N802 - mirrors the dataset's public field name
        """Alias matching the dataset's uppercase ``ID`` field."""

        return self.case_id


def _normalise_case_id(case_id: str) -> str:
    value = str(case_id)
    if value.endswith(".json"):
        value = value[:-5]
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("case_id must be a filename-safe ID, not a path")
    return value


def _json_path(case_id: str, directory: Optional[PathLike], default: Path) -> Path:
    base = default if directory is None else Path(directory).expanduser()
    return base / (case_id + ".json")


def _load_json_object(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError("CTG JSON not found: {}".format(path)) from None
    except json.JSONDecodeError as exc:
        raise DataValidationError(
            "Invalid JSON in {}: {}".format(path, exc)
        ) from exc
    if not isinstance(value, dict):
        raise DataValidationError("Expected a JSON object in {}".format(path))
    return value


def _parse_twin_flag(value: Any, path: Path) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised == "true":
            return True
        if normalised == "false":
            return False
    raise DataValidationError(
        "{} has invalid twins flag {!r}; expected true/false".format(path, value)
    )


def _finite_float(value: Any, field: str, context: str) -> float:
    if isinstance(value, bool):
        raise DataValidationError(
            "{} field {} must be numeric, got boolean".format(context, field)
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "{} field {} must be numeric, got {!r}".format(context, field, value)
        ) from exc
    if not math.isfinite(result):
        raise DataValidationError(
            "{} field {} must be finite, got {!r}".format(context, field, value)
        )
    return result


def _numeric_array(value: Any, field: str, context: str) -> np.ndarray:
    if isinstance(value, str):
        tokens: Sequence[Any] = value.split(",")
        if not value or any(not token.strip() for token in tokens):
            raise DataValidationError(
                "{} field {} contains an empty sample".format(context, field)
            )
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        tokens = value
    else:
        raise DataValidationError(
            "{} field {} must be a comma-separated string or sequence".format(
                context, field
            )
        )

    if len(tokens) == 0:
        raise DataValidationError("{} field {} is empty".format(context, field))
    samples = np.empty(len(tokens), dtype=np.float64)
    for index, token in enumerate(tokens):
        samples[index] = _finite_float(
            token, "{}[{}]".format(field, index), context
        )
    return samples


def _site_from_annotation(annotation: Mapping[str, Any], path: Path) -> str:
    code = annotation.get("code")
    if not isinstance(code, str) or not code.strip():
        raise DataValidationError("{} is missing a valid code".format(path))
    site = code.split("_", 1)[0].strip()
    if not site:
        raise DataValidationError("{} code does not contain a site".format(path))
    return site


def load_case_signal(
    case_id: str, signal_dir: Optional[PathLike] = None
) -> CaseSignal:
    """Load one case and select only the labelled fetus in twin recordings.

    Every source segment is validated, including the non-selected twin trace.
    All segments must have numeric, equal-length FHR/TOCO arrays and share one
    positive ``interval_sec`` value.
    """

    normalised_id = _normalise_case_id(case_id)
    path = _json_path(normalised_id, signal_dir, DEFAULT_SIGNAL_DIR)
    annotation = _load_json_object(path)
    is_twin = _parse_twin_flag(annotation.get("twins"), path)
    site = _site_from_annotation(annotation, path)

    source_segments = annotation.get("data")
    if not isinstance(source_segments, list) or not source_segments:
        raise DataValidationError("{} has no signal segments".format(path))

    twin_token: Optional[str] = None
    if is_twin:
        parts = normalised_id.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in {"1", "2"}:
            raise DataValidationError(
                "Twin case {} must end in _1 or _2".format(normalised_id)
            )
        twin_token = "twin" + parts[1]

    parsed = []
    interval_seconds = []
    for index, segment in enumerate(source_segments):
        context = "{} segment {}".format(path, index)
        if not isinstance(segment, dict):
            raise DataValidationError("{} must be an object".format(context))
        partial_image = segment.get("partial_image")
        if not isinstance(partial_image, str) or not partial_image:
            raise DataValidationError(
                "{} has no valid partial_image".format(context)
            )

        fhr = _numeric_array(segment.get("fhr"), "fhr", context)
        toco = _numeric_array(segment.get("toco"), "toco", context)
        if len(fhr) != len(toco):
            raise DataValidationError(
                "{} has unequal FHR/TOCO lengths: {} vs {}".format(
                    context, len(fhr), len(toco)
                )
            )
        interval = _finite_float(
            segment.get("interval_sec"), "interval_sec", context
        )
        if interval <= 0:
            raise DataValidationError(
                "{} interval_sec must be positive".format(context)
            )
        interval_seconds.append(interval)
        parsed.append((partial_image, fhr, toco))

    first_interval = interval_seconds[0]
    if any(not math.isclose(value, first_interval) for value in interval_seconds[1:]):
        raise DataValidationError(
            "{} contains inconsistent interval_sec values: {}".format(
                path, sorted(set(interval_seconds))
            )
        )

    if twin_token is None:
        selected = parsed
    else:
        selected = [
            segment for segment in parsed if twin_token in segment[0].lower()
        ]
    if not selected:
        raise DataValidationError(
            "{} contains no segments matching {}".format(path, twin_token)
        )

    names = tuple(segment[0] for segment in selected)
    fhr_windows = tuple(segment[1] for segment in selected)
    toco_windows = tuple(segment[2] for segment in selected)
    fhr = np.concatenate(fhr_windows)
    toco = np.concatenate(toco_windows)

    return CaseSignal(
        case_id=normalised_id,
        site=site,
        is_twin=is_twin,
        selected_segments=names,
        fhr_windows=fhr_windows,
        toco_windows=toco_windows,
        fhr=fhr,
        toco=toco,
        dt=first_interval,
        source_segment_count=len(source_segments),
        selected_segment_count=len(selected),
        source_path=path,
    )


def _required_int(labels: Mapping[str, Any], field: str, path: Path) -> int:
    if field not in labels:
        raise DataValidationError("{} is missing label {}".format(path, field))
    value = _finite_float(labels[field], field, str(path))
    if not value.is_integer():
        raise DataValidationError(
            "{} label {} must be integer-valued".format(path, field)
        )
    return int(value)


def load_case_labels(
    case_id: str, label_dir: Optional[PathLike] = None
) -> Dict[str, Union[str, int, float]]:
    """Load normalized supervised targets while retaining ordinal labels.

    Acceleration follows the source convention ``1=Yes, 2=No`` and is mapped
    to ``1/0``.  Deceleration and CA values are mapped to presence indicators;
    BaseLine and Baseline_Variability retain their original target meaning.
    """

    normalised_id = _normalise_case_id(case_id)
    path = _json_path(normalised_id, label_dir, DEFAULT_LABEL_DIR)
    raw = _load_json_object(path)
    payload_id = str(raw.get("ID", ""))
    if payload_id != normalised_id:
        raise DataValidationError(
            "{} ID {!r} does not match filename ID {!r}".format(
                path, payload_id, normalised_id
            )
        )

    if "BaseLine" not in raw:
        raise DataValidationError("{} is missing label BaseLine".format(path))
    baseline_value = raw["BaseLine"]
    _finite_float(baseline_value, "BaseLine", str(path))
    if isinstance(baseline_value, bool):
        raise DataValidationError("{} BaseLine cannot be boolean".format(path))

    variability = _required_int(raw, "Baseline_Variability", path)
    if variability not in {0, 1, 2, 3}:
        raise DataValidationError(
            "{} Baseline_Variability must be in 0..3".format(path)
        )

    acceleration_raw = _required_int(raw, "Acceleration", path)
    if acceleration_raw not in {1, 2}:
        raise DataValidationError(
            "{} Acceleration must use 1=Yes or 2=No".format(path)
        )

    targets: Dict[str, Union[str, int, float]] = {
        "ID": normalised_id,
        "BaseLine": baseline_value,
        "Baseline_Variability": variability,
        "Acceleration": int(acceleration_raw == 1),
        "Abnormality": raw.get("Abnormality", ""),
    }
    for field in (
        "Early_deceleration",
        "Late_deceleration",
        "Variable_deceleration",
        "Prolonged_deceleration",
    ):
        value = _required_int(raw, field, path)
        if value < 0:
            raise DataValidationError("{} {} cannot be negative".format(path, field))
        targets[field] = int(value > 0)

    ca_value = _required_int(raw, "CA", path)
    if ca_value < 0:
        raise DataValidationError("{} CA cannot be negative".format(path))
    targets["CA"] = int(ca_value > 0)
    return targets


def parse_window_abnormality(
    value: Union[str, Mapping[str, Any]],
    selected_windows: Union[int, CaseSignal],
) -> np.ndarray:
    """Parse ``Abnormality`` bits and require one label per selected window.

    ``value`` may be the raw underscore-separated string or a loaded label
    mapping. ``selected_windows`` may be a count or the corresponding case.
    """

    if isinstance(value, Mapping):
        if "Abnormality" not in value:
            raise DataValidationError("Label mapping is missing Abnormality")
        raw_value = value["Abnormality"]
    else:
        raw_value = value
    if not isinstance(raw_value, str) or not raw_value:
        raise DataValidationError("Abnormality must be a non-empty string")

    if isinstance(selected_windows, CaseSignal):
        expected = selected_windows.selected_segment_count
        context = selected_windows.case_id
    elif isinstance(selected_windows, int) and not isinstance(selected_windows, bool):
        expected = selected_windows
        context = "case"
    else:
        raise TypeError("selected_windows must be an int or CaseSignal")
    if expected <= 0:
        raise ValueError("selected_windows must be positive")

    tokens = raw_value.split("_")
    if any(token not in {"0", "1"} for token in tokens):
        raise DataValidationError(
            "{} Abnormality must contain only underscore-separated 0/1 bits".format(
                context
            )
        )
    if len(tokens) != expected:
        raise DataValidationError(
            "{} has {} Abnormality labels for {} selected windows".format(
                context, len(tokens), expected
            )
        )
    return np.asarray([int(token) for token in tokens], dtype=np.int8)


__all__ = [
    "CLASSIFICATION_DATASET_DIR",
    "DEFAULT_LABEL_DIR",
    "DEFAULT_SIGNAL_DIR",
    "CaseSignal",
    "DataValidationError",
    "load_case_labels",
    "load_case_signal",
    "parse_window_abnormality",
]
