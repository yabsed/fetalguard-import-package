"""Export bundled, trusted official weights to portable inference formats.

Maintainer-only, use original torch 1.10.2 / xgboost 1.6.2 environment.
No training, network calls, dataset reads or weights downloaded.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLOv5_AUTOINSTALL"] = "false"
os.environ["YOLOV5_CONFIG_DIR"] = os.environ.get("TMPDIR", "/tmp") + "/fetalguard-yolo-export"
os.environ["MPLCONFIGDIR"] = os.environ.get("TMPDIR", "/tmp") + "/fetalguard-mpl-export"
sys.path.insert(0, str(ROOT / "third_party/yolov5"))


def main():
    import numpy as np
    import torch
    import xgboost as xgb
    torch.set_num_threads(4)
    source = ROOT / "models/official_xgboost/xgb_clf.model"
    booster = xgb.Booster()
    booster.load_model(str(source))
    target = source.with_suffix(".json")
    booster.save_model(str(target))
    restored = xgb.Booster()
    restored.load_model(str(target))
    probe = xgb.DMatrix(np.random.default_rng(42).normal(size=(32, booster.num_features())))
    np.testing.assert_array_equal(booster.predict(probe), restored.predict(probe))
    source_yolo = ROOT / "models/official_yolo/best.pt"
    checkpoint = torch.load(source_yolo, map_location="cpu")
    model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
    # Unfused, fixed-square inference preserves the official model's parameters.
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        eager = model(x)[0]
        traced = torch.jit.trace(model, x, strict=False, check_trace=False)
        probe_yolo = torch.rand(1, 3, 640, 640)
        expected = model(probe_yolo)[0]
        actual = traced(probe_yolo)[0]
        torch.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    target_yolo = ROOT / "models/official_yolo/inference.torchscript"
    traced.save(str(target_yolo))
    result = {"torch": torch.__version__, "xgboost": xgb.__version__,
              "xgboost_features": booster.num_features(), "xgboost_roundtrip_exact": True,
              "yolo_shape": list(eager.shape), "yolo_max_abs_error": float((expected - actual).abs().max()),
              "yolo_input": [1, 3, 640, 640], "yolo_fused": False,
              "artifacts": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (source, target, source_yolo, target_yolo)}}
    (ROOT / "models/export_verification.json").write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
