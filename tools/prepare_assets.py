"""Maintainer only: copy local source/model assets; never copy patient data.

Run in the source repository before import review, NOT in the safe zone.
The two export jobs run with the existing torch 1.10 / XGBoost 1.6 environment.
"""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    dest = Path(__file__).resolve().parents[1]
    sources = []

    def copy(source, target):
        source, target = repo / source, dest / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources.append({"source": str(source.relative_to(repo)),
                        "destination": str(target.relative_to(dest)),
                        "original_sha256": digest(source)})
        return target

    io = copy("shared/src/fetalguard/io.py", "fg/signal_io.py")
    code = io.read_text()
    start, stop = code.index("REPO_ROOT ="), code.index("\n\nclass DataValidationError")
    code = code[:start] + ('CLASSIFICATION_DATASET_DIR = Path("__EXPLICIT_DATA_PATH_REQUIRED__")\n'
        'DEFAULT_SIGNAL_DIR = CLASSIFICATION_DATASET_DIR / "annotation_person"\n'
        'DEFAULT_LABEL_DIR = CLASSIFICATION_DATASET_DIR / "labels"\n') + code[stop:]
    io.write_text(code)
    copy("project/experiment-8/3-lightweight-cnn/ctgnet_mini.py", "fg/ctgnet.py")
    rules = copy("project/experiment-15/2-google-reproduction/features.py", "fg/feature_rules.py")
    code = rules.read_text()
    # Keep feature definitions only; dataset readers and training are deliberately omitted.
    code = code[:code.index("\ndef extract_last30_features")]
    code = code.replace("from preprocess import (\n    FS, crop_last_minutes, impute_short_gaps, smooth_masked,\n)",
                        "from .preprocessing import impute_short_gaps, smooth_masked\nFS = 4")
    code = code.replace("int(ACC_MIN_S * fs)", "int(np.ceil(ACC_MIN_S * fs))")
    code = code.replace("int(DEC_MIN_S * fs)", "int(np.ceil(DEC_MIN_S * fs))")
    rules.write_text(code)
    source = repo / "project/experiment-15/2-google-reproduction/preprocess.py"
    text = source.read_text()
    names = {"gap_runs", "impute_short_gaps", "smooth_masked"}
    selected = [ast.get_source_segment(text, node) for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    (dest / "fg/preprocessing.py").write_text('"""Extracted unchanged from experiment-15/preprocess.py."""\nimport numpy as np\n\n' + '\n\n'.join(selected) + '\n')
    sources.append({"source": str(source.relative_to(repo)), "destination": "fg/preprocessing.py",
                    "original_sha256": digest(source)})
    src = Path("dataset/korean-ctg/docs/1.모델소스코드")
    cls = src / "태아상태진단분류모델_소스코드"
    det = src / "태아이상시점탐지모델_소스코드"
    copy(cls / "model/xgb_clf.model", "models/official_xgboost/xgb_clf.model")
    copy(det / "model/weights/best.pt", "models/official_yolo/best.pt")
    for path in sorted((repo / cls).glob("*.py")):
        copy(path.relative_to(repo), Path("third_party/classifier") / path.name)
    for path in sorted((repo / det / "model/yolov5").rglob("*.py")):
        if "__pycache__" not in path.parts:
            copy(path.relative_to(repo), Path("third_party/yolov5") / path.relative_to(repo / det / "model/yolov5"))
    copy("dataset/korean-ctg/docs/3.모델정보 및 라이선스 가이드/LICENSE", "third_party/LICENSE")
    copy("dataset/korean-ctg/docs/3.모델정보 및 라이선스 가이드/README.md", "third_party/AIHUB_README.md")
    copy("본선/01-패키지-사전신청-회신/requirements-daejeon-py310.txt", "requirements-approved.txt")
    (dest / "SOURCE_MANIFEST.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n')
    print(f"Copied {len(sources)} source/model files; no dataset included.")


if __name__ == "__main__":
    main()
