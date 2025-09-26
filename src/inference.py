

# inference.py
# Inference utilities for IMDB BiLSTM (TextVectorization baked into the .keras model)
# Usage examples:
#   python inference.py --text "Absolutely wonderful and beautifully acted."
#   python inference.py --file samples.txt
#   # from another script:
#   #   from inference import load_artifacts, predict, predict_proba
#   #   load_artifacts() ; labels, probs = predict(["great movie", "awful"])

from __future__ import annotations
import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import tensorflow as tf

# ---------- Default locations (override via env vars or function args) ----------
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/bilstm_textvec_v1.keras")
DEFAULT_THRESH_PATH = os.getenv("THRESHOLD_PATH", "models/decision_threshold.npy")

# ---------- Module-level singletons (lazy-loaded) ----------
_MODEL: tf.keras.Model | None = None
_THRESHOLD: float | None = None


def _load_threshold(threshold_path: str) -> float:
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(
            f"Threshold file not found at '{threshold_path}'. "
            "Expected a numpy file with a single float, e.g., decision_threshold.npy"
        )
    arr = np.load(threshold_path)
    thr = float(arr.squeeze())
    if not (0.0 <= thr <= 1.0):
        raise ValueError(f"Threshold must be in [0,1], got {thr}")
    return thr


def load_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    threshold_path: str = DEFAULT_THRESH_PATH,
    compile: bool = False,
) -> None:
    """
    Loads model and decision threshold into module-level cache.
    Call once at process startup (FastAPI/CLI/etc.).
    """
    global _MODEL, _THRESHOLD

    # Some models were saved with AUC in the graph; provide custom object just in case.
    custom_objects = {"roc_auc": tf.keras.metrics.AUC(name="roc_auc")}

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Expected the full pipeline .keras model (with TextVectorization)."
        )

    _MODEL = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=compile)
    _THRESHOLD = _load_threshold(threshold_path)


def is_ready() -> bool:
    return _MODEL is not None and _THRESHOLD is not None


def predict_proba(texts: List[str], batch_size: int = 256) -> np.ndarray:
    """
    Returns probabilities (P(positive)) for a list of raw texts.
    Requires load_artifacts() to have been called.
    """
    if _MODEL is None:
        raise RuntimeError("Model not loaded. Call load_artifacts() first.")
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list/tuple of strings.")
    ds = tf.data.Dataset.from_tensor_slices(list(texts)).batch(batch_size)
    probs = _MODEL.predict(ds, verbose=0).ravel().astype(float)
    return probs


def predict(
    texts: List[str],
    threshold: float | None = None,
    batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (labels, probs) with labels computed via threshold.
    If threshold is None, uses the saved decision threshold.
    """
    if _THRESHOLD is None and threshold is None:
        raise RuntimeError("Threshold not loaded. Call load_artifacts() or pass threshold explicitly.")
    thr = float(_THRESHOLD if threshold is None else threshold)

    probs = predict_proba(texts, batch_size=batch_size)
    labels = (probs >= thr).astype(int)
    return labels, probs


# ----------------------------- CLI Interface -----------------------------------
def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # drop empties
    return [ln for ln in lines if ln]


def main():
    parser = argparse.ArgumentParser(description="IMDB BiLSTM sentiment inference")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .keras model file")
    parser.add_argument("--threshold", default=DEFAULT_THRESH_PATH, help="Path to decision_threshold.npy")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single text to classify")
    group.add_argument("--file", type=str, help="Path to a text file; each line is a sample")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    load_artifacts(model_path=args.model, threshold_path=args.threshold, compile=False)

    texts: List[str]
    if args.text is not None:
        texts = [args.text]
    else:
        texts = _read_lines(args.file)

    labels, probs = predict(texts, batch_size=args.batch_size)

    if args.json:
        out = [{"text": t, "label": int(l), "prob": float(p)} for t, l, p in zip(texts, labels, probs)]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for t, l, p in zip(texts, labels, probs):
            cls = "positive" if int(l) == 1 else "negative"
            print(f"[{cls:8s}] p={p:.4f}  |  {t}")


if __name__ == "__main__":
    # Make TF less chatty for CLI runs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
