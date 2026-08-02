#!/usr/bin/env python3
"""Task-level 1% log-histogram sketch baseline on Bitbrains.

Builds a DDSketch-style logarithmic histogram representation from the same
complete 12-sample hourly windows used by SATCAP, then evaluates exact event
query fidelity and next-hour forecasting on the common temporal split.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("satcap_analysis", HERE / "satcap_analysis.py")
sa = importlib.util.module_from_spec(spec)
sys.modules["satcap_analysis"] = sa
spec.loader.exec_module(sa)

RAW = Path(os.environ.get("SATCAP_RAW_DIR", str(ROOT / "raw_bitbrains")))
PKL = ROOT / "build" / "hourly_blocks_all_valid.pkl"
REL_ERROR = 0.01
THRESHOLD = 0.90
GAMMA = (1.0 + REL_ERROR) / (1.0 - REL_ERROR)
LOG_GAMMA = math.log(GAMMA)
VALUE_SCALE = 2.0 / (1.0 + GAMMA)


def mapped_values(a: np.ndarray) -> np.ndarray:
    """Return DDSketch logarithmic-mapping representative values."""
    out = np.zeros_like(a, dtype=float)
    pos = a > 0
    keys = np.ceil(np.log(a[pos]) / LOG_GAMMA)
    out[pos] = VALUE_SCALE * np.exp(keys * LOG_GAMMA)
    return np.clip(out, 0.0, 1.0)


def build_sketch_frame() -> pd.DataFrame:
    frames = []
    paths = sorted(RAW.rglob("*.csv"))
    for i, p in enumerate(paths):
        vm = int(p.stem)
        df = pd.read_csv(p, sep=";", engine="c")
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        c = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        x = pd.to_numeric(df.iloc[:, 4], errors="coerce").to_numpy() / 100.0
        ok = np.isfinite(t) & np.isfinite(c) & np.isfinite(x) & (x >= 0) & (x <= 1.000001)
        t = t[ok].astype(np.int64)
        c = c[ok]
        x = np.clip(x[ok], 0, 1)
        order = np.argsort(t, kind="stable")
        t, x, c = t[order], x[order], c[order]
        wid, a, _ = sa.exact_windows(t, x, c, 60)
        if len(wid):
            mapped = mapped_values(a)
            frames.append(pd.DataFrame({
                "vm_id": vm,
                "window_id": wid,
                "sketch_max": mapped.max(axis=1),
                "sketch_count_90": (mapped >= THRESHOLD).sum(axis=1),
                "sketch_bins": np.array([len(np.unique(np.ceil(np.log(row[row > 0]) / LOG_GAMMA))) if np.any(row > 0) else 1 for row in a], dtype=int),
            }))
        if (i + 1) % 250 == 0:
            print(f"processed {i+1}/{len(paths)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return {
        "average_precision": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "mean_log_score": -log_loss(y, p, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, p),
    }


def main() -> None:
    if not PKL.exists():
        raise SystemExit(f"Missing {PKL}")
    sketch = build_sketch_frame()
    blocks = pd.read_pickle(PKL)
    blocks = blocks.merge(sketch, on=["vm_id", "window_id"], how="inner", validate="one_to_one")
    if len(blocks) != 774759:
        raise RuntimeError(f"expected 774759 merged windows, found {len(blocks)}")

    true_event = (blocks.maximum.to_numpy() >= THRESHOLD).astype(int)
    sketch_event = (blocks.sketch_max.to_numpy() >= THRESHOLD).astype(int)
    event_row = {
        "relative_error": REL_ERROR,
        "threshold": THRESHOLD,
        "windows": len(blocks),
        "true_event_windows": int(true_event.sum()),
        "sketch_flagged_windows": int(sketch_event.sum()),
        "event_recall": recall_score(true_event, sketch_event),
        "event_precision": precision_score(true_event, sketch_event),
        "false_positive_windows": int(np.sum((true_event == 0) & (sketch_event == 1))),
        "false_negative_windows": int(np.sum((true_event == 1) & (sketch_event == 0))),
        "median_nonempty_bins": float(np.median(blocks.sketch_bins)),
        "p95_nonempty_bins": float(np.quantile(blocks.sketch_bins, 0.95)),
    }
    pd.DataFrame([event_row]).to_csv(ROOT / "results" / "primary_log_sketch_event_fidelity_v4.csv", index=False)

    tr, yt, _vmt, te, ye, _vme, _counts = sa.temporal_dataset(blocks, THRESHOLD)
    feature_sets = {
        "log_sketch": lambda d: np.c_[
            d.sketch_max.to_numpy(),
            d.sketch_count_90.to_numpy() / d.n.to_numpy(),
        ],
        "maximum_count": lambda d: np.c_[
            d.maximum.to_numpy(),
            d.count_90.to_numpy() / d.n.to_numpy(),
        ],
    }
    rows = []
    for name, fn in feature_sets.items():
        xtr, xte = fn(tr), fn(te)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
        lr.fit(xtr, yt)
        p = lr.predict_proba(xte)[:, 1]
        rows.append({"model": "logistic", "feature_set": name, "train_n": len(yt), "test_n": len(ye), "test_events": int(ye.sum()), **metrics(ye, p)})
        hgb = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=20260801)
        hgb.fit(xtr, yt)
        p = hgb.predict_proba(xte)[:, 1]
        rows.append({"model": "hist_gradient_boosting", "feature_set": name, "train_n": len(yt), "test_n": len(ye), "test_events": int(ye.sum()), **metrics(ye, p)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "primary_log_sketch_forecast_v4.csv", index=False)
    print(pd.DataFrame([event_row]).to_string(index=False))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
