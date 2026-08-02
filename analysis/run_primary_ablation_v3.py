#!/usr/bin/env python3
"""Primary Bitbrains temporal feature/model ablation.

Requires ``build/hourly_blocks_all_valid.pkl``, generated from the checksum-
verified official GWA-T-12 fastStorage archive. The default execution writes the
controlled held-out result table used in the article. An optional VM-cluster
bootstrap can be requested explicitly; it is not needed to reproduce any value
reported in the article and is disabled by default because it is expensive.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("satcap_analysis", HERE / "satcap_analysis.py")
sa = importlib.util.module_from_spec(spec)
sys.modules["satcap_analysis"] = sa
spec.loader.exec_module(sa)

PKL = ROOT / "build" / "hourly_blocks_all_valid.pkl"
if not PKL.exists():
    raise SystemExit(
        "Missing build/hourly_blocks_all_valid.pkl. "
        "Run analysis/run_full_pipeline_v3.py with the official Bitbrains archive first."
    )


def feature_matrix(df: pd.DataFrame, name: str) -> np.ndarray:
    event = (df.maximum.to_numpy() >= 0.90).astype(float)
    count = df.count_90.to_numpy() / df.n.to_numpy()
    maximum = df.maximum.to_numpy()
    snap = df.snapshot_1.to_numpy()
    snap2a = df.snapshot_2a.to_numpy()
    snap2b = df.snapshot_2b.to_numpy()
    if name == "persistence":
        return event[:, None]
    if name == "snapshot":
        return np.c_[snap, snap >= 0.90]
    if name == "two_snapshots":
        return np.c_[
            snap2a,
            snap2b,
            np.maximum(snap2a, snap2b),
            ((snap2a >= 0.90) + (snap2b >= 0.90)) / 2,
        ]
    if name == "maximum":
        return maximum[:, None]
    if name == "count":
        return count[:, None]
    if name == "maximum_event":
        return np.c_[maximum, event]
    if name == "maximum_count":
        return np.c_[maximum, count]
    if name == "satcap":
        return np.c_[maximum, count, event]
    raise KeyError(name)


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p), 1e-8, 1 - 1e-8)
    x = np.c_[np.ones(len(p)), logit(p)]
    beta = np.zeros(2)
    for _ in range(30):
        mu = expit(x @ beta)
        w = np.maximum(mu * (1 - mu), 1e-8)
        z = x @ beta + (y - mu) / w
        h = x.T @ (w[:, None] * x)
        rhs = x.T @ (w * z)
        try:
            new = np.linalg.solve(h, rhs)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    return float(beta[0]), float(beta[1])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-10, 1 - 1e-10)
    ci, cs = calibration(y, p)
    return {
        "average_precision": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "mean_log_score": -log_loss(y, p, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, p),
        "calibration_intercept": ci,
        "calibration_slope": cs,
    }


def optional_cluster_bootstrap(
    y: np.ndarray,
    vm: np.ndarray,
    predictions: dict[str, np.ndarray],
    reps: int,
) -> pd.DataFrame:
    """Optional paired VM bootstrap; not used by the article's reported values."""
    vm_ids = np.unique(vm)
    vm_indices = {v: np.flatnonzero(vm == v) for v in vm_ids}
    rng = np.random.default_rng(20260801)
    rows: list[dict[str, float | int | str]] = []
    for model in ["logistic", "hist_gradient_boosting"]:
        for right in ["persistence", "two_snapshots", "maximum"]:
            left = predictions[f"{model}__satcap"]
            baseline = predictions[f"{model}__{right}"]
            point = average_precision_score(y, left) - average_precision_score(y, baseline)
            values: list[float] = []
            for _ in range(reps):
                sampled = rng.choice(vm_ids, size=len(vm_ids), replace=True)
                idx = np.concatenate([vm_indices[v] for v in sampled])
                if np.unique(y[idx]).size < 2:
                    continue
                values.append(
                    average_precision_score(y[idx], left[idx])
                    - average_precision_score(y[idx], baseline[idx])
                )
            rows.append(
                {
                    "model": model,
                    "comparison": f"satcap_minus_{right}",
                    "ap_difference": point,
                    "cluster_ci_low": np.quantile(values, 0.025),
                    "cluster_ci_high": np.quantile(values, 0.975),
                    "bootstrap_replicates": len(values),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=0,
        help="optional VM-cluster bootstrap replicates (default: 0; article values do not require it)",
    )
    args = parser.parse_args()
    if args.bootstrap_reps < 0:
        raise SystemExit("--bootstrap-reps must be nonnegative")

    blocks = pd.read_pickle(PKL)
    tr, yt, _vmt, te, ye, vme, counts = sa.temporal_dataset(blocks, 0.90)

    rows: list[dict[str, float | int | str]] = []
    predictions: dict[str, np.ndarray] = {}
    global_p = np.full(len(ye), (yt.sum() + 0.5) / (len(yt) + 1))
    panel_p = np.array([(counts[v][0] + 0.5) / (counts[v][1] + 1) for v in vme])
    for label, p in [("global_prior", global_p), ("panel_prior", panel_p)]:
        rows.append(
            {
                "model": "baseline",
                "feature_set": label,
                "train_n": len(yt),
                "test_n": len(ye),
                "test_events": int(ye.sum()),
                **metrics(ye, p),
            }
        )

    features = [
        "persistence",
        "snapshot",
        "two_snapshots",
        "maximum",
        "count",
        "maximum_event",
        "maximum_count",
        "satcap",
    ]
    for feature in features:
        xtr = feature_matrix(tr, feature)
        xte = feature_matrix(te, feature)

        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, class_weight=None, solver="lbfgs"),
        )
        lr.fit(xtr, yt)
        p_lr = lr.predict_proba(xte)[:, 1]
        predictions[f"logistic__{feature}"] = p_lr
        rows.append(
            {
                "model": "logistic",
                "feature_set": feature,
                "train_n": len(yt),
                "test_n": len(ye),
                "test_events": int(ye.sum()),
                **metrics(ye, p_lr),
            }
        )

        hgb = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=20260801,
        )
        hgb.fit(xtr, yt)
        p_hgb = hgb.predict_proba(xte)[:, 1]
        predictions[f"hist_gradient_boosting__{feature}"] = p_hgb
        rows.append(
            {
                "model": "hist_gradient_boosting",
                "feature_set": feature,
                "train_n": len(yt),
                "test_n": len(ye),
                "test_events": int(ye.sum()),
                **metrics(ye, p_hgb),
            }
        )

    out = pd.DataFrame(rows)
    output = ROOT / "results" / "primary_feature_model_ablation_v3.csv"
    out.to_csv(output, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {output.relative_to(ROOT)}")

    if args.bootstrap_reps:
        boot = optional_cluster_bootstrap(ye, vme, predictions, args.bootstrap_reps)
        boot_output = ROOT / "results" / "primary_paired_ap_bootstrap_v3.csv"
        boot.to_csv(boot_output, index=False)
        print(boot.to_string(index=False))
        print(f"Wrote optional {boot_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
