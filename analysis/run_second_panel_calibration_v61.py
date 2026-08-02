#!/usr/bin/env python3
"""Compare within-Rnd and frozen fastStorage-to-Rnd probability calibration.

This executed extension reads the trace-derived hourly caches created by
run_second_panel_and_transfer_v6.py.  It refits the predeclared maximum-plus-
count models, reports discrimination and calibration metrics on the identical
Rnd holdout, and writes equal-frequency reliability bins.  Disk and network
burst thresholds remain Rnd-VM-specific values estimated without labels from
each VM's pre-test segment; only the scaler and classifier coefficients are
transferred from fastStorage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

RESOURCES = ("cpu", "memory", "disk", "network")
LABELS = {"cpu": "CPU", "memory": "Memory", "disk": "Disk burst", "network": "Network burst"}


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = np.finfo(float).eps
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    def objective(theta):
        q = expit(theta[0] + theta[1] * logit)
        return -float(np.sum(y * np.log(np.clip(q, eps, 1)) + (1-y) * np.log(np.clip(1-q, eps, 1))))
    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
    return float(fit.x[0]), float(fit.x[1])


def temporal_transitions(blocks: pd.DataFrame):
    train, test = [], []
    for _, group in blocks.groupby("vm_id", sort=False):
        group = group.sort_values("window_id").reset_index(drop=True)
        if len(group) < 50:
            continue
        candidates = np.where(np.diff(group.window_id.to_numpy()) == 1)[0]
        cut = int(0.70 * len(group))
        left = candidates[candidates + 1 < cut]
        right = candidates[candidates + 1 >= cut]
        if len(left) < 20 or len(right) < 5:
            continue
        x = group.iloc[left].copy(); x["target"] = group.iloc[left + 1].event.to_numpy(np.int8); train.append(x)
        x = group.iloc[right].copy(); x["target"] = group.iloc[right + 1].event.to_numpy(np.int8); test.append(x)
    train = pd.concat(train, ignore_index=True)
    test = pd.concat(test, ignore_index=True)
    y_train = train.pop("target").to_numpy(int)
    y_test = test.pop("target").to_numpy(int)
    return train, y_train, test, y_test


def maximum_count_features(frame: pd.DataFrame) -> np.ndarray:
    maximum = frame.maximum.to_numpy(float)
    count = frame["count"].to_numpy(float) / 12.0
    event = maximum >= frame.threshold_normalized.to_numpy(float)
    return np.c_[maximum, count, event]


def fit_predict(train: pd.DataFrame, y: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    x = maximum_count_features(train)
    xt = maximum_count_features(test)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    x = (x - mean) / scale
    xt = (xt - mean) / scale
    n, dimensions = x.shape
    eps = np.finfo(float).eps
    def objective(theta):
        z = theta[0] + x @ theta[1:]
        p = expit(z)
        loss = -np.mean(y * np.log(np.clip(p, eps, 1)) + (1-y) * np.log(np.clip(1-p, eps, 1)))
        return float(loss + 0.5 * np.dot(theta[1:], theta[1:]) / n)
    fit = minimize(objective, np.zeros(dimensions + 1), method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12})
    return expit(fit.x[0] + xt @ fit.x[1:])


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="stable")
    yy = y[order]
    positives = int(yy.sum())
    if not positives:
        return float("nan")
    precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float(np.sum(precision * yy) / positives)


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average ranks for exact ties.
    values, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_error(y: np.ndarray, p: np.ndarray, bin_count: int = 10) -> float:
    edges = np.linspace(0, 1, bin_count + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1], right=True), bin_count - 1)
    return float(sum((indices == i).mean() * abs(y[indices == i].mean() - p[indices == i].mean())
                     for i in range(bin_count) if np.any(indices == i)))


def metric_values(y: np.ndarray, p: np.ndarray):
    return {
        "average_precision": average_precision(y, p),
        "brier": float(np.mean((y - p) ** 2)),
        "roc_auc": roc_auc(y, p),
        "ece_10": calibration_error(y, p),
    }


def reliability_rows(y: np.ndarray, p: np.ndarray, resource: str, analysis: str, bins: int = 10):
    # Equal-frequency bins avoid nine empty bins for rare CPU/memory events.
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, bins)
    rows = []
    for index, ids in enumerate(groups, 1):
        rows.append({
            "resource": resource,
            "analysis": analysis,
            "bin": index,
            "n": len(ids),
            "mean_predicted_probability": float(np.mean(p[ids])),
            "observed_event_rate": float(np.mean(y[ids])),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    metric_rows, bin_rows = [], []
    predictions_by_resource = {}
    for resource in RESOURCES:
        fast = pd.read_pickle(args.cache_dir / f"faststorage_{resource}_hourly_v6.pkl")
        rnd = pd.read_pickle(args.cache_dir / f"rnd_{resource}_hourly_v6.pkl")
        ftr, fytr, _, _ = temporal_transitions(fast)
        rtr, rytr, rte, ryte = temporal_transitions(rnd)
        p_ind = fit_predict(rtr, rytr, rte)
        p_xfer = fit_predict(ftr, fytr, rte)
        predictions_by_resource[resource] = (ryte, p_ind, p_xfer)
        for analysis, p in (("within_rnd", p_ind), ("frozen_coefficients", p_xfer)):
            intercept, slope = calibration_intercept_slope(ryte, p)
            values = metric_values(ryte, p)
            metric_rows.append({
                "resource": resource,
                "analysis": analysis,
                "test_n": len(ryte),
                "test_events": int(ryte.sum()),
                **values,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            })
            bin_rows.extend(reliability_rows(ryte, p, resource, analysis))

    metrics = pd.DataFrame(metric_rows)
    bins = pd.DataFrame(bin_rows)
    metrics.to_csv(RES / "second_panel_calibration_comparison_v61.csv", index=False)
    bins.to_csv(RES / "second_panel_reliability_bins_v61.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.4))
    for ax, resource in zip(axes.flat, RESOURCES):
        subset = bins[bins.resource == resource]
        for analysis, label, marker in (
            ("within_rnd", "Fit within Rnd", "o"),
            ("frozen_coefficients", "FastStorage coefficients", "s"),
        ):
            rows = subset[subset.analysis == analysis]
            ax.plot(rows.mean_predicted_probability, rows.observed_event_rate,
                    marker=marker, linewidth=1.4, markersize=4.5, label=label)
        limit = max(0.02, 1.08 * float(subset[["mean_predicted_probability", "observed_event_rate"]].max().max()))
        ax.plot([0, limit], [0, limit], color="0.35", linestyle="--", linewidth=1, label="Ideal")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_title(LABELS[resource])
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "Fig10_second_panel_calibration_v61.pdf", bbox_inches="tight")
    print("wrote second-panel calibration comparison, reliability bins, and Fig10")


if __name__ == "__main__":
    main()
