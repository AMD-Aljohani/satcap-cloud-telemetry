#!/usr/bin/env python3
"""Independent GWA-T-12 Rnd validation and fastStorage coefficient transfer.

The official Rnd archive contains 500 VMs in three monthly directories (1,500
CSV files).  Files are therefore grouped by VM filename and concatenated before
window construction.  This corrects the v5 protocol-only validator, which
incorrectly expected 500 CSV files.

For the independent analysis, each Rnd VM uses a chronological 70/30 split.
For the transfer analysis, feature scaling and logistic-regression coefficients
are fitted on fastStorage and applied to the held-out 30% of Rnd without
classifier or calibration refitting. CPU and memory use fixed 90% thresholds
shared across panels. Disk and network use a VM-specific 99th percentile
estimated without outcome labels on each Rnd VM's first 70% and frozen before
Rnd testing. Those resources therefore test coefficient transfer after
unsupervised target-panel threshold estimation, not fully frozen task transfer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
BUILD = ROOT / "build"
for directory in (RES, FIG, BUILD):
    directory.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "mr", ROOT / "analysis" / "run_multiresource_bitbrains_v5.py"
)
mr = importlib.util.module_from_spec(spec)
sys.modules["mr"] = mr
spec.loader.exec_module(mr)

FEATURE_SETS = (
    "persistence", "snapshot", "two_snapshots", "maximum", "count", "maximum_count"
)
SEED = 20260801


def _read_numeric(path: Path) -> np.ndarray | None:
    try:
        frame = pd.read_csv(path, sep=";", engine="c")
    except Exception:
        return None
    if frame.shape[1] < 11:
        return None
    return frame.apply(pd.to_numeric, errors="coerce").to_numpy(float)


def read_vm_group(task: tuple[int, tuple[str, ...]]):
    """Read all monthly files for one VM and construct v5 resource windows."""
    vm, names = task
    arrays = [a for a in (_read_numeric(Path(name)) for name in names) if a is not None]
    if not arrays:
        return None
    a = np.concatenate(arrays, axis=0)
    t_all = a[:, 0]
    cores = a[:, 1]
    raw = {
        "cpu": a[:, 4] / 100.0,
        "memory": np.divide(
            a[:, 6], a[:, 5], out=np.full(len(a), np.nan), where=a[:, 5] > 0
        ),
        "disk": np.maximum(a[:, 7], 0) + np.maximum(a[:, 8], 0),
        "network": np.maximum(a[:, 9], 0) + np.maximum(a[:, 10], 0),
    }
    output = []
    for resource in mr.RESOURCES:
        xraw = raw[resource]
        ok = np.isfinite(t_all) & np.isfinite(xraw)
        if resource == "cpu":
            ok &= np.isfinite(cores) & (xraw >= 0) & (xraw <= 1.000001)
        elif resource == "memory":
            ok &= (xraw >= 0) & (xraw <= 1.000001)
        else:
            ok &= xraw >= 0
        t = t_all[ok].astype(np.int64)
        xraw = xraw[ok]
        if len(t) < 12:
            continue
        order = np.argsort(t, kind="stable")
        t, xraw = t[order], xraw[order]
        # Monthly boundary records can repeat in Rnd. Preserve the original v5
        # single-file fastStorage parser exactly, and deduplicate only when a
        # VM is assembled from multiple monthly files.
        if len(names) > 1:
            keep = np.r_[t[1:] != t[:-1], True]
            t, xraw = t[keep], xraw[keep]
        cut = max(1, int(0.70 * len(t)))
        if resource in ("cpu", "memory"):
            threshold_raw = 0.90
            x = np.clip(xraw, 0, 1)
            threshold_normalized = 0.90
        else:
            positive = xraw[:cut][xraw[:cut] > 0]
            threshold_raw = float(np.quantile(positive, 0.99)) if len(positive) else 1.0
            x = np.clip(xraw / max(threshold_raw, 1e-12), 0, 100)
            threshold_normalized = 1.0
        windows, window_ids = mr.exact_hourly(t, x)
        # Retain every nonempty frame for descriptive event-fidelity totals.
        # The temporal transition routine independently applies the predeclared
        # >=50-window eligibility rule for forecasting.
        if len(windows) == 0:
            continue
        count = (windows >= threshold_normalized).sum(axis=1)
        output.append(pd.DataFrame({
            "vm_id": vm,
            "window_id": window_ids,
            "resource": resource,
            "snapshot_1": windows[:, 0],
            "snapshot_2a": windows[:, 0],
            "snapshot_2b": windows[:, 6],
            "mean": windows.mean(axis=1),
            "p95": np.quantile(windows, 0.95, axis=1),
            "maximum": windows.max(axis=1),
            "count": count,
            "event": (count > 0).astype(np.int8),
            "threshold_raw": threshold_raw,
            "threshold_normalized": threshold_normalized,
        }))
    return output


def group_vm_paths(root: Path, expected_vms: int, panel: str):
    grouped: dict[int, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*.csv")):
        try:
            vm = int(path.stem)
        except ValueError:
            continue
        grouped[vm].append(str(path))
    if len(grouped) != expected_vms:
        raise SystemExit(
            f"{panel}: expected {expected_vms} unique VM filenames, found {len(grouped)} "
            f"across {sum(map(len, grouped.values()))} CSV files under {root}"
        )
    return [(vm, tuple(paths)) for vm, paths in sorted(grouped.items())]


def build_panel(root: Path, expected_vms: int, panel: str, workers: int, force: bool):
    cache = {r: BUILD / f"{panel}_{r}_hourly_v6.pkl" for r in mr.RESOURCES}
    if not force and all(path.exists() for path in cache.values()):
        print(f"{panel}: using cached hourly blocks", flush=True)
        return cache
    tasks = group_vm_paths(root, expected_vms, panel)
    buckets = {r: [] for r in mr.RESOURCES}
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(read_vm_group, tasks, chunksize=2), 1):
            if result:
                for frame in result:
                    buckets[str(frame.resource.iloc[0])].append(frame)
            if i % 100 == 0 or i == len(tasks):
                print(f"{panel}: processed {i}/{len(tasks)} VMs", flush=True)
    for resource in mr.RESOURCES:
        if not buckets[resource]:
            raise RuntimeError(f"{panel}: no valid {resource} windows")
        frame = pd.concat(buckets[resource], ignore_index=True)
        frame.to_pickle(cache[resource])
        print(f"{panel} {resource}: {len(frame):,} windows, "
              f"{frame.vm_id.nunique()} VMs", flush=True)
    return cache


def transitions(blocks: pd.DataFrame):
    train, test = mr.temporal_transitions(blocks)
    y_train = train.pop("target").to_numpy(int)
    y_test = test.pop("target").to_numpy(int)
    return train, y_train, test, y_test


def calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1], right=True), bins - 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += mask.sum() / total * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def metrics(y: np.ndarray, p: np.ndarray):
    return {
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "ece_10": calibration_error(y, p, 10),
    }


def fit_predict(train: pd.DataFrame, y_train: np.ndarray, test: pd.DataFrame, feature: str):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED),
    )
    model.fit(mr.features(train, feature), y_train)
    return model.predict_proba(mr.features(test, feature))[:, 1], model


def cluster_bootstrap(test: pd.DataFrame, y: np.ndarray, predictions: dict[str, np.ndarray],
                      replicates: int = 2000):
    rng = np.random.default_rng(SEED)
    vm = test.vm_id.to_numpy(int)
    ids = np.unique(vm)
    by_vm = {v: np.flatnonzero(vm == v) for v in ids}
    rows = []
    for feature in FEATURE_SETS:
        if feature == "persistence":
            continue
        ap_diff, brier_gain = [], []
        for v in ids:
            idx = by_vm[v]
            yy = y[idx]
            if yy.sum() and yy.sum() < len(yy):
                ap_diff.append(
                    average_precision_score(yy, predictions[feature][idx])
                    - average_precision_score(yy, predictions["persistence"][idx])
                )
            brier_gain.append(
                np.mean((yy - predictions["persistence"][idx]) ** 2)
                - np.mean((yy - predictions[feature][idx]) ** 2)
            )
        ap_diff = np.asarray(ap_diff, float)
        brier_gain = np.asarray(brier_gain, float)
        ap_boot = rng.choice(ap_diff, size=(replicates, len(ap_diff)), replace=True).mean(axis=1)
        br_boot = rng.choice(brier_gain, size=(replicates, len(brier_gain)), replace=True).mean(axis=1)
        rows.append({
            "feature_set": feature,
            "vm_count": len(ids),
            "ap_eligible_vm_count": len(ap_diff),
            "bootstrap_replicates": replicates,
            "mean_vm_ap_difference_vs_persistence": float(ap_diff.mean()),
            "ap_difference_ci_low": float(np.quantile(ap_boot, 0.025)),
            "ap_difference_ci_high": float(np.quantile(ap_boot, 0.975)),
            "mean_vm_brier_improvement_vs_persistence": float(brier_gain.mean()),
            "brier_improvement_ci_low": float(np.quantile(br_boot, 0.025)),
            "brier_improvement_ci_high": float(np.quantile(br_boot, 0.975)),
        })
    return pd.DataFrame(rows)


def event_fidelity(blocks: pd.DataFrame, panel: str, resource: str):
    threshold = blocks.threshold_normalized.to_numpy(float)
    event = blocks.event.to_numpy(bool)
    detected = {
        "snapshot": blocks.snapshot_1.to_numpy() >= threshold,
        "mean": blocks["mean"].to_numpy() >= threshold,
        "p95": blocks.p95.to_numpy() >= threshold,
        "maximum": blocks.maximum.to_numpy() >= threshold,
    }
    return pd.DataFrame([{
        "panel": panel,
        "resource": resource,
        "scheme": scheme,
        "windows": len(blocks),
        "vm_count": blocks.vm_id.nunique(),
        "event_windows": int(event.sum()),
        "event_recall": float(found[event].mean()) if event.any() else np.nan,
        "false_positive_rate": float(found[~event].mean()) if (~event).any() else np.nan,
    } for scheme, found in detected.items()])


def run_evaluation(fast_cache, rnd_cache, replicates: int):
    metric_rows, fidelity_rows, bootstrap_rows, model_rows = [], [], [], []
    for resource in mr.RESOURCES:
        fast = pd.read_pickle(fast_cache[resource])
        rnd = pd.read_pickle(rnd_cache[resource])
        ftr, fytr, _, _ = transitions(fast)
        rtr, rytr, rte, ryte = transitions(rnd)
        fidelity_rows.append(event_fidelity(fast, "GWA-T-12-fastStorage", resource))
        fidelity_rows.append(event_fidelity(rnd, "GWA-T-12-Rnd", resource))

        predictions = {"independent": {}, "frozen_transfer": {}}
        for feature in FEATURE_SETS:
            p_ind, model_ind = fit_predict(rtr, rytr, rte, feature)
            p_xfer, model_xfer = fit_predict(ftr, fytr, rte, feature)
            predictions["independent"][feature] = p_ind
            predictions["frozen_transfer"][feature] = p_xfer
            for analysis, p, model in (
                ("independent_rnd", p_ind, model_ind),
                ("frozen_faststorage_to_rnd", p_xfer, model_xfer),
            ):
                metric_rows.append({
                    "analysis": analysis,
                    "resource": resource,
                    "feature_set": feature,
                    "train_panel": "GWA-T-12-Rnd" if analysis == "independent_rnd" else "GWA-T-12-fastStorage",
                    "test_panel": "GWA-T-12-Rnd",
                    "train_n": len(rytr) if analysis == "independent_rnd" else len(fytr),
                    "test_n": len(ryte),
                    "test_events": int(ryte.sum()),
                    "prevalence": float(ryte.mean()),
                    **metrics(ryte, p),
                })
                lr = model.named_steps["logisticregression"]
                model_rows.append({
                    "analysis": analysis, "resource": resource, "feature_set": feature,
                    "intercept": float(lr.intercept_[0]),
                    "coefficients": json.dumps(lr.coef_[0].tolist()),
                })
        for analysis, pred in predictions.items():
            boot = cluster_bootstrap(rte, ryte, pred, replicates)
            boot.insert(0, "analysis", analysis)
            boot.insert(1, "resource", resource)
            bootstrap_rows.append(boot)
        print(f"evaluated {resource}: Rnd test n={len(ryte):,}, events={ryte.sum():,}", flush=True)

    metrics_out = pd.DataFrame(metric_rows)
    fidelity_out = pd.concat(fidelity_rows, ignore_index=True)
    bootstrap_out = pd.concat(bootstrap_rows, ignore_index=True)
    model_out = pd.DataFrame(model_rows)
    metrics_out.to_csv(RES / "second_panel_and_frozen_transfer_metrics_v6.csv", index=False)
    fidelity_out.to_csv(RES / "second_panel_event_fidelity_v6.csv", index=False)
    bootstrap_out.to_csv(RES / "second_panel_cluster_bootstrap_v6.csv", index=False)
    model_out.to_csv(RES / "second_panel_model_coefficients_v6.csv", index=False)
    return metrics_out, fidelity_out, bootstrap_out


def make_figure(metrics_out: pd.DataFrame, fidelity_out: pd.DataFrame):
    resources = list(mr.RESOURCES)
    labels = {"cpu": "CPU", "memory": "Memory", "disk": "Disk burst", "network": "Network burst"}
    selected = ("persistence", "two_snapshots", "maximum_count")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    x = np.arange(len(resources))
    width = 0.25
    for j, feature in enumerate(selected):
        values = []
        for resource in resources:
            row = metrics_out[
                (metrics_out.analysis == "frozen_faststorage_to_rnd")
                & (metrics_out.resource == resource)
                & (metrics_out.feature_set == feature)
            ]
            values.append(row.average_precision.iloc[0])
        axes[0].bar(x + (j - 1) * width, values, width, label=feature.replace("_", " ").title())
    axes[0].set_xticks(x, [labels[r] for r in resources], rotation=15)
    axes[0].set_ylabel("Frozen-transfer average precision")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8)

    schemes = ("snapshot", "mean", "p95", "maximum")
    width = 0.19
    for j, scheme in enumerate(schemes):
        values = []
        for resource in resources:
            row = fidelity_out[
                (fidelity_out.panel == "GWA-T-12-Rnd")
                & (fidelity_out.resource == resource)
                & (fidelity_out.scheme == scheme)
            ]
            values.append(row.event_recall.iloc[0])
        axes[1].bar(x + (j - 1.5) * width, values, width, label=scheme.upper() if scheme == "p95" else scheme.title())
    axes[1].set_xticks(x, [labels[r] for r in resources], rotation=15)
    axes[1].set_ylabel("Rnd threshold-event recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "Fig9_second_panel_transfer_v6.pdf", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faststorage-dir", required=True, type=Path)
    parser.add_argument("--rnd-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--force-faststorage", action="store_true")
    parser.add_argument("--force-rnd", action="store_true")
    args = parser.parse_args()
    fast_cache = build_panel(
        args.faststorage_dir, 1250, "faststorage", args.workers,
        args.force_rebuild or args.force_faststorage,
    )
    rnd_cache = build_panel(
        args.rnd_dir, 500, "rnd", args.workers,
        args.force_rebuild or args.force_rnd,
    )
    metrics_out, fidelity_out, _ = run_evaluation(fast_cache, rnd_cache, args.bootstrap_replicates)
    make_figure(metrics_out, fidelity_out)
    metadata = {
        "script": "run_second_panel_and_transfer_v6.py",
        "seed": SEED,
        "faststorage_unique_vms": 1250,
        "rnd_unique_vms": 500,
        "rnd_csv_files": 1500,
        "chronological_train_fraction": 0.70,
        "cpu_memory_threshold": 0.90,
        "disk_network_threshold": "VM-specific training-only 99th percentile",
        "bootstrap_replicates": args.bootstrap_replicates,
        "model": "StandardScaler + LogisticRegression(C=1.0, lbfgs)",
        "frozen_transfer_refit_on_rnd": False,
        "task_threshold_transfer": {
            "cpu_memory": "absolute 90% threshold shared across panels",
            "disk_network": "VM-specific 99th percentile estimated without labels from each target VM's pre-test segment",
        },
        "claim_boundary": "disk/network are coefficient transfer after unsupervised target-panel threshold estimation, not fully frozen task transfer",
    }
    (RES / "second_panel_protocol_v6.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("wrote v6 second-panel, coefficient-transfer, uncertainty, model, and figure outputs")


if __name__ == "__main__":
    main()
