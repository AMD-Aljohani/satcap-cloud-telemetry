#!/usr/bin/env python3
"""Multi-resource SATCAP validation on all 1,250 Bitbrains fastStorage VMs.

CPU and memory use absolute 90% capacity thresholds. Disk and network traces do
not expose provisioned throughput capacities; for those resources a VM-specific
99th percentile is estimated using only the first 70% of valid samples and then
frozen before hourly windows and temporal test transitions are evaluated.
"""
from __future__ import annotations

import os
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
RAW = Path(os.environ.get("SATCAP_RAW_DIR", ROOT / "raw_bitbrains"))
RES = ROOT / "results"; FIG = ROOT / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
RESOURCES = ("cpu", "memory", "disk", "network")
SEED = 20260801


def exact_hourly(t: np.ndarray, x: np.ndarray):
    wid = t // 3600
    if len(wid) == 0:
        return np.empty((0, 12)), np.array([], dtype=np.int64)
    starts = np.r_[0, 1 + np.flatnonzero(wid[1:] != wid[:-1])]
    ends = np.r_[starts[1:], len(wid)]
    counts = ends - starts
    starts = starts[counts == 12]
    if len(starts) == 0:
        return np.empty((0, 12)), np.array([], dtype=np.int64)
    idx = starts[:, None] + np.arange(12)[None, :]
    return x[idx], wid[starts]


def read_one(path_str: str):
    """Read one VM with resource-specific validity rules.

    CPU exactly matches the primary all-valid parser: finite timestamp, finite
    core count, and utilization in [0, 100%]. Other resources are evaluated on
    their own valid rows rather than being dropped because an unrelated column
    is missing. This avoids both clipping invalid CPU values and imposing a
    common complete-case filter across all resources.
    """
    p = Path(path_str)
    try:
        d = pd.read_csv(p, sep=";", engine="c")
    except Exception:
        return None
    if d.shape[1] < 11:
        return None
    a = d.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t_all = a[:, 0]
    cores = a[:, 1]
    raw = {
        "cpu": a[:, 4] / 100.0,
        "memory": np.divide(a[:, 6], a[:, 5], out=np.full(len(a), np.nan), where=a[:, 5] > 0),
        "disk": np.maximum(a[:, 7], 0) + np.maximum(a[:, 8], 0),
        "network": np.maximum(a[:, 9], 0) + np.maximum(a[:, 10], 0),
    }
    vm = int(p.stem)
    out = []
    for resource in RESOURCES:
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
        t = t[order]; xraw = xraw[order]
        cut = max(1, int(0.70 * len(t)))
        if resource in ("cpu", "memory"):
            thr = 0.90
            x = np.clip(xraw, 0, 1)
            uthr = 0.90
        else:
            positive = xraw[:cut][xraw[:cut] > 0]
            thr = float(np.quantile(positive, 0.99)) if len(positive) else 1.0
            x = np.clip(xraw / max(thr, 1e-12), 0, 100)
            uthr = 1.0
        w, wid = exact_hourly(t, x)
        # Descriptive event-fidelity totals include every nonempty VM frame.
        # temporal_transitions() applies the >=50-window forecasting rule.
        if len(w) == 0:
            continue
        cnt = (w >= uthr).sum(axis=1)
        frame = pd.DataFrame({
            "vm_id": vm, "window_id": wid, "resource": resource,
            "snapshot_1": w[:, 0], "snapshot_2a": w[:, 0], "snapshot_2b": w[:, 6],
            "mean": w.mean(axis=1), "p95": np.quantile(w, 0.95, axis=1),
            "maximum": w.max(axis=1), "count": cnt, "event": (cnt > 0).astype(np.int8),
            "threshold_raw": thr, "threshold_normalized": uthr,
        })
        out.append(frame)
    return out


def features(df: pd.DataFrame, name: str):
    u = df.threshold_normalized.to_numpy(float)
    if name == "persistence":
        return df.event.to_numpy(float)[:, None]
    if name == "snapshot":
        s = df.snapshot_1.to_numpy(float); return np.c_[s, s >= u]
    if name == "two_snapshots":
        a = df.snapshot_2a.to_numpy(float); b = df.snapshot_2b.to_numpy(float)
        return np.c_[a, b, np.maximum(a, b), ((a >= u) + (b >= u)) / 2]
    if name == "maximum":
        m = df.maximum.to_numpy(float); return np.c_[m, m >= u]
    if name == "count":
        return (df["count"].to_numpy(float) / 12)[:, None]
    if name == "maximum_count":
        m = df.maximum.to_numpy(float); c = df["count"].to_numpy(float) / 12
        return np.c_[m, c, m >= u]
    raise KeyError(name)


def temporal_transitions(blocks: pd.DataFrame):
    tr, te = [], []
    for vm, g in blocks.groupby("vm_id", sort=False):
        g = g.sort_values("window_id").reset_index(drop=True)
        if len(g) < 50:
            continue
        cand = np.where(np.diff(g.window_id.to_numpy()) == 1)[0]
        cut = int(0.70 * len(g))
        a = cand[cand + 1 < cut]; b = cand[cand + 1 >= cut]
        if len(a) < 20 or len(b) < 5:
            continue
        x = g.iloc[a].copy(); x["target"] = g.iloc[a + 1].event.to_numpy(np.int8); tr.append(x)
        x = g.iloc[b].copy(); x["target"] = g.iloc[b + 1].event.to_numpy(np.int8); te.append(x)
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)


def evaluate_resource(blocks: pd.DataFrame):
    tr, te = temporal_transitions(blocks)
    ytr = tr.pop("target").to_numpy(int); yte = te.pop("target").to_numpy(int)
    rows, preds = [], {}
    for name in ("persistence", "snapshot", "two_snapshots", "maximum", "count", "maximum_count"):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
        model.fit(features(tr, name), ytr)
        p = model.predict_proba(features(te, name))[:, 1]
        preds[name] = p
        rows.append({
            "feature_set": name, "train_n": len(ytr), "test_n": len(yte), "test_events": int(yte.sum()),
            "prevalence": float(yte.mean()), "average_precision": average_precision_score(yte, p),
            "brier": brier_score_loss(yte, p),
            "roc_auc": roc_auc_score(yte, p) if np.unique(yte).size == 2 else np.nan,
        })
    # Fast panel-cluster bootstrap for the additive Brier improvement.
    # Average precision is reported as a point contrast because pooled AP is
    # non-additive and a full weighted cluster bootstrap is computationally
    # expensive on this 0.8-million-window extension.
    vm = te.vm_id.to_numpy(int); ids = np.unique(vm); rng = np.random.default_rng(SEED)
    row_gain = (yte - preds["persistence"])**2 - (yte - preds["maximum_count"])**2
    per_vm = pd.DataFrame({"vm_id": vm, "gain": row_gain}).groupby("vm_id").gain.mean()
    boot_vals = []
    values = per_vm.to_numpy(float)
    for _ in range(2000):
        boot_vals.append(float(rng.choice(values, size=len(values), replace=True).mean()))
    boot = {
        "ap_difference_maximum_count_minus_persistence": rows[-1]["average_precision"] - rows[0]["average_precision"],
        "mean_vm_brier_improvement": float(values.mean()),
        "cluster_ci_low": float(np.quantile(boot_vals, 0.025)),
        "cluster_ci_high": float(np.quantile(boot_vals, 0.975)),
        "bootstrap_replicates": len(boot_vals),
    }
    return pd.DataFrame(rows), boot


def main():
    paths = sorted(str(p) for p in RAW.rglob("*.csv"))
    if len(paths) != 1250:
        raise SystemExit(f"Expected 1250 Bitbrains CSV files, found {len(paths)} under {RAW}")
    workers = max(1, min(12, (os.cpu_count() or 2) - 1))
    buckets = {r: [] for r in RESOURCES}
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(read_one, paths, chunksize=4), 1):
            if result:
                for frame in result:
                    buckets[str(frame.resource.iloc[0])].append(frame)
            if i % 100 == 0:
                print(f"processed {i}/{len(paths)}", flush=True)
    all_blocks = {}
    for resource in RESOURCES:
        blocks = pd.concat(buckets[resource], ignore_index=True)
        blocks.to_pickle(ROOT / "build" / f"hourly_blocks_{resource}_v5.pkl")
        all_blocks[resource] = blocks
        print(f"saved {resource}: {len(blocks):,} windows", flush=True)
    del buckets

    metric_rows, forecast_rows, boot_rows = [], [], []
    for resource in RESOURCES:
        blocks = all_blocks[resource]
        u = blocks.threshold_normalized.to_numpy(float)
        ev = blocks.event.to_numpy(bool); n_events = int(ev.sum())
        for scheme, detected in [
            ("snapshot", blocks.snapshot_1.to_numpy() >= u),
            ("mean", blocks["mean"].to_numpy() >= u),
            ("p95", blocks.p95.to_numpy() >= u),
            ("maximum", blocks.maximum.to_numpy() >= u),
        ]:
            metric_rows.append({"resource": resource, "scheme": scheme, "windows": len(blocks),
                                "event_windows": n_events, "event_recall": float(detected[ev].mean()) if n_events else np.nan})
        f, b = evaluate_resource(blocks)
        f.insert(0, "resource", resource); forecast_rows.append(f)
        boot_rows.append({"resource": resource, **b})
        print(resource, f[["feature_set", "average_precision", "brier"]].to_string(index=False), flush=True)
    detection = pd.DataFrame(metric_rows); forecast = pd.concat(forecast_rows, ignore_index=True); boot = pd.DataFrame(boot_rows)
    detection.to_csv(RES / "multiresource_event_recall_v5.csv", index=False)
    forecast.to_csv(RES / "multiresource_forecast_ablation_v5.csv", index=False)
    boot.to_csv(RES / "multiresource_cluster_bootstrap_v5.csv", index=False)

    # Compact journal figure: exactness and forecasting across resources.
    labels = {"cpu":"CPU", "memory":"Memory", "disk":"Disk burst", "network":"Network burst"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    schemes = ["snapshot", "mean", "p95", "maximum"]
    x = np.arange(len(RESOURCES)); width = 0.19
    for j, s in enumerate(schemes):
        vals = [detection[(detection.resource==r)&(detection.scheme==s)].event_recall.iloc[0] for r in RESOURCES]
        axes[0].bar(x + (j-1.5)*width, vals, width, label=s.replace("p95", "P95").title())
    axes[0].set_xticks(x, [labels[r] for r in RESOURCES], rotation=15)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Threshold-event recall")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(axis="y", alpha=.25)
    fs = ["persistence", "two_snapshots", "maximum", "count", "maximum_count"]
    width = 0.16
    for j, s in enumerate(fs):
        vals = [forecast[(forecast.resource==r)&(forecast.feature_set==s)].average_precision.iloc[0] for r in RESOURCES]
        axes[1].bar(x + (j-2)*width, vals, width, label=s.replace("_", " ").title())
    axes[1].set_xticks(x, [labels[r] for r in RESOURCES], rotation=15)
    axes[1].set_ylim(0, 1.0); axes[1].set_ylabel("Held-out average precision")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "Fig7_multiresource_validation_v5.pdf", bbox_inches="tight")
    print("wrote multi-resource results and figure")

if __name__ == "__main__":
    main()
