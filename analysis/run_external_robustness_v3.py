#!/usr/bin/env python3
"""External robustness, feature ablation, and moving-block uncertainty for SATCAP.

This script is self-contained because it uses the aggregate Google and Alibaba
series distributed with the supplement. It does not substitute for the primary
Bitbrains panel analysis; it strengthens cross-trace corroboration.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260801)
DATA = {
    "Google 2019": (ROOT / "data_external" / "google2019_instance_usage.csv", "avg_cpu", 1.0),
    "Alibaba 2018": (ROOT / "data_external" / "alibaba2018_machine_usage.csv", "cpu_util_percent", 100.0),
}


def make_windows(x: np.ndarray, window_samples: int, threshold: float) -> pd.DataFrame:
    n = len(x) // window_samples
    a = np.asarray(x[: n * window_samples], float).reshape(n, window_samples)
    count = (a >= threshold).sum(axis=1)
    second = min(window_samples - 1, window_samples // 2)
    return pd.DataFrame({
        "snapshot": a[:, 0],
        "snapshot_b": a[:, second],
        "mean": a.mean(axis=1),
        "p95": np.quantile(a, 0.95, axis=1, method="linear"),
        "maximum": a.max(axis=1),
        "count_fraction": count / window_samples,
        "event": (count > 0).astype(int),
    })


def build_xy(w: pd.DataFrame, feature_set: str):
    current = w.iloc[:-1].reset_index(drop=True)
    y = w.event.to_numpy(int)[1:]
    if feature_set == "prior":
        X = np.empty((len(current), 0))
    elif feature_set == "persistence":
        X = current[["event"]].to_numpy(float)
    elif feature_set == "snapshot":
        X = np.c_[current.snapshot, current.snapshot >= current.maximum.quantile(0.95)]
    elif feature_set == "two_snapshots":
        X = np.c_[current.snapshot, current.snapshot_b, np.maximum(current.snapshot, current.snapshot_b)]
    elif feature_set == "maximum":
        X = current[["maximum"]].to_numpy(float)
    elif feature_set == "count":
        X = current[["count_fraction"]].to_numpy(float)
    elif feature_set == "maximum_event":
        X = current[["maximum", "event"]].to_numpy(float)
    elif feature_set == "maximum_count":
        X = current[["maximum", "count_fraction"]].to_numpy(float)
    elif feature_set == "satcap":
        X = current[["maximum", "count_fraction", "event"]].to_numpy(float)
    else:
        raise ValueError(feature_set)
    return X, y


def calibration_fit(y: np.ndarray, p: np.ndarray):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = logit(p)
    def objective(b):
        eta = b[0] + b[1] * z
        return float(np.sum(np.logaddexp(0, eta) - y * eta))
    r = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not r.success:
        return np.nan, np.nan
    return float(r.x[0]), float(r.x[1])


def metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    intercept, slope = calibration_fit(y, p)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_log_score": float(-log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def fit_predict(X, y, cut, model_name):
    Xtr, Xte = X[:cut], X[cut:]
    ytr = y[:cut]
    if X.shape[1] == 0:
        p0 = (ytr.sum() + 0.5) / (len(ytr) + 1.0)
        return np.full(len(y) - cut, p0), np.full(cut, p0)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    if model_name == "logistic":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=7,
            min_samples_leaf=max(10, len(ytr) // 100), l2_regularization=1.0,
            random_state=20260801,
        )
    else:
        raise ValueError(model_name)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1], model.predict_proba(Xtr)[:, 1]


def moving_block_indices(n, block_length, rng):
    blocks = int(math.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks)
    idx = np.concatenate([(s + np.arange(block_length)) % n for s in starts])
    return idx[:n]


def paired_ap_bootstrap(y, pa, pb, reps=1000, block_length=12):
    vals = []
    for _ in range(reps):
        idx = moving_block_indices(len(y), block_length, RNG)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        vals.append(average_precision_score(yy, pa[idx]) - average_precision_score(yy, pb[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def recall_bootstrap(w, threshold, reps=1000, block_length=12):
    ev = w.event.to_numpy(bool)
    vals = {k: [] for k in ["snapshot", "mean", "p95", "maximum"]}
    for _ in range(reps):
        idx = moving_block_indices(len(w), min(block_length, len(w)), RNG)
        e = ev[idx]
        if e.sum() == 0:
            continue
        z = w.iloc[idx]
        vals["snapshot"].append(float((z.loc[e, "snapshot"] >= threshold).mean()))
        vals["mean"].append(float((z.loc[e, "mean"] >= threshold).mean()))
        vals["p95"].append(float((z.loc[e, "p95"] >= threshold).mean()))
        vals["maximum"].append(1.0)
    out = {}
    for k, v in vals.items():
        out[k] = (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if v else (np.nan, np.nan)
    return out


def main():
    ablation_rows, comparison_rows, sensitivity_rows, recall_rows = [], [], [], []
    feature_sets = ["prior", "persistence", "snapshot", "two_snapshots", "maximum", "count", "maximum_event", "maximum_count", "satcap"]
    for provider, (path, col, divisor) in DATA.items():
        x = pd.to_numeric(pd.read_csv(path)[col], errors="coerce").dropna().to_numpy(float) / divisor
        x = x[np.isfinite(x)]
        ntrain_fine = int(0.7 * len(x))
        main_threshold = float(np.quantile(x[:ntrain_fine], 0.95))
        w = make_windows(x, 12, main_threshold)
        # Next-window temporal split; the split is frozen at 70% of transitions.
        ntrans = len(w) - 1
        cut = max(20, int(0.7 * ntrans))
        y_test = None
        predictions = {}
        for model_name in ["logistic", "hist_gradient_boosting"]:
            for feature_set in feature_sets:
                X, y = build_xy(w, feature_set)
                p, ptr = fit_predict(X, y, cut, model_name)
                yte = y[cut:]
                y_test = yte
                predictions[(model_name, feature_set)] = p
                row = {
                    "provider": provider, "threshold_quantile": 0.95,
                    "threshold": main_threshold, "window_minutes": 60,
                    "model": model_name, "feature_set": feature_set,
                    "train_n": cut, "test_n": len(yte), "test_events": int(yte.sum()),
                    **metrics(yte, p),
                }
                ablation_rows.append(row)
        for model_name in ["logistic", "hist_gradient_boosting"]:
            full = predictions[(model_name, "satcap")]
            for comparator in ["persistence", "two_snapshots", "maximum"]:
                q = predictions[(model_name, comparator)]
                low, high = paired_ap_bootstrap(y_test, full, q)
                comparison_rows.append({
                    "provider": provider, "model": model_name,
                    "comparison": f"satcap_minus_{comparator}",
                    "ap_difference": float(average_precision_score(y_test, full) - average_precision_score(y_test, q)),
                    "moving_block_ci_low": low, "moving_block_ci_high": high,
                    "block_length_transitions": 12, "bootstrap_replicates": 1000,
                })
        ci = recall_bootstrap(w, main_threshold)
        event_windows = int(w.event.sum())
        for method in ["snapshot", "mean", "p95", "maximum"]:
            if method == "maximum": est = 1.0
            else: est = float((w.loc[w.event.astype(bool), method] >= main_threshold).mean())
            recall_rows.append({
                "provider": provider, "threshold_quantile": .95, "threshold": main_threshold,
                "window_minutes": 60, "method": method, "event_windows": event_windows,
                "recall": est, "moving_block_ci_low": ci[method][0], "moving_block_ci_high": ci[method][1],
            })
        # Sensitivity across provider-specific training quantiles and reporting windows.
        for minutes, samples in [(30, 6), (60, 12), (120, 24)]:
            for tq in [0.80, 0.90, 0.95, 0.99]:
                u = float(np.quantile(x[:ntrain_fine], tq))
                ws = make_windows(x, samples, u)
                ev = ws.event.astype(bool)
                if ev.sum() == 0 or len(ws) < 40:
                    continue
                ntr = len(ws) - 1
                c = max(20, int(.7 * ntr))
                for method in ["snapshot", "mean", "p95", "maximum"]:
                    rec = 1.0 if method == "maximum" else float((ws.loc[ev, method] >= u).mean())
                    sensitivity_rows.append({
                        "provider": provider, "threshold_quantile": tq, "threshold": u,
                        "window_minutes": minutes, "metric": "event_recall", "method": method,
                        "estimate": rec, "test_events": int(ev.sum()),
                    })
                for feature_set in ["persistence", "two_snapshots", "satcap"]:
                    X, y = build_xy(ws, feature_set)
                    if c >= len(y) - 2 or y[c:].sum() == 0:
                        continue
                    p, _ = fit_predict(X, y, c, "logistic")
                    sensitivity_rows.append({
                        "provider": provider, "threshold_quantile": tq, "threshold": u,
                        "window_minutes": minutes, "metric": "forecast_average_precision",
                        "method": feature_set, "estimate": float(average_precision_score(y[c:], p)),
                        "test_events": int(y[c:].sum()),
                    })
    pd.DataFrame(ablation_rows).to_csv(RES / "external_feature_model_ablation_v3.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(RES / "external_paired_ap_bootstrap_v3.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(RES / "external_threshold_window_sensitivity_v3.csv", index=False)
    pd.DataFrame(recall_rows).to_csv(RES / "external_recall_moving_block_ci_v3.csv", index=False)
    print(pd.DataFrame(ablation_rows).to_string(index=False))
    print(pd.DataFrame(comparison_rows).to_string(index=False))

if __name__ == "__main__":
    main()
