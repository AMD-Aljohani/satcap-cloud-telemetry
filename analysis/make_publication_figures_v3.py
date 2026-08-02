#!/usr/bin/env python3
"""Generate publication-quality SATCAP v3 vector figures from distributed CSV files."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
})

COLORS = {
    "snapshot": "#0072B2",
    "mean": "#E69F00",
    "p95": "#009E73",
    "maximum": "#000000",
    "two_snapshots": "#56B4E9",
    "maximum_count": "#D55E00",
    "persistence": "#7F7F7F",
    "satcap": "#D55E00",
    "reactive": "#009E73",
    "no_action": "#7F7F7F",
}
LABELS = {
    "snapshot": "Snapshot",
    "mean": "Mean",
    "p95": "95th percentile",
    "maximum": "Maximum",
    "two_snapshots": "Two snapshots",
    "maximum_count": "Maximum + count",
    "persistence": "Persistence only",
    "satcap": "SATCAP",
    "reactive": "Reactive",
    "no_action": "No action",
}


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


# Figure 1: full-panel descriptive event recall.
recall = pd.DataFrame({
    "method": ["snapshot", "mean", "p95", "two_snapshots", "maximum", "maximum_count"],
    "recall": [0.629, 0.412, 0.884, 0.760, 1.000, 1.000],
    "scalars": [1, 1, 1, 2, 1, 2],
})
order = ["mean", "snapshot", "two_snapshots", "p95", "maximum", "maximum_count"]
recall = recall.set_index("method").loc[order].reset_index()
fig, ax = plt.subplots(figsize=(6.8, 3.55))
y = np.arange(len(recall))
for i, row in recall.iterrows():
    ax.hlines(i, 0, row.recall, color="0.82", lw=1.1)
    ax.plot(row.recall, i, "o", color=COLORS[row.method], markeredgecolor="black", markeredgewidth=0.35)
    ax.text(min(row.recall + 0.018, 1.01), i, f"{row.recall:.3f}", va="center", fontsize=8)
ax.set_yticks(y, [f"{LABELS[m]}  ({s} scalar{'s' if s>1 else ''})" for m, s in zip(recall.method, recall.scalars)])
ax.set_xlim(0, 1.08)
ax.set_xlabel("Recall of hourly windows containing a 90% CPU event")
ax.grid(axis="x", color="0.90", linewidth=0.6)
ax.axvline(1.0, color="0.4", linestyle="--", linewidth=0.8)
fig.tight_layout()
save(fig, "Fig1_event_recall_v3.pdf")


# Figure 2: AP in temporal/unseen holdouts plus paired temporal uncertainty.
temp = pd.read_csv(RESULTS / "all_valid_policy_forecast.csv")
panel = pd.read_csv(RESULTS / "all_valid_panel_forecast.csv")
methods = ["snapshot", "maximum", "two_snapshots", "maximum_count"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 4.05), gridspec_kw={"width_ratios": [1.45, 1]})
y = np.arange(len(methods))
for offset, df, label, marker in [(-0.10, temp, "Temporal holdout", "o"), (0.10, panel, "Unseen-VM holdout", "s")]:
    vals = [float(df.loc[df.scheme == m, "average_precision"].iloc[0]) for m in methods]
    ax1.plot(vals, y + offset, linestyle="none", marker=marker, markerfacecolor="white", markeredgecolor="black", label=label)
    for x, yy in zip(vals, y + offset):
        ax1.text(x + 0.008, yy, f"{x:.3f}", va="center", fontsize=7.4)
ax1.set_yticks(y, [LABELS[m] for m in methods])
ax1.invert_yaxis()
ax1.set_xlim(0.50, 0.84)
ax1.set_xlabel("Average precision")
ax1.grid(axis="x", color="0.90", linewidth=0.6)
ax1.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

comp = pd.read_csv(RESULTS / "all_valid_comparison_summary.csv")
rows = [
    ("Maximum − snapshot", "maximum_vs_snapshot"),
    ("Maximum + count − two snapshots", "maximum_count_vs_two_snapshots"),
]
for i, (label, key) in enumerate(rows):
    r = comp.loc[comp.comparison == key].iloc[0]
    lo = float(r.bootstrap_log_score_difference_low)
    hi = float(r.bootstrap_log_score_difference_high)
    center = (lo + hi) / 2
    ax2.errorbar(center, i, xerr=[[center-lo], [hi-center]], fmt="o", color="black", capsize=3)
    ax2.text(hi + 0.00025, i, f"[{lo:.4f}, {hi:.4f}]", va="center", fontsize=7.1)
ax2.axvline(0, color="0.4", linestyle="--", linewidth=0.8)
ax2.set_yticks(np.arange(2), [r[0] for r in rows])
ax2.invert_yaxis()
ax2.set_xlabel("Paired log-score gain\n(95% VM-cluster bootstrap CI)")
ax2.grid(axis="x", color="0.90", linewidth=0.6)
fig.tight_layout(w_pad=2.0)
save(fig, "Fig2_forecast_and_uncertainty_v3.pdf")


# Figure 3: external logistic ablation with paired AP differences.
ab = pd.read_csv(RESULTS / "external_feature_model_ablation_v3.csv")
ab = ab[ab.model == "logistic"]
boot = pd.read_csv(RESULTS / "external_paired_ap_bootstrap_v3.csv")
boot = boot[boot.model == "logistic"]
external_methods = ["persistence", "snapshot", "two_snapshots", "maximum", "satcap"]
providers = ["Google 2019", "Alibaba 2018"]
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.7), sharex=True)
for ax, provider in zip(axes, providers):
    g = ab[ab.provider == provider].set_index("feature_set")
    vals = [float(g.loc[m, "average_precision"]) for m in external_methods]
    y = np.arange(len(external_methods))
    for i, (m, x) in enumerate(zip(external_methods, vals)):
        ax.hlines(i, 0, x, color="0.87", lw=1)
        ax.plot(x, i, "o", color=COLORS.get(m, "black"), markeredgecolor="black", markeredgewidth=0.35)
        ax.text(x + 0.012, i, f"{x:.3f}", va="center", fontsize=7.3)
    delta = boot[(boot.provider == provider) & (boot.comparison == "satcap_minus_persistence")].iloc[0]
    ax.set_yticks(y, [LABELS[m] for m in external_methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Average precision")
    ax.grid(axis="x", color="0.90", linewidth=0.6)
    ax.set_title(f"{provider}\nSATCAP − persistence = {delta.ap_difference:.3f} [{delta.moving_block_ci_low:.3f}, {delta.moving_block_ci_high:.3f}]", fontsize=8.5, fontweight="normal")
axes[1].tick_params(labelleft=False)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "Fig3_external_ablation_v3.pdf", bbox_inches="tight")
fig.savefig(OUT / "Fig5_external_ablation_v5.pdf", bbox_inches="tight")
plt.close(fig)


# Figure 4: external event recall with moving-block intervals.
er = pd.read_csv(RESULTS / "external_recall_moving_block_ci_v3.csv")
methods_er = ["snapshot", "mean", "p95", "maximum"]
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.35), sharex=True)
for ax, provider in zip(axes, providers):
    g = er[er.provider == provider].set_index("method").loc[methods_er].reset_index()
    y = np.arange(len(g))
    x = g.recall.to_numpy()
    lo = g.moving_block_ci_low.to_numpy()
    hi = g.moving_block_ci_high.to_numpy()
    ax.errorbar(x, y, xerr=[x-lo, hi-x], fmt="o", color="black", capsize=3, linewidth=1.0)
    for xx, yy in zip(x, y):
        ax.text(min(xx + 0.025, 1.01), yy, f"{xx:.3f}", va="center", fontsize=7.3)
    ax.set_yticks(y, [LABELS[m] for m in methods_er])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Event recall (95% moving-block CI)")
    ax.grid(axis="x", color="0.90", linewidth=0.6)
    n_events = int(g.event_windows.iloc[0])
    ax.set_title(f"{provider}  (event windows = {n_events})", fontsize=9, fontweight="normal")
axes[1].tick_params(labelleft=False)
fig.tight_layout(w_pad=1.2)
save(fig, "Fig4_external_recall_ci_v3.pdf")


# Figure 5: threshold/window sensitivity, reported as SATCAP AP gain over persistence.
sens = pd.read_csv(RESULTS / "external_threshold_window_sensitivity_v3.csv")
sens = sens[sens.metric == "forecast_average_precision"]
pivot = sens.pivot_table(index=["provider", "threshold_quantile", "window_minutes"], columns="method", values="estimate").reset_index()
pivot["gain"] = pivot["satcap"] - pivot["persistence"]
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.55), sharey=True)
markers = {30: "o", 60: "s", 120: "^"}
for ax, provider in zip(axes, providers):
    g = pivot[pivot.provider == provider]
    for window in [30, 60, 120]:
        h = g[g.window_minutes == window].sort_values("threshold_quantile")
        ax.plot(h.threshold_quantile * 100, h.gain, marker=markers[window], label=f"{window}-min window")
    ax.axhline(0, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training-defined threshold percentile")
    ax.grid(color="0.90", linewidth=0.6)
    ax.set_title(provider, fontsize=9, fontweight="normal")
axes[0].set_ylabel("AP gain: SATCAP − persistence")
axes[1].legend(frameon=False, loc="best")
fig.tight_layout(w_pad=1.4)
save(fig, "FigS1_threshold_window_sensitivity_v3.pdf")


# Figure 6: 100-seed operational robustness under provisioning delay.
op = pd.read_csv(RESULTS / "operational_robustness_v3_bootstrap_summary.csv")
policies = ["no_action", "reactive", "snapshot", "satcap"]
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.65))
for metric, ax, ylabel in [
    ("p95_latency_ms", axes[0], "p95 latency (ms)"),
    ("sla_violation_fraction", axes[1], "SLA-violation fraction"),
]:
    g = op[op.metric == metric]
    for policy in policies:
        h = g[g.policy == policy].sort_values("provisioning_delay_epochs")
        ax.plot(h.provisioning_delay_epochs, h["mean"], marker="o", label=LABELS[policy], color=COLORS[policy])
        ax.fill_between(h.provisioning_delay_epochs.to_numpy(), h.ci_low.to_numpy(), h.ci_high.to_numpy(), alpha=0.10, color=COLORS[policy], linewidth=0)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Provisioning delay (epochs)")
    ax.set_ylabel(ylabel)
    ax.grid(color="0.90", linewidth=0.6)
axes[1].legend(frameon=False, loc="best")
fig.tight_layout(w_pad=1.6)
save(fig, "Fig5_operational_robustness_v3.pdf")


# Figure 7: actual payload/state and reference throughput.
payload = pd.read_csv(RESULTS / "telemetry_overhead_payload_v3.csv")
timing = pd.read_csv(RESULTS / "telemetry_overhead_timing_v3.csv")
shown = ["snapshot", "mean", "exact_p95", "maximum", "maximum_plus_count", "log_sketch", "raw_12_samples"]
plot_labels = {
    "snapshot": "Snapshot",
    "mean": "Mean",
    "exact_p95": "Exact p95",
    "maximum": "Maximum",
    "maximum_plus_count": "Maximum + count",
    "log_sketch": "Log sketch",
    "raw_12_samples": "Raw 12 samples",
}
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.75))
g = payload.set_index("method").loc[shown].reset_index()
y = np.arange(len(g))
axes[0].barh(y, g.median_binary_payload_bytes, edgecolor="black", linewidth=0.45, color="0.82")
axes[0].set_yticks(y, [plot_labels[m] for m in g.method])
axes[0].invert_yaxis()
axes[0].set_xlabel("Median binary payload (bytes/window)")
axes[0].grid(axis="x", color="0.90", linewidth=0.6)
for yy, value in zip(y, g.median_binary_payload_bytes):
    axes[0].text(value + 1.5, yy, f"{value:.0f}", va="center", fontsize=7.3)

u = timing[timing.operation == "stream_update_and_finalize"].set_index("method").loc[shown[:-1]].reset_index()
y2 = np.arange(len(u))
axes[1].barh(y2, u.million_observations_per_second, edgecolor="black", linewidth=0.45, color="0.82")
axes[1].set_yticks(y2, [plot_labels[m] for m in u.method])
axes[1].invert_yaxis()
axes[1].set_xlabel("Throughput (million obs./s)")
axes[1].grid(axis="x", color="0.90", linewidth=0.6)
for yy, value in zip(y2, u.million_observations_per_second):
    axes[1].text(value + 0.08, yy, f"{value:.2f}", va="center", fontsize=7.3)
fig.tight_layout(w_pad=1.5)
save(fig, "Fig6_telemetry_overhead_v3.pdf")

print(f"Wrote figures to {OUT}")
