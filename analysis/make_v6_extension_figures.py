#!/usr/bin/env python3
"""Rebuild v6 fastStorage/Rnd figures from distributed result tables."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
F = ROOT / "figures"
F.mkdir(exist_ok=True)
fidelity = pd.read_csv(R / "second_panel_event_fidelity_v6.csv")
metrics = pd.read_csv(R / "second_panel_and_frozen_transfer_metrics_v6.csv")
forecast = pd.read_csv(R / "multiresource_forecast_ablation_v5.csv")
resources = ["cpu", "memory", "disk", "network"]
labels = {"cpu": "CPU", "memory": "Memory", "disk": "Disk burst", "network": "Network burst"}

# Corrected fastStorage descriptive universe plus unchanged eligible forecasting set.
fast = fidelity[fidelity.panel == "GWA-T-12-fastStorage"]
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
x = np.arange(len(resources)); width = .19
for j, (scheme, hatch) in enumerate(zip(["snapshot", "mean", "p95", "maximum"], ["///", "...", "xx", "\\\\"])):
    values = [fast[(fast.resource == r) & (fast.scheme == scheme)].event_recall.iloc[0] for r in resources]
    axes[0].bar(x + (j - 1.5) * width, values, width, label=scheme.replace("p95", "P95").title(), hatch=hatch)
axes[0].set_xticks(x, [labels[r] for r in resources], rotation=15)
axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Threshold-event recall")
axes[0].legend(fontsize=8, ncol=2); axes[0].grid(axis="y", alpha=.25)
features = ["persistence", "two_snapshots", "maximum", "count", "maximum_count"]; width = .16
for j, (feature, hatch) in enumerate(zip(features, ["///", "...", "xx", "\\\\", "++"])):
    values = [forecast[(forecast.resource == r) & (forecast.feature_set == feature)].average_precision.iloc[0] for r in resources]
    axes[1].bar(x + (j - 2) * width, values, width, label=feature.replace("_", " ").title(), hatch=hatch)
axes[1].set_xticks(x, [labels[r] for r in resources], rotation=15)
axes[1].set_ylim(0, 1); axes[1].set_ylabel("Held-out average precision")
axes[1].legend(fontsize=7, ncol=2); axes[1].grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(F / "Fig6_multiresource_validation_v6.pdf", bbox_inches="tight"); plt.close(fig)

# Rnd exactness and zero-refit transfer.
rnd = fidelity[fidelity.panel == "GWA-T-12-Rnd"]
selected = ["persistence", "two_snapshots", "maximum_count"]
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1)); width = .25
for j, (feature, hatch) in enumerate(zip(selected, ["///", "...", "++"])):
    values = [metrics[(metrics.analysis == "frozen_faststorage_to_rnd") & (metrics.resource == r) & (metrics.feature_set == feature)].average_precision.iloc[0] for r in resources]
    axes[0].bar(x + (j - 1) * width, values, width, label=feature.replace("_", " ").title(), hatch=hatch)
axes[0].set_xticks(x, [labels[r] for r in resources], rotation=15)
axes[0].set_ylabel("Coefficient-transfer average precision"); axes[0].set_ylim(0, 1)
axes[0].grid(axis="y", alpha=.25); axes[0].legend(fontsize=8)
width = .19
for j, (scheme, hatch) in enumerate(zip(["snapshot", "mean", "p95", "maximum"], ["///", "...", "xx", "\\\\"])):
    values = [rnd[(rnd.resource == r) & (rnd.scheme == scheme)].event_recall.iloc[0] for r in resources]
    axes[1].bar(x + (j - 1.5) * width, values, width, label=scheme.upper() if scheme == "p95" else scheme.title(), hatch=hatch)
axes[1].set_xticks(x, [labels[r] for r in resources], rotation=15)
axes[1].set_ylabel("Rnd threshold-event recall"); axes[1].set_ylim(0, 1.05)
axes[1].grid(axis="y", alpha=.25); axes[1].legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(F / "Fig9_second_panel_transfer_v6.pdf", bbox_inches="tight"); plt.close(fig)

# Rnd reliability diagrams from distributed calibration-bin counts.
bins = pd.read_csv(R / "second_panel_reliability_bins_v61.csv")
fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.4))
for ax, resource in zip(axes.flat, resources):
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
    ax.set_xlim(0, limit); ax.set_ylim(0, limit); ax.set_title(labels[resource])
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed event rate"); ax.grid(alpha=.22)
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(F / "Fig10_second_panel_calibration_v61.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote Fig6, Fig9, and Fig10 v6/v6.1 extension figures")
