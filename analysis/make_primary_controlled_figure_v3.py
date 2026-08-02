#!/usr/bin/env python3
"""Regenerate the primary controlled-ablation vector figure."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "primary_feature_model_ablation_v3.csv"
SKETCH_RESULT = ROOT / "results" / "primary_log_sketch_forecast_v4.csv"
OUT = ROOT / "figures" / "Fig3_primary_controlled_ablation_final.pdf"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
})

labels = {
    "persistence": "Persistence only",
    "snapshot": "One snapshot",
    "two_snapshots": "Two snapshots",
    "maximum": "Maximum",
    "count": "Exceedance count",
    "log_sketch": "1% log sketch",
    "maximum_count": "Maximum + count",
}
order = ["persistence", "snapshot", "two_snapshots", "maximum", "count", "log_sketch", "maximum_count"]
df = pd.read_csv(RESULT)
sketch = pd.read_csv(SKETCH_RESULT)
sketch = sketch[sketch.feature_set == "log_sketch"]
df = pd.concat([df, sketch], ignore_index=True, sort=False)
df = df[df.feature_set.isin(order) & df.model.isin(["logistic", "hist_gradient_boosting"])]

fig, ax = plt.subplots(figsize=(6.9, 4.05))
y = np.arange(len(order))
styles = [
    ("logistic", -0.10, "o", "Logistic regression"),
    ("hist_gradient_boosting", 0.10, "s", "Histogram gradient boosting"),
]
for model, offset, marker, label in styles:
    g = df[df.model == model].set_index("feature_set")
    values = [float(g.loc[item, "average_precision"]) for item in order]
    ax.plot(
        values,
        y + offset,
        linestyle="none",
        marker=marker,
        markerfacecolor="white",
        markeredgecolor="black",
        label=label,
    )
    for value, yy in zip(values, y + offset):
        ax.text(value + 0.006, yy, f"{value:.3f}", va="center", fontsize=7.2)

ax.set_yticks(y, [labels[item] for item in order])
ax.invert_yaxis()
ax.set_xlim(0.40, 0.78)
ax.set_xlabel("Temporal-test average precision")
ax.grid(axis="x", color="0.90", linewidth=0.6)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT.relative_to(ROOT)}")
