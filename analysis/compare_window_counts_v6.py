#!/usr/bin/env python3
"""Compare reconstructed CPU-window counts with the distributed v5 audit."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
blocks = pd.read_pickle(ROOT / "build" / "faststorage_cpu_hourly_v6.pkl")
actual = blocks.groupby("vm_id").size().rename("reconstructed")
reported = pd.read_csv(ROOT / "results" / "all_valid_policy_audit.csv").set_index("vm_id")["complete_hours"]
comparison = pd.concat([reported, actual], axis=1).fillna(0).astype(int)
comparison["difference"] = comparison.reconstructed - comparison.complete_hours
comparison.to_csv(ROOT / "results" / "faststorage_cpu_window_count_audit_v6.csv")
changed = comparison[comparison.difference != 0]
print(f"reported={comparison.complete_hours.sum()} reconstructed={comparison.reconstructed.sum()}")
print(f"differing_vms={len(changed)} total_difference={changed.difference.sum()}")
print(changed.to_string())
