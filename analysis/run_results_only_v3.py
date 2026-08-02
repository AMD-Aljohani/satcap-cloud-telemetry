#!/usr/bin/env python3
"""Audit distributed SATCAP v4 results, or recompute all results not requiring Bitbrains.

The default audit is intentionally fast: it rebuilds every article figure from the
included CSV/JSON files and runs the integrity tests. Use ``--recompute`` to rerun
the bootstrap, 100-seed operational, and microbenchmark analyses before the audit.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAVY = [
    'run_primary_persistence_diagnostic_v3.py',
    'run_external_robustness_v3.py',
    'run_operational_robustness_v3.py',
    'run_telemetry_overhead_v3.py',
]
AUDIT = [
    'make_publication_figures_v3.py',
    'make_primary_controlled_figure_v3.py',
    'make_supplement_figures_v3.py',
]

def run(path: Path) -> None:
    print(f"\n=== {path.relative_to(ROOT)} ===", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true',
                    help='rerun all analyses that do not require the raw Bitbrains archive')
    args = ap.parse_args()
    if args.recompute:
        for name in HEAVY:
            run(ROOT / 'analysis' / name)
    for name in AUDIT:
        run(ROOT / 'analysis' / name)
    run(ROOT / 'tests' / 'test_satcap_v2.py')
    mode = 'recomputation and audit' if args.recompute else 'results audit'
    print(f"\nSATCAP v4 {mode} completed successfully.")

if __name__ == '__main__':
    main()
