#!/usr/bin/env python3
"""Rebuild all distributed v6 figures and run all integrity suites."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*parts: str) -> None:
    print("+", " ".join(parts), flush=True)
    subprocess.run(parts, cwd=ROOT, check=True)


parser = argparse.ArgumentParser()
parser.add_argument("--recompute-local", action="store_true")
parser.add_argument("--calibration-cache-dir", type=Path)
args = parser.parse_args()
if args.recompute_local:
    run(sys.executable, "analysis/run_official_ddsketch_benchmark_v5.py")
    run(sys.executable, "live_testbed/run_opentelemetry_deployment_v5.py")
    run(sys.executable, "analysis/run_delay_aware_hybrid_controller_v5.py")
if args.calibration_cache_dir:
    run(
        sys.executable,
        "analysis/run_second_panel_calibration_v61.py",
        "--cache-dir",
        str(args.calibration_cache_dir.resolve()),
    )
run(sys.executable, "analysis/make_publication_figures_v3.py")
run(sys.executable, "analysis/make_primary_controlled_figure_v3.py")
run(sys.executable, "analysis/make_supplement_figures_v3.py")
run(sys.executable, "analysis/make_v5_extension_figures.py")
run(sys.executable, "analysis/make_v6_extension_figures.py")
run(sys.executable, "tests/test_satcap_v2.py")
run(sys.executable, "tests/test_satcap_v5.py")
run(sys.executable, "tests/test_satcap_v6.py")
print("SATCAP v6 results-only audit completed.")
