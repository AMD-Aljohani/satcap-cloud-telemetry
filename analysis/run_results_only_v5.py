#!/usr/bin/env python3
"""Rebuild all distributed v5 figures and run both integrity suites.

The default mode is self-contained and uses only the included CSV/JSON files.
Use --recompute-local to rerun the official DDSketch benchmark, local OTLP
experiment, and delay-aware simulation. Provider trace recomputation requires
an explicit raw archive and is intentionally kept in the full workflows.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(*parts: str) -> None:
    print('+', ' '.join(parts), flush=True)
    subprocess.run(parts, cwd=ROOT, check=True)

p = argparse.ArgumentParser()
p.add_argument('--recompute-local', action='store_true')
a = p.parse_args()
if a.recompute_local:
    run(sys.executable, 'analysis/run_official_ddsketch_benchmark_v5.py')
    run(sys.executable, 'live_testbed/run_opentelemetry_deployment_v5.py')
    run(sys.executable, 'analysis/run_delay_aware_hybrid_controller_v5.py')
run(sys.executable, 'analysis/make_publication_figures_v3.py')
run(sys.executable, 'analysis/make_primary_controlled_figure_v3.py')
run(sys.executable, 'analysis/make_supplement_figures_v3.py')
run(sys.executable, 'analysis/make_v5_extension_figures.py')
run(sys.executable, 'tests/test_satcap_v2.py')
run(sys.executable, 'tests/test_satcap_v5.py')
print('SATCAP v5 results-only audit completed.')
