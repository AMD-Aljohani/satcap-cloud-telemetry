#!/usr/bin/env python3
"""Compatibility entry point for external and controller analyses.

The former v2 script required an excluded Bitbrains pickle. The v3 workflow is
self-contained: it recomputes external ablations and the 100-seed operational
extension from files distributed in this archive. Historical trace-controller
CSV files remain in results/ for audit.
"""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
for name in ['run_external_robustness_v3.py','run_operational_robustness_v3.py']:
    subprocess.run([sys.executable,str(root/'analysis'/name)],cwd=root,check=True)
print('External and operational v3 analyses completed.')
