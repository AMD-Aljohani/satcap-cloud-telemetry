#!/usr/bin/env python3
"""Run the complete SATCAP v3 workflow, including primary Bitbrains analyses."""
from __future__ import annotations
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def call(script,env=None):
    print(f'\n=== {script} ===',flush=True)
    subprocess.run([sys.executable,str(ROOT/'analysis'/script)],cwd=ROOT,env=env,check=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-zip',required=True); args=ap.parse_args()
    raw=Path(args.raw_zip).resolve(); env=os.environ.copy(); env['SATCAP_RAW_ZIP']=str(raw)
    call('extract_bitbrains.py',env) if False else None
    # extract_bitbrains.py takes a positional argument
    subprocess.run([sys.executable,str(ROOT/'analysis'/'extract_bitbrains.py'),str(raw)],cwd=ROOT,env=env,check=True)
    env['SATCAP_RAW_DIR']=str(ROOT/'raw_bitbrains')
    for script in ['build_all_valid_blocks_fast.py','run_all_valid_policy_fast.py','run_all_valid_panel.py','run_all_valid_uncertainty.py','run_primary_ablation_v3.py','run_primary_log_sketch_baseline_v4.py']:
        call(script,env)
    subprocess.run([sys.executable,str(ROOT/'analysis'/'run_results_only_v3.py'),'--recompute'],cwd=ROOT,env=env,check=True)
    print('\nFull SATCAP v3 workflow completed successfully.')
if __name__=='__main__': main()
