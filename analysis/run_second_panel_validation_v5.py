#!/usr/bin/env python3
"""Deprecated v5 protocol-only validator.

Intended input: the official GWA-T-12 Rnd archive (500 VM CSV files) or another
panel with the same semicolon-delimited schema. The official archive actually
contains 1,500 monthly CSV files for 500 VM identifiers. Use
``run_second_panel_and_transfer_v6.py``; this file is retained for provenance.
"""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mr',ROOT/'analysis/run_multiresource_bitbrains_v5.py');mr=importlib.util.module_from_spec(spec);sys.modules['mr']=mr;spec.loader.exec_module(mr)

def main():
 raise SystemExit('Deprecated: use analysis/run_second_panel_and_transfer_v6.py, which merges the three Rnd monthly partitions by VM and implements frozen transfer.')
 ap=argparse.ArgumentParser();ap.add_argument('--trace-dir',required=True);ap.add_argument('--panel-name',default='GWA-T-12-Rnd');ap.add_argument('--expected-vms',type=int,default=500);args=ap.parse_args()
 paths=sorted(Path(args.trace_dir).rglob('*.csv'))
 if len(paths)!=args.expected_vms:raise SystemExit(f'Expected {args.expected_vms} VM CSVs, found {len(paths)}')
 buckets={r:[] for r in mr.RESOURCES}
 for i,p in enumerate(paths,1):
  result=mr.read_one(str(p))
  if result:
   for frame in result:buckets[str(frame.resource.iloc[0])].append(frame)
  if i%100==0:print(i,flush=True)
 rows=[]
 for resource in mr.RESOURCES:
  blocks=pd.concat(buckets[resource],ignore_index=True);f,b=mr.evaluate_resource(blocks)
  for _,r in f.iterrows():rows.append({'panel':args.panel_name,'resource':resource,**r.to_dict()})
 out=pd.DataFrame(rows);dest=ROOT/'results'/f"second_panel_{args.panel_name.lower().replace('-','_')}_v5.csv";out.to_csv(dest,index=False);print(dest)
if __name__=='__main__':main()
