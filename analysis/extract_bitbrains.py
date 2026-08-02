#!/usr/bin/env python3
"""Verify and extract the Bitbrains fastStorage archive for full reproduction."""
from pathlib import Path
import argparse, hashlib, zipfile
EXPECTED='11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671'
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'raw_bitbrains');a=p.parse_args()
h=hashlib.sha256(a.archive.read_bytes()).hexdigest()
if h!=EXPECTED: raise SystemExit(f'checksum mismatch: {h}')
a.out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.archive) as z:z.extractall(a.out)
print(f'extracted to {a.out}')
