#!/usr/bin/env python3
"""Download and verify the official GWA-T-12 fastStorage archive."""
from __future__ import annotations
import argparse,hashlib,urllib.request
from pathlib import Path
URL='https://atlarge-research.com/gwa-traces/gwa_t_12_fastStorage.zip'
EXPECTED='11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671'
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('output',nargs='?',default='gwa_t_12_fastStorage.zip'); args=ap.parse_args()
    out=Path(args.output)
    print(f'Downloading {URL} -> {out}')
    urllib.request.urlretrieve(URL,out)
    digest=sha256(out)
    if digest != EXPECTED:
        out.unlink(missing_ok=True)
        raise SystemExit(f'Checksum mismatch: {digest}')
    print(f'Verified SHA-256: {digest}')
if __name__=='__main__': main()
