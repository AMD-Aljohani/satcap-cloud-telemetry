#!/usr/bin/env python3
"""Benchmark Datadog's production DDSketch implementation shipped in ddtrace.

The native ``ddtrace.internal.native.DDSketch`` is the Datadog implementation
used by the official Python tracer. Its backend-compatible relative accuracy is
0.775%. We measure actual protobuf payload bytes and native update throughput on
exact Bitbrains hourly CPU windows.
"""
from __future__ import annotations
import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ddtrace.internal.native import DDSketch

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("SATCAP_RAW_DIR", ROOT / "raw_bitbrains"))
RES = ROOT / "results"
MAX_WINDOWS = int(os.environ.get("SATCAP_DDSKETCH_WINDOWS", "100000"))


def windows_from_file(p: Path):
    d = pd.read_csv(p, sep=";", engine="c")
    a = d.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t, x = a[:, 0], a[:, 4] / 100.0
    ok = np.isfinite(t) & np.isfinite(x) & (x >= 0) & (x <= 1.000001)
    t = t[ok].astype(np.int64); x = np.clip(x[ok], 0, 1)
    o = np.argsort(t, kind="stable"); t = t[o]; x = x[o]
    wid = t // 3600
    starts = np.r_[0, 1 + np.flatnonzero(wid[1:] != wid[:-1])]
    ends = np.r_[starts[1:], len(wid)]
    starts = starts[(ends - starts) == 12]
    if not len(starts):
        return np.empty((0, 12))
    return x[starts[:, None] + np.arange(12)[None, :]]


def main():
    arrays=[]; total=0
    for p in sorted(RAW.rglob("*.csv")):
        w=windows_from_file(p)
        if len(w):
            take=min(len(w), MAX_WINDOWS-total); arrays.append(w[:take]); total+=take
        if total>=MAX_WINDOWS: break
    windows=np.vstack(arrays)
    sizes=np.empty(len(windows), dtype=int)
    t0=time.perf_counter()
    for i,row in enumerate(windows):
        s=DDSketch()
        for v in row: s.add(float(v))
        sizes[i]=len(s.to_proto())
    elapsed=time.perf_counter()-t0
    updates=len(windows)*12
    out=pd.DataFrame([{
        "implementation":"Datadog ddtrace native DDSketch",
        "package":"ddtrace",
        "package_version":importlib.metadata.version("ddtrace"),
        "relative_accuracy":0.00775,
        "windows":len(windows),
        "updates":updates,
        "elapsed_seconds":elapsed,
        "update_mobs_per_second":updates/elapsed/1e6,
        "payload_median_bytes":float(np.median(sizes)),
        "payload_p95_bytes":float(np.quantile(sizes,.95)),
        "payload_min_bytes":int(sizes.min()),
        "payload_max_bytes":int(sizes.max()),
    }])
    out.to_csv(RES/"official_ddsketch_benchmark_v5.csv",index=False)
    pd.DataFrame({"payload_bytes":sizes}).to_csv(RES/"official_ddsketch_payload_distribution_v5.csv",index=False)
    print(out.to_string(index=False))

if __name__=="__main__": main()
