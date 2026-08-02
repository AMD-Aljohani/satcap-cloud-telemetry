#!/usr/bin/env python3
"""SATCAP v3 telemetry-state and serialization microbenchmark.

The benchmark distinguishes:
  * collector state needed while a window is open;
  * binary and compact-JSON bytes transmitted once per completed window;
  * Python reference-implementation update throughput;
  * merge throughput for mergeable summaries.

It is a reproducible microbenchmark, not a production OpenTelemetry benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodSpec:
    name: str
    state_bytes: int | None
    payload_value_bytes: int | None
    mergeable: bool
    note: str


METHODS = [
    MethodSpec("snapshot", 8, 8, False, "one designated sample; position-dependent"),
    MethodSpec("mean", 16, 8, True, "running sum and count; transmits final mean"),
    MethodSpec("exact_p95", 96, 8, True, "stores 12 float64 samples until window close"),
    MethodSpec("maximum", 8, 8, True, "running maximum"),
    MethodSpec("maximum_plus_count", 12, 12, True, "float64 maximum plus uint32 exceedance count"),
    MethodSpec("log_sketch", None, None, True, "DDSketch-style relative-error log histogram; variable bins"),
]


def update_snapshot(x: np.ndarray, threshold: float) -> tuple[float]:
    # Fixed sixth position, matching one periodic sample from a 12-sample hour.
    out = 0.0
    for i, value in enumerate(x):
        if i == 5:
            out = float(value)
    return (out,)


def update_mean(x: np.ndarray, threshold: float) -> tuple[float, int]:
    total = 0.0
    count = 0
    for value in x:
        total += float(value)
        count += 1
    return total, count


def update_exact_p95(x: np.ndarray, threshold: float) -> tuple[float, ...]:
    values: list[float] = []
    for value in x:
        values.append(float(value))
    return tuple(values)


def update_maximum(x: np.ndarray, threshold: float) -> tuple[float]:
    maximum = -math.inf
    for value in x:
        value = float(value)
        if value > maximum:
            maximum = value
    return (maximum,)


def update_maximum_count(x: np.ndarray, threshold: float) -> tuple[float, int]:
    maximum = -math.inf
    exceedances = 0
    for value in x:
        value = float(value)
        if value > maximum:
            maximum = value
        if value >= threshold:
            exceedances += 1
    return maximum, exceedances


def log_bucket(value: float, gamma: float) -> int:
    # Values are in [0,1]. A dedicated zero bucket avoids log(0).
    if value <= 0.0:
        return -10_000
    return int(math.ceil(math.log(value) / math.log(gamma)))


def update_log_sketch(x: np.ndarray, threshold: float, rel_error: float = 0.01) -> dict[int, int]:
    gamma = (1.0 + rel_error) / (1.0 - rel_error)
    bins: dict[int, int] = {}
    for value in x:
        key = log_bucket(float(value), gamma)
        bins[key] = bins.get(key, 0) + 1
    return bins


UPDATE_FUNCS: dict[str, Callable[[np.ndarray, float], object]] = {
    "snapshot": update_snapshot,
    "mean": update_mean,
    "exact_p95": update_exact_p95,
    "maximum": update_maximum,
    "maximum_plus_count": update_maximum_count,
    "log_sketch": update_log_sketch,
}


def merge_mean(a: tuple[float, int], b: tuple[float, int]) -> tuple[float, int]:
    return a[0] + b[0], a[1] + b[1]


def merge_exact(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return a + b


def merge_max(a: tuple[float], b: tuple[float]) -> tuple[float]:
    return (max(a[0], b[0]),)


def merge_max_count(a: tuple[float, int], b: tuple[float, int]) -> tuple[float, int]:
    return max(a[0], b[0]), a[1] + b[1]


def merge_log(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
    out = dict(a)
    for key, count in b.items():
        out[key] = out.get(key, 0) + count
    return out


MERGE_FUNCS: dict[str, Callable[[object, object], object]] = {
    "mean": merge_mean,
    "exact_p95": merge_exact,
    "maximum": merge_max,
    "maximum_plus_count": merge_max_count,
    "log_sketch": merge_log,
}


def compact_json_payload(method: str, state: object, vm_id: int, window_start: int) -> bytes:
    if method == "snapshot":
        payload = {"v": vm_id, "t": window_start, "x": round(state[0], 6)}
    elif method == "mean":
        payload = {"v": vm_id, "t": window_start, "x": round(state[0] / state[1], 6)}
    elif method == "exact_p95":
        payload = {"v": vm_id, "t": window_start, "x": round(float(np.quantile(state, 0.95)), 6)}
    elif method == "maximum":
        payload = {"v": vm_id, "t": window_start, "m": round(state[0], 6)}
    elif method == "maximum_plus_count":
        payload = {"v": vm_id, "t": window_start, "m": round(state[0], 6), "c": state[1]}
    elif method == "log_sketch":
        payload = {"v": vm_id, "t": window_start, "b": sorted(state.items())}
    else:
        raise KeyError(method)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def binary_payload_bytes(method: str, state: object) -> int:
    # uint32 VM identifier + int64 window timestamp = 12 metadata bytes.
    metadata = 12
    if method in {"snapshot", "mean", "exact_p95", "maximum"}:
        return metadata + 8
    if method == "maximum_plus_count":
        return metadata + 8 + 4
    if method == "log_sketch":
        # Header: metadata + uint16 number of bins. Each bin: int32 key + uint16 count.
        return metadata + 2 + 6 * len(state)
    raise KeyError(method)


def state_bytes(method: str, state: object) -> int:
    if method == "snapshot":
        return 8
    if method == "mean":
        return 16
    if method == "exact_p95":
        return 8 * len(state)
    if method == "maximum":
        return 8
    if method == "maximum_plus_count":
        return 12
    if method == "log_sketch":
        # Logical serialized state, not Python-object overhead.
        return 2 + 6 * len(state)
    raise KeyError(method)


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    # Beta mixture gives ordinary values plus a small high-utilization component.
    ordinary = rng.beta(2.0, 7.0, size=(args.windows, args.samples)).astype(np.float64)
    bursts = rng.random(size=(args.windows, args.samples)) < 0.025
    ordinary[bursts] = rng.uniform(0.90, 1.0, size=int(bursts.sum()))
    data = ordinary

    timing_rows: list[dict[str, object]] = []
    payload_rows: list[dict[str, object]] = []
    representative_states: dict[str, list[object]] = {}

    for method in [m.name for m in METHODS]:
        fn = UPDATE_FUNCS[method]
        times: list[float] = []
        states_last: list[object] = []
        checksum = 0
        for rep in range(args.repetitions):
            start = time.perf_counter()
            states: list[object] = []
            for row in data:
                state = fn(row, args.threshold)
                states.append(state)
                checksum ^= hash(str(state)) & 0xFFFF
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            states_last = states
        representative_states[method] = states_last[: min(2000, len(states_last))]
        median_seconds = statistics.median(times)
        timing_rows.append({
            "method": method,
            "operation": "stream_update_and_finalize",
            "windows": args.windows,
            "observations": args.windows * args.samples,
            "repetitions": args.repetitions,
            "median_seconds": median_seconds,
            "min_seconds": min(times),
            "max_seconds": max(times),
            "million_observations_per_second": (args.windows * args.samples / median_seconds) / 1e6,
            "checksum": checksum,
        })

        logical_state = [state_bytes(method, s) for s in representative_states[method]]
        binary_sizes = [binary_payload_bytes(method, s) for s in representative_states[method]]
        json_sizes = [len(compact_json_payload(method, s, 17, 1_700_000_000)) for s in representative_states[method]]
        payload_rows.append({
            "method": method,
            "samples_per_window": args.samples,
            "median_collector_state_bytes": float(np.median(logical_state)),
            "p95_collector_state_bytes": float(np.quantile(logical_state, 0.95)),
            "median_binary_payload_bytes": float(np.median(binary_sizes)),
            "p95_binary_payload_bytes": float(np.quantile(binary_sizes, 0.95)),
            "median_compact_json_bytes": float(np.median(json_sizes)),
            "p95_compact_json_bytes": float(np.quantile(json_sizes, 0.95)),
            "mergeable": method != "snapshot",
        })

    # Raw twelve-sample reference includes the same VM/timestamp metadata.
    raw_binary = 12 + 8 * args.samples
    raw_json_sizes = []
    for row in data[: min(2000, args.windows)]:
        payload = {"v": 17, "t": 1_700_000_000, "x": [round(float(v), 6) for v in row]}
        raw_json_sizes.append(len(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")))
    payload_rows.append({
        "method": "raw_12_samples",
        "samples_per_window": args.samples,
        "median_collector_state_bytes": float(8 * args.samples),
        "p95_collector_state_bytes": float(8 * args.samples),
        "median_binary_payload_bytes": float(raw_binary),
        "p95_binary_payload_bytes": float(raw_binary),
        "median_compact_json_bytes": float(np.median(raw_json_sizes)),
        "p95_compact_json_bytes": float(np.quantile(raw_json_sizes, 0.95)),
        "mergeable": True,
    })

    # Merge pairs of two six-sample partial windows.
    split = args.samples // 2
    merge_n = min(args.merge_pairs, args.windows)
    left_data = data[:merge_n, :split]
    right_data = data[:merge_n, split:]
    for method, merge_fn in MERGE_FUNCS.items():
        fn = UPDATE_FUNCS[method]
        left = [fn(row, args.threshold) for row in left_data]
        right = [fn(row, args.threshold) for row in right_data]
        times = []
        checksum = 0
        for _ in range(args.repetitions):
            start = time.perf_counter()
            for a, b in zip(left, right):
                merged = merge_fn(a, b)
                checksum ^= hash(str(merged)) & 0xFFFF
            times.append(time.perf_counter() - start)
        median_seconds = statistics.median(times)
        timing_rows.append({
            "method": method,
            "operation": "merge_two_partial_states",
            "windows": merge_n,
            "observations": merge_n * args.samples,
            "repetitions": args.repetitions,
            "median_seconds": median_seconds,
            "min_seconds": min(times),
            "max_seconds": max(times),
            "million_observations_per_second": (merge_n * args.samples / median_seconds) / 1e6,
            "checksum": checksum,
        })

    payload_df = pd.DataFrame(payload_rows)
    raw_binary_value = float(payload_df.loc[payload_df.method == "raw_12_samples", "median_binary_payload_bytes"].iloc[0])
    raw_json_value = float(payload_df.loc[payload_df.method == "raw_12_samples", "median_compact_json_bytes"].iloc[0])
    payload_df["binary_reduction_vs_raw"] = 1.0 - payload_df["median_binary_payload_bytes"] / raw_binary_value
    payload_df["json_reduction_vs_raw"] = 1.0 - payload_df["median_compact_json_bytes"] / raw_json_value

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(timing_rows).to_csv(out_dir / "telemetry_overhead_timing_v3.csv", index=False)
    payload_df.to_csv(out_dir / "telemetry_overhead_payload_v3.csv", index=False)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "windows": args.windows,
        "samples_per_window": args.samples,
        "threshold": args.threshold,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "interpretation": "Python reference microbenchmark; logical state and serialization bytes exclude transport framing and Python-object overhead.",
    }
    (out_dir / "telemetry_overhead_config_v3.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(payload_df.to_string(index=False))
    print(pd.DataFrame(timing_rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=100_000)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--merge-pairs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    run(parser.parse_args())
