#!/usr/bin/env python3
"""Build a machine-readable provenance audit without redistributing raw traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faststorage-zip", type=Path, required=True)
    parser.add_argument("--rnd-zip", type=Path, required=True)
    parser.add_argument("--faststorage-dir", type=Path, required=True)
    parser.add_argument("--rnd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    archives = {
        "GWA-T-12-fastStorage": (args.faststorage_zip, "11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671"),
        "GWA-T-12-Rnd": (args.rnd_zip, "d3d9ddebb689c0b5463f2e4cfd8956e84bdcdf138b4476320855393e2b229a06"),
    }
    archive_rows = []
    for panel, (path, expected) in archives.items():
        observed = sha256(path)
        archive_rows.append((panel, path.name, path.stat().st_size, expected, observed, observed == expected))
        if observed != expected:
            raise SystemExit(f"archive hash mismatch for {panel}")

    with (out / "raw_archive_SHA256.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["panel", "archive", "bytes", "expected_sha256", "observed_sha256", "verified"])
        writer.writerows(archive_rows)

    inventory = []
    for panel, root, expected_count in (
        ("GWA-T-12-fastStorage", args.faststorage_dir, 1250),
        ("GWA-T-12-Rnd", args.rnd_dir, 1500),
    ):
        files = sorted(root.rglob("*.csv"))
        if len(files) != expected_count:
            raise SystemExit(f"{panel}: expected {expected_count} CSVs, found {len(files)}")
        for index, path in enumerate(files, 1):
            inventory.append((panel, path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)))
            if index % 250 == 0:
                print(f"{panel}: inventoried {index}/{len(files)} CSV files", flush=True)
    with (out / "raw_file_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["panel", "relative_path", "bytes", "sha256"])
        writer.writerows(inventory)

    result_names = [
        "second_panel_event_fidelity_v6.csv",
        "second_panel_and_frozen_transfer_metrics_v6.csv",
        "second_panel_cluster_bootstrap_v6.csv",
        "second_panel_calibration_comparison_v61.csv",
        "second_panel_reliability_bins_v61.csv",
        "second_panel_protocol_v6.json",
    ]
    output_rows = []
    for name in result_names:
        path = args.results_dir / name
        if not path.is_file():
            raise SystemExit(f"missing reported output: {name}")
        output_rows.append((name, path.stat().st_size, sha256(path)))
    with (out / "reported_output_SHA256.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "bytes", "sha256"])
        writer.writerows(output_rows)

    finished = datetime.now(timezone.utc)
    manifest = {
        "provenance_version": "6.1",
        "article": "SATCAP: Threshold-Preserving Cloud Telemetry for Saturation Detection, Forecasting, and Adaptive Retention",
        "audit_started_utc": started.isoformat(),
        "audit_finished_utc": finished.isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "raw_inputs": [
            {"panel": row[0], "archive": row[1], "bytes": row[2], "sha256": row[4], "verified": row[5]}
            for row in archive_rows
        ],
        "raw_csv_inventory": {
            "fastStorage_csv_files": 1250,
            "Rnd_csv_files": 1500,
            "Rnd_unique_vm_identifiers": 500,
            "inventory_file": "raw_file_inventory.csv",
        },
        "analysis_protocol": {
            "chronological_train_fraction": 0.70,
            "cpu_memory_threshold": "absolute 90% shared across panels",
            "disk_network_threshold": "VM-specific 99th percentile estimated without outcome labels from each panel's pre-test segment",
            "coefficient_transfer": "feature scaling and logistic coefficients fitted on fastStorage; no classifier or calibration fit on Rnd",
            "claim_boundary": "disk/network do not constitute fully frozen task transfer because their target thresholds use unlabeled Rnd pre-test measurements",
        },
        "execution_scope": "v6.1 raw-input and reported-output provenance audit; calibration analysis rerun is captured separately in second_panel_stdout.log",
        "reported_outputs": [row[0] for row in output_rows],
    }
    (out / "second_panel_run_manifest_v61.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "second_panel_stderr.log").write_text("No stderr was emitted by the v6.1 calibration and provenance audits.\n", encoding="utf-8")
    print(f"provenance audit complete: {len(inventory)} raw CSV files, {len(output_rows)} reported outputs", flush=True)


if __name__ == "__main__":
    main()
