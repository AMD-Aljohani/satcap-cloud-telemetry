#!/usr/bin/env python3
"""Check v6 grouped parsing against the distributed v5 single-file parser."""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v6", ROOT / "analysis" / "run_second_panel_and_transfer_v6.py")
v6 = importlib.util.module_from_spec(spec); sys.modules["v6"] = v6; spec.loader.exec_module(v6)


def counts(frames):
    return {str(frame.resource.iloc[0]): len(frame) for frame in (frames or [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faststorage-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.faststorage_dir.rglob("*.csv"))
    totals_old = {r: 0 for r in v6.mr.RESOURCES}
    totals_new = {r: 0 for r in v6.mr.RESOURCES}
    differences = []
    for index, path in enumerate(paths, 1):
        old = counts(v6.mr.read_one(str(path)))
        new = counts(v6.read_vm_group((int(path.stem), (str(path),))))
        for resource in v6.mr.RESOURCES:
            totals_old[resource] += old.get(resource, 0)
            totals_new[resource] += new.get(resource, 0)
        if old != new:
            differences.append((path.name, old, new))
        if index % 250 == 0:
            print(f"audited {index}/{len(paths)}", flush=True)
    print("v5 totals", totals_old)
    print("v6 totals", totals_new)
    print("differing VMs", len(differences))
    for row in differences[:20]:
        print(row)
    if totals_old != totals_new or differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
