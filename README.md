# SATCAP v6.1 reproducibility package

**Article:** *SATCAP: Threshold-Preserving Cloud Telemetry for Saturation Detection, Forecasting, and Adaptive Retention*  
**Intended journal:** *Computing*  
**Authors:** Abdulaziz M. D. Aljohani and Khaled M. Alhawiti  
**Affiliation:** University of Tabuk, Tabuk, Saudi Arabia  
**Corresponding author:** Abdulaziz M. D. Aljohani, a-aljohani@ut.edu.sa  
**Corresponding-author ORCID:** 0009-0008-6741-3105

This archive supports three levels of verification.

1. **Results-only audit:** rebuilds every distributed vector figure from included CSV/JSON files and runs both integrity suites.
2. **Local systems recomputation:** reruns the Datadog native DDSketch benchmark, the real four-process OpenTelemetry OTLP/HTTP experiment, and the 100-seed delay-aware controller study.
3. **Provider-trace workflows:** reconstruct the primary and multi-resource Bitbrains analyses when the official archives are supplied.

Only executed or directly auditable evidence is distributed. The local OpenTelemetry experiment and the 500-VM GWA-T-12 Rnd validation were executed and their outputs are included. No Kubernetes or OpenStack protocol or result is included; cluster validation is separate future work.

## Contents

- `analysis/`: primary, external, multi-resource, DDSketch, controller, executed second-panel/coefficient-transfer, calibration, and figure scripts.
- `live_testbed/`: real asynchronous queue test and real OpenTelemetry OTLP/HTTP process deployment.
- `data_external/`: licensed aggregate Google and Alibaba derivatives.
- `results/`: article-reported CSV/JSON outputs.
- `figures/`: vector figures.
- `provenance/`: provider-archive hashes, raw-file inventory, execution log, environment record, and output hashes.
- `tests/`: 29 regression and integrity checks (14 legacy, 10 v5 extension, and 5 v6 extension checks).
- `DATA_LICENSES.md`, `LICENSE`, `CITATION.cff`, `.zenodo.json`, and `SHA256SUMS.txt`.

## Results-only audit

```bash
python -m venv .venv
. .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python analysis/run_results_only_v6.py
```

The audit rebuilds all distributed figures and runs the legacy, v5, and v6 regression and integrity suites.

To rerun the local DDSketch, OTLP, and controller experiments before rebuilding figures:

```bash
python analysis/run_results_only_v5.py --recompute-local
```

The OpenTelemetry campaign uses actual operating-system processes, measured CPU and resident memory, the official Python SDK, and OTLP/HTTP protobuf requests to a local receiver. It is not described as Kubernetes, OpenStack, or commercial-cloud validation.

## Full fastStorage reconstruction

Obtain the official GWA-T-12 fastStorage ZIP and run:

```bash
python analysis/download_bitbrains.py /path/to/gwa_t_12_fastStorage.zip
python analysis/run_full_pipeline_v3.py --raw-zip /path/to/gwa_t_12_fastStorage.zip
SATCAP_RAW_DIR=/path/to/extracted/fastStorage python analysis/run_multiresource_bitbrains_v5.py
```

The expected fastStorage SHA-256 is:

```text
11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671
```

The multi-resource workflow uses absolute 90% thresholds for CPU and memory. Because provisioned disk and network capacities are absent, it freezes VM-specific 99th-percentile burst thresholds from the first 70% of valid samples. Resource-specific filters prevent missing unrelated metrics from deleting an otherwise valid observation.

## Independent machine-level panel and coefficient transfer

The official GWA-T-12 Rnd archive contains 500 VMs stored as 1,500 monthly CSV files. The workflow groups the three monthly files by VM before windowing, fits an independent chronological Rnd analysis, and applies fastStorage-fitted feature scalers and logistic-regression coefficients to Rnd without classifier or calibration refitting:

```bash
python analysis/run_second_panel_and_transfer_v6.py \
  --faststorage-dir /path/to/extracted/fastStorage \
  --rnd-dir /path/to/extracted/rnd \
  --bootstrap-replicates 2000
```

Expected archive SHA-256 values are:

```text
fastStorage  11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671
Rnd          d3d9ddebb689c0b5463f2e4cfd8956e84bdcdf138b4476320855393e2b229a06
```

CPU and memory use the same 90% task thresholds in both panels. Disk and network capacities are unavailable, so their VM-specific burst thresholds are estimated without outcome labels from each panel's first 70% and frozen before testing. Those two resources therefore test coefficient transfer after unsupervised target-panel threshold estimation, not fully frozen task transfer.

The v6.1 calibration audit can be rerun from derived caches created by the full workflow:

```bash
python analysis/run_second_panel_calibration_v61.py --cache-dir /path/to/analysis_cache_v6
```

It reports AP, Brier score, ROC AUC, ten-bin expected calibration error, calibration intercept and slope, and equal-frequency reliability-bin counts for within-Rnd fitting and fastStorage coefficient transfer.

## Citation and archival release

This repository is the reproducibility release for SATCAP v6.1. A DOI will be added after the GitHub release is archived through Zenodo. Please cite the archived release rather than an individual source file.

## Principal v6 additions

- exact lower bound for all-threshold exposure state over a finite alphabet;
- explicit multi-resource marginal/joint-query boundary;
- CPU, memory, disk-burst, and network-burst Bitbrains validation;
- Datadog native DDSketch 4.4.0 benchmark;
- measured OpenTelemetry OTLP/HTTP protobuf request-body payload;
- leakage-free delay-aware hybrid controller;
- executed 500-VM Rnd replication with monthly files merged by VM;
- fastStorage-to-Rnd coefficient transfer without classifier, scaler, or calibration refitting;
- explicit target-threshold boundary and Rnd calibration diagnostics;
- 2,000-replicate VM-cluster AP and Brier contrasts;
- corrected descriptive-window eligibility shared across resources.

## Reproducibility boundary

Raw provider traces and large derived pickle caches are excluded. Every number claimed in the article is represented by a distributed result table and checked by regression or integrity checks. Kubernetes/OpenStack prototype materials are excluded from this submission archive. Code is released under the MIT License; source datasets and third-party libraries retain their original licenses.
