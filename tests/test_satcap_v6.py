#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
F = ROOT / "figures"


def close(a, b, tol=1e-9):
    assert abs(float(a) - float(b)) <= tol, (a, b)


def test_corrected_faststorage_descriptive_universe():
    d = pd.read_csv(R / "second_panel_event_fidelity_v6.csv")
    d = d[d.panel == "GWA-T-12-fastStorage"]
    expected = {"cpu": 774759, "memory": 792754, "disk": 811724, "network": 811724}
    for resource, windows in expected.items():
        g = d[d.resource == resource]
        assert set(g.windows) == {windows}
        close(g[g.scheme == "maximum"].event_recall.iloc[0], 1.0)


def test_rnd_panel_structure_and_exactness():
    d = pd.read_csv(R / "second_panel_event_fidelity_v6.csv")
    d = d[d.panel == "GWA-T-12-Rnd"]
    expected = {
        "cpu": (963340, 10900), "memory": (985076, 1181),
        "disk": (989579, 94121), "network": (989579, 59274),
    }
    for resource, (windows, events) in expected.items():
        g = d[d.resource == resource]
        assert set(g.vm_count) == {500}
        assert set(g.windows) == {windows}
        assert set(g.event_windows) == {events}
        close(g[g.scheme == "maximum"].event_recall.iloc[0], 1.0)


def test_coefficient_transfer_metrics_and_calibration():
    m = pd.read_csv(R / "second_panel_and_frozen_transfer_metrics_v6.csv")
    m = m[m.analysis == "frozen_faststorage_to_rnd"].set_index(["resource", "feature_set"])
    expected_ap = {"cpu": 0.7376428095785746, "memory": 0.3728195625609789,
                   "disk": 0.2951770552823084, "network": 0.35618591047308706}
    for resource, ap in expected_ap.items():
        close(m.loc[(resource, "maximum_count"), "average_precision"], ap)
        assert m.loc[(resource, "maximum_count"), "average_precision"] > m.loc[(resource, "persistence"), "average_precision"]
    assert int(m.loc[("cpu", "maximum_count"), "train_n"]) == 532319
    assert int(m.loc[("cpu", "maximum_count"), "test_n"]) == 284314
    assert int(m.loc[("cpu", "maximum_count"), "test_events"]) == 2453
    c = pd.read_csv(R / "second_panel_calibration_comparison_v61.csv")
    assert set(c.resource) == {"cpu", "memory", "disk", "network"}
    assert set(c.analysis) == {"within_rnd", "frozen_coefficients"}
    assert len(c) == 8
    assert c[["brier", "ece_10", "calibration_intercept", "calibration_slope"]].notna().all().all()


def test_cluster_intervals_and_protocol_freeze():
    b = pd.read_csv(R / "second_panel_cluster_bootstrap_v6.csv")
    b = b[(b.analysis == "frozen_transfer") & (b.feature_set == "maximum_count")]
    assert set(b.resource) == {"cpu", "memory", "disk", "network"}
    assert set(b.bootstrap_replicates) == {2000}
    assert (b.ap_difference_ci_low > 0).all()
    assert (b.brier_improvement_ci_low > 0).all()
    p = json.loads((R / "second_panel_protocol_v6.json").read_text())
    assert p["frozen_transfer_refit_on_rnd"] is False
    assert p["rnd_unique_vms"] == 500 and p["rnd_csv_files"] == 1500
    assert "target-panel threshold" in p["claim_boundary"]


def test_v61_figures_and_no_cluster_claim():
    for name in ["Fig6_multiresource_validation_v6.pdf", "Fig9_second_panel_transfer_v6.pdf", "Fig10_second_panel_calibration_v61.pdf"]:
        path = F / name
        assert path.exists() and path.stat().st_size > 1000
    assert not (R / "kubernetes_campaign_results_v6.csv").exists()


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test(); print("PASS", test.__name__)
    print(f"{len(tests)}/{len(tests)} v6 tests passed")


if __name__ == "__main__":
    main()
