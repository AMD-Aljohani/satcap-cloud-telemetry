#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
F=ROOT/'figures'

def close(a,b,tol=1e-6):
    assert abs(float(a)-float(b))<=tol,(a,b)

def test_multiresource_windows_and_exact_recall():
    d=pd.read_csv(R/'multiresource_event_recall_v5.csv')
    expected={'cpu':774759,'memory':791683,'disk':811570,'network':811570}
    for resource,n in expected.items():
        g=d[d.resource==resource]
        assert set(g.windows)=={n}
        close(g[g.scheme=='maximum'].event_recall.iloc[0],1.0)

def test_multiresource_cpu_matches_primary():
    d=pd.read_csv(R/'multiresource_event_recall_v5.csv')
    cpu=d[d.resource=='cpu'].set_index('scheme')
    close(cpu.loc['snapshot','event_recall'],0.629197301599333)
    close(cpu.loc['mean','event_recall'],0.4115061017206094)
    close(cpu.loc['p95','event_recall'],0.8843326006215417)
    assert int(cpu.loc['maximum','event_windows'])==13193

def test_multiresource_forecasting_boundaries():
    f=pd.read_csv(R/'multiresource_forecast_ablation_v5.csv').set_index(['resource','feature_set'])
    for r in ['cpu','memory','disk','network']:
        assert f.loc[(r,'maximum_count'),'average_precision']>f.loc[(r,'persistence'),'average_precision']
    assert f.loc[('disk','two_snapshots'),'average_precision']>f.loc[('disk','maximum_count'),'average_precision']
    assert f.loc[('disk','maximum_count'),'brier']<f.loc[('disk','two_snapshots'),'brier']

def test_multiresource_cluster_uncertainty():
    b=pd.read_csv(R/'multiresource_cluster_bootstrap_v5.csv').set_index('resource')
    assert set(b.bootstrap_replicates)=={2000}
    assert b.loc['cpu','cluster_ci_low']>0
    assert b.loc['disk','cluster_ci_low']>0
    assert b.loc['network','cluster_ci_low']>0
    assert b.loc['memory','cluster_ci_low']<0<b.loc['memory','cluster_ci_high']

def test_native_ddsketch_benchmark():
    x=pd.read_csv(R/'official_ddsketch_benchmark_v5.csv').iloc[0]
    assert x.package=='ddtrace' and x.package_version=='4.4.0'
    assert int(x.windows)==100000 and int(x.updates)==1200000
    close(x.payload_median_bytes,183)
    close(x.payload_p95_bytes,847)
    assert x.update_mobs_per_second>4

def test_otlp_wire_reduction():
    w=pd.read_csv(R/'opentelemetry_wire_summary_v5.csv').set_index('mode')
    expected=1-w.loc['satcap','wire_bytes']/w.loc['raw','wire_bytes']
    close(w.loc['satcap','wire_reduction_vs_raw'],expected,1e-9)
    assert expected>.90
    assert int(w.loc['raw','requests'])==972 and int(w.loc['satcap','requests'])==60

def test_otlp_event_fidelity():
    e=pd.read_csv(R/'opentelemetry_event_summary_v5.csv').set_index('resource')
    for r in ['cpu','memory']:
        close(e.loc[r,'satcap_recall'],1.0)
        assert e.loc[r,'snapshot_recall']<1.0

def test_delay_aware_hybrid_claim_is_conditional():
    h=pd.read_csv(R/'delay_aware_hybrid_summary_v5.csv')
    g=h[(h.miss_to_action_cost_ratio==10)&(h.provisioning_delay_epochs==0)].set_index('policy')
    assert g.loc['delay_aware_hybrid','normalized_cost']==g.normalized_cost.min()
    g1=h[(h.miss_to_action_cost_ratio==10)&(h.provisioning_delay_epochs==1)].set_index('policy')
    assert g1.loc['always_on','normalized_cost']<g1.loc['delay_aware_hybrid','normalized_cost']

def test_v5_figures_exist():
    for n in ['Fig6_telemetry_overhead_v5.pdf','Fig7_multiresource_validation_v5.pdf','Fig8_otel_hybrid_v5.pdf']:
        p=F/n; assert p.exists() and p.stat().st_size>1000,n

def test_unexecuted_cluster_materials_are_not_claimed_as_results():
    assert (ROOT/'analysis/run_second_panel_validation_v5.py').exists()
    assert not (R/'kubernetes_campaign_results_v5.csv').exists()
    assert not (R/'second_panel_validation_v5.csv').exists()

def main():
    tests=[v for k,v in globals().items() if k.startswith('test_') and callable(v)]
    for t in tests:
        t(); print('PASS',t.__name__)
    print(f'{len(tests)}/{len(tests)} tests passed')
if __name__=='__main__': main()
