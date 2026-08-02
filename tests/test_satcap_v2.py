#!/usr/bin/env python3
from pathlib import Path
from math import comb
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'

def close(a,b,tol=1e-9):
    assert abs(float(a)-float(b))<=tol,(a,b)

def test_sampling_formula():
    m,c,k=12,3,2
    p=1-comb(m-c,k)/comb(m,k)
    # exhaustive subsets
    import itertools
    hits=sum(any(i<c for i in s) for s in itertools.combinations(range(m),k))
    close(p,hits/comb(m,k))

def test_top_b():
    p=np.array([.1,.8,.4,.7,.2]); B=2
    chosen=np.argsort(p)[-B:]
    assert set(chosen)=={1,3}
    # no other two-subset has a higher expected capture
    import itertools
    best=max(sum(p[list(s)]) for s in itertools.combinations(range(len(p)),B))
    close(sum(p[chosen]),best)

def test_primary_values():
    f=pd.read_csv(R/'all_valid_policy_forecast.csv').set_index('scheme')
    close(f.loc['snapshot','average_precision'],0.5902238186930617)
    close(f.loc['maximum_count','average_precision'],0.7488592328974554)
    close(f.loc['maximum_count','brier'],0.005554662356177945)
    s=json.loads((R/'all_valid_policy_summary.json').read_text())['sampling']
    assert s['windows']==774759 and s['event_hours']==13193
    close(s['fixed_snapshot_recall'],0.629197301599333)
    close(s['maximum_recall'],1.0)


def test_primary_controlled_ablation_v3():
    a=pd.read_csv(R/'primary_feature_model_ablation_v3.csv').set_index(['model','feature_set'])
    close(a.loc[('logistic','maximum_count'),'average_precision'],0.7412760566806145,1e-9)
    close(a.loc[('logistic','persistence'),'average_precision'],0.5188175881306408,1e-9)
    close(a.loc[('logistic','two_snapshots'),'average_precision'],0.5995547371618577,1e-9)
    close(a.loc[('hist_gradient_boosting','maximum_count'),'average_precision'],0.7378131161410099,1e-9)
    assert int(a.loc[('logistic','maximum_count'),'test_n'])==228835
    assert int(a.loc[('logistic','maximum_count'),'test_events'])==2927

def test_primary_log_sketch_v4():
    e=pd.read_csv(R/'primary_log_sketch_event_fidelity_v4.csv').iloc[0]
    assert int(e['windows'])==774759
    assert int(e['true_event_windows'])==13193
    assert int(e['false_negative_windows'])==211
    assert int(e['false_positive_windows'])==0
    close(e['event_recall'],0.984006670203896,1e-12)
    close(e['event_precision'],1.0)
    f=pd.read_csv(R/'primary_log_sketch_forecast_v4.csv').set_index(['model','feature_set'])
    close(f.loc[('logistic','log_sketch'),'average_precision'],0.7354144793434363,1e-12)
    close(f.loc[('hist_gradient_boosting','log_sketch'),'average_precision'],0.7362288023313661,1e-12)
    assert f.loc[('logistic','maximum_count'),'average_precision']>f.loc[('logistic','log_sketch'),'average_precision']
    p=pd.read_csv(R/'telemetry_overhead_payload_v3.csv').set_index('method')
    assert p.loc['log_sketch','median_binary_payload_bytes']==86
    assert p.loc['maximum_plus_count','median_binary_payload_bytes']==24

def test_sensitivity():
    s=pd.read_csv(R/'preprocessing_sampling_sensitivity.csv')
    assert set(s['policy'])=={'all_valid','exact_record','short_interval_payload'}
    assert set(s['event_hours'])=={13193}
    assert np.allclose(s['maximum_recall'],1)
    f=pd.read_csv(R/'preprocessing_forecast_sensitivity.csv')
    q=f[f.scheme=='maximum_count'].set_index('policy')['average_precision']
    close(q['all_valid'],0.7488592328974554)
    close(q['exact_record'],0.7487858646737272,1e-9)
    close(q['short_interval_payload'],0.7364252648193974,1e-9)

def test_external():
    s=pd.read_csv(R/'external_sampling_validation.csv')
    for provider in ['Google 2019','Alibaba 2018']:
        g=s[s.provider==provider].set_index('scheme')
        close(g.loc['hourly maximum','recall'],1)
        assert g.loc['one fixed snapshot','recall']<.5
    f=pd.read_csv(R/'external_forecast_validation.csv')
    g=f[f.provider=='Google 2019'].set_index('scheme')
    assert g.loc['maximum_count','average_precision']>g.loc['snapshot','average_precision']

def test_controller_and_live():
    c=pd.read_csv(R/'trace_controller_cost_sensitivity.csv')
    # At miss/action ratio 10, max+count beats snapshot for all providers.
    c=c[np.isclose(c['miss_to_action_cost_ratio'],10)].pivot(index='provider',columns='policy',values='normalized_cost')
    assert np.all(c['maximum_count']<c['snapshot'])
    l=pd.read_csv(R/'live_local_testbed_summary.csv').set_index('policy')
    assert l.loc['satcap','p95_latency_ms']<l.loc['snapshot','p95_latency_ms']
    assert l.loc['satcap','scale_fraction']<l.loc['snapshot','scale_fraction']

def test_uncertainty():
    s=pd.read_csv(R/'all_valid_comparison_summary.csv').set_index('comparison')
    assert int(s.loc['maximum_vs_snapshot','wins'])==150
    assert int(s.loc['maximum_count_vs_two_snapshots','wins'])==152
    assert s.loc['maximum_vs_snapshot','bootstrap_log_score_difference_low']>0
    assert s.loc['maximum_count_vs_two_snapshots','bootstrap_log_score_difference_low']>0

def test_no_raw_trace_bundled():
    names=[p.name.lower() for p in ROOT.rglob('*') if p.is_file()]
    assert 'gwa_t_12_faststorage.zip' not in names
    assert not any(n.endswith('.pkl') for n in names)

def test_external_persistence_and_sensitivity():
    b=pd.read_csv(R/'external_paired_ap_bootstrap_v3.csv')
    q=b[(b.model=='logistic')&(b.comparison=='satcap_minus_persistence')]
    assert len(q)==2 and np.all(q.moving_block_ci_low>0)
    s=pd.read_csv(R/'external_threshold_window_sensitivity_v3.csv')
    a=s[s.metric=='forecast_average_precision'].pivot_table(index=['provider','threshold_quantile','window_minutes'],columns='method',values='estimate')
    assert np.all(a.satcap>a.persistence)
    assert int(np.sum(a.satcap>a.two_snapshots))==22
    assert int(np.sum(a.satcap==a.two_snapshots))==1

def test_operational_robustness_v3():
    o=pd.read_csv(R/'operational_robustness_v3_bootstrap_summary.csv')
    g=o[o.provisioning_delay_epochs==0].pivot(index='policy',columns='metric',values='mean')
    assert g.loc['satcap','p95_latency_ms']<g.loc['snapshot','p95_latency_ms']
    assert g.loc['satcap','sla_violation_fraction']<g.loc['snapshot','sla_violation_fraction']
    assert g.loc['satcap','scale_fraction']<g.loc['snapshot','scale_fraction']

def test_overhead_v3():
    p=pd.read_csv(R/'telemetry_overhead_payload_v3.csv').set_index('method')
    close(p.loc['maximum_plus_count','median_binary_payload_bytes'],24)
    close(p.loc['raw_12_samples','median_binary_payload_bytes'],108)
    assert p.loc['maximum_plus_count','binary_reduction_vs_raw']>.77
    t=pd.read_csv(R/'telemetry_overhead_timing_v3.csv')
    rate=t[(t.method=='maximum_plus_count')&(t.operation=='stream_update_and_finalize')].million_observations_per_second.iloc[0]
    assert rate>1.0

def test_v3_figures_exist():
    required=['Fig1_event_recall_v3.pdf','Fig2_forecast_and_uncertainty_v3.pdf','Fig3_primary_controlled_ablation_final.pdf','Fig3_external_ablation_v3.pdf','Fig4_external_recall_ci_v3.pdf','Fig5_operational_robustness_v3.pdf','Fig6_telemetry_overhead_v3.pdf','FigS1_threshold_window_sensitivity_v3.pdf','FigS2_cost_sensitivity_v3.pdf']
    for name in required:
        p=ROOT/'figures'/name
        assert p.exists() and p.stat().st_size>1000,name

def main():
    tests=[v for k,v in globals().items() if k.startswith('test_') and callable(v)]
    for t in tests:
        t(); print('PASS',t.__name__)
    print(f'{len(tests)}/{len(tests)} tests passed')
if __name__=='__main__': main()
