#!/usr/bin/env python3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)

# Real OTLP deployment and delay-aware controller.
wire=pd.read_csv(R/'opentelemetry_wire_summary_v5.csv')
event=pd.read_csv(R/'opentelemetry_event_summary_v5.csv')
hyb=pd.read_csv(R/'delay_aware_hybrid_summary_v5.csv')
fig,axes=plt.subplots(1,3,figsize=(13.2,4.0))
bars=axes[0].bar(wire['mode'].str.upper(),wire['bytes_per_worker_window']);
for b,h in zip(bars,['///','...']): b.set_hatch(h)
axes[0].set_ylabel('OTLP wire bytes per worker-window');axes[0].grid(axis='y',alpha=.25)
axes[0].text(-0.12,1.04,'a',transform=axes[0].transAxes,fontweight='bold')
res=event.resource.str.upper().tolist();x=np.arange(len(res));w=.34
b1=axes[1].bar(x-w/2,event.snapshot_recall,w,label='First snapshot',hatch='///')
b2=axes[1].bar(x+w/2,event.satcap_recall,w,label='Maximum + count',hatch='...')
axes[1].set_xticks(x,res);axes[1].set_ylim(0,1.05);axes[1].set_ylabel('Measured event recall');axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.25)
axes[1].text(-0.12,1.04,'b',transform=axes[1].transAxes,fontweight='bold')
g=hyb[hyb.miss_to_action_cost_ratio==10]
policies=['always_on','reactive','snapshot_tuned','satcap_tuned','delay_aware_hybrid']
labels=['Always on','Reactive','Snapshot','SATCAP','Delay-aware hybrid']
linestyles=['-','--','-.',':',(0,(3,1,1,1))]
markers=['o','s','^','D','x']
for p,l,ls,mk in zip(policies,labels,linestyles,markers):
 q=g[g.policy==p].sort_values('provisioning_delay_epochs')
 axes[2].plot(q.provisioning_delay_epochs,q.normalized_cost,marker=mk,linestyle=ls,label=l)
axes[2].set_xlabel('Provisioning delay (epochs)');axes[2].set_ylabel('Normalized cost at $\\lambda=10$')
axes[2].set_xticks([0,1,2,3]);axes[2].legend(fontsize=7);axes[2].grid(alpha=.25)
axes[2].text(-0.12,1.04,'c',transform=axes[2].transAxes,fontweight='bold')
fig.tight_layout();fig.savefig(F/'Fig8_otel_hybrid_v5.pdf',bbox_inches='tight');plt.close(fig)

# Reference serialization and production DDSketch benchmark.
timing=pd.read_csv(R/'telemetry_overhead_timing_v3.csv')
payload=pd.read_csv(R/'telemetry_overhead_payload_v3.csv')
off=pd.read_csv(R/'official_ddsketch_benchmark_v5.csv').iloc[0]
dist=pd.read_csv(R/'official_ddsketch_payload_distribution_v5.csv').payload_bytes
fig,axes=plt.subplots(1,3,figsize=(13.2,4.0))
methods=['raw_12_samples','maximum_plus_count','log_sketch']
labels=['Raw 12 samples','Maximum + count','Compact log sketch']
vals=[float(payload.loc[payload.method==m,'median_binary_payload_bytes'].iloc[0]) for m in methods]
bars=axes[0].bar(np.arange(3),vals);
for b,h in zip(bars,['///','...','xx']): b.set_hatch(h)
axes[0].set_xticks(np.arange(3),labels,rotation=18,ha='right');axes[0].set_ylabel('Median compact record (bytes)');axes[0].grid(axis='y',alpha=.25);axes[0].text(-.12,1.04,'a',transform=axes[0].transAxes,fontweight='bold')
ops=['maximum_plus_count','log_sketch'];oplabels=['Maximum + count','Compact log sketch'];opvals=[float(timing[(timing.method==m)&(timing.operation=='stream_update_and_finalize')].million_observations_per_second.iloc[0]) for m in ops]
oplabels.append('Datadog native DDSketch');opvals.append(float(off.update_mobs_per_second))
bars=axes[1].bar(np.arange(3),opvals);
for b,h in zip(bars,['...','xx','///']): b.set_hatch(h)
axes[1].set_xticks(np.arange(3),oplabels,rotation=18,ha='right');axes[1].set_ylabel('Update throughput (M observations/s)');axes[1].grid(axis='y',alpha=.25);axes[1].text(-.12,1.04,'b',transform=axes[1].transAxes,fontweight='bold')
qs=np.quantile(dist,[0,.25,.5,.75,.95,1]);axes[2].boxplot(dist,vert=True,showfliers=False,whis=(5,95));axes[2].set_xticks([1],['Datadog native\nDDSketch protobuf']);axes[2].set_ylabel('Serialized sketch payload (bytes)');axes[2].set_yscale('log');axes[2].grid(axis='y',alpha=.25);axes[2].text(-.12,1.04,'c',transform=axes[2].transAxes,fontweight='bold')
axes[2].annotate(f"median {qs[2]:.0f} B\n95th {qs[4]:.0f} B",xy=(1,qs[2]),xytext=(1.15,qs[3]*1.2),arrowprops={'arrowstyle':'->'},fontsize=8)
fig.tight_layout();fig.savefig(F/'Fig6_telemetry_overhead_v5.pdf',bbox_inches='tight');fig.savefig(F/'Fig7_telemetry_overhead_v5.pdf',bbox_inches='tight');plt.close(fig)
print('wrote Fig6 and Fig8')


# Multi-resource event fidelity and forecasting with grayscale-safe hatching.
detection=pd.read_csv(R/'multiresource_event_recall_v5.csv')
forecast=pd.read_csv(R/'multiresource_forecast_ablation_v5.csv')
resources=['cpu','memory','disk','network']
resource_labels={'cpu':'CPU','memory':'Memory','disk':'Disk burst','network':'Network burst'}
fig,axes=plt.subplots(1,2,figsize=(10.8,4.2))
x=np.arange(len(resources));schemes=['snapshot','mean','p95','maximum'];width=.19
hatches=['///','...','xx','\\\\']
for j,(scheme,hatch) in enumerate(zip(schemes,hatches)):
    vals=[detection[(detection.resource==r)&(detection.scheme==scheme)].event_recall.iloc[0] for r in resources]
    axes[0].bar(x+(j-1.5)*width,vals,width,label=scheme.replace('p95','P95').title(),hatch=hatch)
axes[0].set_xticks(x,[resource_labels[r] for r in resources],rotation=15)
axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Threshold-event recall')
axes[0].legend(fontsize=8,ncol=2);axes[0].grid(axis='y',alpha=.25)
feature_sets=['persistence','two_snapshots','maximum','count','maximum_count'];width=.16
hatches=['///','...','xx','\\\\','++']
for j,(feature,hatch) in enumerate(zip(feature_sets,hatches)):
    vals=[forecast[(forecast.resource==r)&(forecast.feature_set==feature)].average_precision.iloc[0] for r in resources]
    axes[1].bar(x+(j-2)*width,vals,width,label=feature.replace('_',' ').title(),hatch=hatch)
axes[1].set_xticks(x,[resource_labels[r] for r in resources],rotation=15)
axes[1].set_ylim(0,1);axes[1].set_ylabel('Held-out average precision')
axes[1].legend(fontsize=7,ncol=2);axes[1].grid(axis='y',alpha=.25)
fig.tight_layout();fig.savefig(F/'Fig7_multiresource_validation_v5.pdf',bbox_inches='tight');plt.close(fig)
print('wrote Fig6, Fig7, and Fig8')
