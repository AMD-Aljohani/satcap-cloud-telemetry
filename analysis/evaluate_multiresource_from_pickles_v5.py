#!/usr/bin/env python3
from pathlib import Path
import importlib.util, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures'
spec=importlib.util.spec_from_file_location('mr',ROOT/'analysis/run_multiresource_bitbrains_v5.py');mr=importlib.util.module_from_spec(spec);sys.modules['mr']=mr;spec.loader.exec_module(mr)
resources=('cpu','memory','disk','network')
metric_rows=[]; forecast_rows=[]; boot_rows=[]
for resource in resources:
    blocks=pd.read_pickle(ROOT/'build'/f'hourly_blocks_{resource}_v5.pkl')
    u=blocks.threshold_normalized.to_numpy(float);ev=blocks.event.to_numpy(bool);n=int(ev.sum())
    for scheme,det in [('snapshot',blocks.snapshot_1.to_numpy()>=u),('mean',blocks['mean'].to_numpy()>=u),('p95',blocks.p95.to_numpy()>=u),('maximum',blocks.maximum.to_numpy()>=u)]:
        metric_rows.append({'resource':resource,'scheme':scheme,'windows':len(blocks),'event_windows':n,'event_recall':float(det[ev].mean()) if n else np.nan})
    f,b=mr.evaluate_resource(blocks);f.insert(0,'resource',resource);forecast_rows.append(f);boot_rows.append({'resource':resource,**b})
    print(resource,f[['feature_set','average_precision','brier']].to_string(index=False),flush=True)
detection=pd.DataFrame(metric_rows);forecast=pd.concat(forecast_rows,ignore_index=True);boot=pd.DataFrame(boot_rows)
detection.to_csv(RES/'multiresource_event_recall_v5.csv',index=False);forecast.to_csv(RES/'multiresource_forecast_ablation_v5.csv',index=False);boot.to_csv(RES/'multiresource_cluster_bootstrap_v5.csv',index=False)
labels={'cpu':'CPU','memory':'Memory','disk':'Disk burst','network':'Network burst'}
fig,axes=plt.subplots(1,2,figsize=(10.8,4.2));x=np.arange(len(resources));schemes=['snapshot','mean','p95','maximum'];width=.19
for j,s in enumerate(schemes):
 vals=[detection[(detection.resource==r)&(detection.scheme==s)].event_recall.iloc[0] for r in resources];axes[0].bar(x+(j-1.5)*width,vals,width,label=s.replace('p95','P95').title())
axes[0].set_xticks(x,[labels[r] for r in resources],rotation=15);axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Threshold-event recall');axes[0].legend(fontsize=8,ncol=2);axes[0].grid(axis='y',alpha=.25)
fs=['persistence','two_snapshots','maximum','count','maximum_count'];width=.16
for j,s in enumerate(fs):
 vals=[forecast[(forecast.resource==r)&(forecast.feature_set==s)].average_precision.iloc[0] for r in resources];axes[1].bar(x+(j-2)*width,vals,width,label=s.replace('_',' ').title())
axes[1].set_xticks(x,[labels[r] for r in resources],rotation=15);axes[1].set_ylim(0,1);axes[1].set_ylabel('Held-out average precision');axes[1].legend(fontsize=7,ncol=2);axes[1].grid(axis='y',alpha=.25)
fig.tight_layout();fig.savefig(FIG/'Fig7_multiresource_validation_v5.pdf',bbox_inches='tight');plt.close(fig)
print('done')
