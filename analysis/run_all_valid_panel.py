#!/usr/bin/env python3
import importlib.util,sys,json
from pathlib import Path
import os
import numpy as np,pandas as pd
from scipy.special import logit
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('satcap_analysis',HERE/'satcap_analysis.py');sa=importlib.util.module_from_spec(spec);sys.modules['satcap_analysis']=sa;spec.loader.exec_module(sa)
ROOT=HERE.parent;RES=ROOT/'results';blocks=pd.read_pickle(ROOT/'build/hourly_blocks_all_valid.pkl')
fc,trp,tep=sa.evaluate_panel_disjoint(blocks,.9);fc.insert(0,'policy','all_valid');fc.to_csv(RES/'all_valid_panel_forecast.csv',index=False);(RES/'all_valid_panel_split.json').write_text(json.dumps({'seed':20260727,'train_panels':[int(x) for x in trp],'test_panels':[int(x) for x in tep]},indent=2))
# Reproduce predictions for adaptive policy with same split.
k=90;parts=[]
for _,g in blocks.groupby('vm_id',sort=False):
 g=g.sort_values('window_id').reset_index(drop=True);cand=np.where(np.diff(g.window_id.to_numpy())==1)[0]
 if len(cand):p=g.iloc[cand].copy();p['target']=g.iloc[cand+1].event_90.to_numpy(int);parts.append(p)
d=pd.concat(parts,ignore_index=True);tr=d[d.vm_id.isin(set(trp))].reset_index(drop=True);te=d[d.vm_id.isin(set(tep))].reset_index(drop=True);yt=tr.pop('target').to_numpy(int);ye=te.pop('target').to_numpy(int);base=(yt.sum()+.5)/(len(yt)+1);ot=np.full(len(yt),logit(base));oe=np.full(len(ye),logit(base));rows=[]
for s,label,budget in [('snapshot','one snapshot',1),('maximum','maximum',1),('two_snapshots','two snapshots',2),('maximum_count','maximum plus count',2)]:
 a=np.c_[sa.features(tr,s,.9),np.log2(np.maximum(tr.cores.to_numpy(),1))];b=np.c_[sa.features(te,s,.9),np.log2(np.maximum(te.cores.to_numpy(),1))];a,b=sa.std_train_test(a,b);beta=sa.fit_offset_logistic(a,yt,ot);pt=sa.predict_offset(a,ot,beta);pe=sa.predict_offset(b,oe,beta);q=np.quantile(pt,.99);sel=pe>=q;avg=budget+sel.mean()*(12-budget);rows.append({'policy':'all_valid','validation':'panel_disjoint','base_summary':label,'trigger_fraction':sel.mean(),'mean_scalars_per_hour':avg,'storage_reduction_vs_full':1-avg/12,'event_capture':ye[sel].sum()/ye.sum(),'precision':ye[sel].sum()/sel.sum(),'test_n':len(ye),'test_events':int(ye.sum())})
pd.DataFrame(rows).to_csv(RES/'all_valid_panel_adaptive.csv',index=False);print(fc);print(pd.DataFrame(rows))
