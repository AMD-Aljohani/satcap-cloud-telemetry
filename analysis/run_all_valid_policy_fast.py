#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,sys,json,glob
from pathlib import Path
import os
import numpy as np,pandas as pd
from scipy.special import logit
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('satcap_analysis',HERE/'satcap_analysis.py');sa=importlib.util.module_from_spec(spec);sys.modules['satcap_analysis']=sa;spec.loader.exec_module(sa)
ROOT=HERE.parent;RES=ROOT/'results';RAW=Path(os.environ.get('SATCAP_RAW_DIR', str(HERE.parent/'raw_bitbrains')))

def adaptive(blocks):
 u=.9;tr,yt,vmt,te,ye,vme,counts=sa.temporal_dataset(blocks,u);offtr=np.array([logit((counts[v][0]+.5)/(counts[v][1]+1)) for v in vmt]);offte=np.array([logit((counts[v][0]+.5)/(counts[v][1]+1)) for v in vme]);rows=[]
 for scheme,label,base in [('snapshot','one snapshot',1),('maximum','maximum',1),('two_snapshots','two snapshots',2),('maximum_count','maximum plus count',2)]:
  a,b=sa.std_train_test(sa.features(tr,scheme,u),sa.features(te,scheme,u));beta=sa.fit_offset_logistic(a,yt,offtr);pt=sa.predict_offset(a,offtr,beta);pe=sa.predict_offset(b,offte,beta);q=np.quantile(pt,.99);sel=pe>=q;avg=base+sel.mean()*(12-base)
  rows.append({'policy':'all_valid','base_summary':label,'trigger_fraction':sel.mean(),'mean_scalars_per_hour':avg,'storage_reduction_vs_full':1-avg/12,'event_capture':ye[sel].sum()/ye.sum(),'precision':ye[sel].sum()/sel.sum()})
 return pd.DataFrame(rows)

def main():
 frames=[];aud=[];files=sorted(RAW.rglob('*.csv'))
 for i,p in enumerate(files):
  vm=int(p.stem);df=pd.read_csv(p,sep=';',engine='c');n0=len(df);# no duplicate removal
  t=pd.to_numeric(df.iloc[:,0],errors='coerce').to_numpy();c=pd.to_numeric(df.iloc[:,1],errors='coerce').to_numpy();x=pd.to_numeric(df.iloc[:,4],errors='coerce').to_numpy()/100
  ok=np.isfinite(t)&np.isfinite(c)&np.isfinite(x)&(x>=0)&(x<=1.000001);t=t[ok].astype(np.int64);c=c[ok];x=np.clip(x[ok],0,1);o=np.argsort(t,kind='stable');t=t[o];x=x[o];c=c[o]
  h,_=sa.make_hourly_frame(vm,t,x,c)
  if len(h):frames.append(h)
  aud.append({'vm_id':vm,'raw_rows':n0,'nothing_removed':0,'valid_after_exact':len(x),'complete_hours':len(h)})
  if (i+1)%250==0:print(i+1,flush=True)
 blocks=pd.concat(frames,ignore_index=True);pd.DataFrame(aud).to_csv(RES/'all_valid_policy_audit.csv',index=False)
 ev=blocks.event_90.astype(bool);s={'policy':'all_valid','windows':len(blocks),'event_hours':int(ev.sum()),'fixed_snapshot_recall':float((blocks.loc[ev,'snapshot_1']>=.9).mean()),'mean_recall':float((blocks.loc[ev,'mean']>=.9).mean()),'p95_recall':float((blocks.loc[ev,'p95']>=.9).mean()),'maximum_recall':1.,'exact_wall_windows':int(np.isclose(blocks.maximum,1).sum())}
 n00=n01=n10=n11=0
 for _,g in blocks.groupby('vm_id'):
  g=g.sort_values('window_id');adj=np.diff(g.window_id.to_numpy())==1;a=g.event_90.to_numpy()[:-1][adj];b=g.event_90.to_numpy()[1:][adj];n00+=int(((a==0)&(b==0)).sum());n01+=int(((a==0)&(b==1)).sum());n10+=int(((a==1)&(b==0)).sum());n11+=int(((a==1)&(b==1)).sum())
 p0=n01/(n00+n01);p1=n11/(n10+n11);pe={'policy':'all_valid','p_event_given_no_event':p0,'p_event_given_event':p1,'risk_ratio':p1/p0,'n00':n00,'n01':n01,'n10':n10,'n11':n11}
 fc,_=sa.evaluate_temporal(blocks,.9);fc.insert(0,'policy','all_valid');fc.to_csv(RES/'all_valid_policy_forecast.csv',index=False);ad=adaptive(blocks);ad.to_csv(RES/'all_valid_policy_adaptive.csv',index=False)
 (RES/'all_valid_policy_summary.json').write_text(json.dumps({'sampling':s,'persistence':pe},indent=2));print(s);print(pe);print(fc.to_string(index=False));print(ad.to_string(index=False))
if __name__=='__main__':main()
