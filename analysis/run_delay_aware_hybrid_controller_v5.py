#!/usr/bin/env python3
"""Leakage-free delay-aware hybrid controller extension.

The controller predicts a severe low-capacity queueing episode at the actual
action horizon (one reporting step plus provisioning delay). For each seed, the
severity threshold is the 90th percentile of low-capacity epoch p95 latency in
the training segment and is frozen before testing. Decision thresholds are also
selected on training data only. The hybrid combines current severe-event
fallback, predictive risk, and two-level hysteresis. Always-on and no-action
policies are included to prevent a trivial high-action solution from appearing
favorable.
"""
from __future__ import annotations
from pathlib import Path
import heapq, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'
EPOCHS=80; TRAIN=30; SUBS=12; SLOT=.001; SERVICE=.0015; LOW_WORKERS=2
SEEDS=list(range(20260727,20260827)); DELAYS=[0,1,2,3]; LAMBDAS=[1,2,5,10,20,50,100]


def schedule(seed):
 rng=np.random.default_rng(seed); z=-.5; arr=[]
 for _ in range(EPOCHS):
  z=.82*z+rng.normal(0,.6)+(2.0 if rng.random()<.08 else 0)
  lam=float(np.clip(np.exp(z),.4,6)); v=rng.poisson(lam,size=SUBS)
  if rng.random()<.12:v[rng.integers(0,SUBS)]+=rng.integers(4,9)
  arr.append(v)
 return np.asarray(arr,int)

def epoch_p95(counts,workers=LOW_WORKERS):
 avail=[0.0]*workers;heapq.heapify(avail);lat=[]
 for s,n in enumerate(counts):
  arrival=s*SLOT
  for _ in range(int(n)):
   a=heapq.heappop(avail);finish=max(arrival,a)+SERVICE;heapq.heappush(avail,finish);lat.append(finish-arrival)
 return float(np.quantile(lat,.95)) if lat else 0.

def telemetry(arr):
 u=np.clip(arr/5.0,0,1);mx=u.max(1);cnt=(u>=.9).sum(1)/SUBS;snap=u[:,0]
 low=np.array([epoch_p95(v) for v in arr]);threshold=float(np.quantile(low[:TRAIN],.90));severe=(low>threshold).astype(int)
 return mx,cnt,snap,severe,low,threshold

def fit_probs(mx,cnt,snap,target,horizon,kind):
 idx=np.arange(0,TRAIN-horizon);y=target[idx+horizon]
 X=np.c_[snap[idx],snap[idx]>=.9] if kind=='snapshot' else np.c_[mx[idx],cnt[idx],mx[idx]>=.9]
 if np.unique(y).size<2:return np.full(EPOCHS,float(np.mean(y)) if len(y) else 0.)
 m=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,class_weight=None).fit(X,y)
 Xa=np.c_[snap,snap>=.9] if kind=='snapshot' else np.c_[mx,cnt,mx>=.9]
 return m.predict_proba(Xa)[:,1]

def apply_sequence(p,current,q,kind):
 if kind=='predictive':return p>=q
 d=np.zeros(EPOCHS,dtype=bool);active=False;below=0
 for h in range(EPOCHS):
  high=p[h]>=q or bool(current[h]);low=p[h]>=.70*q
  if high:active=True;below=0
  elif active and low:below=0
  elif active:
   below+=1
   if below>=2:active=False;below=0
  d[h]=active
 return d

def map_action(decision,horizon):
 a=np.zeros(EPOCHS,dtype=bool);a[horizon:]=decision[:-horizon];return a

def train_cost(action,target,idx,lam):
 t=target[idx].astype(bool);a=action[idx];return lam*np.mean(t&~a)+np.mean(a)

def select_threshold(p,current,target,horizon,lam,kind):
 idx=np.arange(horizon,TRAIN);candidates=np.unique(np.r_[0,np.quantile(p[:TRAIN-horizon],[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99]),1])
 best=(float('inf'),.5)
 for q in candidates:
  a=map_action(apply_sequence(p,current,float(q),kind),horizon);cost=train_cost(a,target,idx,lam)
  if cost<best[0]-1e-12 or (abs(cost-best[0])<1e-12 and a[idx].mean()<map_action(apply_sequence(p,current,best[1],kind),horizon)[idx].mean()):best=(cost,float(q))
 return best[1]

def evaluate(seed,delay,lam):
 arr=schedule(seed);mx,cnt,snap,target,low,sevthr=telemetry(arr);horizon=1+delay
 ps=fit_probs(mx,cnt,snap,target,horizon,'snapshot');pt=fit_probs(mx,cnt,snap,target,horizon,'satcap')
 qs=select_threshold(ps,target,target,horizon,lam,'predictive');qt=select_threshold(pt,target,target,horizon,lam,'predictive');qh=select_threshold(pt,target,target,horizon,lam,'hybrid')
 policies={
  'no_action':np.zeros(EPOCHS,dtype=bool),'always_on':np.ones(EPOCHS,dtype=bool),
  'reactive':map_action(target.astype(bool),horizon),
  'snapshot_tuned':map_action(apply_sequence(ps,target,qs,'predictive'),horizon),
  'satcap_tuned':map_action(apply_sequence(pt,target,qt,'predictive'),horizon),
  'delay_aware_hybrid':map_action(apply_sequence(pt,target,qh,'hybrid'),horizon),
 }
 idx=np.arange(TRAIN,EPOCHS);t=target[idx].astype(bool);rows=[]
 for name,aall in policies.items():
  a=aall[idx];miss=float(np.mean(t&~a));rate=float(np.mean(a));rows.append({
   'seed':seed,'provisioning_delay_epochs':delay,'horizon_epochs':horizon,'miss_to_action_cost_ratio':lam,'policy':name,
   'test_epochs':len(idx),'severe_event_prevalence':float(t.mean()),'training_severe_p95_threshold_ms':1000*sevthr,
   'missed_event_fraction':miss,'event_coverage_fraction':float(np.sum(t&a)/max(1,np.sum(t))),'action_rate':rate,
   'normalized_cost':lam*miss+rate,'threshold':qh if name=='delay_aware_hybrid' else (qt if name=='satcap_tuned' else (qs if name=='snapshot_tuned' else np.nan))})
 return rows

def main():
 rows=[]
 for seed in SEEDS:
  for d in DELAYS:
   for lam in LAMBDAS:rows.extend(evaluate(seed,d,lam))
 out=pd.DataFrame(rows);out.to_csv(RES/'delay_aware_hybrid_trials_v5.csv',index=False)
 summary=out.groupby(['provisioning_delay_epochs','miss_to_action_cost_ratio','policy']).agg(seeds=('seed','count'),normalized_cost=('normalized_cost','mean'),cost_sd=('normalized_cost','std'),missed_event_fraction=('missed_event_fraction','mean'),event_coverage_fraction=('event_coverage_fraction','mean'),action_rate=('action_rate','mean'),severe_event_prevalence=('severe_event_prevalence','mean')).reset_index()
 summary.to_csv(RES/'delay_aware_hybrid_summary_v5.csv',index=False)
 wins=[]
 for (d,l),g in summary.groupby(['provisioning_delay_epochs','miss_to_action_cost_ratio']):
  b=g.loc[g.normalized_cost.idxmin()];wins.append({'provisioning_delay_epochs':d,'miss_to_action_cost_ratio':l,'best_policy':b.policy,'best_cost':b.normalized_cost})
 pd.DataFrame(wins).to_csv(RES/'delay_aware_hybrid_winners_v5.csv',index=False)
 (RES/'delay_aware_hybrid_config_v5.json').write_text(json.dumps({'training_epochs':TRAIN,'test_epochs':EPOCHS-TRAIN,'seeds':SEEDS,'delays':DELAYS,'cost_ratios':LAMBDAS,'target':'low-capacity epoch p95 latency above training 90th percentile','prediction_horizon':'1 + provisioning delay','threshold_selection':'training only','hybrid':'current severe-event fallback plus predictive risk and two-epoch hysteresis','baselines':['no_action','always_on','reactive','snapshot_tuned','satcap_tuned']},indent=2))
 print(summary.to_string(index=False));print(pd.DataFrame(wins).to_string(index=False))
if __name__=='__main__':main()
