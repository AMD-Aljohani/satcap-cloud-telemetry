#!/usr/bin/env python3
"""Multi-seed discrete-event robustness extension for the local SATCAP testbed.

The original supplement retains two real asyncio executions. This script uses
identical schedules, service time, worker counts, controller fitting, and epoch
logic in a deterministic multi-server queue simulator so that many independent
schedules and provisioning delays can be evaluated reproducibly.
"""
from __future__ import annotations
from pathlib import Path
import json, heapq
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'
EPOCHS=80; TRAIN_EPOCHS=30; SUBS=12; SLOT=0.001; SERVICE=0.0015
LOW_WORKERS=2; HIGH_WORKERS=5; SLA=0.012
SEEDS=list(range(20260727,20260827))  # 100 schedules
POLICIES=['no_action','reactive','snapshot','satcap']; DELAYS=[0,1,2]


def schedule(seed):
 rng=np.random.default_rng(seed); z=-0.5; arr=[]
 for _ in range(EPOCHS):
  z=0.82*z+rng.normal(0,0.6)+(2.0 if rng.random()<0.08 else 0.0)
  lam=float(np.clip(np.exp(z),0.4,6.0)); vals=rng.poisson(lam,size=SUBS)
  if rng.random()<0.12: vals[rng.integers(0,SUBS)]+=rng.integers(4,9)
  arr.append(vals)
 return np.asarray(arr,int)


def telemetry(arr):
 util=np.clip(arr/5.0,0,1); mx=util.max(1); cnt=(util>=.9).sum(1); snap=util[:,0]; event=(cnt>0).astype(int)
 return mx,cnt,snap,event


def fit_controllers(arr):
 mx,cnt,snap,event=telemetry(arr); n=TRAIN_EPOCHS; y=event[1:n]; rate=float(event[:n-1].mean()); models={}
 for name,X in [('snapshot',np.c_[snap[:n-1],snap[:n-1]>=.9]),('satcap',np.c_[mx[:n-1],cnt[:n-1]/SUBS,mx[:n-1]>=.9])]:
  mod=LogisticRegression(C=1,solver='lbfgs',max_iter=1000).fit(X,y); p=mod.predict_proba(X)[:,1]
  models[name]=(mod,float(np.quantile(p,1-rate)) if rate>0 else 1.0)
 return models


def base_decisions(arr,policy):
 models=fit_controllers(arr); mx,cnt,snap,event=telemetry(arr); d=np.zeros(EPOCHS,dtype=bool)
 for h in range(EPOCHS-1):
  if policy=='no_action': nxt=False
  elif policy=='reactive': nxt=bool(event[h])
  elif policy=='snapshot':
   mod,q=models['snapshot']; nxt=bool(mod.predict_proba([[snap[h],float(snap[h]>=.9)]])[0,1]>=q)
  elif policy=='satcap':
   mod,q=models['satcap']; nxt=bool(mod.predict_proba([[mx[h],cnt[h]/SUBS,float(mx[h]>=.9)]])[0,1]>=q)
  else: raise ValueError(policy)
  d[h+1]=nxt
 return d,event


def shifted(d,delay):
 if delay==0:return d.copy()
 out=np.zeros_like(d); out[delay:]=d[:-delay]; return out


def epoch_latencies(counts,workers):
 # FCFS deterministic multi-server queue. Arrival times are subslot boundaries.
 avail=[0.0]*workers; heapq.heapify(avail); lat=[]
 for s,n in enumerate(counts):
  arrival=s*SLOT
  for _ in range(int(n)):
   a=heapq.heappop(avail); start=max(arrival,a); finish=start+SERVICE
   heapq.heappush(avail,finish); lat.append(finish-arrival)
 return np.asarray(lat,float)


def run(arr,policy,delay,seed):
 d,event=base_decisions(arr,policy); action=shifted(d,delay); all_lat=[]; epoch_p95=[]; scaled=[]; events=[]
 for h in range(TRAIN_EPOCHS,EPOCHS):
  workers=HIGH_WORKERS if action[h] else LOW_WORKERS; lv=epoch_latencies(arr[h],workers)
  all_lat.extend(lv.tolist()); epoch_p95.append(np.quantile(lv,.95) if len(lv) else 0); scaled.append(action[h]); events.append(event[h])
 lv=np.asarray(all_lat); scaled=np.asarray(scaled,bool); events=np.asarray(events,bool); ne=max(1,int(events.sum()))
 return {'seed':seed,'policy':policy,'provisioning_delay_epochs':delay,'test_epochs':len(events),'test_requests':len(lv),
  'p50_latency_ms':1000*np.quantile(lv,.5),'p95_latency_ms':1000*np.quantile(lv,.95),'p99_latency_ms':1000*np.quantile(lv,.99),
  'sla_violation_fraction':float(np.mean(lv>SLA)),'scale_fraction':float(scaled.mean()),
  'event_coverage_fraction':float((events&scaled).sum()/ne),'missed_event_fraction':float((events&~scaled).sum()/len(events)),
  'mean_epoch_p95_ms':1000*np.mean(epoch_p95),'worker_epoch_cost':float(np.sum(np.where(scaled,HIGH_WORKERS,LOW_WORKERS)))}


def bootstrap(df,reps=2000):
 rng=np.random.default_rng(20260801); metrics=['p95_latency_ms','p99_latency_ms','sla_violation_fraction','scale_fraction','event_coverage_fraction','worker_epoch_cost']; rows=[]
 for (p,d),g in df.groupby(['policy','provisioning_delay_epochs'],sort=False):
  for m in metrics:
   v=g[m].to_numpy(float); means=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(reps)])
   rows.append({'policy':p,'provisioning_delay_epochs':d,'metric':m,'mean':v.mean(),'ci_low':np.quantile(means,.025),'ci_high':np.quantile(means,.975),'seeds':len(v)})
 return pd.DataFrame(rows)


def main():
 rows=[]
 for seed in SEEDS:
  arr=schedule(seed)
  for policy in POLICIES:
   for delay in DELAYS: rows.append(run(arr,policy,delay,seed))
 out=pd.DataFrame(rows); out.to_csv(RES/'operational_robustness_v3_trials.csv',index=False)
 bootstrap(out).to_csv(RES/'operational_robustness_v3_bootstrap_summary.csv',index=False)
 costs=[]
 for lam in [1,2,5,10,20,50,100]:
  for _,r in out.iterrows(): costs.append({'seed':r.seed,'policy':r.policy,'provisioning_delay_epochs':r.provisioning_delay_epochs,
   'miss_to_action_cost_ratio':lam,'normalized_cost':lam*r.missed_event_fraction+r.scale_fraction,
   'missed_event_fraction':r.missed_event_fraction,'action_rate':r.scale_fraction})
 pd.DataFrame(costs).to_csv(RES/'operational_robustness_v3_cost_sensitivity.csv',index=False)
 (RES/'operational_robustness_v3_config.json').write_text(json.dumps({'type':'deterministic multi-server discrete-event extension','epochs':EPOCHS,
  'training_epochs':TRAIN_EPOCHS,'subslots_per_epoch':SUBS,'slot_seconds':SLOT,'service_seconds':SERVICE,'low_workers':LOW_WORKERS,
  'high_workers':HIGH_WORKERS,'sla_seconds':SLA,'seeds':SEEDS,'provisioning_delays_epochs':DELAYS},indent=2))
 print(bootstrap(out).to_string(index=False))

if __name__=='__main__':main()
