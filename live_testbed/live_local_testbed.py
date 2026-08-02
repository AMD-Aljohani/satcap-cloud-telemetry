#!/usr/bin/env python3
"""Live local closed-loop concurrency-scaling testbed for SATCAP.

Each run launches real asynchronous requests with measured queueing latency.
The controller selects low or high worker concurrency for the next epoch. This
is a local testbed, not a public-cloud deployment.
"""
from __future__ import annotations
import asyncio,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results'
EPOCHS=80;TRAIN_EPOCHS=30;SUBS=12;SLOT=0.001;SERVICE=0.0015
LOW_WORKERS=2;HIGH_WORKERS=5;SLA=0.012

def schedule(seed):
 rng=np.random.default_rng(seed);z=-0.5;arr=[]
 for _ in range(EPOCHS):
  z=0.82*z+rng.normal(0,0.6)+(2.0 if rng.random()<0.08 else 0.0)
  lam=float(np.clip(np.exp(z),0.4,6.0))
  vals=rng.poisson(lam,size=SUBS)
  # Isolated spikes challenge one-position sampling but do not necessarily persist.
  if rng.random()<0.12: vals[rng.integers(0,SUBS)]+=rng.integers(4,9)
  arr.append(vals)
 return np.asarray(arr,int)

def telemetry(arr):
 nominal=5.0
 util=np.clip(arr/max(nominal,1e-9),0,1);mx=util.max(1);cnt=(util>=.9).sum(1);snap=util[:,0];event=(cnt>0).astype(int)
 return mx,cnt,snap,event

def fit_controllers(arr):
 mx,cnt,snap,event=telemetry(arr);n=TRAIN_EPOCHS;y=event[1:n];models={};reactive_rate=float(event[:n-1].mean())
 for name,X in [('snapshot',np.c_[snap[:n-1],snap[:n-1]>=.9]),('satcap',np.c_[mx[:n-1],cnt[:n-1]/SUBS,mx[:n-1]>=.9])]:
  mod=LogisticRegression(C=1,solver='lbfgs',max_iter=1000).fit(X,y);p=mod.predict_proba(X)[:,1];q=float(np.quantile(p,1-reactive_rate)) if reactive_rate>0 else 1.;models[name]=(mod,q)
 return models

async def execute_epoch(arrivals,workers):
 sem=asyncio.Semaphore(workers);lat=[]
 async def one(t0):
  async with sem:
   await asyncio.sleep(SERVICE)
  lat.append(time.perf_counter()-t0)
 tasks=[]
 for n in arrivals:
  now=time.perf_counter()
  tasks.extend(asyncio.create_task(one(now)) for _ in range(int(n)))
  await asyncio.sleep(SLOT)
 await asyncio.gather(*tasks)
 return np.asarray(lat,float)

async def run_policy(arr,policy,seed):
 models=fit_controllers(arr);mx,cnt,snap,event=telemetry(arr);action=False;all_lat=[];rows=[]
 for h in range(EPOCHS):
  workers=HIGH_WORKERS if action else LOW_WORKERS
  lv=await execute_epoch(arr[h],workers)
  if h>=TRAIN_EPOCHS: all_lat.extend(lv.tolist())
  if policy=='no_action': nxt=False
  elif policy=='reactive': nxt=bool(event[h])
  elif policy=='snapshot':
   mod,q=models['snapshot'];nxt=bool(mod.predict_proba([[snap[h],float(snap[h]>=.9)]])[0,1]>=q)
  elif policy=='satcap':
   mod,q=models['satcap'];nxt=bool(mod.predict_proba([[mx[h],cnt[h]/SUBS,float(mx[h]>=.9)]])[0,1]>=q)
  else: raise ValueError(policy)
  if h>=TRAIN_EPOCHS: rows.append({'workers':workers,'scaled':workers==HIGH_WORKERS,'event':event[h],'requests':len(lv),'mean_latency':lv.mean() if len(lv) else 0,'p95_latency':np.quantile(lv,.95) if len(lv) else 0,'sla_fraction':np.mean(lv>SLA) if len(lv) else 0})
  action=nxt
 lv=np.asarray(all_lat);d=pd.DataFrame(rows)
 return {'seed':seed,'policy':policy,'test_epochs':len(d),'test_requests':len(lv),'p50_latency_ms':1000*np.quantile(lv,.5),'p95_latency_ms':1000*np.quantile(lv,.95),'p99_latency_ms':1000*np.quantile(lv,.99),'sla_violation_fraction':float(np.mean(lv>SLA)),'scale_fraction':float(d.scaled.mean()),'mean_epoch_p95_ms':1000*d.p95_latency.mean(),'worker_epoch_cost':float(d.workers.sum())}

async def main_async():
 rows=[]
 for seed in [20260727,20260728]:
  arr=schedule(seed)
  for policy in ['no_action','reactive','snapshot','satcap']:
   print('live',seed,policy,flush=True);rows.append(await run_policy(arr,policy,seed))
 out=pd.DataFrame(rows);out.to_csv(RES/'live_local_testbed_trials.csv',index=False)
 agg=out.groupby('policy').agg(trials=('seed','count'),p95_latency_ms=('p95_latency_ms','mean'),p99_latency_ms=('p99_latency_ms','mean'),sla_violation_fraction=('sla_violation_fraction','mean'),scale_fraction=('scale_fraction','mean'),mean_epoch_p95_ms=('mean_epoch_p95_ms','mean'),worker_epoch_cost=('worker_epoch_cost','mean')).reset_index();agg.to_csv(RES/'live_local_testbed_summary.csv',index=False)
 print(agg.to_string(index=False));(RES/'live_local_testbed_config.json').write_text(json.dumps({'epochs':EPOCHS,'training_epochs':TRAIN_EPOCHS,'subslots_per_epoch':SUBS,'slot_seconds':SLOT,'service_seconds':SERVICE,'low_workers':LOW_WORKERS,'high_workers':HIGH_WORKERS,'sla_seconds':SLA,'seeds':[20260727,20260728]},indent=2))
if __name__=='__main__':asyncio.run(main_async())
