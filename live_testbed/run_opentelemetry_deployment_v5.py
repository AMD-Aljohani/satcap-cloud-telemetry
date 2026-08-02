#!/usr/bin/env python3
"""Real OpenTelemetry multi-process telemetry deployment for SATCAP.

This experiment launches four independent worker processes, generates measured
CPU and resident-memory load, and exports metrics through the official
OpenTelemetry Python SDK using OTLP/HTTP protobuf to a local receiver. Raw mode
exports both gauges every 250 ms. SATCAP mode samples at the same rate but
exports per-5-second maximum and exceedance count. It is a real OpenTelemetry
process deployment; it is deliberately not described as Kubernetes because no
cluster runtime is available in the execution environment.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Observation
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'
SAMPLE_SECONDS=0.25; WINDOW_SAMPLES=20; WORKERS=4; WINDOWS=4; TRIALS=3
CPU_THRESHOLD=0.75; MEMORY_THRESHOLD=0.65

class State:
    lock=threading.Lock()
    current_mode=''; requests=0; bytes=0; points=0
    by_mode=defaultdict(lambda:{'requests':0,'bytes':0,'points':0})

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n=int(self.headers.get('content-length','0')); body=self.rfile.read(n)
        pts=0
        try:
            req=ExportMetricsServiceRequest(); req.ParseFromString(body)
            for rm in req.resource_metrics:
                for sm in rm.scope_metrics:
                    for m in sm.metrics:
                        data=m.WhichOneof('data')
                        if data:
                            pts += len(getattr(m,data).data_points)
        except Exception:
            pass
        with State.lock:
            mode=State.current_mode or 'unknown'; State.requests+=1; State.bytes+=n; State.points+=pts
            State.by_mode[mode]['requests']+=1; State.by_mode[mode]['bytes']+=n; State.by_mode[mode]['points']+=pts
        self.send_response(200); self.send_header('content-type','application/x-protobuf'); self.end_headers(); self.wfile.write(b'')
    def log_message(self,*args): return

def busy_for(seconds:float):
    end=time.perf_counter()+seconds; x=0.0
    while time.perf_counter()<end:
        x=(x*1.0000001+3.14159)%1000
    return x

def worker(worker_id:int, trial:int, mode:str, endpoint:str, outq:mp.Queue):
    rng=np.random.default_rng(20260801+1000*trial+worker_id)
    state={'cpu':0.0,'mem':0.0,'cpu_max':0.0,'cpu_count':0.0,'mem_max':0.0,'mem_count':0.0}
    resource=Resource.create({'service.name':'satcap-otel-worker','worker.id':str(worker_id),'trial':str(trial),'mode':mode})
    exporter=OTLPMetricExporter(endpoint=endpoint, timeout=5)
    reader=PeriodicExportingMetricReader(exporter, export_interval_millis=60000)
    provider=MeterProvider(resource=resource,metric_readers=[reader]); metrics.set_meter_provider(provider)
    meter=provider.get_meter('satcap.v5.otel')
    if mode=='raw':
        meter.create_observable_gauge('process.cpu.utilization',callbacks=[lambda _:[Observation(state['cpu'],{'worker':worker_id})]])
        meter.create_observable_gauge('process.memory.utilization',callbacks=[lambda _:[Observation(state['mem'],{'worker':worker_id})]])
    else:
        meter.create_observable_gauge('satcap.cpu.maximum',callbacks=[lambda _:[Observation(state['cpu_max'],{'worker':worker_id})]])
        meter.create_observable_gauge('satcap.cpu.exceedance_count',callbacks=[lambda _:[Observation(state['cpu_count'],{'worker':worker_id})]])
        meter.create_observable_gauge('satcap.memory.maximum',callbacks=[lambda _:[Observation(state['mem_max'],{'worker':worker_id})]])
        meter.create_observable_gauge('satcap.memory.exceedance_count',callbacks=[lambda _:[Observation(state['mem_count'],{'worker':worker_id})]])
    proc=psutil.Process(os.getpid()); proc.cpu_percent(None)
    baseline=proc.memory_info().rss; memory_budget=48*1024*1024
    rows=[]; cpu_win=[]; mem_win=[]; allocation=bytearray(1)
    for i in range(WINDOWS*WINDOW_SAMPLES):
        phase=i%WINDOW_SAMPLES
        burst=((i//WINDOW_SAMPLES+worker_id+trial)%3==0 and 5<=phase<=9) or (rng.random()<0.05)
        target_cpu=float(np.clip(0.20+0.18*np.sin((i+worker_id)/5)+(.65 if burst else 0)+rng.normal(0,.04),.05,.95))
        target_mem=float(np.clip(.22+(.58 if burst and worker_id%2==0 else 0)+rng.normal(0,.03),.15,.88))
        wanted=max(1,int(target_mem*memory_budget)); allocation=bytearray(wanted)
        t0=time.perf_counter(); busy_for(SAMPLE_SECONDS*target_cpu); remain=SAMPLE_SECONDS-(time.perf_counter()-t0)
        if remain>0: time.sleep(remain)
        cpu=float(np.clip(proc.cpu_percent(None)/100.0,0,1))
        mem=float(np.clip((proc.memory_info().rss-baseline)/memory_budget,0,1))
        state['cpu']=cpu; state['mem']=mem; cpu_win.append(cpu); mem_win.append(mem)
        rows.append({'trial':trial,'worker':worker_id,'mode':mode,'sample':i,'window':i//WINDOW_SAMPLES,'cpu':cpu,'memory':mem})
        if mode=='raw': provider.force_flush(timeout_millis=5000)
        if (i+1)%WINDOW_SAMPLES==0:
            state['cpu_max']=float(max(cpu_win)); state['cpu_count']=float(sum(v>=CPU_THRESHOLD for v in cpu_win))
            state['mem_max']=float(max(mem_win)); state['mem_count']=float(sum(v>=MEMORY_THRESHOLD for v in mem_win))
            if mode=='satcap': provider.force_flush(timeout_millis=5000)
            cpu_win.clear(); mem_win.clear()
    provider.shutdown(); outq.put(rows)

def run_mode(server_port:int,trial:int,mode:str):
    with State.lock:
        State.current_mode=f'{mode}_trial_{trial}'; before=dict(State.by_mode[State.current_mode])
    q=mp.Queue(); endpoint=f'http://127.0.0.1:{server_port}/v1/metrics'
    procs=[mp.Process(target=worker,args=(w,trial,mode,endpoint,q)) for w in range(WORKERS)]
    t0=time.perf_counter()
    for p in procs:p.start()
    rows=[]
    for _ in procs:
        try: rows.extend(q.get(timeout=180))
        except queue.Empty: raise RuntimeError('worker result timeout')
    for p in procs:
        p.join(30)
        if p.exitcode!=0: raise RuntimeError(f'worker exit {p.exitcode}')
    elapsed=time.perf_counter()-t0; time.sleep(.5)
    with State.lock: stats=dict(State.by_mode[f'{mode}_trial_{trial}'])
    stats.update({'trial':trial,'mode':mode,'elapsed_seconds':elapsed,'workers':WORKERS,'samples_per_worker':WINDOWS*WINDOW_SAMPLES})
    return pd.DataFrame(rows),stats

def main():
    mp.set_start_method('fork',force=True)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler); port=server.server_address[1]
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    telemetry=[]; wire=[]
    for trial in range(TRIALS):
        for mode in ('raw','satcap'):
            print('running',trial,mode,flush=True); d,s=run_mode(port,trial,mode); telemetry.append(d); wire.append(s)
    server.shutdown(); server.server_close()
    samples=pd.concat(telemetry,ignore_index=True); wire_df=pd.DataFrame(wire)
    # Event fidelity is computed from the identically sampled measured process values.
    raw=samples[samples['mode']=='raw'].copy()
    rows=[]
    for (trial,worker,window),g in raw.groupby(['trial','worker','window']):
        for r,u in [('cpu',CPU_THRESHOLD),('memory',MEMORY_THRESHOLD)]:
            vals=g[r].to_numpy(); event=bool(np.any(vals>=u));
            rows.append({'trial':trial,'worker':worker,'window':window,'resource':r,'event':event,
                         'snapshot_detected':bool(vals[0]>=u),'maximum_detected':bool(np.max(vals)>=u),
                         'maximum':float(np.max(vals)),'exceedance_count':int(np.sum(vals>=u))})
    fidelity=pd.DataFrame(rows)
    summary=fidelity.groupby('resource').apply(lambda g:pd.Series({
        'windows':len(g),'event_windows':int(g.event.sum()),
        'snapshot_recall':float(g.loc[g.event,'snapshot_detected'].mean()) if g.event.any() else np.nan,
        'satcap_recall':float(g.loc[g.event,'maximum_detected'].mean()) if g.event.any() else np.nan,
    }),include_groups=False).reset_index()
    bymode=wire_df.groupby('mode').agg(trials=('trial','count'),requests=('requests','sum'),wire_bytes=('bytes','sum'),points=('points','sum'),elapsed_seconds=('elapsed_seconds','sum')).reset_index()
    bymode['bytes_per_worker_window']=bymode.wire_bytes/(TRIALS*WORKERS*WINDOWS)
    bymode['requests_per_worker_window']=bymode.requests/(TRIALS*WORKERS*WINDOWS)
    raw_bytes=float(bymode.loc[bymode['mode']=='raw','wire_bytes'].iloc[0]); sat_bytes=float(bymode.loc[bymode['mode']=='satcap','wire_bytes'].iloc[0])
    bymode['wire_reduction_vs_raw']=1-bymode.wire_bytes/raw_bytes
    samples.to_csv(RES/'opentelemetry_measured_samples_v5.csv',index=False)
    wire_df.to_csv(RES/'opentelemetry_wire_trials_v5.csv',index=False)
    fidelity.to_csv(RES/'opentelemetry_event_fidelity_v5.csv',index=False)
    summary.to_csv(RES/'opentelemetry_event_summary_v5.csv',index=False)
    bymode.to_csv(RES/'opentelemetry_wire_summary_v5.csv',index=False)
    (RES/'opentelemetry_deployment_config_v5.json').write_text(json.dumps({'sdk':'OpenTelemetry Python SDK 1.42.1','transport':'OTLP/HTTP protobuf','workers':WORKERS,'trials':TRIALS,'sample_seconds':SAMPLE_SECONDS,'window_samples':WINDOW_SAMPLES,'windows_per_worker':WINDOWS,'cpu_threshold':CPU_THRESHOLD,'memory_threshold':MEMORY_THRESHOLD,'environment':'local independent OS processes; no Kubernetes runtime'},indent=2))
    print(bymode.to_string(index=False)); print(summary.to_string(index=False))

if __name__=='__main__':main()
