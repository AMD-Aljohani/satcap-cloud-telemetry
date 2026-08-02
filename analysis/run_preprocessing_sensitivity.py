#!/usr/bin/env python3
"""Three-policy preprocessing sensitivity for SATCAP headline results.

Policies:
  all_valid: retain every valid row.
  exact_record: remove only consecutive byte-identical complete records,
                including an identical timestamp.
  short_interval_payload: remove a record when the full non-time payload is
                identical to the preceding retained record and timestamp gap
                is 0--180 seconds (the v1 rule).
"""
from __future__ import annotations
import importlib.util, json, sys, zipfile, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('satcap_analysis', HERE/'satcap_analysis.py')
sa=importlib.util.module_from_spec(spec);sys.modules['satcap_analysis']=sa;spec.loader.exec_module(sa)
ROOT=HERE.parent;RESULTS=ROOT/'results';FIGURES=ROOT/'figures';BUILD=ROOT/'build'
RESULTS.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True);BUILD.mkdir(exist_ok=True)
RAW=Path(os.environ.get('SATCAP_RAW_ZIP', str(ROOT/'gwa_t_12_fastStorage.zip')))
POLICIES=('all_valid','exact_record','short_interval_payload')


def read_variants(zf,name):
    vm=int(Path(name).stem); lines=zf.read(name).splitlines()[1:]
    rec=[]
    for line in lines:
        p=line.split(b';\t')
        if len(p)<5: continue
        try: t=int(float(p[0])); c=float(p[1]); x=float(p[4])/100.0
        except (ValueError,OverflowError): continue
        if not(np.isfinite(x) and 0<=x<=1.000001 and np.isfinite(c)): continue
        rec.append((t,min(max(x,0.),1.),c,line,b';\t'.join(p[1:])))
    out={}
    # No removal.
    for policy in POLICIES:
        kept=[];prev_line=None;prev_t=None;prev_payload=None
        for t,x,c,line,payload in rec:
            drop=False
            if policy=='exact_record':
                drop=(line==prev_line)
            elif policy=='short_interval_payload':
                drop=(prev_t is not None and 0<=t-prev_t<=180 and payload==prev_payload)
            if not drop:
                kept.append((t,x,c));prev_t=t;prev_payload=payload
            prev_line=line
        if kept:
            a=np.asarray(kept,float); tt=a[:,0].astype(np.int64); xx=a[:,1];cc=a[:,2]
            if np.any(np.diff(tt)<0):
                o=np.argsort(tt,kind='stable');tt=tt[o];xx=xx[o];cc=cc[o]
        else: tt=np.array([],np.int64);xx=np.array([]);cc=np.array([])
        out[policy]=(tt,xx,cc,len(rec)-len(kept))
    return vm,len(rec),out


def adaptive_temporal(blocks):
    u=sa.MAIN_THRESHOLD
    tr,yt,vmt,te,ye,vme,counts=sa.temporal_dataset(blocks,u)
    offtr=np.array([logit((counts[v][0]+.5)/(counts[v][1]+1)) for v in vmt])
    offte=np.array([logit((counts[v][0]+.5)/(counts[v][1]+1)) for v in vme])
    rows=[]
    for scheme,label,base in [('snapshot','one snapshot',1),('maximum','maximum',1),('two_snapshots','two snapshots',2),('maximum_count','maximum plus count',2)]:
        xa,xb=sa.std_train_test(sa.features(tr,scheme,u),sa.features(te,scheme,u))
        beta=sa.fit_offset_logistic(xa,yt,offtr,l2=1.0)
        ptr=sa.predict_offset(xa,offtr,beta);pte=sa.predict_offset(xb,offte,beta)
        q=float(np.quantile(ptr,.99));sel=pte>=q;tp=int(ye[sel].sum());pos=int(ye.sum())
        avg=base+sel.mean()*(12-base)
        rows.append({'base_summary':label,'base_scalars':base,'risk_threshold':q,'trigger_fraction':sel.mean(),'mean_scalars_per_hour':avg,'storage_reduction_vs_full':1-avg/12,'event_capture':tp/pos if pos else np.nan,'precision':tp/sel.sum() if sel.sum() else np.nan,'test_n':len(ye),'test_events':pos})
    return pd.DataFrame(rows)


def headline(blocks,policy):
    u=.9;k=90;ev=blocks.event_90.to_numpy(bool);q=blocks.count_90.to_numpy();
    samp={'policy':policy,'windows':len(blocks),'event_hours':int(ev.sum()),'fixed_snapshot_recall':float(np.mean(blocks.loc[ev,'snapshot_1']>=u)),'mean_recall':float(np.mean(blocks.loc[ev,'mean']>=u)),'p95_recall':float(np.mean(blocks.loc[ev,'p95']>=u)),'maximum_recall':1.0,'exact_wall_windows':int(np.isclose(blocks.maximum,1.).sum())}
    # Persistence.
    n00=n01=n10=n11=0
    for _,g in blocks.groupby('vm_id',sort=False):
        g=g.sort_values('window_id');adj=np.diff(g.window_id.to_numpy())==1;a=g.event_90.to_numpy()[:-1][adj];b=g.event_90.to_numpy()[1:][adj]
        n00+=int(np.sum((a==0)&(b==0)));n01+=int(np.sum((a==0)&(b==1)));n10+=int(np.sum((a==1)&(b==0)));n11+=int(np.sum((a==1)&(b==1)))
    p0=n01/(n00+n01);p1=n11/(n10+n11)
    pers={'policy':policy,'n00':n00,'n01':n01,'n10':n10,'n11':n11,'p_event_given_no_event':p0,'p_event_given_event':p1,'risk_ratio':p1/p0}
    fc,pred=sa.evaluate_temporal(blocks,u)
    fc.insert(0,'policy',policy)
    ad=adaptive_temporal(blocks);ad.insert(0,'policy',policy)
    return samp,pers,fc,ad


def main():
    if sa.sha256(RAW)!=sa.EXPECTED_SHA256: raise RuntimeError('raw checksum mismatch')
    lists={p:[] for p in POLICIES};audit=[]
    with zipfile.ZipFile(RAW) as z:
        names=sorted(n for n in z.namelist() if n.endswith('.csv'))
        for i,name in enumerate(names):
            vm,raw,variants=read_variants(z,name)
            row={'vm_id':vm,'raw_valid':raw}
            for p,(t,x,c,removed) in variants.items():
                h,_=sa.make_hourly_frame(vm,t,x,c)
                if len(h):lists[p].append(h)
                row[f'{p}_removed']=removed;row[f'{p}_rows']=len(x);row[f'{p}_complete_hours']=len(h)
            audit.append(row)
            if (i+1)%250==0: print('processed',i+1,flush=True)
    pd.DataFrame(audit).to_csv(RESULTS/'preprocessing_policy_audit_by_vm.csv',index=False)
    S=[];P=[];F=[];A=[]
    for p in POLICIES:
        blocks=pd.concat(lists[p],ignore_index=True)
        print(p,len(blocks),flush=True)
        s,per,fc,ad=headline(blocks,p);S.append(s);P.append(per);F.append(fc);A.append(ad)
        # Save compact pickle only for current primary policy.
        if p=='all_valid': blocks.to_pickle(BUILD/'hourly_blocks_all_valid.pkl')
    pd.DataFrame(S).to_csv(RESULTS/'preprocessing_sampling_sensitivity.csv',index=False)
    pd.DataFrame(P).to_csv(RESULTS/'preprocessing_persistence_sensitivity.csv',index=False)
    pd.concat(F,ignore_index=True).to_csv(RESULTS/'preprocessing_forecast_sensitivity.csv',index=False)
    pd.concat(A,ignore_index=True).to_csv(RESULTS/'preprocessing_adaptive_sensitivity.csv',index=False)
    # Focused figure.
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    d=pd.DataFrame(S).set_index('policy').loc[list(POLICIES)]
    labels=['All valid','Exact-record only','Short-interval payload']
    x=np.arange(3);w=.22
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    for j,(col,lab) in enumerate([('fixed_snapshot_recall','Snapshot'),('p95_recall','P95'),('maximum_recall','Maximum')]):
        ax.bar(x+(j-1)*w,d[col],w,label=lab)
    ax.set_xticks(x,labels);ax.set_ylim(0,1.05);ax.set_ylabel('Recall of 90% event windows');ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIGURES/'Fig3_preprocessing_sensitivity.pdf');plt.close(fig)
    summary={'policies':POLICIES,'sampling':S,'persistence':P}
    (RESULTS/'PREPROCESSING_SENSITIVITY_SUMMARY.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
