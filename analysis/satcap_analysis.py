#!/usr/bin/env python3
"""SATCAP reproducibility analysis for the Bitbrains fastStorage trace.

The workflow studies equal-budget telemetry aggregation, next-hour saturation
forecasting, and adaptive detailed-retention policies.
"""
from __future__ import annotations
import hashlib, json, math, os, zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"; FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
RAW_ZIP = Path(os.environ.get("SATCAP_RAW_ZIP", str(ROOT / "gwa_t_12_fastStorage.zip")))
EXPECTED_SHA256 = "11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671"
THRESHOLDS = (0.80, 0.90, 0.95, 0.99)
MAIN_THRESHOLD = 0.90
RNG = np.random.default_rng(20260727)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read_clean_series(zf: zipfile.ZipFile, name: str):
    """Return vm_id,timestamp,cpu,cores and cleaning counts.

    A row is removed only when its complete non-time payload is byte-identical
    to the preceding row and the timestamps are no more than 180 seconds apart.
    The nominal sampling separation is 300 seconds.
    """
    vm=int(Path(name).stem); lines=zf.read(name).splitlines()[1:]
    ts=[]; xs=[]; cs=[]; raw=removed=0; prev_t=None; prev_payload=None
    for line in lines:
        p=line.split(b';\t')
        if len(p)<5:continue
        try:t=int(float(p[0])); c=float(p[1]); x=float(p[4])/100.0
        except (ValueError,OverflowError):continue
        if not(np.isfinite(x) and 0<=x<=1.000001 and np.isfinite(c)):continue
        raw+=1; payload=b';\t'.join(p[1:])
        if prev_t is not None and 0<=t-prev_t<=180 and payload==prev_payload:
            removed+=1;continue
        ts.append(t);xs.append(min(max(x,0.0),1.0));cs.append(c)
        prev_t=t;prev_payload=payload
    t=np.asarray(ts,np.int64);x=np.asarray(xs,float);c=np.asarray(cs,float)
    if len(t) and np.any(np.diff(t)<0):
        o=np.argsort(t,kind='stable');t=t[o];x=x[o];c=c[o]
    return vm,t,x,c,raw,removed


def exact_windows(t: np.ndarray, x: np.ndarray, cores: np.ndarray, minutes: int):
    expected=minutes//5; wid=t//(minutes*60)
    if len(wid)==0:return np.array([],np.int64),np.empty((0,expected)),np.array([])
    starts=np.r_[0,1+np.flatnonzero(wid[1:]!=wid[:-1])]
    ends=np.r_[starts[1:],len(wid)];counts=ends-starts
    starts=starts[counts==expected]
    if len(starts)==0:return np.array([],np.int64),np.empty((0,expected)),np.array([])
    idx=starts[:,None]+np.arange(expected)[None,:]
    return wid[starts],x[idx],np.median(cores[idx],axis=1)


def make_hourly_frame(vm,t,x,c):
    wid,a,cc=exact_windows(t,x,c,60)
    if len(wid)==0:return pd.DataFrame(), np.empty((0,12))
    out={
      'vm_id':np.full(len(wid),vm,dtype=int),'window_id':wid,'n':np.full(len(wid),12,dtype=int),'cores':cc,
      'snapshot_1':a[:,0],'snapshot_2a':a[:,0],'snapshot_2b':a[:,6],
      'mean':a.mean(1),'p95':np.quantile(a,.95,axis=1,method='linear'),'maximum':a.max(1)
    }
    for u in THRESHOLDS:
        k=int(round(100*u)); count=(a>=u).sum(1)
        out[f'count_{k}']=count;out[f'event_{k}']=(count>0).astype(int)
        out[f'snapshot1_event_{k}']=(a[:,0]>=u).astype(int)
        out[f'snapshot2_count_{k}']=((a[:,0]>=u)+(a[:,6]>=u)).astype(int)
    return pd.DataFrame(out),a


def features(df,scheme,u):
    k=int(round(100*u))
    if scheme=='snapshot':
        v=df.snapshot_1.to_numpy();return np.c_[v,v>=u]
    if scheme=='mean':
        v=df['mean'].to_numpy();return np.c_[v,v>=u]
    if scheme=='p95':
        v=df.p95.to_numpy();return np.c_[v,v>=u]
    if scheme=='maximum':
        v=df.maximum.to_numpy();return np.c_[v,v>=u]
    if scheme=='two_snapshots':
        a=df.snapshot_2a.to_numpy();b=df.snapshot_2b.to_numpy();q=df[f'snapshot2_count_{k}'].to_numpy()/2
        return np.c_[a,b,np.maximum(a,b),q,q>0]
    if scheme=='maximum_count':
        m=df.maximum.to_numpy();q=df[f'count_{k}'].to_numpy()/df.n.to_numpy()
        return np.c_[m,q,m>=u]
    raise ValueError(scheme)


def std_train_test(a,b):
    mu=a.mean(0);sd=a.std(0);sd[sd<1e-8]=1
    return (a-mu)/sd,(b-mu)/sd


def fit_offset_logistic(X,y,offset,l2=1.0):
    """Penalized logistic regression with a fixed panel-specific offset."""
    X=np.c_[np.ones(len(X)),X]
    def fun(beta):
        eta=offset+X@beta
        nll=np.sum(np.logaddexp(0,eta)-y*eta)+.5*l2*np.dot(beta[1:],beta[1:])
        grad=X.T@(expit(eta)-y);grad[1:]+=l2*beta[1:]
        return float(nll),grad
    res=minimize(lambda b:fun(b),np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12})
    if not res.success:raise RuntimeError(res.message)
    return res.x


def predict_offset(X,offset,beta):return expit(offset+np.c_[np.ones(len(X)),X]@beta)


def metrics(y,p,alert_fraction=.01):
    p=np.clip(np.asarray(p),1e-12,1-1e-12);y=np.asarray(y,int);nalert=max(1,int(np.ceil(alert_fraction*len(y))))
    top=np.argpartition(p,-nalert)[-nalert:];pos=int(y.sum());tp=int(y[top].sum())
    return {'n':len(y),'positives':pos,'prevalence':y.mean(),'mean_log_score':-log_loss(y,p,labels=[0,1]),
      'brier':brier_score_loss(y,p),'roc_auc':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
      'average_precision':average_precision_score(y,p) if pos else np.nan,'alert_fraction':alert_fraction,
      'alerts':nalert,'alert_recall':tp/pos if pos else np.nan,'alert_precision':tp/nalert}


def temporal_dataset(blocks,u):
    k=int(round(100*u));tr_parts=[];te_parts=[];counts={}
    for vm,g in blocks.groupby('vm_id',sort=False):
        g=g.sort_values('window_id').reset_index(drop=True)
        if len(g)<50:continue
        cand=np.where(np.diff(g.window_id.to_numpy())==1)[0]
        cut=int(.7*len(g));a=cand[cand+1<cut];b=cand[cand+1>=cut]
        if len(a)<20 or len(b)<5:continue
        trp=g.iloc[a].copy(); trp['target']=g.iloc[a+1][f'event_{k}'].to_numpy(int)
        tep=g.iloc[b].copy(); tep['target']=g.iloc[b+1][f'event_{k}'].to_numpy(int)
        yy=trp.target.to_numpy(int);counts[vm]=(int(yy.sum()),len(yy))
        tr_parts.append(trp);te_parts.append(tep)
    tr=pd.concat(tr_parts,ignore_index=True);te=pd.concat(te_parts,ignore_index=True)
    yt=tr.pop('target').to_numpy(int);ye=te.pop('target').to_numpy(int)
    return tr,yt,tr.vm_id.to_numpy(int),te,ye,te.vm_id.to_numpy(int),counts

def evaluate_temporal(blocks,u=MAIN_THRESHOLD):
    tr,yt,vmt,te,ye,vme,counts=temporal_dataset(blocks,u)
    pbase=np.array([(counts[v][0]+.5)/(counts[v][1]+1) for v in vme])
    offtr=np.array([logit((counts[v][0]+.5)/(counts[v][1]+1)) for v in vmt])
    offte=logit(np.clip(pbase,1e-8,1-1e-8))
    rows=[{'validation':'temporal','scheme':'smoothed_panel_rate','budget_scalars':0,**metrics(ye,pbase)}]
    pred=pd.DataFrame({'vm_id':vme,'y':ye,'smoothed_panel_rate':pbase})
    budgets={'snapshot':1,'mean':1,'p95':1,'maximum':1,'two_snapshots':2,'maximum_count':2}
    for s in budgets:
        a,b=std_train_test(features(tr,s,u),features(te,s,u));beta=fit_offset_logistic(a,yt,offtr,l2=1.0);p=predict_offset(b,offte,beta)
        rows.append({'validation':'temporal','scheme':s,'budget_scalars':budgets[s],**metrics(ye,p)})
        pred[s]=p
    return pd.DataFrame(rows),pred


def evaluate_panel_disjoint(blocks,u=MAIN_THRESHOLD):
    k=int(round(100*u));pan=np.array(sorted(blocks.vm_id.unique()));RNG.shuffle(pan);cut=int(.7*len(pan));train=set(pan[:cut]);test=set(pan[cut:])
    parts=[]
    for vm,g in blocks.groupby('vm_id',sort=False):
        g=g.sort_values('window_id').reset_index(drop=True);cand=np.where(np.diff(g.window_id.to_numpy())==1)[0]
        if len(cand)==0:continue
        p=g.iloc[cand].copy();p['target']=g.iloc[cand+1][f'event_{k}'].to_numpy(int);parts.append(p)
    d=pd.concat(parts,ignore_index=True);tr=d[d.vm_id.isin(train)].reset_index(drop=True);te=d[d.vm_id.isin(test)].reset_index(drop=True)
    yt=tr.pop('target').to_numpy(int);ye=te.pop('target').to_numpy(int);base=(yt.sum()+.5)/(len(yt)+1);offt=np.full(len(yt),logit(base));offe=np.full(len(ye),logit(base))
    out=[]
    for s in ['snapshot','maximum','two_snapshots','maximum_count']:
        a=np.c_[features(tr,s,u),np.log2(np.maximum(tr.cores.to_numpy(),1))];b=np.c_[features(te,s,u),np.log2(np.maximum(te.cores.to_numpy(),1))]
        a,b=std_train_test(a,b);beta=fit_offset_logistic(a,yt,offt,l2=1.0);p=predict_offset(b,offe,beta)
        out.append({'validation':'panel_disjoint','scheme':s,'budget_scalars':1 if s in ['snapshot','maximum'] else 2,**metrics(ye,p)})
    return pd.DataFrame(out),sorted(train),sorted(test)

def perpanel(pred,left,right):
    rr=[]
    for vm,g in pred.groupby('vm_id'):
        y=g.y.to_numpy();pl=np.clip(g[left].to_numpy(),1e-12,1-1e-12);pr=np.clip(g[right].to_numpy(),1e-12,1-1e-12)
        sl=np.mean(y*np.log(pl)+(1-y)*np.log(1-pl));sr=np.mean(y*np.log(pr)+(1-y)*np.log(1-pr))
        rr.append({'vm_id':vm,'test_n':len(g),'test_positives':int(y.sum()),'left':left,'right':right,'gain_left_minus_right':sl-sr})
    return pd.DataFrame(rr)


def bootstrap(pred,left,right,reps=1000):
    stats=[]
    for vm,g in pred.groupby('vm_id'):
        y=g.y.to_numpy();pl=np.clip(g[left].to_numpy(),1e-12,1-1e-12);pr=np.clip(g[right].to_numpy(),1e-12,1-1e-12)
        stats.append([len(g),np.sum(y*np.log(pl)+(1-y)*np.log(1-pl)),np.sum(y*np.log(pr)+(1-y)*np.log(1-pr)),np.sum((y-pl)**2),np.sum((y-pr)**2)])
    a=np.asarray(stats);out=[]
    for r in range(reps):
        q=a[RNG.integers(0,len(a),len(a))].sum(0);out.append({'replicate':r,'comparison':f'{left}_vs_{right}','log_score_difference':(q[1]-q[2])/q[0],'brier_difference':(q[3]-q[4])/q[0]})
    return pd.DataFrame(out)


def figures(sampling,forecast,persist,pp):
    one=sampling[(sampling.budget_scalars==1)&sampling.scheme.isin(['one fixed snapshot','hourly mean','hourly 95th percentile','hourly maximum'])]
    fig,ax=plt.subplots(figsize=(7.2,4.4));schemes=list(one.scheme.unique());x=np.arange(4);w=.19
    for i,s in enumerate(schemes):
        g=one[one.scheme==s].sort_values('threshold');ax.bar(x+(i-1.5)*w,g.recall,w,label=s)
    ax.set_xticks(x,[f'{int(100*u)}%' for u in THRESHOLDS]);ax.set_ylim(0,1.05);ax.set_ylabel('Recall of windows containing an event');ax.set_xlabel('CPU-utilization threshold');ax.legend(ncol=2,fontsize=8);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIGURES/'Fig1_equal_budget_detection.pdf');plt.close(fig)
    g=forecast[(forecast.validation=='temporal')&forecast.scheme.isin(['snapshot','mean','p95','maximum','two_snapshots','maximum_count'])].set_index('scheme').loc[['snapshot','mean','p95','maximum','two_snapshots','maximum_count']]
    fig,ax=plt.subplots(figsize=(7.3,4.4));ax.bar(np.arange(6),g.average_precision);ax.set_xticks(np.arange(6),['Snapshot','Mean','P95','Maximum','2 snapshots','Maximum+count'],rotation=22,ha='right');ax.set_ylabel('Average precision for next-hour >=90% event');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIGURES/'Fig2_forecasting_average_precision.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4.3));ax.plot(persist.threshold*100,persist.p_event_given_no_event,marker='o',label='after no event');ax.plot(persist.threshold*100,persist.p_event_given_event,marker='o',label='after event');ax.set_xlabel('CPU-utilization threshold (%)');ax.set_ylabel('Next-hour event probability');ax.set_ylim(0,1);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIGURES/'Fig3_saturation_persistence.pdf');plt.close(fig)
    active=pp[pp.test_positives>0]
    fig,ax=plt.subplots(figsize=(6.8,4.3));v=np.sort(active.gain_left_minus_right);ax.plot(np.arange(1,len(v)+1),v);ax.axhline(0,ls='--',lw=1);ax.set_xlabel('Event-active VM panels ordered by log-score gain');ax.set_ylabel('Maximum+count minus two-snapshot log score');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIGURES/'Fig4_panel_forecast_gains.pdf');plt.close(fig)


def main():
    digest=sha256(RAW_ZIP)
    if digest!=EXPECTED_SHA256:raise RuntimeError('checksum mismatch')
    hourly=[];clean=[];offset_num={(u,j):0 for u in THRESHOLDS for j in range(12)};offset_den={u:0 for u in THRESHOLDS}
    win={(m,u):{'events':0,'fixed':0,'random':0.,'p95':0} for m in [30,60,120] for u in THRESHOLDS}
    with zipfile.ZipFile(RAW_ZIP) as z:
        names=sorted(n for n in z.namelist() if n.endswith('.csv'))
        for ii,name in enumerate(names):
            vm,t,x,c,raw,rem=read_clean_series(z,name);h,a=make_hourly_frame(vm,t,x,c)
            if len(h):
                hourly.append(h)
                for u in THRESHOLDS:
                    ev=a.max(1)>=u;offset_den[u]+=int(ev.sum())
                    for j in range(12):offset_num[(u,j)]+=int(np.sum(a[ev,j]>=u))
            for m in [30,60,120]:
                wid,b,_=exact_windows(t,x,c,m)
                if len(wid)==0:continue
                for u in THRESHOLDS:
                    ev=b.max(1)>=u
                    if not np.any(ev):continue
                    q=(b[ev]>=u).sum(1);d=win[(m,u)];d['events']+=int(ev.sum());d['fixed']+=int(np.sum(b[ev,0]>=u));d['random']+=float(np.sum(q/(m//5)));d['p95']+=int(np.sum(np.quantile(b[ev],.95,axis=1)>=u))
            clean.append({'vm_id':vm,'raw_valid_rows':raw,'removed_repeated_payload_rows':rem,'clean_rows':len(x),'exact_wall_rows':int(np.isclose(x,1,atol=1e-12).sum()),'complete_hourly_windows':len(h)})
            if (ii+1)%250==0:print(f'loaded {ii+1}/{len(names)}',flush=True)
    blocks=pd.concat(hourly,ignore_index=True);clean=pd.DataFrame(clean);clean.to_csv(RESULTS/'cleaning_audit.csv',index=False)
    # Sampling summaries.
    sr=[];dur=[]
    for u in THRESHOLDS:
        k=int(100*u);ev=blocks[f'event_{k}'].to_numpy(bool);q=blocks[f'count_{k}'].to_numpy();ne=int(ev.sum());m=12
        r1=float(np.mean(blocks.loc[ev,'snapshot_1']>=u));rr1=float(np.sum(q[ev]/m)/ne);miss2=((m-q[ev])*(m-q[ev]-1))/(m*(m-1));rr2=float(np.mean(1-miss2));r2=float(np.mean(blocks.loc[ev,f'snapshot2_count_{k}']>0))
        for s,b,r in [('one fixed snapshot',1,r1),('one random snapshot (expected)',1,rr1),('hourly mean',1,float(np.mean(blocks.loc[ev,'mean']>=u))),('hourly 95th percentile',1,float(np.mean(blocks.loc[ev,'p95']>=u))),('hourly maximum',1,1.),('two fixed snapshots',2,r2),('two random snapshots (expected)',2,rr2),('maximum plus exceedance count',2,1.)]:sr.append({'threshold':u,'scheme':s,'budget_scalars':b,'event_hours':ne,'recall':r})
        true=q/m;one=blocks[f'snapshot1_event_{k}'].to_numpy();two=blocks[f'snapshot2_count_{k}'].to_numpy()/2
        dur += [{'threshold':u,'scheme':'one fixed snapshot','duration_mae':np.mean(abs(one-true))},{'threshold':u,'scheme':'two fixed snapshots','duration_mae':np.mean(abs(two-true))},{'threshold':u,'scheme':'maximum plus exceedance count','duration_mae':0.}]
    sampling=pd.DataFrame(sr);sampling.to_csv(RESULTS/'sampling_equal_budget.csv',index=False);pd.DataFrame(dur).to_csv(RESULTS/'duration_estimation.csv',index=False)
    pd.DataFrame([{'threshold':u,'offset':j,'event_hours':offset_den[u],'recall':offset_num[(u,j)]/offset_den[u]} for u in THRESHOLDS for j in range(12)]).to_csv(RESULTS/'snapshot_offset_sensitivity.csv',index=False)
    pd.DataFrame([{'window_minutes':m,'threshold':u,'event_windows':d['events'],'fixed_snapshot_recall':d['fixed']/d['events'],'random_snapshot_expected_recall':d['random']/d['events'],'p95_recall':d['p95']/d['events'],'maximum_recall':1.} for (m,u),d in win.items()]).to_csv(RESULTS/'window_sensitivity.csv',index=False)
    # Persistence.
    prs=[]
    for u in THRESHOLDS:
        k=int(100*u);n00=n01=n10=n11=0
        for _,g in blocks.groupby('vm_id'):
            g=g.sort_values('window_id');adj=np.diff(g.window_id.to_numpy())==1;a=g[f'event_{k}'].to_numpy()[:-1][adj];b=g[f'event_{k}'].to_numpy()[1:][adj]
            n00+=int(np.sum((a==0)&(b==0)));n01+=int(np.sum((a==0)&(b==1)));n10+=int(np.sum((a==1)&(b==0)));n11+=int(np.sum((a==1)&(b==1)))
        p0=n01/(n00+n01);p1=n11/(n10+n11);prs.append({'threshold':u,'n00':n00,'n01':n01,'n10':n10,'n11':n11,'p_event_given_no_event':p0,'p_event_given_event':p1,'risk_ratio':p1/p0})
    persist=pd.DataFrame(prs);persist.to_csv(RESULTS/'event_persistence.csv',index=False)
    blocks.to_pickle(ROOT/'build/hourly_blocks.pkl')
    print('saved hourly blocks', len(blocks), flush=True)
    if __import__('os').environ.get('SATCAP_PREP_ONLY')=='1': return
    ft,pred=evaluate_temporal(blocks);fc,trp,tep=evaluate_panel_disjoint(blocks);forecast=pd.concat([ft,fc],ignore_index=True);forecast.to_csv(RESULTS/'forecast_metrics.csv',index=False);(RESULTS/'panel_disjoint_split.json').write_text(json.dumps({'seed':20260727,'train_panels':trp,'test_panels':tep},indent=2))
    pp1=perpanel(pred,'maximum','snapshot');pp1['comparison']='maximum_vs_snapshot';pp2=perpanel(pred,'maximum_count','two_snapshots');pp2['comparison']='maximum_count_vs_two_snapshots';pp=pd.concat([pp1,pp2]);pp.to_csv(RESULTS/'per_panel_forecast_comparisons.csv',index=False)
    b1=bootstrap(pred,'maximum','snapshot');b2=bootstrap(pred,'maximum_count','two_snapshots');boot=pd.concat([b1,b2]);boot.to_csv(RESULTS/'cluster_bootstrap.csv',index=False)
    cs=[]
    for comp,d,b in [('maximum_vs_snapshot',pp1,b1),('maximum_count_vs_two_snapshots',pp2,b2)]:
        a=d[d.test_positives>0];w=int(np.sum(a.gain_left_minus_right>0));n=len(a);cs.append({'comparison':comp,'active_panels':n,'wins':w,'win_fraction':w/n,'median_panel_log_score_gain':a.gain_left_minus_right.median(),'two_sided_sign_test_p':binomtest(w,n,.5).pvalue,'bootstrap_log_score_difference_low':np.quantile(b.log_score_difference,.025),'bootstrap_log_score_difference_high':np.quantile(b.log_score_difference,.975),'bootstrap_brier_difference_low':np.quantile(b.brier_difference,.025),'bootstrap_brier_difference_high':np.quantile(b.brier_difference,.975)})
    comps=pd.DataFrame(cs);comps.to_csv(RESULTS/'comparison_summary.csv',index=False)
    conc=[]
    for u in THRESHOLDS:
        k=int(100*u);w=blocks.groupby('vm_id')[f'count_{k}'].sum().to_numpy(float);tot=w.sum();conc.append({'threshold':u,'total_exceedance_samples':int(tot),'contributing_panels':int(np.sum(w>0)),'kish_effective_panels':tot*tot/np.sum(w*w),'largest_panel_share':w.max()/tot})
    pd.DataFrame(conc).to_csv(RESULTS/'panel_event_concentration.csv',index=False)
    figures(sampling,forecast,persist,pp2)
    summary={'raw_archive_sha256':digest,'vm_panels':clean.vm_id.nunique(),'raw_valid_rows':int(clean.raw_valid_rows.sum()),'removed_repeated_payload_rows':int(clean.removed_repeated_payload_rows.sum()),'clean_rows':int(clean.clean_rows.sum()),'complete_hourly_windows':len(blocks),'panels_with_complete_hours':blocks.vm_id.nunique(),'exact_wall_rows_after_cleaning':int(clean.exact_wall_rows.sum()),'sampling_at_90':sampling[sampling.threshold==.9].to_dict('records'),'forecast_temporal':forecast[forecast.validation=='temporal'].to_dict('records'),'comparison_summary':comps.to_dict('records')}
    (RESULTS/'SATCAP_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['sampling_at_90','forecast_temporal']},indent=2))
if __name__=='__main__':main()
