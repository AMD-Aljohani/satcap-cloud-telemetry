#!/usr/bin/env python3
"""Compute an all-transition two-state persistence diagnostic from shipped counts.

This is descriptive and is not a replacement for a held-out feature ablation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

root=Path(__file__).resolve().parents[1]
counts=pd.read_csv(root/'results'/'preprocessing_persistence_sensitivity.csv')
rows=[]
for _,r in counts.iterrows():
    n00,n01,n10,n11=[int(r[x]) for x in ['n00','n01','n10','n11']]
    p0=n01/(n00+n01); p1=n11/(n10+n11)
    y=np.r_[np.zeros(n00,dtype=np.uint8),np.ones(n01,dtype=np.uint8),np.zeros(n10,dtype=np.uint8),np.ones(n11,dtype=np.uint8)]
    p=np.r_[np.full(n00,p0),np.full(n01,p0),np.full(n10,p1),np.full(n11,p1)]
    rows.append({
        'policy':r['policy'],'n_transitions':len(y),'events':int(y.sum()),
        'p_next_given_no_event':p0,'p_next_given_event':p1,
        'average_precision':average_precision_score(y,p),
        'brier':brier_score_loss(y,p),
        'mean_log_score':-log_loss(y,p),
        'roc_auc':roc_auc_score(y,p),
        'interpretation':'descriptive all-transition Markov baseline; not a temporal holdout estimate'
    })
out=pd.DataFrame(rows)
out.to_csv(root/'results'/'primary_persistence_diagnostic_v3.csv',index=False)
print(out.to_string(index=False))
