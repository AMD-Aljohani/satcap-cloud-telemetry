#!/usr/bin/env python3
from pathlib import Path
import importlib.util,sys
import numpy as np,pandas as pd
from scipy.stats import binomtest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('satcap_analysis',HERE/'satcap_analysis.py');sa=importlib.util.module_from_spec(spec);sys.modules['satcap_analysis']=sa;spec.loader.exec_module(sa)
ROOT=HERE.parent; RES=ROOT/'results'
blocks=pd.read_pickle(ROOT/'build/hourly_blocks_all_valid.pkl')
_,pred=sa.evaluate_temporal(blocks,.9)
comparisons=[('maximum','snapshot'),('maximum_count','two_snapshots')]
pps=[];boots=[];summ=[]
for left,right in comparisons:
    pp=sa.perpanel(pred,left,right); pp['comparison']=f'{left}_vs_{right}'; pps.append(pp)
    boot=sa.bootstrap(pred,left,right,reps=2000); boots.append(boot)
    active=pp[pp.test_positives>0]
    wins=int((active.gain_left_minus_right>0).sum()); n=len(active)
    summ.append({
        'comparison':f'{left}_vs_{right}', 'active_panels':n,'wins':wins,
        'win_fraction':wins/n,'median_panel_log_score_gain':active.gain_left_minus_right.median(),
        'two_sided_sign_test_p':binomtest(wins,n,.5).pvalue,
        'bootstrap_log_score_difference_low':np.quantile(boot.log_score_difference,.025),
        'bootstrap_log_score_difference_high':np.quantile(boot.log_score_difference,.975),
        'bootstrap_brier_difference_low':np.quantile(boot.brier_difference,.025),
        'bootstrap_brier_difference_high':np.quantile(boot.brier_difference,.975),
    })
pd.concat(pps,ignore_index=True).to_csv(RES/'all_valid_per_panel_comparisons.csv',index=False)
pd.concat(boots,ignore_index=True).to_csv(RES/'all_valid_cluster_bootstrap.csv',index=False)
pd.DataFrame(summ).to_csv(RES/'all_valid_comparison_summary.csv',index=False)
print(pd.DataFrame(summ).to_string(index=False))
