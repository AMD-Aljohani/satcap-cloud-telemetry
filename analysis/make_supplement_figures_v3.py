#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors={'no_action':'#7F7F7F','reactive':'#009E73','snapshot':'#0072B2','satcap':'#D55E00'}
labels={'no_action':'No action','reactive':'Reactive','snapshot':'Snapshot','satcap':'SATCAP'}
df=pd.read_csv(RES/'operational_robustness_v3_cost_sensitivity.csv')
df=df[df.provisioning_delay_epochs==0]
g=df.groupby(['policy','miss_to_action_cost_ratio']).normalized_cost.agg(mean='mean',lo=lambda x:x.quantile(.025),hi=lambda x:x.quantile(.975)).reset_index()
fig,ax=plt.subplots(figsize=(6.8,3.8))
for policy in ['no_action','reactive','snapshot','satcap']:
 h=g[g.policy==policy].sort_values('miss_to_action_cost_ratio')
 ax.plot(h.miss_to_action_cost_ratio,h['mean'],marker='o',label=labels[policy],color=colors[policy])
 ax.fill_between(h.miss_to_action_cost_ratio.to_numpy(),h.lo.to_numpy(),h.hi.to_numpy(),alpha=.10,color=colors[policy],linewidth=0)
ax.set_xscale('log'); ax.set_xlabel(r'Missed-event-to-action cost ratio, $\lambda$'); ax.set_ylabel('Normalized decision cost'); ax.grid(which='both',color='.90',linewidth=.6); ax.legend(frameon=False,ncol=2)
fig.tight_layout(); fig.savefig(OUT/'FigS2_cost_sensitivity_v3.pdf',bbox_inches='tight')
print('wrote',OUT/'FigS2_cost_sensitivity_v3.pdf')
