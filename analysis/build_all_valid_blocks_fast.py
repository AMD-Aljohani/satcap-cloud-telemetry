#!/usr/bin/env python3
from pathlib import Path
import os
import importlib.util,sys
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('satcap_analysis',HERE/'satcap_analysis.py');sa=importlib.util.module_from_spec(spec);sys.modules['satcap_analysis']=sa;spec.loader.exec_module(sa)
RAW=Path(os.environ.get('SATCAP_RAW_DIR', str(HERE.parent/'raw_bitbrains')));ROOT=HERE.parent;frames=[]
for i,p in enumerate(sorted(RAW.rglob('*.csv'))):
 vm=int(p.stem);df=pd.read_csv(p,sep=';',engine='c');t=pd.to_numeric(df.iloc[:,0],errors='coerce').to_numpy();c=pd.to_numeric(df.iloc[:,1],errors='coerce').to_numpy();x=pd.to_numeric(df.iloc[:,4],errors='coerce').to_numpy()/100
 ok=np.isfinite(t)&np.isfinite(c)&np.isfinite(x)&(x>=0)&(x<=1.000001);t=t[ok].astype(np.int64);c=c[ok];x=np.clip(x[ok],0,1);o=np.argsort(t,kind='stable');t=t[o];x=x[o];c=c[o];h,_=sa.make_hourly_frame(vm,t,x,c)
 if len(h):frames.append(h)
 if (i+1)%250==0:print(i+1,flush=True)
blocks=pd.concat(frames,ignore_index=True);blocks.to_pickle(ROOT/'build/hourly_blocks_all_valid.pkl');print(len(blocks))
