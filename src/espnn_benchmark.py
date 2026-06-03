#!/usr/bin/env python3
"""
Optional independent benchmark vs ESPNN (atomic targets).
ESPNN pins old deps, so install with --no-deps and patch the RNG seed for NumPy 2.x:

    pip install ESPNN==1.0.1 --no-deps
    pip install pyvalem==2.5.7 --no-deps

Requires outputs/features_full.parquet and outputs/bnn_weights.pt from pipeline.py.
"""
import os, glob, numpy as np, pandas as pd, random
from scipy.interpolate import interp1d

OUT="outputs"
# patch seeds (ESPNN passes numpy types that NumPy 2.x / Py3.12 reject)
_rs=random.seed
random.seed=lambda a=None,*x,**k:(_rs(a) if a is None else _rs(int(a)))
_ns=np.random.seed
def _safe(a=None,*x,**k):
    try: return _ns(a)
    except Exception: return _ns(int(a)%(2**32))
np.random.seed=_safe
import ESPNN

FEAT=["Z1","A1","v","v_red","Z1_16","gam","logE","reg","Z2","A2","I2","Z2_13","Z1Z2","lss","bbb","bt","uca"]
df=pd.read_parquet(f"{OUT}/features_full.parquet")
te=df[(df.split=="test")&df.target_is_element].copy()
te["tgt"]=te.target_name.replace({"Carbon":"C","Graphite":"C","Diamond":"C","Vitreous Carbon":"C"})
te=te[te.tgt.str.match(r'^[A-Z][a-z]?$')]
counts=te.groupby(["projectile_name","tgt"]).size().sort_values(ascending=False)
sub=te[te.set_index(["projectile_name","tgt"]).index.isin(set(counts.head(40).index))].copy()

rows=[]
for (proj,tgt),g in sub.groupby(["projectile_name","tgt"]):
    emin=max(g.E.min()*0.8,1e-4); emax=min(g.E.max()*1.25,1e3)
    if emax<=emin: continue
    fp=f"{OUT}/{proj}{tgt}_prediction.dat"
    try:
        if os.path.exists(fp): os.remove(fp)
        ESPNN.run_NN(projectile=proj,target=tgt,emin=float(emin),emax=float(emax),npoints=40,outdir=OUT,plot=False)
        d=pd.read_csv(fp,sep=None,engine="python"); eg=d.iloc[:,0].values; sg=d.iloc[:,1].values; mk=(sg>0)&(eg>0)
        itp=interp1d(np.log10(eg[mk]),np.log10(sg[mk]),bounds_error=False,fill_value=np.nan)
        for _,r in g.iterrows():
            se=10**itp(np.log10(r.E))
            if np.isfinite(se): rows.append((r.S,se))
    except Exception: pass
arr=np.array(rows)
print(f"points compared: {len(arr)}  ESPNN MAPE={np.mean(np.abs(arr[:,1]-arr[:,0])/arr[:,0])*100:.1f}%")
