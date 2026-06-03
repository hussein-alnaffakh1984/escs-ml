#!/usr/bin/env python3
"""
ESCS-ML — end-to-end reproducible pipeline.
Run top-to-bottom (e.g. on a GPU notebook). Reproduces every number in the paper.

Inputs (place in ./data or /kaggle/input):
  StoppingPower.csv            (IAEA-derived; NOT redistributed here — see data/README.md)
  StoppingPower_refs.csv
  target_composition_table.csv (provided in this repo)

Outputs (./outputs):
  features_full.parquet, results_*.json, *_weights.pt, ngboost_model.pkl, figures
"""
import os, glob, re, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import DBSCAN
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import norm

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
def find(name):
    for base in ("data", "/kaggle/input"):
        hits = glob.glob(f"{base}/**/{name}", recursive=True)
        if hits: return hits[0]
    raise FileNotFoundError(name)

FEAT=["Z1","A1","v","v_red","Z1_16","gam","logE","reg","Z2","A2","I2","Z2_13","Z1Z2","lss","bbb","bt","uca"]
Zmap={"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,"Al":13,
 "Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,
 "Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,"Y":39,"Zr":40,
 "Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,"Ag":47,"Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,
 "La":57,"Ce":58,"Nd":60,"Pm":61,"Sm":62,"W":74,"Au":79,"Pb":82,"Bi":83,"Th":90,"U":92}

# ---------------- STEP 1: data -> features -> split ----------------
def cE(e,u,A):
    if u=="MeV/u": return e
    if u=="keV/u": return e/1e3
    if u=="MeV":   return e/A if A>0 else np.nan
    if u=="keV":   return (e/1e3)/A if A>0 else np.nan
    return np.nan
def pe(x):
    if pd.isna(x): return np.nan
    s=str(x).strip()
    if re.fullmatch(r"\d+(\.\d+)?",s): return float(s)
    m=re.fullmatch(r"(\d+(\.\d+)?)\s*-\s*(\d+(\.\d+)?)",s)
    return (float(m.group(1))+float(m.group(3)))/2 if m else np.nan

df=pd.read_csv(find("StoppingPower.csv"),low_memory=False)
comp=pd.read_csv(find("target_composition_table.csv")).set_index("target")
df["E"]=[cE(e,u,A) for e,u,A in zip(df.energy,df.energy_unit,df.ion_isotope)]
df["S"]=pd.to_numeric(df.stopping_power_converted,errors="coerce")
df["pe"]=df.percentage_error.apply(pe).fillna(5.0)
df["Z1"]=df.projectile_name.map(Zmap); df["A1"]=df.ion_isotope
df=df[(df.S>0)&df.E.notna()&df.Z1.notna()].copy()
df["logS"]=np.log10(df.S); df["logE"]=np.log10(df.E)
df["Z2"]=df.target_name.map(comp.Z2); df["A2"]=df.target_name.map(comp.A2); df["I2"]=df.target_name.map(comp.I2_eV)
df=df[df.Z2.notna()&df.A2.notna()&df.I2.notna()].copy()
df["target_is_element"]=df.target_name.map(comp.method).isin(["element","element_alias"])
df["v"]=6.348*np.sqrt(df.E); df["v_red"]=df.v**2/(df.v**2+df.Z1**(2/3)); df["Z1_16"]=df.Z1**(1/6)
df["gam"]=df.Z1*(1-np.exp(-df.v/df.Z1**(2/3))); df["reg"]=df.v**2/df.Z1**(2/3); df["Z2_13"]=df.Z2**(1/3)
df["Z1Z2"]=df.Z1*df.Z2; df["lss"]=df.Z1**(2/3)/df.v**2
df["bbb"]=(df.Z1**2/df.v**2)*np.log(np.clip(2*df.v**2/(df.I2/27.2114),1.01,None))
df["bt"]=np.sqrt(df.Z1*df.Z2); df["uca"]=df.v/(df.Z1**(2/3)+df.Z2**(2/3))
df["outlier"]=False
for (i,t),idx in df.groupby(["projectile_name","target_name"]).groups.items():
    sub=df.loc[idx]
    if len(sub)<5: continue
    X=np.column_stack([(sub.logE-sub.logE.mean())/(sub.logE.std()+1e-9),(sub.logS-sub.logS.mean())/(sub.logS.std()+1e-9)])
    df.loc[sub.index[DBSCAN(eps=0.5,min_samples=3).fit_predict(X)==-1],"outlier"]=True
rng=np.random.default_rng(42); df["edec"]=np.floor(df.logE).astype(int); df["strat"]=df.projectile_name+"_"+df.edec.astype(str)
df["split"]="train"
for _,idx in df.groupby("strat").groups.items():
    idx=np.array(idx); rng.shuffle(idx); n=len(idx)
    df.loc[idx[int(.85*n):],"split"]="test"; df.loc[idx[int(.70*n):int(.85*n)],"split"]="val"
df.to_parquet(f"{OUT}/features_full.parquet")
print(f"[STEP1] {len(df)} rows | elemental {int(df.target_is_element.sum())} | compound {int((~df.target_is_element).sum())}")

tr=df[(df.split=="train")&(~df.outlier)]; va=df[df.split=="val"]; te=df[df.split=="test"]
yt=te.logS.values; L=[.5,.7,.8,.9,.95,.99]
def mape(p,y): return float(np.mean(np.abs(10**p-10**y)/10**y)*100)

# ---------------- STEP 2: GBM baseline + deep ensemble ----------------
w=1/np.clip(tr.pe.values/100,1e-3,None)**2
gbm=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.08,l2_regularization=1.0,early_stopping=True,random_state=0).fit(tr[FEAT].values,tr.logS.values,sample_weight=w)
pg=gbm.predict(te[FEAT].values)
print(f"[STEP2] GBM R2={r2_score(yt,pg):.3f} MAPE={mape(pg,yt):.1f}%")
sc=StandardScaler().fit(tr[FEAT].values)
P=np.array([MLPRegressor((128,128),alpha=1e-4,learning_rate_init=2e-3,max_iter=200,early_stopping=True,random_state=s).fit(sc.transform(tr[FEAT].values),tr.logS.values).predict(sc.transform(te[FEAT].values)) for s in range(5)])
mu=P.mean(0); ep=P.std(0); al=(te.pe.values/100)/np.log(10); sig=np.sqrt(ep**2+al**2); z=(yt-mu)/sig
ens_ece=float(np.mean([abs(np.mean(np.abs(z)<=norm.ppf(0.5+l/2))-l) for l in L]))
print(f"[STEP2] Ensemble MAPE={mape(mu,yt):.1f}% ECE={ens_ece:.3f}")

# ---------------- STEP 3+4: variational BNN + temperature calibration ----------------
import torch, torch.nn as nn
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
lo=tr[FEAT].quantile(.005); hi=tr[FEAT].quantile(.995)
Wn=lambda D: D[FEAT].clip(lo,hi,axis=1).values
xs=StandardScaler().fit(Wn(tr)); ym,ysd=tr.logS.mean(),tr.logS.std()
TT=lambda D: torch.tensor(xs.transform(Wn(D)),dtype=torch.float32,device=dev)
Xtr=TT(tr); ytr=torch.tensor(((tr.logS-ym)/ysd).values,dtype=torch.float32,device=dev).view(-1,1); N=len(tr)
class BLin(nn.Module):
    def __init__(s,i,o,ps=1.0):
        super().__init__(); s.ps=ps
        s.wm=nn.Parameter(torch.randn(o,i)*0.05); s.wr=nn.Parameter(torch.full((o,i),-5.0))
        s.bm=nn.Parameter(torch.zeros(o)); s.br=nn.Parameter(torch.full((o,),-5.0))
    def forward(s,x):
        ws=torch.log1p(torch.exp(s.wr)); bs=torch.log1p(torch.exp(s.br))
        return x@(s.wm+ws*torch.randn_like(ws)).t()+(s.bm+bs*torch.randn_like(bs))
    def kl(s):
        f=lambda m,r:(np.log(s.ps)-torch.log(torch.log1p(torch.exp(r)))+(torch.log1p(torch.exp(r))**2+m**2)/(2*s.ps**2)-0.5).sum()
        return f(s.wm,s.wr)+f(s.bm,s.br)
class BNN(nn.Module):
    def __init__(s,d,h=128):
        super().__init__(); s.l1=BLin(d,h); s.l2=BLin(h,h); s.o=BLin(h,2)
    def forward(s,x):
        x=torch.relu(s.l1(x)); x=torch.relu(s.l2(x)); o=s.o(x); return o[:,:1],torch.clamp(o[:,1:2],-6,2)
    def kl(s): return s.l1.kl()+s.l2.kl()+s.o.kl()
net=BNN(len(FEAT)).to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
for ep in range(400):
    perm=torch.randperm(N,device=dev); klw=min(1.0,ep/60)
    for i in range(0,N,1024):
        idx=perm[i:i+1024]; m_,lv=net(Xtr[idx])
        loss=(0.5*torch.exp(-lv)*(ytr[idx]-m_)**2+0.5*lv).mean()+klw*net.kl()/N
        opt.zero_grad(); loss.backward(); opt.step()
torch.save(net.state_dict(),f"{OUT}/bnn_weights.pt")
def bnn_infer(D,T=200):
    X=TT(D); net.eval(); ms=[]; av=[]
    with torch.no_grad():
        for _ in range(T): m_,lv=net(X); ms.append(m_.cpu().numpy()[:,0]); av.append(np.exp(lv.cpu().numpy())[:,0])
    ms=np.array(ms); av=np.array(av); return ms.mean(0)*ysd+ym, np.sqrt(ms.var(0)+av.mean(0))*ysd
pv,sv=bnn_infer(va); pt,st=bnn_infer(te)
ece=lambda res,s,k:float(np.mean([abs(np.mean(np.abs(res/(s*k))<=norm.ppf(0.5+l/2))-l) for l in L]))
sstar=float(np.linspace(0.4,3,131)[np.argmin([ece(va.logS.values-pv,sv,k) for k in np.linspace(0.4,3,131)])])
print(f"[STEP3/4] BNN R2={r2_score(yt,pt):.3f} MAPE={mape(pt,yt):.1f}% ECE {ece(yt-pt,st,1.0):.3f}->{ece(yt-pt,st,sstar):.3f} (s*={sstar:.2f})")

# ---------------- STEP 10: NGBoost ----------------
try:
    from ngboost import NGBRegressor; from ngboost.distns import Normal
    import pickle
    ngb=NGBRegressor(Dist=Normal,n_estimators=500,learning_rate=0.04,minibatch_frac=0.5,verbose=False).fit(tr[FEAT].values,tr.logS.values)
    d2=ngb.pred_dist(te[FEAT].values); mt,stn=d2.loc,d2.scale
    ng_ece=float(np.mean([abs(np.mean(np.abs((yt-mt)/stn)<=norm.ppf(0.5+l/2))-l) for l in L]))
    pickle.dump(ngb,open(f"{OUT}/ngboost_model.pkl","wb"))
    print(f"[STEP10] NGBoost R2={r2_score(yt,mt):.3f} MAPE={mape(mt,yt):.1f}% native ECE={ng_ece:.3f}")
except ImportError:
    print("[STEP10] ngboost not installed; skipping")

# ---------------- STEP 5b: effective Z1 exponent alpha(E) ----------------
def feats(Z1,A1,E,Z2,A2,I2):
    v=6.348*np.sqrt(E)
    return pd.DataFrame({"Z1":Z1,"A1":A1,"v":v,"v_red":v**2/(v**2+Z1**(2/3)),"Z1_16":Z1**(1/6),
      "gam":Z1*(1-np.exp(-v/Z1**(2/3))),"logE":np.log10(E),"reg":v**2/Z1**(2/3),"Z2":Z2,"A2":A2,"I2":I2,
      "Z2_13":Z2**(1/3),"Z1Z2":Z1*Z2,"lss":Z1**(2/3)/v**2,
      "bbb":(Z1**2/v**2)*np.log(np.clip(2*v**2/(I2/27.2114),1.01,None)),"bt":np.sqrt(Z1*Z2),"uca":v/(Z1**(2/3)+Z2**(2/3))})
Z1g=np.arange(2,37); lZ=np.log10(Z1g); alpha=[]
for E in [0.01,0.03,0.1,0.3,1,3,10,30]:
    p,_=bnn_infer(feats(Z1g,2.0*Z1g,E,6,12.011,78.0)); alpha.append((E,round(float(np.polyfit(lZ,p,1)[0]),3)))
pd.DataFrame(alpha,columns=["E_MeV_u","alpha_eff"]).to_csv(f"{OUT}/alpha_vs_energy.csv",index=False)
print("[STEP5b] alpha(E):", alpha)
print("Done. Outputs in ./outputs")
