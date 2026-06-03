#!/usr/bin/env python3
"""
Quantitative comparison against reference stopping-power codes.

Two modes:
  1) Analytic Bethe/ICRU-49 reference (built in, no external code) — faithful at high energy
     (protons E >= ~10 MeV/u), where it reproduces PSTAR/experiment to ~3%.
  2) External SRIM/PSTAR/ASTAR: drop a CSV of reference predictions and this script
     computes the code's MAPE on the same test points and prints the comparison row.

External CSV format (one row per test point you evaluated in SRIM/PSTAR):
     projectile,target,energy_MeV_u,S_ref_MeV_per_mg_cm2

Requires outputs/features_full.parquet from pipeline.py.
"""
import sys, numpy as np, pandas as pd
Me, Mp = 0.511, 938.272
def bethe_mass(z, E, Z2, A2, I_eV):           # MeV cm^2 / g
    g = 1 + E/Mp; b2 = 1 - 1/g**2; I = I_eV*1e-6
    return 0.307075*(Z2/A2)*(z**2/b2)*(np.log(2*Me*b2*g**2/I) - b2)
mape = lambda pred, obs: float(np.mean(np.abs(pred-obs)/obs)*100)

df = pd.read_parquet("outputs/features_full.parquet")
te = df[df.split == "test"].copy()

# Mode 1: analytic reference (protons, high energy)
zmap = {"H":1, "He":2}
for proj, z, thr in [("H",1,10.0), ("He",2,10.0)]:
    s = te[(te.projectile_name==proj) & (te.E>=thr)]
    if len(s) < 15: continue
    Sref = bethe_mass(z, s.E.values, s.Z2.values, s.A2.values, s.I2.values)/1000.0
    print(f"[analytic] {proj} E>={thr} MeV/u  n={len(s)}  Bethe/ICRU-49 MAPE={mape(Sref, s.S.values):.1f}%")

# Mode 2: external SRIM/PSTAR CSV (optional)
if len(sys.argv) > 1:
    ref = pd.read_csv(sys.argv[1])
    key = ["projectile","target","energy_MeV_u"]
    te2 = te.rename(columns={"projectile_name":"projectile","target_name":"target","E":"energy_MeV_u"})
    m = te2.merge(ref, on=key, how="inner")
    if len(m):
        print(f"[external] {sys.argv[1]}  n={len(m)}  code MAPE={mape(m.S_ref_MeV_per_mg_cm2.values, m.S.values):.1f}%")
    else:
        print("[external] no matching (projectile,target,energy) rows found; check formatting/units.")
