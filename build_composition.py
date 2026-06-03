#!/usr/bin/env python3
"""
Phase 1 dependency (built in-house) — TARGET COMPOSITION TABLE.

For every non-elemental target: resolve composition -> mean Z2, mean A2,
I2 (Bragg additivity from elemental I), density, n_e.
Every row carries method/source flags and validated_by_physics=False so the
physics team reviews rather than builds.

Resolution tiers:
  element_alias  : Carbon/Diamond/Graphite/... -> C ; D2 -> H ; lanthanides etc.
  formula        : parsed chemical formula (incl. parentheses)
  name_map       : organic/named compound -> molecular formula
  weight_frac    : alloys & NIST standard materials -> element mass fractions
  UNRESOLVED     : flagged for physics team (low-confidence / exotic)
"""
import pandas as pd, numpy as np, re
N_A = 6.02214076e23

# ---- atomic data: Z, standard atomic weight A, ICRU-37 mean excitation I[eV] ----
# I marked None where not confidently known -> Bloch fallback 10*Z, flagged.
ATOM = {
 "H":(1,1.008,19.2),"D":(1,2.014,19.2),"He":(2,4.003,41.8),"Li":(3,6.94,40.0),
 "Be":(4,9.012,63.7),"B":(5,10.81,76.0),"C":(6,12.011,78.0),"N":(7,14.007,82.0),
 "O":(8,15.999,95.0),"F":(9,18.998,115.0),"Ne":(10,20.18,137.0),"Na":(11,22.99,149.0),
 "Mg":(12,24.305,156.0),"Al":(13,26.982,166.0),"Si":(14,28.085,173.0),"P":(15,30.974,173.0),
 "S":(16,32.06,180.0),"Cl":(17,35.45,174.0),"Ar":(18,39.948,188.0),"K":(19,39.098,190.0),
 "Ca":(20,40.078,191.0),"Sc":(21,44.956,216.0),"Ti":(22,47.867,233.0),"V":(23,50.942,245.0),
 "Cr":(24,51.996,257.0),"Mn":(25,54.938,272.0),"Fe":(26,55.845,286.0),"Co":(27,58.933,297.0),
 "Ni":(28,58.693,311.0),"Cu":(29,63.546,322.0),"Zn":(30,65.38,330.0),"Ga":(31,69.723,334.0),
 "Ge":(32,72.63,350.0),"As":(33,74.922,347.0),"Se":(34,78.971,348.0),"Br":(35,79.904,343.0),
 "Kr":(36,83.798,352.0),"Rb":(37,85.468,363.0),"Sr":(38,87.62,366.0),"Y":(39,88.906,379.0),"Zr":(40,91.224,393.0),"Nb":(41,92.906,417.0),
 "Mo":(42,95.95,424.0),"Tc":(43,98.0,428.0),"Ru":(44,101.07,441.0),"Rh":(45,102.906,449.0),
 "Pd":(46,106.42,470.0),"Ag":(47,107.868,470.0),"Cd":(48,112.414,469.0),"In":(49,114.818,488.0),
 "Sn":(50,118.71,488.0),"Sb":(51,121.76,487.0),"Te":(52,127.6,485.0),"I":(53,126.904,491.0),
 "Xe":(54,131.293,482.0),"Cs":(55,132.905,488.0),"Ba":(56,137.327,491.0),"La":(57,138.905,501.0),
 "Ce":(58,140.116,523.0),"Pr":(59,140.908,535.0),"Nd":(60,144.242,546.0),"Pm":(61,145.0,560.0),"Sm":(62,150.36,574.0),
 "Gd":(64,157.25,591.0),"Tb":(65,158.925,614.0),"Dy":(66,162.5,628.0),"Ho":(67,164.93,650.0),
 "Er":(68,167.259,658.0),"Tm":(69,168.934,674.0),"Yb":(70,173.045,684.0),"Lu":(71,174.967,694.0),
 "Hf":(72,178.49,705.0),"Ta":(73,180.948,718.0),"W":(74,183.84,727.0),"Re":(75,186.207,736.0),
 "Ir":(77,192.217,757.0),"Pt":(78,195.084,790.0),"Au":(79,196.967,790.0),"Pb":(82,207.2,823.0),
 "Bi":(83,208.98,823.0),"Th":(90,232.038,847.0),"U":(92,238.029,890.0)}

# ---- tier 1: element aliases (carbon allotropes differ only in density) ----
ALIAS = {"Carbon":("C",2.20),"Graphite":("C",2.23),"Diamond":("C",3.515),
         "Vitreous Carbon":("C",1.50),"Graphite Oxide":None,  # compound -> handle below
         "D2":("D",None)}
ELEMENT_NAMES = {"Dy","Er","Gd","Hf","Ho","Ir","Lu","Pt","Re","Ta","Tb","Tm","Yb"}

# ---- tier 3: name -> molecular formula (organics & named compounds) ----
NAME = {
 # alkanes
 "Ethane":"C2H6","Propane":"C3H8","Butane":"C4H10","Pentane":"C5H12","Hexane":"C6H14","Heptane":"C7H16","Octane":"C8H18",
 "isooctane":"C8H18","Nonane":"C9H20","Decane":"C10H22","Undecane":"C11H24","Dodecane":"C12H26",
 "Tridecane":"C13H28","Tetradecane":"C14H30","Pentadecane":"C15H32",
 # alcohols
 "Methanol":"CH4O","Ethanol":"C2H6O","Propanol":"C3H8O","Propyl alcohol":"C3H8O","Butanol":"C4H10O",
 "Pentanol":"C5H12O","Hexanol":"C6H14O","Heptanol":"C7H16O","Octanol":"C8H18O","Nonanol":"C9H20O",
 "Decanol":"C10H22O","Undecanol":"C11H24O",
 # alkenes / dienes
 "Ethylene":"C2H4","Propylene":"C3H6","Butene":"C4H8","Pentene":"C5H10","Hexene":"C6H12",
 "Heptene":"C7H14","Octene":"C8H16","Decene":"C10H20","Butadiene":"C4H6","Allene":"C3H4",
 "Styrene":"C8H8","2-methyl-1 3-butadiene":"C5H8","Difluoroethene":"C2H2F2",
 # alkynes
 "Acetylene":"C2H2","Propyne":"C3H4","Butyne":"C4H6","Pentyne":"C5H8","Hexyne":"C6H10",
 "Heptyne":"C7H12","Ethynylbenzene":"C8H6",
 # cyclics
 "Cyclopropane":"C3H6","Cyclopentane":"C5H10","Cyclohexane":"C6H12","Cycloheptane":"C7H14",
 "Cyclooctane":"C8H16","Cyclopentene":"C5H8","Cyclohexene":"C6H10","Cyclohexadiene":"C6H8",
 "Cycloheptatriene":"C7H8","cis-cyclooctene":"C8H14","Cyclohexanone":"C6H10O",
 "bicycloheptadiene":"C7H8",
 # aromatics
 "Benzene":"C6H6","Toluene":"C7H8","Terphenyl":"C18H14","Anthracene":"C14H10","p-Dioxane":"C4H8O2",
 # ketones / aldehydes / ethers
 "Acetone":"C3H6O","2-Butanone":"C4H8O","3-Pentanone":"C5H10O","Acetaldehyde":"C2H4O",
 "Butyraldehyde":"C4H8O","Diethyl ether":"C4H10O","Dimethyl ether":"C2H6O",
 "Vinyl methyl ether":"C3H6O","Ethylene oxide":"C2H4O","Propylene oxide":"C3H6O",
 "Dimethyl sulfite":"C2H6O3S",
 # amines
 "Methylamine":"CH5N","Ethylamine":"C2H7N","Dimethylamine":"C2H7N","Trimethylamine":"C3H9N",
 "Ammonia":"NH3",
 # S-compounds
 "Thiophene":"C4H4S","Dimethyl sulfide":"C2H6S","Dimethyl disulfide":"C2H6S2",
 "Ethylene sulfide":"C2H4S","Propylene sulfide":"C3H6S","Trimethylene sulfide":"C3H6S",
 # halocarbons (named)
 "Bromoethane":"C2H5Br","Chloroform":"CHCl3","Ethyl iodide":"C2H5I","Difluoroethane":"C2H4F2",
 "Trifluoroethane":"C2H3F3","Hexafluoroethane":"C2F6","Octafluoropropane":"C3F8",
 "Octafluorocyclobutane":"C4F8","Dibromotetrafluoroethane":"C2Br2F4","Fluoroform":"CHF3",
 "Vinyl chloride":"C2H3Cl","Vinyl bromide":"C2H3Br","Ethyl Cellulose":"C12H22O5",
 # diols / polyols
 "1,2-ethanediol":"C2H6O2","1,3-propanediol":"C3H8O2","1,4-butanediol":"C4H10O2",
 "1,5-pentanediol":"C5H12O2","Glycerol":"C3H8O3",
 # misc small molecules / oxides named
 "Carbon monoxide":"CO","Boric-acid trimethyl ester":"C3H9BO3",
 "D2O":"H2O",  # Z identical to water; A handled via formula (uses H, slight underestimate of A)
 # polymers (repeat unit)
 "Polyethylene":"C2H4","Polypropylene":"C3H6","Polystyrene":"C8H8","Polyvinyl chloride":"C2H3Cl",
 "Polyvinyl toluene":"C9H10","Polycarbonate":"C16H14O3","Mylar":"C10H8O4","Kapton":"C22H10N2O5",
 "Teflon":"C2F4","Polysulfone":"C27H22O4S","Poly ether ether ketone":"C19H12O3",
 "Polyhydroxybutyrate":"C4H6O2","CR-39":"C12H18O7","NE-111":"C9H10",
 "Hydroxyapatite":"Ca10P6O26H2","Formvar":"C8H14O5",
}

# ---- tier 4: alloys & standard materials by ELEMENT MASS FRACTION (+density, +measured I) ----
WFRAC = {
 "Havar":      ({"Co":0.425,"Cr":0.20,"Ni":0.13,"Fe":0.187,"W":0.028,"Mo":0.022,"Mn":0.016,"C":0.002},8.3,None),
 "Eurofer97":  ({"Fe":0.885,"Cr":0.09,"W":0.011,"Mn":0.004,"V":0.002,"Ta":0.0012,"C":0.0011,"Si":0.0005},7.8,None),
 "Mu metal":   ({"Ni":0.77,"Fe":0.16,"Cu":0.05,"Mo":0.02},8.7,None),
 "Permalloy":  ({"Ni":0.80,"Fe":0.20},8.7,None),
 "Air":        ({"N":0.7553,"O":0.2318,"Ar":0.0128,"C":0.0001},1.205e-3,85.7),
 "A-150":      ({"H":0.1013,"C":0.7755,"N":0.0351,"O":0.0523,"F":0.0174,"Ca":0.0184},1.127,65.1),
 "Tissue eq":  ({"H":0.1019,"C":0.1001,"N":0.0270,"O":0.7551,"Na":0.0011,"P":0.0011,"S":0.0011,"Cl":0.0011,"K":0.0011,"Ca":0.0092},1.0,74.9),
 "bone":       ({"H":0.0639,"C":0.278,"N":0.027,"O":0.410,"Mg":0.002,"P":0.070,"S":0.002,"Ca":0.147},1.85,91.9),
 "mica":       ({"K":0.0998,"Al":0.2046,"Si":0.2127,"O":0.4790,"H":0.0039},2.82,None),  # muscovite approx
}

# ---- low-confidence / exotic -> UNRESOLVED for physics team ----
UNRESOLVED = {"Celluloid","LR-115","Pliolite S-5A","EP-PTCDI","Graphite Oxide",
              "Vinylchloride vinylacetate copolymer"}

tok = re.compile(r'([A-Z][a-z]?)(\d+(?:\.\d+)?)?')
def parse_formula(f):
    """expand one level of parentheses, return {element: count}"""
    def expand(s):
        while '(' in s:
            m = re.search(r'\(([^()]*)\)(\d+(?:\.\d+)?)?', s)
            if not m: break
            inner, mult = m.group(1), float(m.group(2) or 1)
            rep = "".join(f"{el}{(float(n or 1))*mult:g}" for el,n in tok.findall(inner))
            s = s[:m.start()] + rep + s[m.end():]
        return s
    s = expand(f); d={}
    for el,n in tok.findall(s):
        if el: d[el] = d.get(el,0)+float(n or 1)
    return d

def metrics(counts):
    """counts: {el: atom_count} -> mean Z2, A2, I2(additivity eV), n_imputed"""
    tot=sum(counts.values()); Zm=Am=0; num=den=0; imp=0
    for el,n in counts.items():
        if el not in ATOM: return None
        Z,A,I = ATOM[el]
        if I is None: I=10*Z; imp+=1
        Zm += n*Z/tot; Am += n*A/tot
        num += n*Z*np.log(I); den += n*Z
    return Zm, Am, float(np.exp(num/den)), imp

def wfrac_metrics(wf):
    """mass fractions -> atom fractions -> metrics"""
    counts={el: w/ATOM[el][1] for el,w in wf.items() if el in ATOM}  # moles ~ w/A
    return metrics(counts)

df = pd.read_csv("/mnt/project/StoppingPower.csv", low_memory=False)
elem_syms=set(ATOM)
targets = sorted(df.target_name.unique())
rows=[]
for t in targets:
    method=Zm=Am=I2=dens=imp=None; src="additivity"
    if t in elem_syms or t in ELEMENT_NAMES:
        Z,A,I=ATOM[t]; Zm,Am,I2=Z,A,(I if I else 10*Z); method="element"; imp=int(I is None)
    elif t in ALIAS and ALIAS[t]:
        sym,dens=ALIAS[t]; Z,A,I=ATOM[sym]; Zm,Am,I2=Z,A,I; method="element_alias"; imp=0
    elif t in WFRAC:
        wf,dens,Imeas=WFRAC[t]; r=wfrac_metrics(wf)
        if r: Zm,Am,I2,imp=r; method="weight_frac"
        if Imeas: I2=Imeas; src="NIST_measured"
    elif t in NAME:
        r=metrics(parse_formula(NAME[t]))
        if r: Zm,Am,I2,imp=r; method="name_map"
    elif t in UNRESOLVED:
        method="UNRESOLVED"
    else:
        r=metrics(parse_formula(t))            # try as raw formula (SiO2, Al2O3, ...)
        if r: Zm,Am,I2,imp=r; method="formula"
        else: method="UNRESOLVED"
    rows.append(dict(target=t, method=method, Z2=Zm, A2=Am, I2_eV=I2, density=dens,
                     I2_source=src if method not in(None,"UNRESOLVED")else None,
                     I2_imputed_elems=imp, validated_by_physics=False,
                     n_records=int((df.target_name==t).sum())))

tab=pd.DataFrame(rows)
tab["n_e_1e23"]=tab.density*N_A*tab.Z2/tab.A2/1e23
tab.to_csv("/home/claude/target_composition_table.csv", index=False)

# ---- coverage report ----
comp_only = tab[~tab.target.isin(elem_syms) & ~tab.target.isin(ELEMENT_NAMES)]
tot_comp_rows = comp_only.n_records.sum()
res = comp_only[comp_only.method!="UNRESOLVED"]
unr = comp_only[comp_only.method=="UNRESOLVED"]
print("=== TARGET COMPOSITION TABLE — coverage ===")
print(f"All targets in table: {len(tab)}")
print(f"Compound/mixture targets: {len(comp_only)}  | records they cover: {tot_comp_rows:,}")
print(f"  RESOLVED: {len(res)} targets / {res.n_records.sum():,} records "
      f"({100*res.n_records.sum()/tot_comp_rows:.1f}% of compound records)")
print(f"  UNRESOLVED (need physics team): {len(unr)} targets / {unr.n_records.sum():,} records")
print("  unresolved list:", list(unr.sort_values('n_records',ascending=False).target))
print(f"\nmethod breakdown:\n{comp_only.method.value_counts().to_string()}")
print(f"\nDensity missing (need value): {int(res.density.isna().sum())} resolved targets")
print(f"I2 with Bloch-imputed elements (need ESTAR check): {int((res.I2_imputed_elems>0).sum())} targets")
print("\nTop-15 compound targets by records (verify these first):")
print(comp_only.sort_values('n_records',ascending=False)
      [['target','method','Z2','A2','I2_eV','n_records']].head(15).to_string(index=False))
