#!/usr/bin/env python3
"""
Population-weighted load disaggregation for BiHGrid.
Replaces the uniform-split placeholder: zone load is distributed to buses in
proportion to the 2013 census population of each bus's `nearest_place`
municipality (shared equally when several buses map to the same place).
Validated by AC power flow and compared to the uniform baseline.

Population figures: 2013 Census of Population, Households and Dwellings in Bosnia
and Herzegovina (Agency for Statistics of BiH), municipality (opstina) totals.
Two Neretva HPP settlements (Salakovac, Grabovica) are dam sites, not load
centres, and are assigned nominal populations (documented heuristic).
"""
import pandas as pd, numpy as np, math, warnings, json
import pandapower as pp
warnings.filterwarnings("ignore")

# 2013 census municipality populations (Agency for Statistics of BiH)
POP = {
 "Prijedor":89397, "Kakanj":37441, "Visoko":39938, "Sarajevo":275524,
 "Rogatica":10723, "Višegrad":10668, "Prozor":14280, "Jablanica":10072,
 "Banja Luka":185042, "Posušje":20698, "Mostar":105797, "Gacko":8990,
 "Čapljina":26157, "Trebinje":31433, "Bihać":56261, "Živinice":57765,
 "Zvornik":58856, "Zenica":110663, "Ugljevik":15590, "Jajce":27258,
 "Gradačac":41836, "Lukavac":44520,
 "Salakovac":500, "Grabovica":300,     # HPP dam settlements — nominal
}

DS="/mnt/user-data/uploads/FarukClaude/Projects/bih-power-data/dataset"
nodes=pd.read_csv(f"{DS}/nodes.csv"); edges=pd.read_csv(f"{DS}/edges.csv")
ba=nodes[nodes.role=="ba"].copy()
assert set(ba.nearest_place)<=set(POP), set(ba.nearest_place)-set(POP)

# per-bus weight: population of place / (# buses sharing that place), then normalize
share = ba.nearest_place.map(POP).astype(float)
counts = ba.nearest_place.map(ba.nearest_place.value_counts())
w = share/counts
w = (w/w.sum()).values
ba["w_pop"]=w; ba["w_uniform"]=1.0/len(ba)
weights_pop = dict(zip(ba.bus_id, ba.w_pop))
json.dump(weights_pop, open("/tmp/load_weights_pop.json","w"), indent=2)

# --- concentration summary ---
print("Load share by place (population-weighted), top 12:")
byplace=ba.groupby("nearest_place").w_pop.sum().sort_values(ascending=False)
for p,v in byplace.head(12).items(): print(f"  {p:12s} {v*100:5.1f}%   (uniform each bus {100/len(ba):.1f}%)")
# Gini-ish: top-5 places' share
print(f"\nTop-5 places carry {byplace.head(5).sum()*100:.1f}% of load (population); "
      f"uniform would give those {ba.nearest_place.isin(byplace.head(5).index).sum()/len(ba)*100:.1f}%.")

# --- build network and validate both disaggregations via AC power flow ---
net=pp.create_empty_network(sn_mva=100.0); bus_idx={}
for _,r in nodes.iterrows(): bus_idx[r.bus_id]=pp.create_bus(net,vn_kv=float(r.voltage),name=r.bus_id)
for _,r in edges[edges.kind=="line"].iterrows():
    if r.bus0 not in bus_idx or r.bus1 not in bus_idx: continue
    L=max(float(r.length_km),0.1); vn=float(r.voltage)
    pp.create_line_from_parameters(net,bus_idx[r.bus0],bus_idx[r.bus1],length_km=L,
        r_ohm_per_km=float(r.r)/L,x_ohm_per_km=max(float(r.x)/L,1e-4),
        c_nf_per_km=max((float(r.b)/(2*math.pi*50.0))/L*1e9,0.0),
        max_i_ka=max(float(r.s_nom)/(math.sqrt(3)*vn),0.05),
        parallel=int(r.circuits) if not pd.isna(r.circuits) else 1)
for _,r in edges[edges.kind=="transformer"].iterrows():
    if r.bus0 not in bus_idx or r.bus1 not in bus_idx: continue
    hv,lv=bus_idx[r.bus1],bus_idx[r.bus0]
    if net.bus.at[hv,"vn_kv"]<net.bus.at[lv,"vn_kv"]: hv,lv=lv,hv
    pp.create_transformer_from_parameters(net,hv_bus=hv,lv_bus=lv,sn_mva=float(r.s_nom),
        vn_hv_kv=net.bus.at[hv,"vn_kv"],vn_lv_kv=net.bus.at[lv,"vn_kv"],
        vkr_percent=0.3,vk_percent=12.0,pfe_kw=0.0,i0_percent=0.0)
pp.create_ext_grid(net,bus_idx["BA4-400"],vm_pu=1.0)
ba_buses=list(ba.bus_id)

# generation weights + border (same as pipeline) to make scenes realistic
wgen=pd.read_csv(f"{DS}/gen_bus_weights.csv"); wvec={t:dict(zip(g.bus_id,g.weight)) for t,g in wgen.groupby("entsoe_type")}
GEN_MAP={"gen_fossil_brown_coal_lignite_mw":"Fossil coal (combined)","gen_fossil_hard_coal_mw":"Fossil coal (combined)",
 "gen_hydro_water_reservoir_mw":"Hydro Water Reservoir","gen_hydro_pumped_storage_mw":"Hydro Pumped Storage",
 "gen_hydro_run_of_river_and_poundage_mw":"Hydro Run-of-river and poundage","gen_wind_onshore_mw":"Wind Onshore"}
border=nodes[nodes.role=="border"]; bbc={c:[b for b in g.bus_id if b in bus_idx] for c,g in border.groupby("country")}
BORDER_COL={"HR":"net_import_HR_mw","RS":"net_import_RS_mw","ME":"net_import_ME_mw"}
z=pd.read_parquet(f"{DS}/zone_series.parquet"); zc=z.dropna(subset=["load_actual_mw"]).sample(n=200,random_state=1)

def scene(row, wmode):
    net.load.drop(net.load.index,inplace=True); net.sgen.drop(net.sgen.index,inplace=True)
    Lt=row["load_actual_mw"]
    for bid in ba_buses:
        frac = weights_pop[bid] if wmode=="pop" else 1.0/len(ba_buses)
        p=Lt*frac; pp.create_load(net,bus_idx[bid],p_mw=p,q_mvar=p*0.31)
    inj={}
    for col,typ in GEN_MAP.items():
        g=row.get(col,np.nan)
        if np.isfinite(g) and typ in wvec:
            for b,wt in wvec[typ].items():
                if b in bus_idx: inj[b]=inj.get(b,0.0)+g*wt
    for b,p in inj.items(): pp.create_sgen(net,bus_idx[b],p_mw=p,q_mvar=0.0)
    for c,col in BORDER_COL.items():
        v=row.get(col,np.nan); bs=bbc.get(c,[])
        if np.isfinite(v) and bs:
            for b in bs: pp.create_sgen(net,bus_idx[b],p_mw=v/len(bs),q_mvar=0.0)

def validate(wmode):
    conv=0; vmins=[]; vmaxs=[]; maxloads=[]
    for _,row in zc.iterrows():
        scene(row,wmode)
        try: pp.runpp(net,max_iteration=30)
        except Exception: continue
        if net.res_bus.vm_pu.isna().any(): continue
        conv+=1; vmins.append(net.res_bus.vm_pu.min()); vmaxs.append(net.res_bus.vm_pu.max())
        maxloads.append(net.res_line.loading_percent.max())
    return dict(conv=conv,total=len(zc),vmin=float(np.min(vmins)),vmax=float(np.max(vmaxs)),
                vmin_iqr=float(np.percentile(vmins,25)),median_maxload=float(np.median(maxloads)))

vu=validate("uniform"); vp=validate("pop")
print("\nAC power-flow validation (200 hours):")
print(f"  uniform:    converged {vu['conv']}/{vu['total']}, vmin {vu['vmin']:.3f}, median max-load {vu['median_maxload']:.1f}%")
print(f"  population: converged {vp['conv']}/{vp['total']}, vmin {vp['vmin']:.3f}, median max-load {vp['median_maxload']:.1f}%")
json.dump({"uniform":vu,"population":vp,"weights_by_place":{k:float(v) for k,v in byplace.items()}},
          open("/tmp/load_validation.json","w"),indent=2)

# --- figure: load share by place, population vs uniform ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE="#2a78d6"; GREY="#9a9a94"; SURF="#fcfcfb"; INK="#0b0b0b"; MUT="#52514e"
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"text.color":INK,"axes.labelcolor":INK,
 "xtick.color":MUT,"ytick.color":MUT,"axes.grid":True,"grid.color":"#ececea","axes.spines.top":False,"axes.spines.right":False})
top=byplace.head(14)[::-1]
uni_share=ba.groupby("nearest_place").w_uniform.sum().reindex(top.index)
fig,ax=plt.subplots(figsize=(7.2,5.4))
y=np.arange(len(top))
ax.barh(y+0.2,top.values*100,height=0.4,color=BLUE,label="population-weighted",edgecolor=SURF)
ax.barh(y-0.2,uni_share.values*100,height=0.4,color=GREY,label="uniform (placeholder)",edgecolor=SURF)
ax.set_yticks(y); ax.set_yticklabels(top.index,fontsize=9)
ax.set_xlabel("share of national load (%)")
ax.set_title("Population-weighted load disaggregation concentrates demand\nin the real load centres (2013 census municipalities)",loc="left",fontsize=11)
ax.legend(frameon=False,loc="lower right",fontsize=9.5)
fig.tight_layout(); fig.savefig("/tmp/figI_loadweights.png",dpi=150)
print("wrote /tmp/figI_loadweights.png")
