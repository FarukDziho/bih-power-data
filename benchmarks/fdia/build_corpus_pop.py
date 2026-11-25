#!/usr/bin/env python3
"""
Stage 1 of the BiHGrid FDIA recreation: build a labelled corpus.

For each sampled real hour:
  - dress the network from the zone series (load/gen/border heuristics),
  - AC power flow -> true operating state (Vm, Va, injections),
  - emit an HONEST sample (measurement noise, no attack), and
  - emit a STEALTH-FDIA sample: a topology-consistent (unobservable) localized
    state perturbation on a bus + its neighbourhood, so the classical bad-data
    residual is unchanged but a spatial anomaly is injected. Per-bus labels mark
    the attacked region (localization); graph label marks any attack (detection).

Population-weighted load variant (load_weights_pop.json), i.e. the configuration
of the saved run in gnn_results_pop.json. Paths come from BIHGRID_DATASET,
BIHGRID_LOAD_WEIGHTS and BIHGRID_CORPUS (defaults are relative to this folder).
"""
import numpy as np, pandas as pd, math, warnings, time
import pandapower as pp
import networkx as nx
warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)

import json, os, sys
DS = os.environ.get("BIHGRID_DATASET", "../../bih-power-data/dataset")
OUT = os.environ.get("BIHGRID_CORPUS", "corpus_bih.npz")
LOAD_W = json.load(open(os.environ.get("BIHGRID_LOAD_WEIGHTS", "load_weights_pop.json")))
nodes = pd.read_csv(f"{DS}/nodes.csv")
edges = pd.read_csv(f"{DS}/edges.csv")
w = pd.read_csv(f"{DS}/gen_bus_weights.csv")
z = pd.read_parquet(f"{DS}/zone_series.parquet")

# ----- network (same construction as the feasibility prototype) --------------
net = pp.create_empty_network(sn_mva=100.0)
bus_idx = {}
for _, r in nodes.iterrows():
    bus_idx[r.bus_id] = pp.create_bus(net, vn_kv=float(r.voltage), name=r.bus_id)
for _, r in edges[edges.kind == "line"].iterrows():
    if r.bus0 not in bus_idx or r.bus1 not in bus_idx: continue
    L = max(float(r.length_km), 0.1); vn = float(r.voltage)
    pp.create_line_from_parameters(net, bus_idx[r.bus0], bus_idx[r.bus1],
        length_km=L, r_ohm_per_km=float(r.r)/L, x_ohm_per_km=max(float(r.x)/L,1e-4),
        c_nf_per_km=max((float(r.b)/(2*math.pi*50.0))/L*1e9,0.0),
        max_i_ka=max(float(r.s_nom)/(math.sqrt(3)*vn),0.05),
        parallel=int(r.circuits) if not pd.isna(r.circuits) else 1, name=r.edge_id)
for _, r in edges[edges.kind == "transformer"].iterrows():
    if r.bus0 not in bus_idx or r.bus1 not in bus_idx: continue
    hv, lv = bus_idx[r.bus1], bus_idx[r.bus0]
    if net.bus.at[hv,"vn_kv"] < net.bus.at[lv,"vn_kv"]: hv, lv = lv, hv
    pp.create_transformer_from_parameters(net, hv_bus=hv, lv_bus=lv, sn_mva=float(r.s_nom),
        vn_hv_kv=net.bus.at[hv,"vn_kv"], vn_lv_kv=net.bus.at[lv,"vn_kv"],
        vkr_percent=0.3, vk_percent=12.0, pfe_kw=0.0, i0_percent=0.0, name=r.edge_id)

slack = bus_idx["BA4-400"]
pp.create_ext_grid(net, slack, vm_pu=1.0)
ba_buses = [b for b in nodes[nodes.role=="ba"].bus_id if b in bus_idx]
border = nodes[nodes.role=="border"]
border_by_country = {c:[b for b in g.bus_id if b in bus_idx] for c,g in border.groupby("country")}
nb = len(net.bus)

# edge_index (undirected, both directions) for the GNN
ei = []
for _, ln in net.line.iterrows(): ei += [(int(ln.from_bus),int(ln.to_bus)),(int(ln.to_bus),int(ln.from_bus))]
for _, tr in net.trafo.iterrows(): ei += [(int(tr.hv_bus),int(tr.lv_bus)),(int(tr.lv_bus),int(tr.hv_bus))]
edge_index = np.array(ei).T

# DC B matrix (nb x nb) from lines + trafos -> exact stealth construction
base = net.sn_mva
B = np.zeros((nb, nb))
def add_branch(f,t,b):
    B[f,f]+=b; B[t,t]+=b; B[f,t]-=b; B[t,f]-=b
for _, ln in net.line.iterrows():
    vn = net.bus.at[int(ln.from_bus),"vn_kv"]
    x_pu = (ln.x_ohm_per_km*ln.length_km)/(vn**2/base)/max(ln.parallel,1)
    add_branch(int(ln.from_bus),int(ln.to_bus),1.0/x_pu)
for _, tr in net.trafo.iterrows():
    x_pu = (tr.vk_percent/100.0)*(base/tr.sn_mva)
    add_branch(int(tr.hv_bus),int(tr.lv_bus),1.0/x_pu)

# adjacency for choosing localized attack regions
G = nx.Graph(); G.add_nodes_from(range(nb))
for f,t in zip(edge_index[0],edge_index[1]): G.add_edge(int(f),int(t))
deg = np.array([G.degree(i) for i in range(nb)], float)

GEN_MAP = {
 "gen_fossil_brown_coal_lignite_mw":"Fossil coal (combined)",
 "gen_fossil_hard_coal_mw":"Fossil coal (combined)",
 "gen_hydro_water_reservoir_mw":"Hydro Water Reservoir",
 "gen_hydro_pumped_storage_mw":"Hydro Pumped Storage",
 "gen_hydro_run_of_river_and_poundage_mw":"Hydro Run-of-river and poundage",
 "gen_wind_onshore_mw":"Wind Onshore"}
wvec = {t:dict(zip(g.bus_id,g.weight)) for t,g in w.groupby("entsoe_type")}
BORDER_COL = {"HR":"net_import_HR_mw","RS":"net_import_RS_mw","ME":"net_import_ME_mw"}

def build_scene(row):
    net.load.drop(net.load.index, inplace=True)
    net.sgen.drop(net.sgen.index, inplace=True)
    Lt = row.get("load_actual_mw", np.nan)
    if not np.isfinite(Lt) or Lt<=0: return False
    for b in ba_buses:
        per = Lt*LOAD_W[b]
        pp.create_load(net, bus_idx[b], p_mw=per, q_mvar=per*0.31)
    inj={}
    for col,typ in GEN_MAP.items():
        g=row.get(col,np.nan)
        if not np.isfinite(g) or typ not in wvec: continue
        for b,wt in wvec[typ].items():
            if b in bus_idx: inj[b]=inj.get(b,0.0)+g*wt
    for b,p in inj.items(): pp.create_sgen(net, bus_idx[b], p_mw=p, q_mvar=0.0)
    for c,col in BORDER_COL.items():
        v=row.get(col,np.nan); bs=border_by_country.get(c,[])
        if not np.isfinite(v) or not bs: continue
        for b in bs: pp.create_sgen(net, bus_idx[b], p_mw=v/len(bs), q_mvar=0.0)
    return True

# ------- corpus generation ---------------------------------------------------
N_HOURS = 2200
zc = z.dropna(subset=["load_actual_mw"]).copy()
sample = zc.sample(n=min(N_HOURS, len(zc)), random_state=3).sort_index()

X_list, ynode_list, ygraph_list, ts_list, J_list = [], [], [], [], []
pnorm_list = []   # per-sample divisor of the injection feature (see note at the end)
noise_ang = 0.005   # rad measurement noise on angle-derived features
noise_p   = 0.01    # 1% on injection
t0=time.time(); done=0; fail=0
keep = [i for i in range(nb) if i!=slack]

for ts, row in sample.iterrows():
    if not build_scene(row):
        fail+=1; continue
    try:
        pp.runpp(net, algorithm="nr", max_iteration=30)
    except Exception:
        fail+=1; continue
    if net.res_bus.vm_pu.isna().any():
        fail+=1; continue
    Vm = net.res_bus.vm_pu.values.copy()
    Va = np.deg2rad(net.res_bus.va_degree.values.copy())     # true angles (rad)
    theta = Va - Va[slack]
    Pdc = B @ theta                                           # DC injections (pu of base? consistent)
    # normalise features to comparable scales
    def feats(vm, th, p, lab):
        f = np.stack([
            (vm-1.0)*10.0,                # centred voltage magnitude
            np.sin(th), np.cos(th),       # angle on the circle
            p/ (np.abs(Pdc).mean()+1e-6), # normalised injection
            deg/deg.max()                 # static topology cue
        ], axis=1).astype(np.float32)
        return f
    # residual objective for BDD baseline (measurement = Pdc)
    sigma = noise_p*(np.abs(Pdc).mean()+1e-6)
    Hr = B[np.ix_(range(nb), keep)]
    # honest
    zt = Pdc + rng.normal(0, sigma, nb)
    W = np.eye(nb)/sigma**2
    Gm = Hr.T@W@Hr
    xh = np.linalg.solve(Gm, Hr.T@W@zt)
    rh = zt - Hr@xh; Jh = float(rh.T@W@rh)
    th_h = np.zeros(nb); th_h[keep]=xh
    Vm_h = Vm + rng.normal(0, 0.002, nb)
    X_list.append(feats(Vm_h, th_h+rng.normal(0,noise_ang,nb), Hr@xh, 0))
    ynode_list.append(np.zeros(nb, np.int64)); ygraph_list.append(0)
    ts_list.append(str(ts)); J_list.append(Jh); pnorm_list.append(np.abs(Pdc).mean()+1e-6)
    # stealth attack: localized c on a region (bus + neighbours)
    center = int(rng.choice([i for i in range(nb) if i!=slack]))
    region = set([center]) | set(list(G.neighbors(center))[:rng.integers(1,4)])
    region.discard(slack)
    region = list(region)
    c = np.zeros(nb)
    mag = rng.uniform(0.03, 0.12) * rng.choice([-1,1])
    for b in region: c[b] = mag * rng.uniform(0.6,1.0)
    theta_a = theta + c                       # consistent perturbed state
    a = B @ theta_a - Pdc                      # a = Hc  (stealth injection)
    za = zt + a
    xha = np.linalg.solve(Gm, Hr.T@W@za)
    ra = za - Hr@xha; Ja = float(ra.T@W@ra)    # ~= Jh by construction
    th_a = np.zeros(nb); th_a[keep]=xha
    lab = np.zeros(nb, np.int64)
    for b in region: lab[b]=1
    X_list.append(feats(Vm_h, th_a+rng.normal(0,noise_ang,nb), Hr@xha, 1))
    ynode_list.append(lab); ygraph_list.append(1)
    ts_list.append(str(ts)); J_list.append(Ja); pnorm_list.append(np.abs(Pdc).mean()+1e-6)
    done+=1
    if done % 300 == 0:
        print(f"  {done} hours, {time.time()-t0:.0f}s, fail={fail}")

X = np.stack(X_list); ynode = np.stack(ynode_list)
ygraph = np.array(ygraph_list, np.int64); J = np.array(J_list)
ts_arr = np.array(ts_list)
print(f"[corpus] {X.shape[0]} samples ({done} hours x2), features {X.shape}, "
      f"attacked-bus rate {ynode.mean():.3f}")
# NOTE on feature 3 (injection): it is divided by the mean absolute injection
# of the hour's UNATTACKED state (pnorm). A detector would not know that
# number, and it makes honest samples average exactly 1 on this feature while
# attacked samples do not. pnorm is saved so that train_fixed_scaling.py can
# undo the division and scale by a training-set constant instead.
np.savez_compressed(OUT, X=X, ynode=ynode, ygraph=ygraph,
                    J=J, ts=ts_arr, edge_index=edge_index.astype(np.int64),
                    slack=slack, nb=nb, pnorm=np.array(pnorm_list))
print("saved", OUT)
