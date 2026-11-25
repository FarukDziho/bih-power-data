#!/usr/bin/env python3
"""
Sensor-coverage test, rerun (replaces sweep_sparsity.py and scada_sweep.py).

What changed against the two original sweeps:

1. No hidden-bus information in the features. scada_sweep.py divided the
   observed injections by the mean absolute injection over ALL buses of the
   true state, including buses the detector could not see. Here the scale is
   one constant per run, the mean absolute observed injection over the
   training hours. The state estimate is (H'H + lam I)^-1 H'z from observed
   injections only, with a fixed lam.
2. Five random seeds per coverage level. A seed sets the sensor masks, the
   measurement noise and the model initialisation. Results are mean and
   standard deviation over seeds.
3. Detection is reported as the share of attacked hours detected at a fixed
   5% false-alarm rate. The threshold is the 95th percentile of the scores of
   honest validation hours; the realised false-alarm rate on the test hours
   is reported next to it.

4. The localisation loss weight is computed from training hours only.

Not changed, and worth knowing: the simulated sensor noise is 1% of the mean
absolute injection over all buses of the hour's true state. That is part of
the simulator. The detector's features are scaled with observed training
measurements only.

Unchanged: the corpus (population-weighted load, build_corpus_pop.py), the
time-ordered 70/15/15 split by hour, the ChebConv GNN and the MLP, the two
measurement models (PMU = voltage phasor at observed buses, SCADA = power
injection at observed buses), per-hour random sensor masks, coverage levels.

Run:  python coverage_rerun.py            (about 15 minutes on 2 CPU cores)
Env:  BIHGRID_DATASET, BIHGRID_CORPUS, BIHGRID_OUT, BIHGRID_SEEDS
"""
import json, os, time
import numpy as np, pandas as pd, networkx as nx, torch
import torch.nn as nn, torch.nn.functional as Fn
from torch_geometric.nn import ChebConv
from sklearn.metrics import roc_auc_score, f1_score

DS = os.environ.get("BIHGRID_DATASET", "../../bih-power-data/dataset")
CORPUS = os.environ.get("BIHGRID_CORPUS", "corpus_bih.npz")
OUT = os.environ.get("BIHGRID_OUT", ".")
SEEDS = [int(s) for s in os.environ.get("BIHGRID_SEEDS", "0,1,2,3,4").split(",")]
DENS = [1.0, 0.7, 0.5, 0.35, 0.2]
FAR = 0.05          # fixed false-alarm rate for the detection operating point
LAM = 0.05          # ridge term of the injection-only state estimate
torch.set_num_threads(2)

# ---------------------------------------------------------------- network
nodes = pd.read_csv(f"{DS}/nodes.csv"); edges = pd.read_csv(f"{DS}/edges.csv")
bus_ids = list(nodes.bus_id); pos = {b: i for i, b in enumerate(bus_ids)}; nb = len(bus_ids)
base = 100.0; B = np.zeros((nb, nb))
def addB(f, t, b): B[f, f] += b; B[t, t] += b; B[f, t] -= b; B[t, f] -= b
for _, r in edges[edges.kind == "line"].iterrows():
    vn = float(r.voltage)
    x_pu = float(r.x) / (vn**2 / base) / (int(r.circuits) if not pd.isna(r.circuits) else 1)
    if x_pu > 0: addB(pos[r.bus0], pos[r.bus1], 1.0 / x_pu)
for _, r in edges[edges.kind == "transformer"].iterrows():
    addB(pos[r.bus0], pos[r.bus1], 1.0 / ((12.0 / 100.0) * (base / float(r.s_nom))))
slack = pos["BA4-400"]; keep = [i for i in range(nb) if i != slack]
G = nx.Graph(); G.add_nodes_from(range(nb))
for i in range(nb):
    for j in range(i + 1, nb):
        if abs(B[i, j]) > 0: G.add_edge(i, j)
degn = np.array([G.degree(i) for i in range(nb)], float); degn /= degn.max()
ei = np.array([(i, j) for i, j in G.edges()] + [(j, i) for i, j in G.edges()]).T
edge_index = torch.tensor(ei, dtype=torch.long)

# ---------------------------------------------------------------- corpus
d = np.load(CORPUS, allow_pickle=True)
X, ynode, ygraph, ts = d["X"], d["ynode"], d["ygraph"], d["ts"]
theta = np.arctan2(X[:, :, 1], X[:, :, 2]); Vm = X[:, :, 0] / 10.0 + 1.0
hon = ygraph == 0; att = ygraph == 1
oh = np.argsort(ts[hon]); oa = np.argsort(ts[att])
assert (ts[hon][oh] == ts[att][oa]).all()
theta_t, Vm_t = theta[hon][oh], Vm[hon][oh]
theta_a, Vm_a = theta[att][oa], Vm[att][oa]
label_a = ynode[att][oa]; hours = ts[hon][oh]; H = len(hours)
n_tr, n_va = int(.70 * H), int(.85 * H)            # hours are sorted in time
split = np.zeros(H, int); split[n_tr:n_va] = 1; split[n_va:] = 2
grp = np.repeat(np.arange(H), 2)                   # sample -> hour
yg = np.tile(np.array([0, 1]), H)                  # honest, attacked
yn = np.stack([np.zeros((H, nb), np.int64), label_a], 1).reshape(2 * H, nb)
S = split[grp]
# localisation loss weight from TRAINING hours only (19.6286; the first rerun
# used all hours, 19.6324)
tr_lab = label_a[split == 0]
pos_w = float((tr_lab == 0).sum() / max((tr_lab == 1).sum(), 1))
print(f"{H} hours, {2*H} samples; train/val/test samples "
      f"{(S==0).sum()}/{(S==1).sum()}/{(S==2).sum()}", flush=True)

# ---------------------------------------------------------------- features
def pmu_features(observed):
    """Observed buses report their voltage phasor; the rest report nothing."""
    obs = np.repeat(observed, 2, axis=0).astype(np.float32)          # [2H, nb]
    th = np.stack([theta_t, theta_a], 1).reshape(2 * H, nb)
    vm = np.stack([Vm_t, Vm_a], 1).reshape(2 * H, nb)
    return np.stack([(vm - 1.0) * 10.0 * obs, np.sin(th) * obs, np.cos(th) * obs,
                     obs, np.broadcast_to(degn, obs.shape)], 2).astype(np.float32)

def scada_features(observed, rng):
    """Observed buses report their power injection. The state is estimated
    from those injections alone; unobserved buses contribute nothing."""
    zobs = np.zeros((2 * H, nb)); th_hat = np.zeros((2 * H, nb))
    for i in range(H):
        oi = np.where(observed[i])[0]
        Hm = B[np.ix_(oi, keep)]
        A = np.linalg.inv(Hm.T @ Hm + LAM * np.eye(len(keep))) @ Hm.T
        for k, st in enumerate((theta_t[i], theta_a[i])):
            z_full = B @ st
            sigma = 0.01 * (np.abs(z_full).mean() + 1e-6)   # simulated sensor noise
            z = z_full[oi] + rng.normal(0, sigma, len(oi))
            zobs[2 * i + k, oi] = z
            th_hat[2 * i + k, keep] = A @ z
    obs = np.repeat(observed, 2, axis=0)
    # scale from observed training measurements only
    scale = np.abs(zobs[S == 0][obs[S == 0]]).mean() + 1e-9
    f = np.stack([zobs / scale, obs.astype(float), np.sin(th_hat), np.cos(th_hat),
                  np.broadcast_to(degn, obs.shape)], 2).astype(np.float32)
    return f, float(scale)

# ---------------------------------------------------------------- models
class GNN(nn.Module):
    def __init__(self, f=5, h=64, K=3):
        super().__init__()
        self.c1 = ChebConv(f, h, K); self.c2 = ChebConv(h, h, K); self.c3 = ChebConv(h, h, K)
        self.b1 = nn.BatchNorm1d(h); self.b2 = nn.BatchNorm1d(h)
        self.loc = nn.Linear(h, 1)
        self.det = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x):                           # x: [Bt, nb, F]
        Bt = x.shape[0]
        off = (torch.arange(Bt) * nb).repeat_interleave(edge_index.shape[1])
        e_idx = edge_index.repeat(1, Bt) + off      # one disjoint graph per sample
        e = x.reshape(Bt * nb, -1)
        e = Fn.relu(self.b1(self.c1(e, e_idx)))
        e = Fn.relu(self.b2(self.c2(e, e_idx)))
        e = Fn.relu(self.c3(e, e_idx)).reshape(Bt, nb, -1)
        return (self.det(torch.cat([e.mean(1), e.max(1).values], 1)).squeeze(-1),
                self.loc(e).squeeze(-1))

class MLP(nn.Module):
    def __init__(self, f=5, h=64):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.loc = nn.Linear(h, 1)
        self.det = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x):
        e = self.node(x)
        return (self.det(torch.cat([e.mean(1), e.max(1).values], 1)).squeeze(-1),
                self.loc(e).squeeze(-1))

def train_eval(model_cls, feats, seed, epochs, bs, patience):
    torch.manual_seed(seed); rs = np.random.default_rng(1000 + seed)
    Xs = torch.tensor(feats); Yg = torch.tensor(yg).float(); Yn = torch.tensor(yn).float()
    model = model_cls(); opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pw = torch.tensor([pos_w])
    itr, iva, ite = (np.where(S == k)[0] for k in range(3))
    def run(idx, train):
        model.train(train); tot = 0.0
        idx = rs.permutation(idx) if train else idx
        for i in range(0, len(idx), bs):
            bi = idx[i:i + bs]; det, loc = model(Xs[bi])
            loss = (Fn.binary_cross_entropy_with_logits(det, Yg[bi]) +
                    Fn.binary_cross_entropy_with_logits(loc, Yn[bi], pos_weight=pw))
            if train: opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(bi)
        return tot / len(idx)
    best, state, pat = 1e9, None, 0
    for _ in range(epochs):
        run(itr, True)
        with torch.no_grad(): vl = run(iva, False)
        if vl < best - 1e-4: best, pat = vl, 0; state = {k: v.clone() for k, v in model.state_dict().items()}
        else: pat += 1
        if pat >= patience: break
    model.load_state_dict(state); model.eval()
    @torch.no_grad()
    def pred(idx):
        D, L = [], []
        for i in range(0, len(idx), 256):
            det, loc = model(Xs[idx[i:i + 256]]); D.append(torch.sigmoid(det).numpy()); L.append(torch.sigmoid(loc).numpy())
        return np.concatenate(D), np.concatenate(L)
    dv, lv = pred(iva); dt, lt = pred(ite)
    ygv, ygt, ynv, ynt = yg[iva], yg[ite], yn[iva], yn[ite]
    # detection operating point: threshold from honest VALIDATION hours
    thr = np.quantile(dv[ygv == 0], 1 - FAR)
    tpr = float((dt[ygt == 1] > thr).mean()); far = float((dt[ygt == 0] > thr).mean())
    # localisation, on attacked hours; F1 threshold tuned on validation
    yv, pv = ynv[ygv == 1].ravel(), lv[ygv == 1].ravel()
    bt, bf = 0.5, -1.0
    for t_ in np.linspace(0.1, 0.95, 35):
        f = f1_score(yv, (pv > t_).astype(int)) if (pv > t_).any() else 0.0
        if f > bf: bf, bt = f, t_
    yt, pt = ynt[ygt == 1].ravel(), lt[ygt == 1].ravel()
    return dict(det_auc=float(roc_auc_score(ygt, dt)), det_tpr_at_far=tpr, det_far_test=far,
                loc_auc=float(roc_auc_score(yt, pt)),
                loc_f1=float(f1_score(yt, (pt > bt).astype(int))))

# ---------------------------------------------------------------- sweep
runs = []; t0 = time.time()
for seed in SEEDS:
    for p in DENS:
        rng = np.random.default_rng(seed * 100 + int(p * 100))
        k = max(int(round(p * nb)), 3)
        observed = np.zeros((H, nb), bool)
        for i in range(H): observed[i, rng.choice(nb, size=k, replace=False)] = True
        fp = pmu_features(observed); fs, scale = scada_features(observed, rng)
        for meas, feats in (("PMU", fp), ("SCADA", fs)):
            for name, cls, ep, bs, pat in (("GNN", GNN, 20, 64, 6), ("MLP", MLP, 25, 128, 6)):
                m = train_eval(cls, feats, seed, ep, bs, pat)
                runs.append(dict(seed=seed, coverage=p, sensors=k, measurement=meas, model=name, **m))
        last = {(r["measurement"], r["model"]): r for r in runs[-4:]}
        print(f"[seed {seed} cov {int(p*100):3d}%] " + " | ".join(
            f"{a} {b}: det@5% {last[a,b]['det_tpr_at_far']:.2f} loc {last[a,b]['loc_auc']:.3f}"
            for a in ("PMU", "SCADA") for b in ("GNN", "MLP")) + f"  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(runs)
metrics = ["det_tpr_at_far", "det_far_test", "det_auc", "loc_auc", "loc_f1"]
summ = df.groupby(["measurement", "coverage", "model"])[metrics].agg(["mean", "std"])
summary = []
for (meas, cov, model), row in summ.iterrows():
    e = dict(measurement=meas, coverage=cov, model=model)
    for m in metrics: e[m + "_mean"] = round(float(row[m, "mean"]), 3); e[m + "_std"] = round(float(row[m, "std"]), 3)
    summary.append(e)
json.dump(dict(
    protocol=dict(seeds=SEEDS, coverage=DENS, false_alarm_rate=FAR, ridge_lambda=LAM,
                  hours=int(H), test_samples=int((S == 2).sum()),
                  scaling="mean |observed injection| over training hours",
                  localisation_loss_weight=pos_w,
                  loss_weight_from="training hours",
                  noise="1% of the mean |injection| over all buses of the true state (simulator)",
                  threshold="95th percentile of honest validation scores",
                  masks="random observed set per hour, shared by the honest/attacked pair"),
    summary=summary, runs=runs), open(f"{OUT}/coverage_rerun_results.json", "w"), indent=1)
print(f"saved {OUT}/coverage_rerun_results.json ({time.time()-t0:.0f}s)")
