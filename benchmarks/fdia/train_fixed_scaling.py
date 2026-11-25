#!/usr/bin/env python3
"""
Full-observability benchmark with the injection feature rescaled.

train_gnn.py (the saved run in gnn_results_pop.json) uses the corpus features
as built: the injection feature is divided by the mean absolute injection of
the hour's UNATTACKED state. A detector would not know that number, and it
makes honest samples average exactly 1 on this feature, so one statistic of
it separates attacked from honest hours almost perfectly.

This script undoes that division (corpus field `pnorm`) and divides by one
constant instead, the mean absolute injection estimate over the training
samples. Everything else follows train_gnn.py: same corpus, same split, same
models and losses, 40 epochs with patience 8. Five seeds; mean and standard
deviation are reported. Detection is given as ROC-AUC, F1 at a 0.5 threshold
(as in the saved run) and the share of attacked hours detected at a 5%
false-alarm rate with the threshold set on honest validation hours.

Env: BIHGRID_CORPUS (needs `pnorm`, written by build_corpus_pop.py), BIHGRID_OUT
"""
import json, os, time
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as Fn
from torch_geometric.nn import ChebConv
from sklearn.metrics import roc_auc_score, f1_score

CORPUS = os.environ.get("BIHGRID_CORPUS", "corpus_bih.npz")
OUT = os.environ.get("BIHGRID_OUT", ".")
SEEDS = [int(s) for s in os.environ.get("BIHGRID_SEEDS", "0,1,2,3,4").split(",")]
FAR = 0.05
torch.set_num_threads(2)

d = np.load(CORPUS, allow_pickle=True)
X, yn, yg, ts, pnorm = d["X"].copy(), d["ynode"], d["ygraph"], d["ts"], d["pnorm"]
edge_index = torch.tensor(d["edge_index"], dtype=torch.long); nb = int(d["nb"])
uh = sorted(set(ts.tolist())); n = len(uh)
tr_h, va_h = set(uh[:int(.70 * n)]), set(uh[int(.70 * n):int(.85 * n)])
S = np.array([0 if t in tr_h else 1 if t in va_h else 2 for t in ts])
itr, iva, ite = (np.where(S == k)[0] for k in range(3))

def one_stat_auc(feat):                      # detection from mean |injection feature| alone
    return float(roc_auc_score(yg[ite], np.abs(feat[ite]).mean(1)))
diag = {"saved_scaling": one_stat_auc(X[:, :, 3])}
raw = X[:, :, 3] * pnorm[:, None]            # undo the per-hour division
const = float(np.abs(raw[itr]).mean())       # one constant, from training samples
diag["train_constant_scaling"] = one_stat_auc(raw / const)
# BIHGRID_SAVED_SCALING=1 keeps the saved run's features, to separate the
# effect of the scaling from the effect of the trainer.
SAVED_SCALING = os.environ.get("BIHGRID_SAVED_SCALING", "0") == "1"
if not SAVED_SCALING:
    X[:, :, 3] = raw / const
print("detection ROC-AUC from mean |injection feature| alone:", diag, flush=True)

class GNN(nn.Module):
    def __init__(self, f, h=64, K=3):
        super().__init__()
        self.c1 = ChebConv(f, h, K); self.c2 = ChebConv(h, h, K); self.c3 = ChebConv(h, h, K)
        self.b1 = nn.BatchNorm1d(h); self.b2 = nn.BatchNorm1d(h)
        self.loc = nn.Linear(h, 1)
        self.det = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x):
        Bt = x.shape[0]
        off = (torch.arange(Bt) * nb).repeat_interleave(edge_index.shape[1])
        e_idx = edge_index.repeat(1, Bt) + off
        e = x.reshape(Bt * nb, -1)
        e = Fn.relu(self.b1(self.c1(e, e_idx))); e = Fn.relu(self.b2(self.c2(e, e_idx)))
        e = Fn.relu(self.c3(e, e_idx)).reshape(Bt, nb, -1)
        return (self.det(torch.cat([e.mean(1), e.max(1).values], 1)).squeeze(-1),
                self.loc(e).squeeze(-1))

class MLP(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.loc = nn.Linear(h, 1)
        self.det = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x):
        e = self.node(x)
        return (self.det(torch.cat([e.mean(1), e.max(1).values], 1)).squeeze(-1),
                self.loc(e).squeeze(-1))

Xs = torch.tensor(X); Yg = torch.tensor(yg).float(); Yn = torch.tensor(yn).float()
pos_w = torch.tensor([float((yn[itr] == 0).sum() / max((yn[itr] == 1).sum(), 1))])

def train_eval(cls, seed, bs):
    torch.manual_seed(seed); rs = np.random.default_rng(1000 + seed)
    model = cls(X.shape[2]); opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    def run(idx, train):
        model.train(train); tot = 0.0
        idx = rs.permutation(idx) if train else idx
        for i in range(0, len(idx), bs):
            bi = idx[i:i + bs]; det, loc = model(Xs[bi])
            loss = (Fn.binary_cross_entropy_with_logits(det, Yg[bi]) +
                    Fn.binary_cross_entropy_with_logits(loc, Yn[bi], pos_weight=pos_w))
            if train: opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(bi)
        return tot / len(idx)
    best, state, pat = 1e9, None, 0
    for _ in range(40):
        run(itr, True)
        with torch.no_grad(): vl = run(iva, False)
        if vl < best - 1e-4: best, pat = vl, 0; state = {k: v.clone() for k, v in model.state_dict().items()}
        else: pat += 1
        if pat >= 8: break
    model.load_state_dict(state); model.eval()
    @torch.no_grad()
    def pred(idx):
        D, L = [], []
        for i in range(0, len(idx), 256):
            det, loc = model(Xs[idx[i:i + 256]]); D.append(torch.sigmoid(det).numpy()); L.append(torch.sigmoid(loc).numpy())
        return np.concatenate(D), np.concatenate(L)
    dv, lv = pred(iva); dt, lt = pred(ite)
    ygv, ygt = yg[iva], yg[ite]
    thr = np.quantile(dv[ygv == 0], 1 - FAR)
    yv, pv = yn[iva][ygv == 1].ravel(), lv[ygv == 1].ravel(); bt, bf = 0.5, -1.0
    for t_ in np.linspace(0.1, 0.95, 35):
        f = f1_score(yv, (pv > t_).astype(int)) if (pv > t_).any() else 0.0
        if f > bf: bf, bt = f, t_
    yt, pt = yn[ite][ygt == 1].ravel(), lt[ygt == 1].ravel()
    return dict(det_auc=float(roc_auc_score(ygt, dt)),
                det_f1=float(f1_score(ygt, (dt > 0.5).astype(int))),
                det_tpr_at_far=float((dt[ygt == 1] > thr).mean()),
                det_far_test=float((dt[ygt == 0] > thr).mean()),
                loc_auc=float(roc_auc_score(yt, pt)),
                loc_f1=float(f1_score(yt, (pt > bt).astype(int))))

runs = []; t0 = time.time()
for seed in SEEDS:
    for name, cls, bs in (("GNN", GNN, 64), ("MLP", MLP, 128)):
        r = train_eval(cls, seed, bs); runs.append(dict(seed=seed, model=name, **r))
        print(f"[seed {seed}] {name}: " + " ".join(f"{k} {v:.3f}" for k, v in r.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)
keys = [k for k in runs[0] if k not in ("seed", "model")]
summary = {m: {k: {"mean": round(float(np.mean([r[k] for r in runs if r["model"] == m])), 3),
                   "std": round(float(np.std([r[k] for r in runs if r["model"] == m], ddof=1)), 3)}
               for k in keys} for m in ("GNN", "MLP")}
json.dump(dict(protocol=dict(seeds=SEEDS, test_samples=int(len(ite)), scaling_constant=const,
                             false_alarm_rate=FAR,
                             scaling=("as in the saved run (per-hour, from the unattacked state)" if SAVED_SCALING else
                                      "injection feature divided by mean |injection estimate| over training samples"),
                             training="one disjoint graph per sample, mini-batches of 64 (GNN) and 128 (MLP)"),
               one_statistic_detection_auc=diag, summary=summary, runs=runs),
          open(f"{OUT}/" + ("gnn_results_saved_scaling_batched.json" if SAVED_SCALING
                            else "gnn_results_fixed_scaling.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
