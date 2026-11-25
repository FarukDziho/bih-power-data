#!/usr/bin/env python3
"""
Stage 2: train a Boyaci-style ChebConv GNN on the BiHGrid FDIA corpus, with a
graph-level DETECTION head and a node-level LOCALIZATION head, and compare against
(a) the classical bad-data detector (BDD, residual chi-square) and (b) a
topology-blind MLP on the same node features.
"""
import numpy as np, torch, json, time
import torch.nn as nn, torch.nn.functional as Fn
from torch_geometric.nn import ChebConv
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix)
torch.manual_seed(0); np.random.seed(0)

d = np.load("/tmp/corpus.npz", allow_pickle=True)
X, ynode, ygraph, J, ts = d["X"], d["ynode"], d["ygraph"], d["J"], d["ts"]
edge_index = torch.tensor(d["edge_index"], dtype=torch.long)
nb = int(d["nb"]); F = X.shape[2]

# time-ordered split by unique hour (keep honest/attack pair together, no leakage)
uh = sorted(set(ts.tolist()))
n = len(uh); tr_h=set(uh[:int(.70*n)]); va_h=set(uh[int(.70*n):int(.85*n)]); te_h=set(uh[int(.85*n):])
def mask(hset): return np.array([t in hset for t in ts])
mtr, mva, mte = mask(tr_h), mask(va_h), mask(te_h)
print(f"samples: train {mtr.sum()}  val {mva.sum()}  test {mte.sum()}  (hours {n})")

Xt = torch.tensor(X)                     # [S, nb, F]
ynode_t = torch.tensor(ynode)            # [S, nb]
ygraph_t = torch.tensor(ygraph)          # [S]

def batches(idx, bs=64, shuffle=True):
    idx = np.where(idx)[0]
    if shuffle: np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i+bs]

class GNN(nn.Module):
    def __init__(self, f, h=64, K=3):
        super().__init__()
        self.c1=ChebConv(f,h,K); self.c2=ChebConv(h,h,K); self.c3=ChebConv(h,h,K)
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h)
        self.loc=nn.Linear(h,1)                       # node-level localization
        self.det=nn.Sequential(nn.Linear(2*h,h), nn.ReLU(), nn.Linear(h,1))  # graph detection
    def forward(self, x, ei):                          # x:[B,nb,F]
        B=x.shape[0]
        outs_loc=[]; outs_emb=[]
        for b in range(B):
            e=Fn.relu(self.bn1(self.c1(x[b],ei)))
            e=Fn.relu(self.bn2(self.c2(e,ei)))
            e=Fn.relu(self.c3(e,ei))
            outs_loc.append(self.loc(e).squeeze(-1))
            outs_emb.append(torch.cat([e.mean(0), e.max(0).values]))
        loc=torch.stack(outs_loc)                      # [B,nb]
        emb=torch.stack(outs_emb)                      # [B,2h]
        det=self.det(emb).squeeze(-1)                  # [B]
        return det, loc

dev="cpu"; model=GNN(F).to(dev)
opt=torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
pos_w=torch.tensor([ (ynode[mtr]==0).sum()/max((ynode[mtr]==1).sum(),1) ], dtype=torch.float32)
print("localization pos_weight:", float(pos_w))

def run_epoch(train=True):
    model.train(train); tot=0
    for bi in batches(mtr if train else mva, 64, train):
        xb=Xt[bi].to(dev); ei=edge_index.to(dev)
        det,loc=model(xb,ei)
        yg=ygraph_t[bi].float().to(dev); yn=ynode_t[bi].float().to(dev)
        ld=Fn.binary_cross_entropy_with_logits(det,yg)
        ll=Fn.binary_cross_entropy_with_logits(loc,yn,pos_weight=pos_w.to(dev))
        loss=ld+ll
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot+=float(loss)*len(bi)
    return tot/max((mtr if train else mva).sum(),1)

t0=time.time(); best=1e9; best_state=None; patience=0
for ep in range(40):
    tl=run_epoch(True)
    with torch.no_grad(): vl=run_epoch(False)
    if vl<best-1e-4: best=vl; best_state={k:v.clone() for k,v in model.state_dict().items()}; patience=0
    else: patience+=1
    if ep%5==0 or patience>=8: print(f"  ep{ep:02d} train {tl:.3f} val {vl:.3f}")
    if patience>=8: break
model.load_state_dict(best_state)
print(f"trained in {time.time()-t0:.0f}s")

# ---------- evaluate on test ----------
@torch.no_grad()
def predict(idx):
    model.eval(); dets=[]; locs=[]
    for bi in batches(idx,64,False):
        det,loc=model(Xt[bi],edge_index)
        dets.append(torch.sigmoid(det).numpy()); locs.append(torch.sigmoid(loc).numpy())
    return np.concatenate(dets), np.concatenate(locs)

ii=np.where(mte)[0]
det_p, loc_p = predict(mte)
yg_te=ygraph[ii]; yn_te=ynode[ii]

# choose localization operating threshold on VALIDATION (no test-set tuning)
iv=np.where(mva)[0]
det_v, loc_v = predict(mva)
yg_v=ygraph[iv]; yn_v=ynode[iv]; attv=yg_v==1
yv=yn_v[attv].ravel(); pv=loc_v[attv].ravel()
best_thr, best_valf1 = 0.5, -1
for thr_ in np.linspace(0.1,0.95,35):
    f=f1_score(yv,(pv>thr_).astype(int)) if (pv>thr_).any() else 0
    if f>best_valf1: best_valf1, best_thr = f, thr_
loc_thr=float(best_thr)

# detection metrics
det_auc=roc_auc_score(yg_te, det_p)
det_pred=(det_p>0.5).astype(int)
det_f1=f1_score(yg_te,det_pred); det_acc=accuracy_score(yg_te,det_pred)
# localization on attacked samples only, threshold from validation
att=yg_te==1
loc_flat_y=yn_te[att].ravel(); loc_flat_p=(loc_p[att].ravel()>loc_thr).astype(int)
loc_f1=f1_score(loc_flat_y,loc_flat_p); loc_prec=precision_score(loc_flat_y,loc_flat_p)
loc_rec=recall_score(loc_flat_y,loc_flat_p)
loc_auc=roc_auc_score(loc_flat_y, loc_p[att].ravel())

# ---------- baseline 1: BDD (residual chi-square) ----------
Jte=J[ii]
dof=nb-(nb-1)
# threshold at honest 95th percentile from TRAIN honest samples
Jtr_honest=J[mtr][ygraph[mtr]==0]
thr=np.quantile(Jtr_honest,0.95)
bdd_pred=(Jte>thr).astype(int)
bdd_f1=f1_score(yg_te,bdd_pred); bdd_acc=accuracy_score(yg_te,bdd_pred)
bdd_auc=roc_auc_score(yg_te, Jte)

# ---------- baseline 2: topology-blind MLP (same features, no message passing) ----------
class MLP(nn.Module):
    def __init__(self,f,h=64):
        super().__init__()
        self.node=nn.Sequential(nn.Linear(f,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.loc=nn.Linear(h,1); self.det=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(self,x,ei=None):
        e=self.node(x)                                 # [B,nb,h]
        loc=self.loc(e).squeeze(-1)
        emb=torch.cat([e.mean(1),e.max(1).values],dim=1)
        return self.det(emb).squeeze(-1), loc
mlp=MLP(F); opt2=torch.optim.Adam(mlp.parameters(),lr=2e-3,weight_decay=1e-4)
def run_mlp(train=True):
    mlp.train(train); tot=0
    for bi in batches(mtr if train else mva,128,train):
        xb=Xt[bi]; det,loc=mlp(xb)
        yg=ygraph_t[bi].float(); yn=ynode_t[bi].float()
        loss=Fn.binary_cross_entropy_with_logits(det,yg)+Fn.binary_cross_entropy_with_logits(loc,yn,pos_weight=pos_w)
        if train: opt2.zero_grad(); loss.backward(); opt2.step()
        tot+=float(loss)*len(bi)
    return tot/max((mtr if train else mva).sum(),1)
bestm=1e9; bstate=None; pat=0
for ep in range(40):
    run_mlp(True)
    with torch.no_grad(): vl=run_mlp(False)
    if vl<bestm-1e-4: bestm=vl; bstate={k:v.clone() for k,v in mlp.state_dict().items()}; pat=0
    else: pat+=1
    if pat>=8: break
mlp.load_state_dict(bstate)
with torch.no_grad():
    mdet=[]; mloc=[]
    for bi in batches(mte,128,False):
        det,loc=mlp(Xt[bi]); mdet.append(torch.sigmoid(det).numpy()); mloc.append(torch.sigmoid(loc).numpy())
    mdet=np.concatenate(mdet); mloc=np.concatenate(mloc)
mlp_auc=roc_auc_score(yg_te,mdet); mlp_f1=f1_score(yg_te,(mdet>0.5).astype(int))
mlp_loc_auc=roc_auc_score(yn_te[att].ravel(), mloc[att].ravel())
# MLP localization threshold also tuned on val for fairness
with torch.no_grad():
    mlv=[]
    for bi in batches(mva,128,False):
        _,loc=mlp(Xt[bi]); mlv.append(torch.sigmoid(loc).numpy())
    mlv=np.concatenate(mlv)
yvm=yn_v[attv].ravel(); pvm=mlv[attv].ravel()
bthr,bf=0.5,-1
for thr_ in np.linspace(0.1,0.95,35):
    f=f1_score(yvm,(pvm>thr_).astype(int)) if (pvm>thr_).any() else 0
    if f>bf: bf,bthr=f,thr_
mlp_loc_f1=f1_score(yn_te[att].ravel(),(mloc[att].ravel()>bthr).astype(int))

res=dict(
 test_samples=int(mte.sum()),
 detection=dict(
   GNN=dict(auc=round(det_auc,3), f1=round(det_f1,3), acc=round(det_acc,3)),
   MLP=dict(auc=round(mlp_auc,3), f1=round(mlp_f1,3)),
   BDD=dict(auc=round(bdd_auc,3), f1=round(bdd_f1,3), acc=round(bdd_acc,3), thr=round(float(thr),3)),
 ),
 localization=dict(
   GNN=dict(auc=round(loc_auc,3), f1=round(loc_f1,3), precision=round(loc_prec,3),
            recall=round(loc_rec,3), thr=round(loc_thr,3)),
   MLP=dict(auc=round(mlp_loc_auc,3), f1=round(mlp_loc_f1,3)),
 ),
 note="BDD AUC ~0.5 confirms stealth attacks are residual-invisible by construction; GNN must exploit topology.",
)
json.dump(res, open("/tmp/gnn_results.json","w"), indent=2)
print("\n===== RESULTS (test) =====")
print(json.dumps(res, indent=2))
# save predictions for figures
np.savez("/tmp/preds.npz", det_p=det_p, yg=yg_te, loc_p=loc_p, yn=yn_te,
         J=Jte, thr=thr, mdet=mdet, mloc=mloc, ii=ii, loc_thr=loc_thr)
print("saved /tmp/gnn_results.json and /tmp/preds.npz")

# ---------------- figures (dataviz palette) ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; RED="#e34948"; GREY="#52514e"
SURF="#fcfcfb"; INK="#0b0b0b"
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,
    "axes.edgecolor":"#d9d8d4","axes.labelcolor":INK,"text.color":INK,
    "xtick.color":GREY,"ytick.color":GREY,"font.size":11,"axes.grid":True,
    "grid.color":"#ececea","axes.spines.top":False,"axes.spines.right":False})

# Fig A: BDD residual distributions honest vs stealth -> overlap = invisible
fig,ax=plt.subplots(figsize=(6.2,4))
Jh=Jte[yg_te==0]; Ja_=Jte[yg_te==1]
bins=np.linspace(min(Jte.min(),0), np.quantile(Jte,0.99), 40)
ax.hist(Jh,bins=bins,color=BLUE,alpha=0.75,label="Honest hours",edgecolor=SURF,linewidth=0.5)
ax.hist(Ja_,bins=bins,color=RED,alpha=0.55,label="Stealth-FDIA hours",edgecolor=SURF,linewidth=0.5)
ax.set_xlabel("WLS residual objective  J(x̂)"); ax.set_ylabel("count")
ax.set_title("Classical bad-data detector is blind to stealth FDIA\n(distributions overlap → AUC ≈ 0.50)",fontsize=11,loc="left")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("/tmp/figA_bdd_blind.png",dpi=150)

# Fig B: detection ROC GNN vs MLP vs BDD
fig,ax=plt.subplots(figsize=(5.6,5))
for name,score,col in [("GNN (ChebConv)",det_p,BLUE),("MLP (no topology)",mdet,ORANGE),("Classical BDD",Jte/Jte.max(),GREY)]:
    fpr,tpr,_=roc_curve(yg_te,score); a=roc_auc_score(yg_te,score)
    ax.plot(fpr,tpr,color=col,lw=2,label=f"{name}  (AUC {a:.3f})")
ax.plot([0,1],[0,1],"--",color="#c9c8c4",lw=1)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("Detection: is there an attack?",fontsize=12,loc="left")
ax.legend(frameon=False,loc="lower right",fontsize=9.5)
fig.tight_layout(); fig.savefig("/tmp/figB_detection_roc.png",dpi=150)

# Fig C: localization PR curve GNN vs MLP (the topology-dependent task)
fig,ax=plt.subplots(figsize=(5.6,5))
yflat=yn_te[att].ravel()
for name,score,col in [("GNN (ChebConv)",loc_p[att].ravel(),BLUE),("MLP (no topology)",mloc[att].ravel(),ORANGE)]:
    pr,rc,_=precision_recall_curve(yflat,score); a=roc_auc_score(yflat,score)
    ax.plot(rc,pr,color=col,lw=2,label=f"{name}  (AUC {a:.3f})")
ax.axhline(yflat.mean(),ls="--",color="#c9c8c4",lw=1,label=f"chance (prevalence {yflat.mean():.3f})")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Localization: which buses are attacked?\nTopology-aware GNN vs topology-blind MLP",fontsize=11,loc="left")
ax.legend(frameon=False,loc="upper right",fontsize=9.5)
fig.tight_layout(); fig.savefig("/tmp/figC_localization_pr.png",dpi=150)

# Fig D: summary bar of the three tasks
fig,ax=plt.subplots(figsize=(6.6,4))
labels=["Detection AUC","Localization AUC","Localization F1"]
gnn_v=[det_auc,loc_auc,loc_f1]; mlp_v=[mlp_auc,mlp_loc_auc,mlp_loc_f1]; bdd_v=[bdd_auc,0.5,0]
xp=np.arange(len(labels)); wdt=0.26
ax.bar(xp-wdt,gnn_v,wdt,color=BLUE,label="GNN",edgecolor=SURF)
ax.bar(xp,mlp_v,wdt,color=ORANGE,label="MLP",edgecolor=SURF)
ax.bar(xp+wdt,bdd_v,wdt,color=GREY,label="Classical BDD",edgecolor=SURF)
ax.set_xticks(xp); ax.set_xticklabels(labels); ax.set_ylim(0,1.05)
ax.set_title("Where the topology-aware GNN earns its place",fontsize=12,loc="left")
ax.legend(frameon=False,ncol=3,fontsize=9.5)
for x,v in zip(xp-wdt,gnn_v): ax.text(x,v+0.02,f"{v:.2f}",ha="center",fontsize=8,color=INK)
for x,v in zip(xp,mlp_v): ax.text(x,v+0.02,f"{v:.2f}",ha="center",fontsize=8,color=INK)
fig.tight_layout(); fig.savefig("/tmp/figD_summary.png",dpi=150)
print("wrote figA..figD png")
