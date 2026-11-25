#!/usr/bin/env python3
"""Figure and tables for the rerun sensor-coverage test.
Reads coverage_rerun_results.json; writes figJ_coverage_rerun.png,
coverage_table.md and coverage_table.tex next to it."""
import json, os
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = os.environ.get("BIHGRID_OUT", ".")
res = json.load(open(f"{OUT}/coverage_rerun_results.json"))
s = pd.DataFrame(res["summary"]); nseed = len(res["protocol"]["seeds"])
cov = sorted(s.coverage.unique(), reverse=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight"})
panels = [("det_tpr_at_far", "attacked test hours detected (share)\nthreshold: 5% false alarms on validation", (0, 1.02)),
          ("loc_auc", "bus localisation ROC-AUC", (0.5, 1.0))]
names = {"PMU": "Voltage phasors at observed buses (PMU)",
         "SCADA": "Power injections at observed buses (SCADA)"}
fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4), sharex=True, sharey="row")
x = [int(round(c * 100)) for c in cov]
for j, meas in enumerate(["PMU", "SCADA"]):
    for i, (metric, ylab, ylim) in enumerate(panels):
        ax = axes[i, j]
        ends = {}
        for model, color, label in (("GNN", BLUE, "GNN"), ("MLP", ORANGE, "MLP (no graph)")):
            d = s[(s.measurement == meas) & (s.model == model)].set_index("coverage").loc[cov]
            mu, sd = d[metric + "_mean"].values, d[metric + "_std"].values
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, mu, "-o", color=color, lw=2, ms=5.5, mec=SURF, mew=1.2, label=label)
            ends[model] = mu[-1]
        # value labels at the last point; pushed apart when the two lines meet
        close = abs(ends["GNN"] - ends["MLP"]) < 0.06 * (ylim[1] - ylim[0])
        for model, v in ends.items():
            up = (v == max(ends.values()))
            ax.annotate(f"{v:.2f}", (x[-1], v), xytext=(6, (5 if up else -5) if close else 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        if metric == "det_tpr_at_far":
            ax.axhline(0.05, color=MUTED, lw=0.8, ls=(0, (3, 3)))
            if j == 1: ax.text(x[0], 0.065, "5% reference", fontsize=7.5, color=MUTED, va="bottom")
        ax.set_ylim(*ylim); ax.set_xlim(x[0] + 4, x[-1] - 9); ax.set_xticks(x)
        if i == 0: ax.set_title(names[meas], fontsize=9.5, loc="left")
        if j == 0: ax.set_ylabel(ylab)
        if i == 1: ax.set_xlabel("buses with a sensor (%)")
axes[0, 0].legend(loc="upper right", fontsize=8.5, ncol=1)
fig.suptitle(f"Sensor coverage: mean of {nseed} seeds, band = ±1 standard deviation",
             fontsize=10.5, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(f"{OUT}/figJ_coverage_rerun.png"); plt.close(fig)

def cell(meas, c, model, metric):
    r = s[(s.measurement == meas) & (s.coverage == c) & (s.model == model)].iloc[0]
    return r[metric + "_mean"], r[metric + "_std"]
# Detection is reported with the false-alarm rate actually realised on the
# honest test hours, next to the 5% target that set the threshold.
md = ["| measurement | coverage | detected, GNN | test false alarms, GNN | detected, MLP | test false alarms, MLP | localisation AUC, GNN | localisation AUC, MLP |",
      "|---|---|---|---|---|---|---|---|"]
tex = []
pm = lambda a, b, nd: f"{a:.{nd}f} ± {b:.{nd}f}"
pct = lambda a: f"{100*a:.1f}%"
for meas in ["PMU", "SCADA"]:
    tex.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{names[meas]}}}}} \\\\")
    for c in cov:
        v = {(m, k): cell(meas, c, m, k) for m in ("GNN", "MLP") for k in ("det_tpr_at_far", "det_far_test", "loc_auc")}
        cells = [pm(*v["GNN", "det_tpr_at_far"], 2), pct(v["GNN", "det_far_test"][0]),
                 pm(*v["MLP", "det_tpr_at_far"], 2), pct(v["MLP", "det_far_test"][0]),
                 pm(*v["GNN", "loc_auc"], 3), pm(*v["MLP", "loc_auc"], 3)]
        md.append(f"| {meas} | {int(round(c*100))}% | " + " | ".join(cells) + " |")
        tex.append(f"{int(round(c*100))}\\% & " + " & ".join(x.replace("±", "$\\pm$").replace("%", "\\%") for x in cells) + " \\\\")
    if meas == "PMU": tex.append("\\addlinespace")
open(f"{OUT}/coverage_table.md", "w").write("\n".join(md) + "\n")
open(f"{OUT}/coverage_table.tex", "w").write("\n".join(tex) + "\n")
far = s.det_far_test_mean
print("\n".join(md)); print(f"realised test false-alarm rate: {far.min():.3f} to {far.max():.3f} (mean {far.mean():.3f})")
