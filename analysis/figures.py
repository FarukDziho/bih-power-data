#!/usr/bin/env python3
"""Paper figures for the bihgrid data descriptor (dataviz reference palette)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import requests

import os
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("BIHGRID_REPO", HERE.parent.parent / "bih-power-data"))
OUT = Path(os.environ.get("BIHGRID_OUT", HERE / "out"))
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
    "#4a3aa7", "#e34948")
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

zone = pd.read_parquet(REPO / "dataset/zone_series.parquet")
nodes = pd.read_csv(REPO / "dataset/nodes.csv", dtype={"bus_id": str})
edges = pd.read_csv(REPO / "dataset/edges.csv", dtype={"bus0": str, "bus1": str})
plants = pd.read_csv(REPO / "topology/plants.csv", dtype={"bus_id": str})
geo = json.loads((REPO / "topology/out/ba_topology.geojson").read_text())

# ================================================== FIG 1: map
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.grid(False)
ax.set_aspect(1 / np.cos(np.radians(43.9)))
try:
    r = requests.get("https://raw.githubusercontent.com/johan/world.geo.json/"
                     "master/countries/BIH.geo.json", timeout=30)
    for poly in r.json()["features"][0]["geometry"]["coordinates"]:
        arr = np.array(poly[0] if isinstance(poly[0][0], list) else poly)
        ax.fill(arr[:, 0], arr[:, 1], color="#f0efec", zorder=0)
        ax.plot(arr[:, 0], arr[:, 1], color=BASE, lw=0.8, zorder=1)
except Exception as e:
    print("border skipped:", e)
for f in geo["features"]:
    if f["geometry"]["type"] == "LineString":
        arr = np.array(f["geometry"]["coordinates"])
        v = f["properties"].get("voltage")
        ax.plot(arr[:, 0], arr[:, 1], lw=1.6 if v == 400 else 1.0,
                color=BLUE if v == 400 else ORANGE, alpha=0.9, zorder=2)
for role, mk, sz in [("ba", "o", 22), ("border", "s", 20)]:
    sel = nodes[nodes.role == role]
    ax.scatter(sel.x, sel.y, s=sz, marker=mk, c=INK2 if role == "border" else INK,
               edgecolors=SURF, linewidths=0.7, zorder=4)
pl = plants.merge(nodes[["bus_id", "x", "y"]], on="bus_id")
kind_color = {"Fossil": VIOLET, "Hydro": AQUA, "Wind": MAGENTA}
for kind, color in kind_color.items():
    sel = pl[pl.entsoe_type.str.startswith(kind)]
    ax.scatter(sel.x, sel.y, s=sel.capacity_mw / 4, facecolors="none",
               edgecolors=color, linewidths=1.6, zorder=5)
handles = [
    Line2D([], [], color=BLUE, lw=1.6, label="400 kV line"),
    Line2D([], [], color=ORANGE, lw=1.0, label="220 kV line"),
    Line2D([], [], marker="o", ls="", color=INK, ms=5, label="BA bus"),
    Line2D([], [], marker="s", ls="", color=INK2, ms=5, label="Border bus"),
    Line2D([], [], marker="o", ls="", markerfacecolor="none",
           markeredgecolor=VIOLET, ms=8, label="Coal plant"),
    Line2D([], [], marker="o", ls="", markerfacecolor="none",
           markeredgecolor=AQUA, ms=8, label="Hydro plant"),
    Line2D([], [], marker="o", ls="", markerfacecolor="none",
           markeredgecolor=MAGENTA, ms=8, label="Wind farm"),
]
ax.legend(handles=handles, loc="upper left", fontsize=7.5, ncol=2)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("bihgrid topology: ≥220 kV network, plants sized by capacity",
             fontsize=10, color=INK, loc="left")
fig.savefig(OUT / "fig1_map.png"); plt.close(fig)

# ================================================== FIG 2: coverage heatmap
sig = [c for c in zone.columns if not c.startswith("sched_")]
lab = {c: c.replace("gen_", "").replace("_mw", "").replace("_", " ")
       for c in sig}
monthly = zone[sig].notna().resample("MS").mean() * 100
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.grid(False)
im = ax.imshow(monthly.T.values, aspect="auto", cmap=CMAP, vmin=0, vmax=100,
               interpolation="nearest")
years = pd.date_range(monthly.index[0], monthly.index[-1], freq="YS")
ax.set_xticks([monthly.index.get_indexer([y], method="nearest")[0] for y in years])
ax.set_xticklabels([y.year for y in years], fontsize=7.5)
ax.set_yticks(range(len(sig)))
ax.set_yticklabels([lab[c] for c in sig], fontsize=7.5)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
cb.set_label("hours with data (%)", fontsize=8)
cb.outline.set_visible(False)
ax.set_title("Monthly data availability per signal", fontsize=10, loc="left")
fig.savefig(OUT / "fig2_coverage.png"); plt.close(fig)

# ================================================== FIG 3: consistency
gen_cols = [c for c in zone.columns if c.startswith("gen_")]
z = zone.copy()
z["gen_total"] = z[gen_cols].sum(axis=1, min_count=1)
z["supply"] = z["gen_total"] - z["net_position_mw"]
# same-hours comparison (see validate.py): months are averaged over hours
# where both load and supply are reported
both = z.dropna(subset=["load_actual_mw", "supply"])
mo = both[["load_actual_mw", "supply"]].resample("MS").mean().loc["2017-03":]
fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True,
                         height_ratios=[2, 1])
ax = axes[0]
ax.plot(mo.index, mo.load_actual_mw, color=BLUE, lw=1.4, label="load")
ax.plot(mo.index, mo.supply, color=ORANGE, lw=1.4,
        label="generation − net export")
ax.set_ylabel("monthly mean MW")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_title("Load and generation minus net export, hours with both reported",
             fontsize=10, loc="left")
ax2 = axes[1]
gap = 100 * (mo.supply - mo.load_actual_mw) / mo.supply
ax2.axhline(0, color=BASE, lw=0.8)
ax2.fill_between(gap.index, gap, 0, color=INK2, alpha=0.25, linewidth=0)
ax2.plot(gap.index, gap, color=INK2, lw=1.0)
ax2.set_ylabel("gap (% of supply)")
fig.savefig(OUT / "fig3_consistency.png"); plt.close(fig)

# ================================================== FIG 4: generation mix
groups = {
    "coal": (["gen_fossil_brown_coal_lignite_mw", "gen_fossil_hard_coal_mw"], BLUE),
    "hydro reservoir": (["gen_hydro_water_reservoir_mw"], ORANGE),
    "hydro RoR": (["gen_hydro_run_of_river_and_poundage_mw"], AQUA),
    "pumped storage": (["gen_hydro_pumped_storage_mw"], YELLOW),
    "wind": (["gen_wind_onshore_mw"], MAGENTA),
    "solar": (["gen_solar_mw"], GREEN),
}
mix = pd.DataFrame({k: zone[cols].sum(axis=1, min_count=1)
                    for k, (cols, _) in groups.items()})
mm = mix.resample("MS").mean().fillna(0).loc["2017-03":] / 1000
fig, ax = plt.subplots(figsize=(8.2, 3.6))
ax.stackplot(mm.index, [mm[k] for k in groups], labels=list(groups),
             colors=[c for _, c in groups.values()], lw=0)
ax.set_ylabel("monthly mean GW")
ax.legend(loc="upper right", fontsize=7.5, ncol=3)
ax.set_title("Generation mix (ENTSO-E per-type, as reported)", fontsize=10,
             loc="left")
ax.margins(x=0)
fig.savefig(OUT / "fig4_genmix.png"); plt.close(fig)

# ================================================== FIG 5: graph stats
import networkx as nx
G = nx.MultiGraph()
for _, r in nodes.iterrows():
    G.add_node(r.bus_id, role=r.role)
for _, r in edges.iterrows():
    G.add_edge(r.bus0, r.bus1)
deg = pd.Series(dict(G.degree()))
lines = edges[edges.kind == "line"].copy()
lines["r_km"] = lines.r / lines.length_km
lines["x_km"] = lines.x / lines.length_km
fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7))
ax = axes[0]
ax.hist(deg.values, bins=range(1, deg.max() + 2), color=BLUE,
        edgecolor=SURF, linewidth=1.2, align="left", rwidth=0.92)
ax.set_xlabel("bus degree"); ax.set_ylabel("count")
ax.set_title("Degree distribution", fontsize=9, loc="left")
for i, (col, name, lit) in enumerate([
        ("r_km", "resistance (Ω/km)", (0.02, 0.04, 0.05, 0.12)),
        ("x_km", "reactance (Ω/km)", (0.25, 0.34, 0.38, 0.43))]):
    ax = axes[i + 1]
    for v, color, dx in [(400, BLUE, -0.12), (220, ORANGE, 0.12)]:
        sel = lines[lines.voltage == v][col].dropna()
        xj = np.full(len(sel), (0 if v == 400 else 1))
        ax.scatter(xj + np.random.default_rng(1).uniform(-0.1, 0.1, len(sel)),
                   sel, s=14, color=color, alpha=0.75, edgecolors="none")
    lo4, hi4, lo2, hi2 = lit
    ax.fill_between([-0.25, 0.25], lo4, hi4, color=BLUE, alpha=0.12, lw=0)
    ax.fill_between([0.75, 1.25], lo2, hi2, color=ORANGE, alpha=0.12, lw=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["400 kV", "220 kV"])
    ax.set_title(name, fontsize=9, loc="left")
    ax.set_xlim(-0.5, 1.5)
axes[1].set_ylim(0, 0.2); axes[2].set_ylim(0, 0.6)
fig.suptitle("Line parameters vs typical OHL ranges (shaded)", fontsize=10,
             x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig5_graphstats.png"); plt.close(fig)

print("figures written:", sorted(p.name for p in OUT.glob("fig*.png")))
