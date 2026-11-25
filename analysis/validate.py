#!/usr/bin/env python3
"""Technical-validation analyses + paper figures for the bihgrid data paper."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import networkx as nx

import os
# Point BIHGRID_REPO at a checkout of bih-power-data at the release being
# validated (the paper uses the v0.1.3 tag); outputs go to BIHGRID_OUT.
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("BIHGRID_REPO", HERE.parent.parent / "bih-power-data"))
OUT = Path(os.environ.get("BIHGRID_OUT", HERE / "out"))
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (dataviz reference, light mode) ----
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
    "#4a3aa7", "#e34948")
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

results = {}

# ================================================================ load data
zone = pd.read_parquet(REPO / "dataset/zone_series.parquet")
nodes = pd.read_csv(REPO / "dataset/nodes.csv", dtype={"bus_id": str})
edges = pd.read_csv(REPO / "dataset/edges.csv", dtype={"bus0": str, "bus1": str})
plants = pd.read_csv(REPO / "topology/plants.csv", dtype={"bus_id": str})
summary = json.loads((REPO / "dataset/dataset_summary.json").read_text())

# ============================================== 1. topology validation
lines = edges[edges.kind == "line"].copy()
lines["r_km"] = lines["r"] / lines["length_km"]
lines["x_km"] = lines["x"] / lines["length_km"]
G = nx.MultiGraph()
for _, r in nodes.iterrows():
    G.add_node(r.bus_id, role=r.role)
for _, r in edges.iterrows():
    G.add_edge(r.bus0, r.bus1)
ba_nodes = [n for n, d in G.nodes(data=True) if d["role"] == "ba"]
GBA = G.subgraph(ba_nodes)
deg = dict(G.degree())
simple_ba = nx.Graph(GBA)

by_v = lines.groupby("voltage").agg(
    n=("r_km", "size"), km=("length_km", "sum"),
    r_km_med=("r_km", "median"), x_km_med=("x_km", "median")).round(4)
internal = lines[~lines.cross_border.astype(bool)]
results["topology"] = {
    "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
    "connected": nx.is_connected(G),
    "ba_connected": nx.is_connected(GBA),
    "ba_diameter": nx.diameter(simple_ba),
    "mean_degree": round(np.mean(list(deg.values())), 2),
    "max_degree": int(max(deg.values())),
    "line_km_total": round(lines.length_km.sum(), 0),
    "line_km_internal": round(internal.length_km.sum(), 0),
    "per_voltage": by_v.reset_index().to_dict("records"),
    # literature typical OHL params (ohm/km): 400kV r 0.02-0.04, x 0.25-0.34;
    # 220kV r 0.05-0.12, x 0.38-0.43 (Kundur; ENTSO-E standard line data)
}

# ============================================== 2. coverage
sig_cols = [c for c in zone.columns if not c.startswith("sched_")]
monthly = zone[sig_cols].notna().resample("MS").mean() * 100
results["coverage"] = {
    "rows": len(zone),
    "per_signal_pct": {c: round(100 * zone[c].notna().mean(), 1) for c in sig_cols},
    "cleaning": summary["zone_series"].get("generation_outliers_set_nan"),
    "load_zero_hours": summary["zone_series"].get("load_zero_hours_set_nan"),
}

# ============================================== 3. internal consistency
gen_cols = [c for c in zone.columns if c.startswith("gen_") and
            "forecast" not in c]
z = zone.copy()
z["gen_total"] = z[gen_cols].sum(axis=1, min_count=1)
z["supply"] = z["gen_total"] - z["net_position_mw"]  # gen - net export
# Compare load and supply on the SAME hours. Averaging each series over its
# own available hours (as the first version of this script did) mixes
# different parts of the year whenever one series has gaps: in 2024 load is
# missing for 624 hours, which moved the annual gap from +3.5% to -3.7%.
both = z.dropna(subset=["load_actual_mw", "supply"])
ann = both[["load_actual_mw", "supply"]].resample("YE").mean().dropna()
ann["gap_pct"] = 100 * (ann["supply"] - ann["load_actual_mw"]) / ann["supply"]
ann["matched_hours"] = both["supply"].resample("YE").count()
ann = ann[ann.index.year >= 2017]  # load starts 2017-02
results["consistency"] = {
    "method": "hours with both load and supply reported",
    "annual_gap_pct": {str(i.year): round(v, 1) for i, v in ann.gap_pct.items()},
    "matched_hours": {str(i.year): int(v) for i, v in ann.matched_hours.items()},
    "gap_mean": round(ann.gap_pct.mean(), 1),
    "gap_std": round(ann.gap_pct.std(), 1),
}
# flows vs schedules
corr = {}
for b in ["HR", "RS", "ME"]:
    pair = zone[[f"net_import_{b}_mw", f"sched_net_import_{b}_mw"]].dropna()
    corr[b] = round(pair.corr().iloc[0, 1], 3)
results["consistency"]["flow_vs_sched_corr"] = corr

# ============================================== 4. external cross-check
# window matching lowcarbonpower (May 2025 - Apr 2026), shares of total gen
win = z.loc["2025-05-01":"2026-04-30", gen_cols]
tot = win.sum().sum()
share = lambda cols: round(100 * win[cols].sum().sum() / tot, 1)
ours = {
    "coal": share([c for c in gen_cols if "fossil" in c]),
    "hydro": share([c for c in gen_cols if "hydro" in c]),
    "wind": share(["gen_wind_onshore_mw"]),
    "solar": share(["gen_solar_mw"]),
}
results["external"] = {
    "window": "2025-05..2026-04",
    "our_shares_pct": ours,
    "lowcarbonpower_pct": {"coal": 53.6, "hydro": 38.3, "wind": 4.6, "solar": 3.6},
    "annual_gen_twh": {str(i.year): round(v, 2) for i, v in
                       (z["gen_total"].resample("YE").sum() / 1e6).items()
                       if i.year >= 2018},
}

# Regulator cross-check for 2024 and 2025. Dataset totals are annual mean MW
# times hours in the year (coverage-adjusted). Regulator figures are the
# SERC/DERK annual reports: hydro, wind and solar are plants connected to
# the transmission system; small plants on the distribution system are
# reported separately (710 GWh in 2024, 1,042 GWh in 2025) and are not in
# these rows.
REG = {
    "2024": {"coal": 8483.0, "hydro": 4732.0, "wind": 393.0, "solar": 209.0,
             "consumption": 12131.0},
    "2025": {"coal": 7933.9, "hydro": 4432.6, "wind": 648.4, "solar": 474.2,
             "consumption": 12330.4},
}
COLS = {
    "coal": [c for c in gen_cols if "fossil" in c],
    "hydro": [c for c in gen_cols if "hydro" in c],
    "wind": ["gen_wind_onshore_mw"],
    "solar": ["gen_solar_mw"],
}
reg_out = {}
for yr, reg in REG.items():
    y = z.loc[yr]
    hours = 8784 if int(yr) % 4 == 0 else 8760
    ours = {k: float(y[cols].sum(axis=1, min_count=1).mean() * hours / 1e3)
            for k, cols in COLS.items()}
    ours["consumption"] = float(y["load_actual_mw"].mean() * hours / 1e3)
    row = {k: {"dataset_gwh": round(ours[k]), "regulator_gwh": round(reg[k]),
               "diff_pct": round(100 * (ours[k] - reg[k]) / reg[k], 1)}
           for k in reg}
    gen_ours = sum(ours[k] for k in COLS)
    gen_reg = sum(reg[k] for k in COLS)
    row["generation_transmission_connected"] = {
        "dataset_gwh": round(gen_ours), "regulator_gwh": round(gen_reg),
        "diff_pct": round(100 * (gen_ours - gen_reg) / gen_reg, 1)}
    reg_out[yr] = row
results["regulator"] = reg_out

# ============================================== 5. mapping audit
plants["low_conf"] = plants.connection_confidence == "low"
audit = plants.groupby("entsoe_type").apply(
    lambda g: pd.Series({
        "capacity_mw": g.capacity_mw.sum(),
        "low_conf_mw": g.loc[g.low_conf, "capacity_mw"].sum(),
    }), include_groups=False).round(1)
audit["low_conf_share_pct"] = (100 * audit.low_conf_mw / audit.capacity_mw).round(1)
results["mapping_audit"] = audit.reset_index().to_dict("records")
results["mapping_coverage"] = summary["generation_mapping_coverage"]

json_path = OUT / "validation_numbers.json"
json_path.write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1)[:3000])
print("... numbers written to", json_path)
