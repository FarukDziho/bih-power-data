#!/usr/bin/env python3
"""
Joins the ENTSO-E BA-zone time series (data/*.parquet) onto the >=220kV
transmission topology (topology/out/) and writes a graph-ready dataset
into dataset/.

Design decisions (documented in dataset/DESIGN.md, generated here):

1. BA is ONE bidding zone -> hourly signals are zone-level. We ship them as
   GRAPH-LEVEL features (dataset/zone_series.parquet, UTC index), untouched.
2. Generation is ADDITIONALLY attached to buses via a curated plant table
   (topology/plants.csv): each ENTSO-E production type gets a bus-weight
   vector (capacity share of mapped plants). Unmapped capacity per type is
   reported explicitly — users can renormalise or keep a zone residual.
   Load is NOT disaggregated (no defensible sub-zone data); nodes carry the
   nearest_place label so users can apply their own population heuristic.
3. Parallel circuits are PRESERVED: edges.csv has one row per line record
   (62 lines + transformers), and ba_graph_multi.graphml is a MultiGraph.
   The `circuits` attribute is kept as an edge feature.
4. Obvious sensor errors in generation_per_type (values > OUTLIER_FACTOR x
   installed capacity of that type) are set to NaN and counted, never
   silently dropped.

Everything reads from the repo checkout; no network access needed.
Run:  python src/join_graph.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOPO = ROOT / "topology" / "out"
DATA = ROOT / "data"
OUT = ROOT / "dataset"

OUTLIER_FACTOR = 3.0  # x installed capacity => treat as bad datapoint

ZONE_FILES = {
    "load_actual": ("load_actual.parquet", "Actual Load", "load_actual_mw"),
    "load_forecast": ("load_forecast.parquet", None, "load_forecast_mw"),
    "generation_forecast": ("generation_forecast.parquet", None, "generation_forecast_mw"),
}
BORDERS = ["HR", "RS", "ME"]


def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_zone_series() -> tuple[pd.DataFrame, dict]:
    frames, notes = [], {}

    for key, (fname, colname, outname) in ZONE_FILES.items():
        df = to_utc(pd.read_parquet(DATA / fname))
        col = colname if colname in df.columns else df.columns[0]
        frames.append(df[[col]].rename(columns={col: outname}))

    # generation per type, with outlier cleaning against installed capacity
    gen = to_utc(pd.read_parquet(DATA / "generation_per_type.parquet"))
    cap = to_utc(pd.read_parquet(DATA / "installed_capacity.parquet"))
    cleaned = {}
    for c in gen.columns:
        s = gen[c].astype(float)
        if c in cap.columns and cap[c].notna().any():
            # annual capacity, forward-filled onto the hourly index
            cap_h = cap[c].reindex(gen.index.union(cap.index)).ffill().reindex(gen.index)
            # capacity can be NaN before first reporting year -> use overall max
            cap_h = cap_h.fillna(cap[c].max())
            bad = s > OUTLIER_FACTOR * cap_h
            cleaned[c] = int(bad.sum())
            s = s.mask(bad)
        gen[c] = s
    gen.columns = ["gen_" + c.lower().replace(" ", "_").replace("/", "_")
                   .replace("-", "_") + "_mw" for c in gen.columns]
    frames.append(gen)
    notes["generation_outliers_set_nan"] = cleaned
    notes["outlier_rule"] = f"value > {OUTLIER_FACTOR} x installed capacity of type"

    # cross-border physical flows -> per-border net import + zone net position
    for b in BORDERS:
        exp = to_utc(pd.read_parquet(DATA / f"flow_BA_{b}.parquet")).iloc[:, 0]
        imp = to_utc(pd.read_parquet(DATA / f"flow_{b}_BA.parquet")).iloc[:, 0]
        frames.append(pd.DataFrame({f"net_import_{b}_mw": imp.sub(exp, fill_value=np.nan)}))
        frames.append(pd.DataFrame({f"sched_net_import_{b}_mw":
            to_utc(pd.read_parquet(DATA / f"sched_{b}_BA.parquet")).iloc[:, 0]
            .sub(to_utc(pd.read_parquet(DATA / f"sched_BA_{b}.parquet")).iloc[:, 0],
                 fill_value=np.nan)}))

    zone = pd.concat(frames, axis=1).sort_index()

    # load cleaning: zero-load hours are metering gaps, not demand; spikes
    # far above any plausible BA peak (~2.2 GW) are sensor errors
    la = zone["load_actual_mw"]
    zero_hours = int((la == 0).sum())
    hi = 2 * la[la > 0].quantile(0.999)
    spike_hours = int((la > hi).sum())
    zone["load_actual_mw"] = la.mask((la == 0) | (la > hi))
    notes["load_zero_hours_set_nan"] = zero_hours
    notes["load_spikes_set_nan"] = {"count": spike_hours,
                                    "threshold_mw": round(float(hi), 1)}
    zone["net_position_mw"] = -zone[[f"net_import_{b}_mw" for b in BORDERS]].sum(
        axis=1, min_count=1)  # positive = net exporter
    zone.index.name = "time_utc"
    return zone, notes


def build_nodes_and_weights() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    buses = pd.read_csv(TOPO / "ba_buses.csv", dtype={"bus_id": str})
    places = pd.read_csv(ROOT / "topology" / "ba_bus_places.csv",
                         dtype={"bus_id": str})[["bus_id", "place", "dist_km"]]
    places = places.rename(columns={"place": "nearest_place",
                                    "dist_km": "nearest_place_km"})
    nodes = buses.merge(places, on="bus_id", how="left")

    plants = pd.read_csv(ROOT / "topology" / "plants.csv", dtype={"bus_id": str})
    unknown = set(plants["bus_id"]) - set(nodes["bus_id"])
    if unknown:
        raise SystemExit(f"plants.csv references unknown buses: {unknown}")

    # per-bus installed capacity per ENTSO-E type (static node features)
    pivot = plants.pivot_table(index="bus_id", columns="entsoe_type",
                               values="capacity_mw", aggfunc="sum")
    pivot.columns = ["plant_" + c.lower().replace(" ", "_").replace("/", "_") + "_mw"
                     for c in pivot.columns]
    nodes = nodes.merge(pivot.reset_index(), on="bus_id", how="left")
    for c in pivot.columns:
        nodes[c] = nodes[c].fillna(0.0)

    # weight matrix: share of MAPPED capacity of each type, per bus
    w = plants.pivot_table(index="bus_id", columns="entsoe_type",
                           values="capacity_mw", aggfunc="sum").fillna(0.0)
    weights = (w / w.sum()).reset_index().melt(
        id_vars="bus_id", var_name="entsoe_type", value_name="weight")
    weights = weights[weights["weight"] > 0].sort_values(
        ["entsoe_type", "weight"], ascending=[True, False])

    # ENTSO-E BA splits coal between "Fossil Brown coal/Lignite" and
    # "Fossil Hard coal" inconsistently over the years (same 5 plants,
    # shifting classification; hard-coal capacity stops being reported).
    # Provide a combined-coal weight vector so users can join the SUM of
    # both fossil columns without worrying about the split.
    coal = plants[plants["entsoe_type"].str.startswith("Fossil")]
    cw = coal.groupby("bus_id")["capacity_mw"].sum()
    cw = (cw / cw.sum()).reset_index()
    cw["entsoe_type"] = "Fossil coal (combined)"
    cw = cw.rename(columns={"capacity_mw": "weight"})
    weights = pd.concat([weights, cw[["bus_id", "entsoe_type", "weight"]]],
                        ignore_index=True)

    # mapped-capacity coverage vs latest ENTSO-E installed capacity
    cap = pd.read_parquet(DATA / "installed_capacity.parquet")
    latest = cap.ffill().iloc[-1]
    coverage = {}
    for t, mapped in w.sum().items():
        installed = float(latest.get(t, np.nan))
        coverage[t] = {
            "mapped_mw": round(float(mapped), 1),
            "entsoe_installed_mw": None if np.isnan(installed) else round(installed, 1),
            "mapped_fraction": None if (np.isnan(installed) or installed == 0)
            else round(float(mapped) / installed, 3),
        }
    if "Fossil Hard coal" not in coverage:
        coverage["Fossil Hard coal"] = {
            "mapped_mw": 0.0, "entsoe_installed_mw": None, "mapped_fraction": None,
            "note": "same 5 coal plants as Lignite under older ENTSO-E "
                    "classification; sum both fossil columns and use the "
                    "'Fossil coal (combined)' weights"}
    coverage["Solar"] = {"mapped_mw": 0.0,
                         "entsoe_installed_mw": round(float(latest.get("Solar", np.nan)), 1),
                         "mapped_fraction": 0.0,
                         "note": "distributed small-scale; deliberately unmapped"}
    return nodes, weights, coverage


def build_edges_and_graph(nodes: pd.DataFrame):
    import networkx as nx

    lines = pd.read_csv(TOPO / "ba_lines.csv", dtype={"bus0": str, "bus1": str})
    trafos = pd.read_csv(TOPO / "ba_transformers.csv", dtype={"bus0": str, "bus1": str})

    keep_l = ["line_id", "bus0", "bus1", "voltage", "circuits", "s_nom", "r", "x", "b",
              "length", "underground", "cross_border"]
    e_lines = lines[[c for c in keep_l if c in lines.columns]].copy()
    e_lines["kind"] = "line"
    e_lines = e_lines.rename(columns={"line_id": "edge_id"})
    e_lines["length_km"] = e_lines["length"] / 1000.0  # source column is METERS

    id_col = "transformer_id" if "transformer_id" in trafos.columns else trafos.columns[0]
    e_tr = trafos[[c for c in [id_col, "bus0", "bus1", "voltage_bus0", "voltage_bus1",
                               "s_nom", "r", "x", "b"] if c in trafos.columns]].copy()
    e_tr = e_tr.rename(columns={id_col: "edge_id"})
    e_tr["kind"] = "transformer"
    e_tr["cross_border"] = False

    edges = pd.concat([e_lines, e_tr], ignore_index=True)

    G = nx.MultiGraph()
    ncols = ["voltage", "x", "y", "country", "role", "nearest_place", "nearest_place_km"]
    ncols += [c for c in nodes.columns if c.startswith("plant_")]
    for _, r in nodes.iterrows():
        G.add_node(r["bus_id"], **{c: r[c] for c in ncols if pd.notna(r.get(c))})
    for _, r in edges.iterrows():
        attrs = {k: r[k] for k in ["kind", "voltage", "circuits", "s_nom", "r", "x",
                                   "b", "length_km", "cross_border"]
                 if k in edges.columns and pd.notna(r.get(k))}
        G.add_edge(r["bus0"], r["bus1"], key=str(r["edge_id"]), **attrs)
    return edges, G


def main():
    OUT.mkdir(exist_ok=True)
    import networkx as nx

    zone, notes = load_zone_series()
    nodes, weights, coverage = build_nodes_and_weights()
    edges, G = build_edges_and_graph(nodes)

    zone.to_parquet(OUT / "zone_series.parquet")
    nodes.drop(columns=[c for c in ("geometry", "tags", "symbol", "under_construction",
                                    "dc") if c in nodes.columns]) \
         .to_csv(OUT / "nodes.csv", index=False)
    edges.to_csv(OUT / "edges.csv", index=False)
    weights.to_csv(OUT / "gen_bus_weights.csv", index=False)
    nx.write_graphml(G, OUT / "ba_graph_multi.graphml")

    ba = G.subgraph(n for n, d in G.nodes(data=True) if d.get("role") == "ba")
    summary = {
        "zone_series": {
            "rows": int(len(zone)),
            "start_utc": str(zone.index.min()),
            "end_utc": str(zone.index.max()),
            "columns": list(zone.columns),
            **notes,
        },
        "graph": {
            "nodes": G.number_of_nodes(),
            "edges_multigraph": G.number_of_edges(),
            "note": "MultiGraph keeps parallel circuits as separate edges "
                    "(nx.Graph would collapse them); `circuits` kept as edge attr",
            "connected": bool(nx.is_connected(G)),
            "ba_subgraph_connected": bool(nx.is_connected(ba)),
        },
        "generation_mapping_coverage": coverage,
        "load_disaggregation": "none (zone-level only) — see DESIGN.md",
    }
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("done — outputs in dataset/")


if __name__ == "__main__":
    main()
