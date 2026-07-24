#!/usr/bin/env python3
"""
Builds the BiH grid-topology dataset from the OSM-based prebuilt European
transmission network (PyPSA-Eur), Zenodo record 14144752.

Downloads buses/lines/links/transformers/converters, filters to Bosnia and
Herzegovina plus first-hop cross-border neighbours, and writes graph-ready
artifacts into topology/out/:

  ba_buses.csv         nodes (role = "ba" | "border")
  ba_lines.csv         AC edges with electrical parameters
  ba_transformers.csv  intra-substation voltage-level edges
  ba_converters.csv    AC/DC converter edges (if any touch BA)
  ba_links.csv         DC edges (if any touch BA)
  ba_graph.graphml     the whole thing as a NetworkX graph
  ba_topology.geojson  buses as points + lines as linestrings (for maps)
  summary.json         counts, voltage levels, total circuit-km

Run by .github/workflows/topology.yml (manual dispatch) or locally:
  python topology/build_topology.py
"""
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

RECORD = "14144752"
API = f"https://zenodo.org/api/records/{RECORD}"
NEEDED = ["buses.csv", "lines.csv", "links.csv", "transformers.csv", "converters.csv"]
OUT = Path(__file__).resolve().parent / "out"
COUNTRY = "BA"
NEIGHBORS = {"HR", "RS", "ME"}


def fetch_files() -> dict:
    print(f"querying zenodo record {RECORD} …", flush=True)
    rec = requests.get(API, timeout=60).json()
    files = {f["key"]: f["links"]["self"] for f in rec.get("files", [])}
    out = {}
    for name in NEEDED:
        url = files.get(name)
        if not url:
            print(f"  ! {name} not in record (available: {list(files)[:10]})")
            continue
        print(f"  downloading {name} …", flush=True)
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        # this Zenodo release wraps geometry in single quotes ('LINESTRING (…)'),
        # whose embedded commas break the default double-quote CSV dialect —
        # detect it from the bytes (exception-based fallback is unreliable:
        # pandas can silently mis-parse instead of raising)
        kwargs = {"low_memory": False}
        head = r.content[:500000]
        if b"'LINESTRING" in head or b"'POINT" in head or b"'MULTILINESTRING" in head:
            kwargs["quotechar"] = "'"
        try:
            out[name] = pd.read_csv(io.BytesIO(r.content), **kwargs)
        except pd.errors.ParserError:
            out[name] = pd.read_csv(io.BytesIO(r.content), low_memory=False, quotechar="'")
        print(f"    {len(out[name]):,} rows, cols: {list(out[name].columns)[:12]}", flush=True)
    if "buses.csv" not in out or "lines.csv" not in out:
        sys.exit("ERROR: buses.csv and lines.csv are required.")
    return out


def col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_linestring(wkt):
    """'LINESTRING (lon lat, lon lat, …)' -> [[lon, lat], …] or None."""
    if not isinstance(wkt, str) or "LINESTRING" not in wkt.upper():
        return None
    nums = re.findall(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", wkt)
    return [[float(a), float(b)] for a, b in nums] or None


def main():
    OUT.mkdir(exist_ok=True)
    data = fetch_files()

    buses = data["buses.csv"]
    id_col = col(buses, "bus_id", "id", "name") or buses.columns[0]
    country_col = col(buses, "country", "country_code")
    if not country_col:
        sys.exit(f"ERROR: no country column in buses.csv (cols: {list(buses.columns)})")
    xcol = col(buses, "x", "lon", "longitude")
    ycol = col(buses, "y", "lat", "latitude")

    buses[id_col] = buses[id_col].astype(str)
    ba_ids = set(buses.loc[buses[country_col] == COUNTRY, id_col])
    print(f"BA buses: {len(ba_ids)}")
    if not ba_ids:
        sys.exit("ERROR: no BA buses found — check the country column values.")

    # collect edges touching BA from every edge table
    edge_tables, border_ids = {}, set()
    for name in ["lines.csv", "transformers.csv", "converters.csv", "links.csv"]:
        df = data.get(name)
        if df is None or df.empty:
            edge_tables[name] = pd.DataFrame()
            continue
        b0, b1 = col(df, "bus0"), col(df, "bus1")
        if not b0 or not b1:
            print(f"  ! {name}: no bus0/bus1 columns, skipped")
            edge_tables[name] = pd.DataFrame()
            continue
        df[b0] = df[b0].astype(str)
        df[b1] = df[b1].astype(str)
        touches = df[df[b0].isin(ba_ids) | df[b1].isin(ba_ids)].copy()
        for _, row in touches.iterrows():
            for e in (row[b0], row[b1]):
                if e not in ba_ids:
                    border_ids.add(e)
        touches["cross_border"] = ~(touches[b0].isin(ba_ids) & touches[b1].isin(ba_ids))
        edge_tables[name] = touches
        print(f"{name}: {len(touches)} edges touch BA ({int(touches['cross_border'].sum())} cross-border)")

    keep = ba_ids | border_ids
    nodes = buses[buses[id_col].isin(keep)].copy()
    nodes["role"] = nodes[id_col].map(lambda i: "ba" if i in ba_ids else "border")
    nodes.to_csv(OUT / "ba_buses.csv", index=False)

    names_map = {"lines.csv": "ba_lines.csv", "transformers.csv": "ba_transformers.csv",
                 "converters.csv": "ba_converters.csv", "links.csv": "ba_links.csv"}
    for src, dst in names_map.items():
        edge_tables[src].to_csv(OUT / dst, index=False)

    # ---- graphml ----
    import networkx as nx
    G = nx.Graph()
    for _, r in nodes.iterrows():
        attrs = {"role": r["role"], "country": str(r[country_col])}
        vcol = col(nodes, "voltage", "v_nom")
        if vcol and pd.notna(r.get(vcol)):
            attrs["voltage"] = float(r[vcol])
        if xcol and ycol and pd.notna(r.get(xcol)):
            attrs["x"] = float(r[xcol]); attrs["y"] = float(r[ycol])
        G.add_node(r[id_col], **attrs)
    for src, kind in [("lines.csv", "line"), ("transformers.csv", "transformer"),
                      ("converters.csv", "converter"), ("links.csv", "link")]:
        df = edge_tables[src]
        if df.empty:
            continue
        b0, b1 = col(df, "bus0"), col(df, "bus1")
        for _, r in df.iterrows():
            attrs = {"kind": kind, "cross_border": bool(r["cross_border"])}
            for a in ("voltage", "v_nom", "length", "s_nom", "i_nom", "r", "x", "b", "circuits", "underground"):
                if a in df.columns and pd.notna(r.get(a)):
                    try:
                        attrs[a] = float(r[a])
                    except (TypeError, ValueError):
                        attrs[a] = str(r[a])
            G.add_edge(r[b0], r[b1], **attrs)
    nx.write_graphml(G, OUT / "ba_graph.graphml")

    # ---- geojson ----
    features = []
    for _, r in nodes.iterrows():
        if not (xcol and ycol) or pd.isna(r.get(xcol)):
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r[xcol]), float(r[ycol])]},
            "properties": {"id": r[id_col], "role": r["role"], "country": str(r[country_col]),
                           "voltage": float(r[col(nodes,'voltage','v_nom')]) if col(nodes,'voltage','v_nom') and pd.notna(r.get(col(nodes,'voltage','v_nom'))) else None},
        })
    coords_by_id = {str(r[id_col]): [float(r[xcol]), float(r[ycol])]
                    for _, r in nodes.iterrows() if xcol and pd.notna(r.get(xcol))}
    for src, kind in [("lines.csv", "line"), ("links.csv", "link")]:
        df = edge_tables[src]
        if df.empty:
            continue
        b0, b1 = col(df, "bus0"), col(df, "bus1")
        gcol = col(df, "geometry", "wkt", "linestring")
        for _, r in df.iterrows():
            path = parse_linestring(r.get(gcol)) if gcol else None
            if path is None:
                a, b = coords_by_id.get(r[b0]), coords_by_id.get(r[b1])
                if not (a and b):
                    continue
                path = [a, b]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": path},
                "properties": {"kind": kind, "bus0": r[b0], "bus1": r[b1],
                               "cross_border": bool(r["cross_border"]),
                               "voltage": float(r["voltage"]) if "voltage" in df.columns and pd.notna(r.get("voltage")) else None},
            })
    (OUT / "ba_topology.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    # ---- summary ----
    lines = edge_tables["lines.csv"]
    vcol_l = col(lines, "voltage", "v_nom") if not lines.empty else None
    lencol = col(lines, "length", "length_km") if not lines.empty else None
    summary = {
        "source": f"zenodo.org/records/{RECORD} (PyPSA-Eur OSM prebuilt network)",
        "buses_ba": len(ba_ids & set(nodes[id_col])),
        "buses_border": len(border_ids & set(nodes[id_col])),
        "lines": int(len(lines)),
        "lines_cross_border": int(lines["cross_border"].sum()) if not lines.empty else 0,
        "transformers": int(len(edge_tables["transformers.csv"])),
        "converters": int(len(edge_tables["converters.csv"])),
        "dc_links": int(len(edge_tables["links.csv"])),
        "voltage_levels_kv": sorted(set(float(v) for v in lines[vcol_l].dropna())) if vcol_l else [],
        "total_line_km": (lambda v, med: round(v / 1000.0, 1) if med > 2000 else round(v, 1))(
            float(lines[lencol].sum()), float(lines[lencol].median())) if lencol else None,  # source lengths are meters
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("done — outputs in topology/out/")


if __name__ == "__main__":
    main()
