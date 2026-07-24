# bihgrid dataset — design decisions

How the ENTSO-E BA-zone time series are joined onto the ≥220 kV topology,
and why. Produced by `src/join_graph.py`; outputs live in `dataset/`.

## 1. Zone signals stay zone-level (graph-level features)

Bosnia and Herzegovina is a **single bidding zone**, so every hourly ENTSO-E
signal (load, generation per type, forecasts, cross-border flows) is measured
for the zone as a whole. `dataset/zone_series.parquet` ships these untouched
(UTC index, one column per signal, `net_position_mw` positive = net exporter).
This is the ground truth; everything below is a documented heuristic on top.

## 2. Generation is attached to buses via a curated plant table

`topology/plants.csv` maps 25 named plants (coal, hydro, pumped storage, wind)
to their electrically-nearest bus in the ≥220 kV extract, with a
`connection_confidence` rating. `dataset/gen_bus_weights.csv` turns this into
per-type bus-weight vectors (capacity share of *mapped* plants), so a node-level
generation signal is `weight[bus, type] × zone_series[gen_type]`.

Caveats, deliberately explicit:

- **Coverage is not 100%.** `dataset_summary.json` →
  `generation_mapping_coverage` reports mapped MW vs ENTSO-E installed MW per
  type. Solar (~244 MW) is entirely unmapped (distributed small-scale);
  run-of-river is ~59% mapped; wind ~71%.
- **Small plants and renewables mostly connect at 110 kV**, which is below this
  topology's voltage cutoff — their bus assignment means "nearest ≥220 kV bus",
  not "actual point of connection". `connection_confidence` = low flags these.
- **TE Stanari** (300 MW) has no nearby bus in the extract (Doboj area absent
  at ≥220 kV); it is assigned to the Banja Luka 400 kV bus ~40 km away.
- **Coal classification drift:** ENTSO-E BA moves the same five coal plants
  between "Fossil Brown coal/Lignite" and "Fossil Hard coal" across years
  (hard-coal installed capacity stops being reported entirely). For analysis,
  sum both fossil columns and use the `Fossil coal (combined)` weight vector.

## 3. Load is NOT disaggregated

There is no defensible public sub-zone load data for BA. Rather than invent a
population heuristic and bake it in, `nodes.csv` carries `nearest_place` (from
the curated gazetteer `topology/ba_bus_places.csv`) so users can apply their own
disaggregation (population, GDP, uniform) — and the zone series remains the
reference. Cite this choice when benchmarking: models consuming "node load"
on this dataset are consuming a user-chosen heuristic, not measurements.

## 4. Parallel circuits are preserved

`nx.Graph` collapses parallel circuits (62 line records → 59 simple edges).
The dataset instead keeps **one edge per line record**: `edges.csv` has 62
lines + 5 transformers, and `ba_graph_multi.graphml` is a `networkx.MultiGraph`
keyed by `edge_id`. `circuits` is kept as an edge attribute; `length_km` is
converted from the source's meters.

## 5. Data cleaning (counted, never silent)

- Generation values > 3 × installed capacity of that type → NaN
  (counts in `dataset_summary.json`).
- Zero-load hours (metering gaps) and load spikes > 2 × the 99.9th percentile
  of nonzero load → NaN (counts in `dataset_summary.json`).
- Known, documented, *not* corrected: ENTSO-E load vs (generation − net export)
  differ by ~14%/yr (losses + distribution-side accounting).

## Files

| file | contents |
|---|---|
| `zone_series.parquet` | hourly zone-level signals, UTC, 2015→present |
| `nodes.csv` | 51 buses: voltage, coords, role, nearest_place, per-type plant MW |
| `edges.csv` | 67 edges (62 lines + 5 transformers), electrical params, per-circuit |
| `gen_bus_weights.csv` | per-type bus weight vectors for generation attachment |
| `ba_graph_multi.graphml` | MultiGraph with all node/edge attributes |
| `dataset_summary.json` | shapes, coverage, cleaning counts, connectivity checks |
