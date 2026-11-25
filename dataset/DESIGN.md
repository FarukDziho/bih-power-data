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
- **Direct and proxy assignments.** `plants.csv` records for every plant
  the connection voltage (`connection_kv`), the distance from the plant to
  its bus (`bus_distance_km`) and a `mapping_basis`: `direct` when the plant
  connects at ≥220 kV and the bus is its own switchyard or the adjacent
  substation, `proxy` when the nearest ≥220 kV bus stands in for a
  connection the extract does not contain (a 110 kV connection, or a
  switchyard missing from the extract). Direct capacity by type: coal 85.5%,
  reservoir hydro 69.9%, pumped storage 100%, run-of-river 0%, wind 0%
  (`dataset_summary.json` → `generation_mapping_basis`). The source for each
  row is listed in `topology/PLANTS_SOURCES.md`.
- **Commissioning dates.** `plants.csv` has a `commissioned` year, and
  `gen_bus_weights_by_year.csv` holds one weight vector per type and calendar
  year built only from plants in service that year. `gen_bus_weights.csv`
  stays the static vector for the current fleet. Within the reported series
  this changes wind only: 2018–2020 all wind sits on the Posušje bus
  (Mesihovina, then Jelovača), Podveležje (Mostar bus) enters in 2021, and
  Ivovik in 2024. Stanari (2016) predates the first coal hour (2017-02), and
  Bočac 2 (2018) predates the first run-of-river hour (2021).
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

`nx.Graph` collapses parallel circuits (67 branch records → 59 simple edges;
the 62 line records alone → 54).
The dataset instead keeps **one edge per line record**: `edges.csv` has 62
lines + 5 transformers, and `ba_graph_multi.graphml` is a `networkx.MultiGraph`
keyed by `edge_id`. `circuits` is kept as an edge attribute; `length_km` is
converted from the source's meters.

## 5. Data cleaning (counted, never silent)

- Generation values > 3 × installed capacity of that type → NaN
  (counts in `dataset_summary.json`).
- Zero-load hours (metering gaps) and load spikes > 2 × the 99.9th percentile
  of nonzero load → NaN (counts in `dataset_summary.json`).
- Known, documented, *not* corrected: ENTSO-E load and (generation − net
  export) do not balance. Compared on hours where both are reported, the
  annual gap runs from −23% (2017) to +6% (2021); it was +3.5% in 2024 and
  −15.6% in 2025.

## Files

| file | contents |
|---|---|
| `zone_series.parquet` | hourly zone-level signals, UTC, 2015→present |
| `nodes.csv` | 51 buses: voltage, coords, role, nearest_place, per-type plant MW |
| `edges.csv` | 67 edges (62 lines + 5 transformers), electrical params, per-circuit |
| `gen_bus_weights.csv` | per-type bus weight vectors for generation attachment (current fleet) |
| `gen_bus_weights_by_year.csv` | the same per calendar year, counting plants from their commissioning year |
| `ba_graph_multi.graphml` | MultiGraph with all node/edge attributes |
| `dataset_summary.json` | shapes, coverage, cleaning counts, connectivity checks |
