# bih-power-data

Continuously refreshed dataset and monthly report for the **Bosnia and Herzegovina power system**
(BA control area, operated by [NOSBiH](https://www.nosbih.ba/en/)), collected from the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).

**Live report:** `report/index.html` (enable GitHub Pages on this repo to serve it).

## What's collected

| dataset | description |
|---|---|
| `load_actual` | actual total load, hourly |
| `load_forecast` | day-ahead load forecast |
| `generation_per_type` | actual generation per production type (lignite, hydro ×3, wind, solar…) |
| `generation_forecast` | day-ahead generation forecast |
| `installed_capacity` | installed capacity per type (annual) |
| `flow_BA_XX` / `flow_XX_BA` | cross-border physical flows with HR / RS / ME, both directions |
| `sched_BA_XX` / `sched_XX_BA` | scheduled commercial exchanges with HR / RS / ME |
| `day_ahead_prices`, `hydro_reservoir_storage` | attempted; BA may not report these (status noted in `data/state.json`) |

Everything is stored as parquet in `data/`, full history from 2015 where available.
`data/state.json` tracks coverage and per-dataset status.

## Setup

1. Register (free) at [transparency.entsoe.eu](https://transparency.entsoe.eu/) and generate an
   API security token in your account settings.
2. First full collection (locally):

   ```bash
   pip install -r requirements.txt
   ENTSOE_TOKEN=your-token python src/collect.py        # full history on first run (takes a while)
   python src/build_report.py                            # writes report/index.html
   ```

3. Push to GitHub, then add the token so the weekly refresh works:
   **Settings → Secrets and variables → Actions → New repository secret** → name `ENTSOE_TOKEN`.

The included workflow (`.github/workflows/refresh.yml`) then re-collects incrementally every
Monday 03:17 UTC and commits any new data plus a rebuilt report. You can also trigger it manually
from the Actions tab (**Run workflow**).

## Pipeline test without a token

```bash
python src/collect.py --demo && python src/build_report.py
```

generates synthetic data so the whole pipeline and report can be exercised end-to-end.

## Graph-ready dataset

`src/join_graph.py` joins the ENTSO-E time series onto the ≥220 kV topology
(`topology/out/`) and writes a graph-ready dataset into `dataset/`: hourly
zone-level series (UTC), 51 buses with `nearest_place` labels and per-type
plant capacities (from the curated `topology/plants.csv`), 67 per-circuit
edges with electrical parameters, generation→bus weight vectors, and a
MultiGraph GraphML. All design decisions (zone-level vs node-level signals,
plant mapping coverage, cleaning rules) are documented in
[`dataset/DESIGN.md`](dataset/DESIGN.md). Runs offline from the repo checkout:

```bash
python src/join_graph.py
```

## `bihgrid` — PyTorch Geometric package

The dataset ships as a PyG `InMemoryDataset` (auto-downloads from the GitHub
release or raw files, caches locally):

```bash
pip install bihgrid
```

```python
from bihgrid import BiHGrid

ds = BiHGrid(root="data/bihgrid")        # revision="v0.1.0" pins a release
data = ds[0]
data.x           # [51, 9]  voltage, coords, role, per-type plant MW
data.edge_index  # [2, 134] 67 undirected per-circuit edges, both directions
data.edge_attr   # [134, 8] r, x, b, s_nom, length_km, circuits, flags
data.zone_series # [T, 18]  hourly zone signals, UTC (NaN = gap/cleaned)
ds.zone_dataframe()                       # same as pandas DataFrame
ds.node_generation("Fossil coal (combined)")  # [T, 51] heuristic — see DESIGN.md
```

Feature orders: `bihgrid.dataset.NODE_FEATURES` / `EDGE_FEATURES`. Read
[`dataset/DESIGN.md`](dataset/DESIGN.md) and [`DATASET_CARD.md`](DATASET_CARD.md)
before treating node-level signals as measurements — they are documented
heuristics over zone-level data. Tests: `python -m pytest tests/`.

## Releasing

`git tag vX.Y.Z && git push origin vX.Y.Z` → tests, GitHub release with
dataset zip, PyPI publish, Zenodo DOI. One-time account setup in
[`PUBLISHING.md`](PUBLISHING.md).

## Roadmap

- [x] Combine with grid topology (PyPSA-Eur / OpenStreetMap extract for BiH) into a
      graph-ready dataset — candidate for publication. → `dataset/`, `src/join_graph.py`
- [x] PyG package `bihgrid` (InMemoryDataset, auto-download, PyPI). → `bihgrid/`
- [x] Zenodo deposit with versioned DOI + dataset card. → `.zenodo.json`,
      `DATASET_CARD.md`, automated via `release.yml` (one-time setup in `PUBLISHING.md`)
- [ ] Data paper (Scientific Data / NeurIPS D&B).
- [ ] Companion collectors for EIA + ERCOT following the same layout.

## License / attribution

Data © ENTSO-E Transparency Platform, reported by NOSBiH; reuse subject to ENTSO-E's
[terms](https://transparency.entsoe.eu/content/static_content/Static%20content/terms%20and%20conditions/terms%20and%20conditions.html).
Collection and report code: MIT.
