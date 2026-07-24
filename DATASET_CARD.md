# bihgrid — dataset card

Real-topology graph benchmark of the Bosnia and Herzegovina (BA) transmission
grid with 11+ years of hourly ENTSO-E zone-level time series. Built as a
non-synthetic alternative to IEEE test cases for GNN research on power grids.

## Contents

- **Topology** (≥220 kV): 40 BA buses (10×400 kV, 30×220 kV) + 11 first-hop
  border buses (HR/RS/ME), 62 AC lines with electrical parameters
  (r, x, b, s_nom, length, circuits), 5 transformers, 12 cross-border ties.
  Fully connected; the BA-only subgraph is also connected (diameter 10).
  Every bus carries a curated `nearest_place` label.
- **Time series** (hourly, UTC, 2015→present, refreshed weekly): actual load
  (from 2017-02), load/generation forecasts (from 2015-12), generation per
  production type (from 2017-02), installed capacity per year, physical flows
  and scheduled exchanges with HR/RS/ME in both directions, zone net position.
- **Join artifacts**: curated plant→bus table (25 plants, confidence-rated),
  per-type generation→bus weight vectors, MultiGraph GraphML, and
  `dataset/DESIGN.md` documenting every design decision.
- **Access**: files in `dataset/`, or `pip install bihgrid` for a
  `torch_geometric` `InMemoryDataset`.

## Sources

- Time series: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu),
  reported by NOSBiH (BA TSO). Collected via `src/collect.py`.
- Topology: PyPSA-Eur OSM prebuilt European network,
  [Zenodo record 14144752](https://zenodo.org/records/14144752), filtered to
  BA + first-hop neighbours by `topology/build_topology.py`. Underlying data
  © OpenStreetMap contributors.
- Plant table and gazetteer: curated by the author from public sources.

## Licenses

- **Code** (collectors, join, package): MIT.
- **ENTSO-E time series**: reuse subject to the
  [ENTSO-E Transparency Platform terms](https://transparency.entsoe.eu/content/static_content/Static%20content/terms%20and%20conditions/terms%20and%20conditions.html)
  — free reuse with attribution ("Source: ENTSO-E Transparency Platform").
- **Topology**: derived from OpenStreetMap via PyPSA-Eur →
  [ODbL 1.0](https://opendatacommons.org/licenses/odbl/). Attribution:
  © OpenStreetMap contributors, PyPSA-Eur team.

## Known limitations (read before benchmarking)

1. **≥220 kV only** — the 110 kV network is absent; small plants and most
   renewables physically connect below the cutoff. Bus assignments for those
   mean "nearest ≥220 kV bus", flagged `connection_confidence: low`.
2. **Zone-level signals** — BA is one bidding zone; node-level generation in
   the package is capacity-weighted zone data (documented heuristic), and
   load is deliberately NOT disaggregated.
3. **OSM-derived electrical parameters** — heuristically cleaned, not
   TSO-validated. Line lengths derive from OSM geometry (source meters).
4. **Accounting gap** — ENTSO-E load vs (generation − net export) differ by
   ~14%/yr (losses + distribution-side accounting). Documented, not corrected.
5. **Data quality** — counted-and-NaN'd: ~1,229 zero-load hours, 2 load
   spikes, 72 generation outlier hours (>3× installed capacity). Coal is
   inconsistently split between two ENTSO-E fossil categories across years —
   sum both columns (weights provided as "Fossil coal (combined)").
6. **No day-ahead prices, no reservoir levels** — BA does not report them.

## Citation

Until the data paper is out, cite the Zenodo DOI (see repo badge) or:

> Dziho, F. (2026). bihgrid: the transmission grid of Bosnia and Herzegovina as
> a graph learning dataset. https://github.com/FarukDziho/bih-power-data

## Maintenance

Weekly automated ENTSO-E refresh (GitHub Actions, Mondays 03:17 UTC).
Contact: Faruk Dziho, faruk.dziho@gmail.com.
