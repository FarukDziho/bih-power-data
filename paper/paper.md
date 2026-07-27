---
title: 'bihgrid: the transmission grid of Bosnia and Herzegovina as a PyTorch Geometric dataset'
tags:
  - Python
  - power systems
  - graph neural networks
  - PyTorch Geometric
  - open data
  - ENTSO-E
authors:
  - name: Faruk Dziho
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: University of Texas at San Antonio, San Antonio, TX, United States
    index: 1
date: 24 July 2026
bibliography: paper.bib
---

# Summary

`bihgrid` provides the real ≥220 kV transmission network of Bosnia and
Herzegovina (BA) as a machine-learning-ready graph dataset, together with the
fully open pipeline that builds and continuously refreshes it. The topology —
40 domestic and 11 border substations, 62 circuit-resolved lines with
electrical parameters (resistance, reactance, susceptance, thermal rating,
length), and 5 transformers — is extracted from the OpenStreetMap-based
prebuilt European network of PyPSA-Eur [@Xiong2025; @Horsch2018]. It is
joined with more than a decade (2015–present) of hourly measurements from the
ENTSO-E Transparency Platform [@Hirth2018]: load, generation by production
type, day-ahead forecasts, and cross-border physical and scheduled exchanges
with Croatia, Serbia, and Montenegro. A curated table maps the country's 25
largest power plants to their nearest network buses with per-assignment
confidence ratings, from which capacity-share weight vectors provide
node-level generation signals alongside the authoritative zone-level series.

The package exposes all of this as a `torch_geometric` [@Fey2019]
`InMemoryDataset`: one line of code downloads a versioned release, caches it,
and returns a `Data` object with static node features, an 8-dimensional edge
attribute matrix over 134 directed edge entries, generation weight matrices,
and the full hourly zone-series tensor with a UTC time index. The underlying
files (Parquet, CSV, GraphML) are also usable without PyTorch. Data are
archived on Zenodo with versioned DOIs [@Dziho2026]; a GitHub Actions
pipeline refreshes the time series weekly and rebuilds the joined dataset, so
the resource stays current without manual effort.

# Statement of need

Graph neural networks are now applied across power-system tasks — power-flow
approximation [@Donon2020], forecasting, and the detection of false data
injection attacks on grid telemetry [@Boyaci2022] — as surveyed by
@Liao2022. The evaluation basis of this literature is, however, dominated by
synthetic IEEE test cases (14/118/300 bus): stylised topologies with no real
geography and, critically for learning research, no measured time series, so
temporal behaviour must be simulated [@Birchfield2017]. At the other
extreme, open measured data such as the ENTSO-E Transparency Platform
describe entire bidding zones as single points, with no intra-zonal network.
Researchers who want both — a real grid graph *and* real aligned
measurements — currently must assemble them ad hoc, negotiating OSM
extraction, ENTSO-E API quirks, timezone and classification drift, and
plant-to-bus attribution, all of which embed silent modelling choices.

`bihgrid` packages one complete national system with those choices made
explicit and documented. BA is well suited to the benchmark role: it is a
single bidding zone (so zone measurements have a well-defined electrical
footprint), compact enough for expensive experiments and manual inspection,
yet a real meshed grid in the synchronous Continental European system with
three interconnected neighbours and a distinctive half-lignite, half-hydro
generation mix. Every heuristic in the zone-to-node bridging layer is
labelled (confidence-rated plant mapping, published coverage fractions), the
raw zone series are always shipped untouched, and all cleaning actions are
counted rather than silent — properties aimed at making the dataset safe to
build claims on. The intended audience is researchers in graph machine
learning for energy systems (forecasting, state estimation, robustness and
grid-cybersecurity studies) and power-system data scientists who need a
realistic, continuously updated testbed smaller than country-scale models
like PyPSA-Eur but richer than static test cases.

# Functionality

The repository contains three connected layers. The **collection layer**
(`src/collect.py`) incrementally downloads seventeen ENTSO-E series and
maintains a coverage manifest. The **topology layer**
(`topology/build_topology.py`) filters the PyPSA-Eur prebuilt European
network to BA plus first-hop border buses and emits CSV/GraphML/GeoJSON
artifacts. The **join layer** (`src/join_graph.py`) merges the two — UTC
alignment, outlier accounting, plant-to-bus weight construction, multigraph
preservation of parallel circuits — and writes the `dataset/` release
artifacts consumed by the Python package. Design decisions and their limits
are documented in `dataset/DESIGN.md` and `DATASET_CARD.md`, including the
≥220 kV voltage cutoff, the zone-level nature of all measurements, and the
OSM provenance of electrical parameters. The package is tested end-to-end in
CI, and tagged releases are archived automatically (GitHub release → PyPI →
Zenodo DOI).

# Acknowledgements

Time-series data © ENTSO-E Transparency Platform, reported by NOSBiH;
topology derived from OpenStreetMap via the PyPSA-Eur project (ODbL).
Portions of the code, documentation, and this manuscript were drafted with
the assistance of Anthropic's Claude (Cowork); all outputs were reviewed,
tested, and verified by the author.

# References
