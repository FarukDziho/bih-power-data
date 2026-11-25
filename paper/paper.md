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
    orcid: 0009-0007-9117-9315
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
generation mix. The intended audience is researchers in graph machine
learning for energy systems (forecasting, state estimation, robustness and
grid-cybersecurity studies) and power-system data scientists who need a
realistic, continuously updated testbed smaller than country-scale models
like PyPSA-Eur but richer than static test cases.

# State of the field

Open resources for learning on power grids fall into three groups, and
`bihgrid` sits in the gap between them. *Simulation-derived benchmarks*
provide labelled graphs at scale: PowerGraph [@Varbella2024] offers
cascading-failure datasets and OPFData [@Lovett2024] offers solved AC optimal
power flow instances, the latter distributed through PyTorch Geometric. Their
samples are produced by simulation on test-case or model networks, as are
studies built on the IEEE cases shipped with tools such as pandapower
[@Thurner2018], so none contains measured operation. *Open measured data*,
such as Open Power System Data [@Wiese2019] and the ENTSO-E Transparency
Platform itself, provide long observed series, but per bidding zone and
without a network. *Open network models*, chiefly PyPSA-Eur, provide real
topology with electrical parameters as input to optimisation studies, without
aligned measurements or learning-oriented packaging.

`bihgrid` builds on these projects instead of duplicating them: it consumes
the PyPSA-Eur prebuilt network and retrieves ENTSO-E series through the
`entsoe-py` client. What it adds does not fit upstream in either. Country-specific
curation (the plant-to-bus table, the substation gazetteer, the handling of
classification drift) and a continuously refreshed measurement join are
outside the scope of a continental optimisation workflow, and PyTorch
Geometric hosts dataset loaders, not the collection pipelines behind them.
The loader follows the `InMemoryDataset` conventions of PyTorch Geometric so
that it can be proposed upstream once the schema has stabilised.

# Software design

The software is organised as three layers that can be run independently. The
**collection layer** (`src/collect.py`) incrementally downloads seventeen
ENTSO-E series and maintains a coverage manifest. The **topology layer**
(`topology/build_topology.py`) filters the PyPSA-Eur prebuilt network to BA
plus first-hop border buses. The **join layer** (`src/join_graph.py`) aligns
the two and writes the release artifacts that the `bihgrid` package loads.
Four trade-offs shaped the design.

*Measurements are never fabricated.* Every ENTSO-E signal is measured for the
zone as a whole, so the zone series ship untouched as graph-level ground
truth. Node-level generation is offered as an explicit heuristic
(capacity-share weights from the confidence-rated plant table), and load is
deliberately not disaggregated: no defensible public sub-zone load data exist
for BA, so the dataset supplies a nearest-place label per bus and leaves the
choice of disaggregation to the user. Users get fewer ready-made node
features, and in exchange no result on the dataset silently rests on an
invented signal.

*A ≥220 kV cutoff.* OpenStreetMap coverage of the BA 110 kV network was
judged too unreliable to publish with electrical parameters. The cost is that
most wind, solar, and small hydro plants connect below the cutoff; their bus
assignments mean "nearest ≥220 kV bus" and are rated low confidence, and
mapped-capacity coverage per production type is published with every release.

*Parallel circuits are preserved.* A simple graph would collapse the 67 branch
records to 59 edges (the 62 line records to 54). The dataset keeps one edge per record, because
circuit-level parameters matter for power-flow studies.

*Files first, framework second.* Releases are plain Parquet, CSV, and
GraphML, and the PyTorch Geometric class is a thin consumer of them, so the
data remain usable without PyTorch and every release can be pinned by tag.
The weekly refresh commits data to the repository through GitHub Actions,
which grows the repository over time and gives full provenance with no
infrastructure to maintain. Cleaning rules are few and counted (the number of
values each rule sets to missing is written to a summary file), and known
inconsistencies in the source are documented and left uncorrected.

Design decisions and their limits are recorded in `dataset/DESIGN.md` and
`DATASET_CARD.md`. The package is tested end-to-end in continuous
integration, and tagged releases propagate automatically to PyPI and Zenodo.

# Research impact statement

`bihgrid` was first released in July 2026, and its near-term significance
rests on the following evidence. The release and refresh machinery is in
production use: versioned releases are on PyPI and archived on Zenodo
[@Dziho2026], and the hourly series have been refreshed every week by the
automated pipeline since release. The same pipeline generates a public report
on the BA power system in English and Bosnian, rebuilt at each refresh, which
exercises the collection layer end to end on live data.

**[DRAFT — keep this paragraph only once the benchmark scripts are public in
the repository, and replace the July prototype figures with the released
ones.]** The dataset also supports a reproducible benchmark, shipped with the
package: a replication of the stealth false-data-injection study of
@Boyaci2022 on the real BA topology and measured hours. A classical
residual-based bad-data detector is blind to the constructed attacks (ROC-AUC
0.50); a topology-blind neural baseline matches a graph neural network on
detecting that an attack occurred (0.997 against 0.992); and the graph model
leads on localising the attacked buses (0.959 against 0.906).

**[AUTHOR: add any external use you can point to — downloads, citations,
issues or pull requests from other users, courses or projects using the
data. JOSS asks for specific evidence here and rejects purely aspirational
statements.]**

# AI usage disclosure

Generative AI was used in this project. Anthropic's Claude **[AUTHOR: list
the model versions used]**, accessed through the Claude desktop application,
drafted portions of the Python code, tests, continuous-integration
workflows, documentation, and this manuscript.

**[AUTHOR TO COMPLETE IN YOUR OWN WORDS. JOSS requires this section to state
accurately (1) where and how much AI assistance was used, (2) that you made
the primary architectural and design decisions, naming them, and (3) how you
reviewed and validated the AI-generated code and text. JOSS treats an
incomplete or inaccurate disclosure as an ethics violation, so describe what
actually happened.]**

# Acknowledgements

Time-series data © ENTSO-E Transparency Platform, reported by NOSBiH;
topology derived from OpenStreetMap via the PyPSA-Eur project (ODbL).
**[AUTHOR: confirm]** This work received no dedicated funding.

# References
