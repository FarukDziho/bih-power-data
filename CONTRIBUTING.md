# Contributing to bih-power-data / bihgrid

Thanks for your interest! Contributions of every kind are welcome — bug
reports, data-quality findings, plant-mapping corrections, new features, and
documentation fixes.

## Reporting issues

Open a [GitHub issue](https://github.com/FarukDziho/bih-power-data/issues).
Especially valuable:

- **Data-quality reports** — an hour range and signal where values look
  wrong, with what you expected and why.
- **Plant-mapping corrections** — if you have better knowledge of where a
  BA plant connects to the ≥220 kV grid, cite a source; every row of
  `topology/plants.csv` carries a `connection_confidence` we'd love to raise.
- **Topology errors** — missing/extra lines or substations vs reality
  (remember the deliberate ≥220 kV cutoff before reporting the 110 kV
  network as missing).

## Pull requests

1. Fork, branch from `main`, make your change.
2. `pip install -r requirements.txt` plus `torch`/`torch_geometric` for
   package work.
3. Run the tests: `python -m pytest tests/`. Data-pipeline changes should
   keep `python src/join_graph.py` running clean from a repo checkout.
4. Keep the documentation honest: any new heuristic or cleaning rule must be
   described in `dataset/DESIGN.md` with its counts/limits, in the same
   spirit as the existing ones.
5. Open the PR — CI runs the test suite.

## Releases

Maintainer releases are tagged `vX.Y.Z` (see `PUBLISHING.md`); the pipeline
publishes the GitHub release, PyPI package, and Zenodo DOI automatically.
Version numbers follow semver-ish pragmatism: data-only refreshes don't bump
the package version; schema or API changes do.

## Questions

Open an issue, or email faruk.dziho@gmail.com.
