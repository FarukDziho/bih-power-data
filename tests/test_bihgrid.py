"""End-to-end test for the BiHGrid dataset, using the repo's local dataset/
files as the raw cache (no network)."""
import shutil
from pathlib import Path

import pytest
import torch

from bihgrid import BiHGrid
from bihgrid.dataset import EDGE_FEATURES, NODE_FEATURES, RAW_FILES

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    root = tmp_path_factory.mktemp("bihgrid")
    raw = root / "raw"
    raw.mkdir()
    for name in RAW_FILES:
        shutil.copy(REPO_ROOT / "dataset" / name, raw / name)
    return BiHGrid(root=root)


def test_graph_shapes(ds):
    d = ds[0]
    assert d.num_nodes == 51
    assert d.x.shape == (51, len(NODE_FEATURES))
    # 67 undirected edges stored in both directions
    assert d.edge_index.shape == (2, 134)
    assert d.edge_attr.shape == (134, len(EDGE_FEATURES))
    assert d.pos.shape == (51, 2)
    assert torch.isfinite(d.x).all()
    assert torch.isfinite(d.edge_attr).all()


def test_parallel_circuits_preserved(ds):
    d = ds[0]
    pairs = d.edge_index.t().tolist()
    # MultiGraph: at least one bus pair appears more than once per direction
    seen, dup = set(), False
    for p in map(tuple, pairs[:67]):
        if p in seen:
            dup = True
        seen.add(p)
    assert dup, "expected parallel circuits as duplicate bus pairs"


def test_zone_series(ds):
    d = ds[0]
    T, F = d.zone_series.shape
    assert T > 100_000 and F == len(ds.zone_columns)
    assert d.time_unix.shape == (T,)
    assert (d.time_unix.diff() > 0).all()
    df = ds.zone_dataframe()
    assert str(df.index.tz) == "UTC"
    # cleaned load: no zeros, no >2x-p99.9 spikes
    load = df["load_actual_mw"].dropna()
    assert (load > 0).all() and load.max() < 5000


def test_gen_weights_and_node_generation(ds):
    d = ds[0]
    types = ds.meta["gen_weight_types"]
    assert d.gen_weights.shape == (51, len(types))
    # each type's weights sum to 1
    assert torch.allclose(d.gen_weights.sum(0),
                          torch.ones(len(types)), atol=1e-4)
    ng = ds.node_generation("Fossil coal (combined)")
    assert ng.shape == (d.zone_series.shape[0], 51)
    # per-bus series sums back to the zone total where defined
    cols = ds.zone_columns
    sel = [cols.index("gen_fossil_brown_coal_lignite_mw"),
           cols.index("gen_fossil_hard_coal_mw")]
    zone = d.zone_series[:, sel].nansum(dim=1)
    assert torch.allclose(ng.sum(1), zone, rtol=1e-4, atol=0.5)


def test_metadata(ds):
    assert len(ds.bus_ids) == 51
    assert "nearest_place" in ds.meta
    assert ds.meta["summary"]["graph"]["ba_subgraph_connected"] is True
