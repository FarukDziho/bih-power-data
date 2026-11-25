"""End-to-end test for the BiHGrid dataset, using the repo's local dataset/
files as the raw cache (no network)."""
import shutil
from pathlib import Path

import pytest
import torch

from bihgrid import BiHGrid
from bihgrid.dataset import (EDGE_FEATURES, NODE_FEATURES,
                             OPTIONAL_RAW_FILES, RAW_FILES)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    root = tmp_path_factory.mktemp("bihgrid")
    raw = root / "raw"
    raw.mkdir()
    for name in RAW_FILES + OPTIONAL_RAW_FILES:
        shutil.copy(REPO_ROOT / "dataset" / name, raw / name)
    return BiHGrid(root=root)


@pytest.fixture(scope="module")
def ds_static(tmp_path_factory):
    """A release without per-year weights (v0.1.3 and earlier)."""
    root = tmp_path_factory.mktemp("bihgrid_static")
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
    assert df.index[0].year == 2014 and df.index[-1].year >= 2026
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
    ng = ds.node_generation("Fossil coal (combined)", by_commissioning=False)
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


def test_node_generation_respects_commissioning(ds):
    d = ds[0]
    years = ds.meta["gen_weight_years"]
    types = ds.meta["gen_weight_types"]
    assert years[0] == 2015 and d.gen_weights_by_year.shape == (
        len(years), 51, len(types))
    # in every year a type's weights sum to 1, or to 0 before its first plant
    sums = d.gen_weights_by_year.sum(1)
    assert (torch.isclose(sums, torch.ones_like(sums), atol=1e-4)
            | (sums == 0)).all()
    # the last year equals the static (current fleet) weights
    assert torch.allclose(d.gen_weights_by_year[-1], d.gen_weights, atol=1e-6)

    year = torch.as_tensor(ds.zone_dataframe().index.year.to_numpy())
    bus = ds.bus_ids
    wind = ds.node_generation("Wind Onshore")
    # Podvelezje (Mostar bus BA22-220) was commissioned in 2021: no wind
    # there before, a share from 2021 on
    podv = wind[:, bus.index("BA22-220")]
    assert torch.nan_to_num(podv[year < 2021]).abs().sum() == 0
    assert torch.nan_to_num(podv[year >= 2021]).sum() > 0
    # Stanari (Banja Luka bus BA2-400) was commissioned in 2016
    coal = ds.node_generation("Fossil coal (combined)")
    assert torch.nan_to_num(coal[year < 2016, bus.index("BA2-400")]).sum() == 0
    # the per-bus series still sums to the zone total wherever wind is reported
    zone = d.zone_series[:, ds.zone_columns.index("gen_wind_onshore_mw")]
    ok = ~torch.isnan(zone) & (year >= 2018)
    assert torch.allclose(wind.sum(1)[ok], zone[ok], rtol=1e-4, atol=0.5)
    # static weights put wind on the Mostar bus in 2018; the default does not
    static = ds.node_generation("Wind Onshore", by_commissioning=False)
    assert torch.nan_to_num(static[year == 2018, bus.index("BA22-220")]).sum() > 0


def test_static_fallback_without_yearly_weights(ds_static):
    d = ds_static[0]
    assert ds_static.meta["gen_weight_years"] == []
    a = ds_static.node_generation("Wind Onshore")
    b = ds_static.node_generation("Wind Onshore", by_commissioning=False)
    assert a.shape == (d.zone_series.shape[0], 51)
    assert torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))
