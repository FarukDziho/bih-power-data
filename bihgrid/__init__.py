"""bihgrid — the transmission grid of Bosnia and Herzegovina as a PyG dataset.

Real ≥220 kV grid topology (PyPSA-Eur / OSM) joined with ENTSO-E hourly
zone-level time series (2015→present), packaged as a
`torch_geometric.data.InMemoryDataset`.

    from bihgrid import BiHGrid
    ds = BiHGrid(root="data/bihgrid")
    data = ds[0]          # static graph + time-series tensors
    ds.node_generation()  # [T, N] heuristic per-bus generation

See https://github.com/FarukDziho/bih-power-data for sources, licenses and
the design decisions behind every heuristic (dataset/DESIGN.md).
"""
from .dataset import BiHGrid

__version__ = "0.1.3"
__all__ = ["BiHGrid", "__version__"]
