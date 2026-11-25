"""BiHGrid: torch_geometric InMemoryDataset for the BiH transmission grid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

REPO = "FarukDziho/bih-power-data"
RAW_FILES = [
    "nodes.csv",
    "edges.csv",
    "gen_bus_weights.csv",
    "zone_series.parquet",
    "dataset_summary.json",
]

# Shipped from the release after v0.1.3; older revisions do not have it, so
# it is fetched when present and the loader falls back to the static weights.
OPTIONAL_RAW_FILES = ["gen_bus_weights_by_year.csv"]

# static node features, in x column order
NODE_FEATURES = [
    "voltage_kv",
    "lon",
    "lat",
    "is_ba",  # 1 = BA bus, 0 = border bus
    "plant_coal_mw",
    "plant_hydro_reservoir_mw",
    "plant_hydro_pumped_storage_mw",
    "plant_hydro_ror_mw",
    "plant_wind_mw",
]
# edge features, in edge_attr column order
EDGE_FEATURES = ["r", "x", "b", "s_nom", "length_km", "circuits",
                 "is_transformer", "cross_border"]

_PLANT_COLS = {
    "plant_coal_mw": ["plant_fossil_brown_coal_lignite_mw", "plant_fossil_hard_coal_mw"],
    "plant_hydro_reservoir_mw": ["plant_hydro_water_reservoir_mw"],
    "plant_hydro_pumped_storage_mw": ["plant_hydro_pumped_storage_mw"],
    "plant_hydro_ror_mw": ["plant_hydro_run-of-river_and_poundage_mw",
                           "plant_hydro_run_of_river_and_poundage_mw"],
    "plant_wind_mw": ["plant_wind_onshore_mw"],
}


class BiHGrid(InMemoryDataset):
    """The ≥220 kV transmission grid of Bosnia and Herzegovina.

    One static graph (51 buses, 67 per-circuit edges incl. transformers)
    plus hourly ENTSO-E zone-level series (UTC) and a curated
    generation→bus weight matrix.

    Args:
        root: cache directory.
        revision: git revision of ``FarukDziho/bih-power-data`` to download
            the dataset files from. A tag like ``"v0.1.0"`` first tries the
            GitHub-release asset zip, then falls back to raw files at that
            revision; ``"main"`` (default) uses the latest raw files.
        transform, pre_transform: standard PyG hooks.

    Node/edge feature orders are ``bihgrid.dataset.NODE_FEATURES`` and
    ``bihgrid.dataset.EDGE_FEATURES``. The heuristics and their limits are
    documented in the repo's ``dataset/DESIGN.md`` — read it before using
    node-level generation as ground truth (it is capacity-weighted
    zone data, not per-plant metering).
    """

    def __init__(self, root: str | Path, revision: str = "main",
                 transform=None, pre_transform=None, force_reload: bool = False):
        self.revision = revision
        super().__init__(str(root), transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])
        with open(Path(self.processed_dir) / "meta.json") as f:
            self._meta = json.load(f)

    # ------------------------------------------------------------------ names
    @property
    def raw_file_names(self):
        return RAW_FILES

    @property
    def processed_file_names(self):
        return ["data.pt", "meta.json"]

    # ------------------------------------------------------------------ download
    def download(self):
        import io
        import zipfile
        import requests

        raw_dir = Path(self.raw_dir)
        if self.revision.startswith("v"):
            url = (f"https://github.com/{REPO}/releases/download/"
                   f"{self.revision}/bihgrid-dataset.zip")
            r = requests.get(url, timeout=300)
            if r.ok:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    for name in RAW_FILES:
                        member = next((m for m in z.namelist()
                                       if m.endswith("dataset/" + name)), None)
                        if member is None:
                            break
                        (raw_dir / name).write_bytes(z.read(member))
                    else:
                        for name in OPTIONAL_RAW_FILES:
                            member = next((m for m in z.namelist()
                                           if m.endswith("dataset/" + name)), None)
                            if member is not None:
                                (raw_dir / name).write_bytes(z.read(member))
                        return
        base = (f"https://raw.githubusercontent.com/{REPO}/"
                f"{self.revision}/dataset/")
        for name in RAW_FILES:
            r = requests.get(base + name, timeout=300)
            r.raise_for_status()
            (raw_dir / name).write_bytes(r.content)
        for name in OPTIONAL_RAW_FILES:
            r = requests.get(base + name, timeout=300)
            if r.ok:
                (raw_dir / name).write_bytes(r.content)

    # ------------------------------------------------------------------ process
    def process(self):
        raw = Path(self.raw_dir)
        nodes = pd.read_csv(raw / "nodes.csv", dtype={"bus_id": str})
        edges = pd.read_csv(raw / "edges.csv", dtype={"bus0": str, "bus1": str})
        weights = pd.read_csv(raw / "gen_bus_weights.csv", dtype={"bus_id": str})
        zone = pd.read_parquet(raw / "zone_series.parquet")
        summary = json.loads((raw / "dataset_summary.json").read_text())

        bus_ids = nodes["bus_id"].tolist()
        idx = {b: i for i, b in enumerate(bus_ids)}

        # ---- static node features
        feats = pd.DataFrame(index=nodes.index)
        feats["voltage_kv"] = nodes["voltage"].astype(float)
        feats["lon"] = nodes["x"].astype(float)
        feats["lat"] = nodes["y"].astype(float)
        feats["is_ba"] = (nodes["role"] == "ba").astype(float)
        for out_col, candidates in _PLANT_COLS.items():
            cols = [c for c in candidates if c in nodes.columns]
            feats[out_col] = nodes[cols].sum(axis=1) if cols else 0.0
        x = torch.tensor(feats[NODE_FEATURES].to_numpy(), dtype=torch.float)
        pos = torch.tensor(nodes[["x", "y"]].to_numpy(), dtype=torch.float)

        # ---- edges (undirected -> both directions; parallel circuits kept)
        e = edges.copy()
        e["is_transformer"] = (e["kind"] == "transformer").astype(float)
        e["cross_border"] = e["cross_border"].astype(bool).astype(float)
        for c in ["r", "x", "b", "s_nom", "length_km", "circuits"]:
            e[c] = pd.to_numeric(e.get(c), errors="coerce").fillna(0.0)
        src = e["bus0"].map(idx).to_numpy()
        dst = e["bus1"].map(idx).to_numpy()
        attr = e[EDGE_FEATURES].to_numpy(dtype=np.float32)
        edge_index = torch.tensor(np.concatenate([np.stack([src, dst]),
                                                  np.stack([dst, src])], axis=1),
                                  dtype=torch.long)
        edge_attr = torch.tensor(np.concatenate([attr, attr]), dtype=torch.float)

        # ---- generation weight matrix [N, n_types]
        types = sorted(weights["entsoe_type"].unique())
        W = np.zeros((len(bus_ids), len(types)), dtype=np.float32)
        for _, r in weights.iterrows():
            W[idx[r["bus_id"]], types.index(r["entsoe_type"])] = r["weight"]

        # ---- per-year weights [Y, N, n_types]: plants counted from their
        #      commissioning year (absent in releases up to v0.1.3)
        yearly_path = raw / "gen_bus_weights_by_year.csv"
        weight_years, W_year = [], None
        if yearly_path.exists():
            wy = pd.read_csv(yearly_path, dtype={"bus_id": str})
            weight_years = sorted(int(y) for y in wy["year"].unique())
            W_year = np.zeros((len(weight_years), len(bus_ids), len(types)),
                              dtype=np.float32)
            for _, r in wy.iterrows():
                W_year[weight_years.index(int(r["year"])), idx[r["bus_id"]],
                       types.index(r["entsoe_type"])] = r["weight"]

        # ---- zone-level hourly series [T, F] (NaN kept as NaN)
        zone = zone.sort_index()
        zone_t = torch.tensor(zone.to_numpy(dtype=np.float32))
        # seconds since the epoch, whatever resolution the index is stored in
        # (pandas >= 3 reads Parquet timestamps as microseconds, not ns)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        time_unix = torch.tensor(
            ((zone.index - epoch) // pd.Timedelta(seconds=1)).to_numpy(),
            dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos,
                    gen_weights=torch.tensor(W),
                    zone_series=zone_t, time_unix=time_unix)
        data.num_nodes = len(bus_ids)
        if W_year is not None:
            data.gen_weights_by_year = torch.tensor(W_year)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save([data], self.processed_paths[0])

        meta = {
            "bus_ids": bus_ids,
            "nearest_place": nodes.get("nearest_place",
                                       pd.Series([None] * len(nodes))).tolist(),
            "node_features": NODE_FEATURES,
            "edge_features": EDGE_FEATURES,
            "gen_weight_types": types,
            "gen_weight_years": weight_years,
            "zone_columns": zone.columns.tolist(),
            "revision": self.revision,
            "summary": summary,
        }
        (Path(self.processed_dir) / "meta.json").write_text(json.dumps(meta))

    # ------------------------------------------------------------------ helpers
    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def bus_ids(self) -> list[str]:
        return self._meta["bus_ids"]

    @property
    def zone_columns(self) -> list[str]:
        return self._meta["zone_columns"]

    def zone_dataframe(self) -> pd.DataFrame:
        """Zone-level hourly series as a pandas DataFrame (UTC index)."""
        d = self[0]
        index = pd.to_datetime(d.time_unix.numpy(), unit="s", utc=True)
        return pd.DataFrame(d.zone_series.numpy(), index=index,
                            columns=self.zone_columns)

    def node_generation(self, gen_type: str = "Fossil coal (combined)",
                        by_commissioning: bool = True) -> torch.Tensor:
        """Heuristic per-bus generation series ``[T, N]`` for one type.

        This is zone generation spread over buses by installed-capacity
        share of *mapped* plants — a documented heuristic, not metering.
        For ``"Fossil coal (combined)"`` the two ENTSO-E fossil columns are
        summed first (their classification drifts across years).

        With ``by_commissioning=True`` (default) each hour uses the weights
        of its calendar year, so a plant gets no share of zone generation
        before its commissioning year (``topology/plants.csv``). Releases up
        to v0.1.3 do not ship per-year weights; for those, or with
        ``by_commissioning=False``, the static weights of the current fleet
        are applied to every hour.
        """
        d = self[0]
        types = self._meta["gen_weight_types"]
        if gen_type not in types:
            raise ValueError(f"unknown type {gen_type!r}; choose from {types}")
        years = self._meta.get("gen_weight_years") or []
        if by_commissioning and years and "gen_weights_by_year" in d:
            hour_year = pd.to_datetime(d.time_unix.numpy(), unit="s",
                                       utc=True).year.to_numpy()
            pos = np.clip(hour_year - years[0], 0, len(years) - 1)
            w = d.gen_weights_by_year[torch.as_tensor(pos), :,
                                      types.index(gen_type)]  # [T, N]
        else:
            w = d.gen_weights[:, types.index(gen_type)].unsqueeze(0)  # [1, N]
        cols = self.zone_columns
        if gen_type == "Fossil coal (combined)":
            sel = [cols.index("gen_fossil_brown_coal_lignite_mw"),
                   cols.index("gen_fossil_hard_coal_mw")]
            zone = d.zone_series[:, sel].nansum(dim=1)  # [T]
        else:
            name = ("gen_" + gen_type.lower().replace(" ", "_")
                    .replace("/", "_").replace("-", "_") + "_mw")
            zone = d.zone_series[:, cols.index(name)]
        return zone.unsqueeze(1) * w
