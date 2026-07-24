#!/usr/bin/env python3
"""
Incremental ENTSO-E Transparency Platform collector for Bosnia and Herzegovina (BA zone).

Collects every dataset the BA control area actually reports, stores each as parquet
under data/, and keeps a small state.json with coverage info. Safe to re-run: each
run refetches a small overlap window and deduplicates, so weekly cron keeps data current.

Usage:
  ENTSOE_TOKEN=... python src/collect.py            # incremental (or full on first run)
  ENTSOE_TOKEN=... python src/collect.py --full     # force full re-pull
  python src/collect.py --demo                      # generate synthetic data (pipeline test, no token)
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE = DATA / "state.json"

COUNTRY = "BA"
TZ = "Europe/Sarajevo"
HISTORY_START = "20150101"            # ENTSO-E TP data generally begins 2015
NEIGHBORS = ["HR", "RS", "ME"]
OVERLAP_DAYS = 10                     # refetch window to catch late corrections
CHUNK_DAYS = 365                      # query in ~yearly chunks

# ----------------------------------------------------------------------------- datasets

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = df.to_frame("value")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" / ".join(str(x) for x in c if str(x) != "").strip() for c in df.columns]
    df.columns = [str(c) for c in df.columns]
    return df


def make_datasets(client):
    """name -> fetch(start, end) -> DataFrame indexed by tz-aware timestamp."""
    ds = {
        "load_actual": lambda s, e: client.query_load(COUNTRY, start=s, end=e),
        "load_forecast": lambda s, e: client.query_load_forecast(COUNTRY, start=s, end=e),
        "generation_per_type": lambda s, e: client.query_generation(COUNTRY, start=s, end=e, psr_type=None),
        "generation_forecast": lambda s, e: client.query_generation_forecast(COUNTRY, start=s, end=e),
        "installed_capacity": lambda s, e: client.query_installed_generation_capacity(COUNTRY, start=s, end=e, psr_type=None),
    }
    for n in NEIGHBORS:
        ds[f"flow_{COUNTRY}_{n}"] = (lambda n_: lambda s, e: client.query_crossborder_flows(COUNTRY, n_, start=s, end=e))(n)
        ds[f"flow_{n}_{COUNTRY}"] = (lambda n_: lambda s, e: client.query_crossborder_flows(n_, COUNTRY, start=s, end=e))(n)
        ds[f"sched_{COUNTRY}_{n}"] = (lambda n_: lambda s, e: client.query_scheduled_exchanges(COUNTRY, n_, start=s, end=e))(n)
        ds[f"sched_{n}_{COUNTRY}"] = (lambda n_: lambda s, e: client.query_scheduled_exchanges(n_, COUNTRY, start=s, end=e))(n)
    # Datasets BA may or may not report — attempted, skipped gracefully if empty:
    ds["day_ahead_prices"] = lambda s, e: client.query_day_ahead_prices(COUNTRY, start=s, end=e)
    ds["hydro_reservoir_storage"] = lambda s, e: client.query_aggregate_water_reservoirs_and_hydro_storage(COUNTRY, start=s, end=e)
    return ds


# ----------------------------------------------------------------------------- helpers

def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"datasets": {}}


def save_state(state: dict):
    state["last_run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True))


def read_existing(name: str) -> pd.DataFrame | None:
    f = DATA / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    return None


def merge_save(name: str, old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    frames = [f for f in (old, new) if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(DATA / f"{name}.parquet")
    return df


def fetch_range(fetch, start: pd.Timestamp, end: pd.Timestamp, name: str) -> pd.DataFrame | None:
    """Fetch [start, end) in chunks; tolerate empty periods; return None if nothing at all."""
    from entsoe.exceptions import NoMatchingDataError
    pieces = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=CHUNK_DAYS), end)
        for attempt in range(3):
            try:
                out = fetch(cur, nxt)
                if out is not None and len(out):
                    pieces.append(_flatten_cols(out))
                break
            except NoMatchingDataError:
                break
            except Exception as ex:  # rate limit / transient
                if attempt == 2:
                    print(f"    ! {name} {cur:%Y-%m-%d}->{nxt:%Y-%m-%d}: {type(ex).__name__}: {ex}", flush=True)
                else:
                    time.sleep(8 * (attempt + 1))
        cur = nxt
        time.sleep(0.4)  # stay far below rate limits
    if not pieces:
        return None
    return pd.concat(pieces)


# ----------------------------------------------------------------------------- demo mode

def demo_data():
    """Synthetic-but-plausible BA data so the whole pipeline can be tested without a token."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", "2026-07-20 23:00", freq="h", tz=TZ)
    hours = idx.hour.values
    doy = idx.dayofyear.values
    base = 1150 + 260 * np.cos((doy - 15) / 365 * 2 * np.pi)          # winter peak
    daily = 140 * np.sin((hours - 7) / 24 * 2 * np.pi)
    load = base + daily + rng.normal(0, 45, len(idx))
    hydro = np.clip(520 + 320 * np.sin((doy - 100) / 365 * 2 * np.pi) + rng.normal(0, 90, len(idx)), 60, None)
    lignite = np.clip(load * 0.62 - hydro * 0.45 + rng.normal(0, 60, len(idx)), 350, 1450)
    wind = np.clip(rng.gamma(2.0, 28, len(idx)), 0, 135)
    solar = np.clip(95 * np.sin((hours - 6) / 13 * np.pi), 0, None) * (doy % 9 != 0)
    gen = pd.DataFrame({
        "Fossil Brown coal/Lignite / Actual Aggregated": lignite,
        "Hydro Water Reservoir / Actual Aggregated": hydro * 0.55,
        "Hydro Run-of-river and poundage / Actual Aggregated": hydro * 0.35,
        "Hydro Pumped Storage / Actual Aggregated": hydro * 0.10,
        "Wind Onshore / Actual Aggregated": wind,
        "Solar / Actual Aggregated": solar,
    }, index=idx).round(1)
    total_gen = gen.sum(axis=1)
    net_pos = total_gen - load
    out = {
        "load_actual": pd.DataFrame({"Actual Load": load.round(1)}, index=idx),
        "load_forecast": pd.DataFrame({"Forecasted Load": (load + rng.normal(0, 55, len(idx))).round(1)}, index=idx),
        "generation_per_type": gen,
        "installed_capacity": pd.DataFrame(
            {"Fossil Brown coal/Lignite": [1765], "Hydro Water Reservoir": [1505],
             "Hydro Run-of-river and poundage": [600], "Hydro Pumped Storage": [420],
             "Wind Onshore": [135], "Solar": [107]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz=TZ)])),
    }
    for n, w in zip(NEIGHBORS, (0.5, 0.3, 0.2)):
        exp = np.clip(net_pos * w + rng.normal(0, 30, len(idx)), 0, None)
        imp = np.clip(-net_pos * w + rng.normal(0, 30, len(idx)), 0, None)
        out[f"flow_BA_{n}"] = pd.DataFrame({"value": exp.round(1)}, index=idx)
        out[f"flow_{n}_BA"] = pd.DataFrame({"value": imp.round(1)}, index=idx)
    return out


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="force full historical re-pull")
    ap.add_argument("--demo", action="store_true", help="write synthetic data (no token needed)")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    state = load_state()

    if args.demo:
        print("DEMO MODE — writing synthetic data")
        for name, df in demo_data().items():
            merge_save(name, None, df)
            state["datasets"][name] = {
                "rows": int(len(df)), "start": str(df.index.min()), "end": str(df.index.max()),
                "status": "demo",
            }
        state["mode"] = "demo"
        save_state(state)
        print("done.")
        return

    token = os.environ.get("ENTSOE_TOKEN", "").strip()
    if not token:
        sys.exit("ERROR: set ENTSOE_TOKEN environment variable (get one at transparency.entsoe.eu).")

    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=token)
    datasets = make_datasets(client)

    now = pd.Timestamp.now(tz=TZ).ceil("h")
    hist_start = pd.Timestamp(HISTORY_START, tz=TZ)

    for name, fetch in datasets.items():
        old = None if args.full else read_existing(name)
        prev = state["datasets"].get(name, {})
        if old is not None and len(old) and prev.get("status") != "demo":
            start = pd.Timestamp(old.index.max()).tz_convert(TZ) - pd.Timedelta(days=OVERLAP_DAYS)
        else:
            old = None
            start = hist_start
        print(f"[{name}] {start:%Y-%m-%d} -> {now:%Y-%m-%d}", flush=True)
        try:
            new = fetch_range(fetch, start, now, name)
        except Exception:
            traceback.print_exc()
            new = None
        if new is None and old is None:
            f = DATA / f"{name}.parquet"
            if f.exists():
                f.unlink()  # drop stale demo file
            state["datasets"][name] = {"rows": 0, "status": "not_reported_by_BA"}
            print("    (no data — BA does not appear to report this)", flush=True)
            continue
        df = merge_save(name, old, new if new is not None else pd.DataFrame())
        state["datasets"][name] = {
            "rows": int(len(df)), "start": str(df.index.min()), "end": str(df.index.max()),
            "status": "ok",
        }
        print(f"    {len(df):,} rows total, through {df.index.max()}", flush=True)

    state["mode"] = "live"
    save_state(state)
    print("done.")


if __name__ == "__main__":
    main()
