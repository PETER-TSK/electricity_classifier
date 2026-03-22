"""
Phase 2 — Preprocessing.

Loads all 8 raw CSVs, aligns them on a common daily date index,
forward-fills commodity gaps, and drops days with too many missing values.
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    COMMODITY_COLS,
    FFILL_LIMIT,
    MAX_NAN_ALLOWED,
    PROC_DIR,
    RAW_DIR,
)


def _load_raw() -> dict[str, pd.DataFrame]:
    """Load all raw CSVs and normalise their indices to tz-naive date timestamps."""
    files = {
        "gr_price": "gr_dam_price.csv",
        "res":      "res.csv",
        "load":     "load.csv",
        "balance":  "energy_balance.csv",
        "ttf":      "ttf.csv",
        "coal":     "coal.csv",
        "co2":      "co2_eua.csv",
        "wx":       "weather_athens.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        df = pd.read_csv(RAW_DIR / filename, index_col=0, parse_dates=True)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        frames[key] = df
    return frames


def run_preprocessing() -> pd.DataFrame:
    """
    Merge all raw sources into a single daily DataFrame, forward-fill
    commodity prices over non-trading days, and drop data-sparse rows.

    Returns the merged DataFrame and saves it to data/processed/merged_daily.csv.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    raw = _load_raw()

    # ── Merge on common date index ─────────────────────────────────────────────
    df = pd.DataFrame({
        "gr_price":        raw["gr_price"].squeeze(),
        "res_MWh":         raw["res"].squeeze(),
        "load_MWh":        raw["load"].squeeze(),
        "net_imports_MWh": raw["balance"]["net_imports_MWh"],
        "hydro_MWh":       raw["balance"]["hydro_MWh"],
        "ttf":             raw["ttf"].squeeze(),
        "coal":            raw["coal"].squeeze(),
        "co2_eua":         raw["co2"].squeeze(),
        "tmax":            raw["wx"]["tmax"],
        "tmin":            raw["wx"]["tmin"],
    })
    df.index.name = "date"

    print(f"Merged shape  : {df.shape}")
    print(f"Date range    : {df.index.min().date()} → {df.index.max().date()}")
    print(f"\nMissing before ffill:\n{df.isnull().sum().to_string()}")

    # ── Forward-fill commodity prices (no trading on weekends / holidays) ──────
    df[COMMODITY_COLS] = df[COMMODITY_COLS].ffill(limit=FFILL_LIMIT)

    # ── Drop rows with too many missing columns ────────────────────────────────
    df = df.dropna(thresh=len(df.columns) - MAX_NAN_ALLOWED)

    print(f"\nShape after cleaning : {df.shape}")
    print(f"Date range           : {df.index.min().date()} → {df.index.max().date()}")
    print(f"\nMissing after ffill + dropna:\n{df.isnull().sum().to_string()}")

    # ── Save ───────────────────────────────────────────────────────────────────
    out = PROC_DIR / "merged_daily.csv"
    df.to_csv(out)
    print(f"\nSaved → {out.relative_to(out.parents[2])}")
    return df
