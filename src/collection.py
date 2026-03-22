"""
Phase 1 — Raw data collection.

Downloads and saves all external data sources to data/raw/.
All functions are idempotent: re-running overwrites existing files.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from src.config import (
    COLLECT_END,
    COLLECT_START,
    DATAGOV_BASE,
    DATAGOV_DATASETS,
    HYDRO_FUEL,
    NET_IMPORTS_FUEL,
    OPENMETEO_LAT,
    OPENMETEO_LON,
    OPENMETEO_TZ,
    OPENMETEO_URL,
    RAW_DIR,
    YFINANCE_TICKERS,
)

# ─── Commodities (yfinance) ───────────────────────────────────────────────────


def collect_commodities(start: str = COLLECT_START) -> dict[str, pd.Series]:
    """Download daily close prices for TTF, coal, and CO2 EUA proxy via yfinance."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.Series] = {}

    for name, ticker in YFINANCE_TICKERS.items():
        series = (
            yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
            .squeeze()
        )
        series.index.name = "date"
        series.name = name
        series.to_csv(RAW_DIR / f"{name}.csv")
        print(
            f"  {name}.csv: {len(series)} rows  "
            f"{series.index.min().date()} → {series.index.max().date()}"
        )
        results[name] = series

    return results


# ─── Weather (open-meteo) ─────────────────────────────────────────────────────


def collect_weather(
    start: str = COLLECT_START,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily Tmax/Tmin for Athens from the open-meteo archive API."""
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    end = end or date.today().isoformat()

    session = requests_cache.CachedSession(".cache", expire_after=3600)
    session = retry(session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=session)

    params = {
        "latitude":   OPENMETEO_LAT,
        "longitude":  OPENMETEO_LON,
        "start_date": start,
        "end_date":   end,
        "daily":      ["temperature_2m_max", "temperature_2m_min"],
        "timezone":   OPENMETEO_TZ,
    }

    response = client.weather_api(OPENMETEO_URL, params=params)[0]
    daily    = response.Daily()

    tmax   = daily.Variables(0).ValuesAsNumpy()
    tmin   = daily.Variables(1).ValuesAsNumpy()
    n_days = len(tmax)

    idx = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True)
                  .tz_convert(OPENMETEO_TZ)
                  .normalize(),
        periods=n_days,
        freq="D",
    )

    df = pd.DataFrame({"tmax": tmax, "tmin": tmin}, index=idx)
    df.index = df.index.tz_localize(None)   # drop tz for CSV compatibility
    df.index.name = "date"
    df.to_csv(RAW_DIR / "weather_athens.csv")
    print(f"  weather_athens.csv: {len(df)} days  {df.index.min().date()} → {df.index.max().date()}")
    return df


# ─── Greek grid data (data.gov.gr) ───────────────────────────────────────────


def _fetch_datagov(dataset_tag: str, year_start: int, year_end: int) -> pd.DataFrame:
    """Fetch one data.gov.gr dataset year-by-year and return a concatenated DataFrame."""
    rows: list[dict] = []
    for year in range(year_start, year_end + 1):
        params = {"date_from": f"{year}-01-01", "date_to": f"{year}-12-31"}
        response = requests.get(
            f"{DATAGOV_BASE}/{dataset_tag}",
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        batch = response.json()
        rows.extend(batch)
        print(f"    {dataset_tag} {year}: {len(batch):,} records")
    return pd.DataFrame(rows)


def collect_grid_data(
    start: str = COLLECT_START,
    end: str = COLLECT_END,
) -> dict[str, pd.DataFrame]:
    """Download RES generation, system load, and energy balance from data.gov.gr."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    year_start = int(start[:4])
    year_end   = int(end[:4])
    results: dict[str, pd.DataFrame] = {}

    # ── RES generation ────────────────────────────────────────────────────────
    print("  Fetching RES generation...")
    df_res = _fetch_datagov(DATAGOV_DATASETS["res"], year_start, year_end)
    df_res["date"] = pd.to_datetime(df_res["date"]).dt.normalize()
    res_daily = df_res.groupby("date")["energy_mwh"].sum().rename("energy_mwh")
    res_daily.to_csv(RAW_DIR / "res.csv")
    print(f"    → res.csv: {len(res_daily)} days")
    results["res"] = res_daily.to_frame()

    # ── System load ───────────────────────────────────────────────────────────
    print("  Fetching system load...")
    df_load = _fetch_datagov(DATAGOV_DATASETS["load"], year_start, year_end)
    df_load["date"] = pd.to_datetime(df_load["date"]).dt.normalize()
    load_daily = df_load.groupby("date")["energy_mwh"].sum().rename("energy_mwh")
    load_daily.to_csv(RAW_DIR / "load.csv")
    print(f"    → load.csv: {len(load_daily)} days")
    results["load"] = load_daily.to_frame()

    # ── Energy balance (net imports + hydro) ──────────────────────────────────
    print("  Fetching energy balance...")
    df_bal = _fetch_datagov(DATAGOV_DATASETS["energy_balance"], year_start, year_end)
    df_bal["date"] = pd.to_datetime(df_bal["date"]).dt.normalize()

    net_imports = (
        df_bal[df_bal["fuel"] == NET_IMPORTS_FUEL]
        .groupby("date")["energy_mwh"].sum()
        .rename("net_imports_MWh")
    )
    hydro = (
        df_bal[df_bal["fuel"] == HYDRO_FUEL]
        .groupby("date")["energy_mwh"].sum()
        .rename("hydro_MWh")
    )
    balance = pd.concat([net_imports, hydro], axis=1).sort_index()
    balance.to_csv(RAW_DIR / "energy_balance.csv")
    print(f"    → energy_balance.csv: {len(balance)} days")
    results["energy_balance"] = balance

    return results


# ─── Validation ───────────────────────────────────────────────────────────────

REQUIRED_RAW_FILES = [
    "gr_dam_price.csv",
    "ttf.csv",
    "coal.csv",
    "co2_eua.csv",
    "weather_athens.csv",
    "res.csv",
    "load.csv",
    "energy_balance.csv",
]


def validate_raw() -> bool:
    """Check all required raw files are present. Returns True if all found."""
    missing = [f for f in REQUIRED_RAW_FILES if not (RAW_DIR / f).exists()]
    if missing:
        print("MISSING raw files:")
        for f in missing:
            print(f"  ✗ {f}")
        return False
    print("All 8 raw files present — ready for Phase 2.")
    for f in REQUIRED_RAW_FILES:
        p = RAW_DIR / f
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        print(f"  ✓ {f:<30} {len(df):>5} rows")
    return True


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def collect_all() -> None:
    """Run the full Phase 1 data collection pipeline."""
    print("=== Phase 1 — Data Collection ===\n")

    print("[1/3] Commodities (yfinance)...")
    collect_commodities()

    print("\n[2/3] Athens weather (open-meteo)...")
    collect_weather()

    print("\n[3/3] Greek grid data (data.gov.gr)...")
    collect_grid_data()

    print("\n--- Validation ---")
    validate_raw()
