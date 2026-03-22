"""
Phase 3 — Feature Engineering.

Builds the full feature matrix from the merged daily DataFrame and
the ENEX DAM Aggregate Curves file.

Feature groups
--------------
A. Price lags & momentum
B. Commodity cost-pressure features  (Halužan et al., 2020)
   + dynamic commodity regime ratios (replaces static is_energy_crisis flag)
C. Grid balance features              (Demir et al., 2020)
D. Weather — HDD / CDD               (base 18 °C, Athens standard)
E. Calendar — cyclical encoding      (Jędrzejewski et al., 2022)
F. ENEX DAM market-structure features (current-day + t-1 lags)
G. Target variables                  (D+1 price regime)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    HDD_CDD_BASE,
    HOLIDAY_YEARS,
    PRICE_BINS,
    PRICE_LABELS,
    PROC_DIR,
    RAW_DIR,
)


# ─── Feature group builders ───────────────────────────────────────────────────


def _price_lags(df: pd.DataFrame) -> pd.DataFrame:
    """A. Lagged prices and 7-day momentum."""
    df = df.copy()
    df["price_lag_1d"]  = df["gr_price"].shift(1)
    df["price_lag_7d"]  = df["gr_price"].shift(7)
    df["price_lag_14d"] = df["gr_price"].shift(14)
    df["price_mom_7d"]  = (
        (df["gr_price"].shift(1) - df["gr_price"].shift(8))
        / df["gr_price"].shift(8)
    )
    return df


def _commodity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    B. 7-day rolling commodity prices, cost-pressure interaction terms,
    and dynamic regime ratios (commodity price vs. its 12-month median).

    The regime ratios replace the old static is_energy_crisis flag.
    Values > 1 indicate an elevated-cost environment; values < 1 indicate
    a relaxed market — this signal adapts continuously as the market evolves.
    """
    df = df.copy()
    df["ttf_roll7"]  = df["ttf"].rolling(7, min_periods=1).mean()
    df["coal_roll7"] = df["coal"].rolling(7, min_periods=1).mean()
    df["ttf_x_load"] = df["ttf"]     * df["load_MWh"] / 1e6
    df["co2_x_load"] = df["co2_eua"] * df["load_MWh"] / 1e6

    # Dynamic commodity regime ratios (12-month rolling median as baseline)
    ttf_med_12m  = df["ttf"].rolling(365, min_periods=90).median()
    co2_med_12m  = df["co2_eua"].rolling(365, min_periods=90).median()
    price_med_12m = df["gr_price"].rolling(365, min_periods=90).median()

    df["ttf_vs_median_12m"]   = (df["ttf"]     / ttf_med_12m.replace(0, np.nan))
    df["co2_vs_median_12m"]   = (df["co2_eua"] / co2_med_12m.replace(0, np.nan))
    df["price_vs_median_12m"] = (df["price_lag_1d"] / price_med_12m.replace(0, np.nan))

    return df


def _grid_balance(df: pd.DataFrame) -> pd.DataFrame:
    """C. Residual load, RES penetration, hydro anomaly, import dependency."""
    df = df.copy()
    df["residual_load"] = df["load_MWh"] - df["res_MWh"]
    df["res_share"]     = df["res_MWh"]  / df["load_MWh"]
    df["hydro_deficit"] = df["hydro_MWh"] - df["hydro_MWh"].rolling(30, min_periods=1).mean()
    df["import_share"]  = df["net_imports_MWh"] / df["load_MWh"]
    return df


def _weather_hdd_cdd(df: pd.DataFrame) -> pd.DataFrame:
    """D. Heating / Cooling Degree Days, 7-day rolling means, and weather×grid interaction."""
    df = df.copy()
    df["t_mean"] = (df["tmax"] + df["tmin"]) / 2
    df["HDD"]    = (HDD_CDD_BASE - df["t_mean"]).clip(lower=0)
    df["CDD"]    = (df["t_mean"] - HDD_CDD_BASE).clip(lower=0)
    df["HDD_7d"] = df["HDD"].rolling(7).mean()
    df["CDD_7d"] = df["CDD"].rolling(7).mean()
    # Post-crisis High driver: cold snap (HDD) × grid stress (residual_load)
    # residual_load already computed in _grid_balance() which runs before this function
    df["hdd_x_residual_load"] = df["HDD"] * df["residual_load"] / 1e6
    return df


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    """E. Cyclical calendar encoding and Greek public holidays."""
    import holidays as _hol

    df = df.copy()
    gr_holidays = _hol.Greece(years=list(HOLIDAY_YEARS))

    holiday_index = pd.DatetimeIndex(gr_holidays.keys())
    df["is_holiday"]  = df.index.normalize().isin(holiday_index).astype(int)
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    df["month_sin"]   = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"]       / 12)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    return df


def _aggrcurves_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    F. ENEX DAM market-structure features derived from the aggregate
    supply/demand curves (aggrcurves_dam.csv).

    Current-day features (8):
    - price_spread_mean   : mean over 24 h of (max − min price) per hour
    - buy_volume_total    : total buy-side volume (MWh)
    - sell_volume_total   : total sell-side volume (MWh)
    - buy_sell_imbalance  : (buy − sell) / (buy + sell)
    - curve_steepness     : avg ΔPrice/ΔQty between 25th and 75th pct volume
    - sell_price_p25vol   : avg sell price at 25% of daily volume
    - sell_price_p50vol   : avg sell price at 50% of daily volume
    - sell_price_p75vol   : avg sell price at 75% of daily volume

    t-1 lag features (8): all above features shifted by 1 day.
    Yesterday's market structure is known before today's auction closes
    and provides additional predictive signal for the D+1 regime.
    """
    print("  Loading aggrcurves_dam.csv (1.2 GB)...")
    aggr = pd.read_csv(
        RAW_DIR / "aggrcurves_dam.csv",
        parse_dates=["date"],
        usecols=["date", "SIDE_DESCR", "SORT", "QUANTITY", "UNITPRICE"],
    )
    print(f"  {len(aggr):,} rows  {aggr['date'].min().date()} → {aggr['date'].max().date()}")

    buy  = aggr[aggr["SIDE_DESCR"] == "Buy"]
    sell = aggr[aggr["SIDE_DESCR"] == "Sell"].copy()

    # Price spread per hour → daily mean
    price_spread_mean = (
        aggr.groupby(["date", "SORT"])["UNITPRICE"]
        .agg(lambda x: x.max() - x.min())
        .groupby("date").mean()
        .rename("price_spread_mean")
    )

    # Total bid / offer volumes (max cumulative qty per hour, summed over 24 h)
    buy_vol  = (
        buy.groupby(["date", "SORT"])["QUANTITY"].max()
        .groupby("date").sum()
        .rename("buy_volume_total")
    )
    sell_vol = (
        sell.groupby(["date", "SORT"])["QUANTITY"].max()
        .groupby("date").sum()
        .rename("sell_volume_total")
    )
    buy_sell_imbalance = (
        (buy_vol - sell_vol) / (buy_vol + sell_vol)
    ).rename("buy_sell_imbalance")

    # Supply-curve steepness: ΔPrice / ΔQty between 25th and 75th pct of volume
    sell["max_qty"] = sell.groupby(["date", "SORT"])["QUANTITY"].transform("max")

    def _at_pct(pct: float) -> pd.Series:
        """Min price at or above `pct`× max daily volume, averaged over 24 h."""
        mask = sell["QUANTITY"] >= pct * sell["max_qty"]
        return (
            sell[mask]
            .groupby(["date", "SORT"])["UNITPRICE"].min()
            .groupby("date").mean()
        )

    def _qty_at_pct(pct: float) -> pd.Series:
        mask = sell["QUANTITY"] >= pct * sell["max_qty"]
        return (
            sell[mask]
            .groupby(["date", "SORT"])["QUANTITY"].min()
            .groupby("date").mean()
        )

    p25_price = _at_pct(0.25)
    p50_price = _at_pct(0.50)
    p75_price = _at_pct(0.75)
    p25_qty   = _qty_at_pct(0.25)
    p75_qty   = _qty_at_pct(0.75)

    curve_steepness = (
        (p75_price - p25_price) / (p75_qty - p25_qty).replace(0, np.nan)
    ).rename("curve_steepness")

    feats = pd.concat(
        [
            price_spread_mean,
            buy_vol,
            sell_vol,
            buy_sell_imbalance,
            curve_steepness,
            p25_price.rename("sell_price_p25vol"),
            p50_price.rename("sell_price_p50vol"),
            p75_price.rename("sell_price_p75vol"),
        ],
        axis=1,
    )
    feats.index = pd.to_datetime(feats.index)

    df = df.join(feats, how="left")

    # t-1 lags of all market-structure features
    aggr_cols = [
        "price_spread_mean", "buy_volume_total", "sell_volume_total",
        "buy_sell_imbalance", "curve_steepness",
        "sell_price_p25vol", "sell_price_p50vol", "sell_price_p75vol",
    ]
    for col in aggr_cols:
        df[f"{col}_lag1"] = df[col].shift(1)

    print(f"  Added {len(aggr_cols) * 2} market-structure features (current + t-1 lags).")
    return df


def _targets(df: pd.DataFrame) -> pd.DataFrame:
    """G. D+1 price-regime target (Low / Normal / High / Spike)."""
    df = df.copy()
    df["target_1d"] = pd.cut(
        df["gr_price"].shift(-1),
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
    )
    return df


# ─── Public API ───────────────────────────────────────────────────────────────


def build_features() -> pd.DataFrame:
    """
    Build the full feature matrix from merged_daily.csv + aggrcurves_dam.csv.

    Applies all feature groups A–G, drops rows without a valid D+1 target or
    oldest price lag, and saves the result to data/processed/features_final.csv.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        PROC_DIR / "merged_daily.csv",
        index_col=0,
        parse_dates=True,
    )
    print(f"Loaded merged_daily: {df.shape}")

    print("[A] Price lags & momentum...")
    df = _price_lags(df)

    print("[B] Commodity features + dynamic regime ratios...")
    df = _commodity_features(df)

    print("[C] Grid balance features...")
    df = _grid_balance(df)

    print("[D] Weather HDD/CDD...")
    df = _weather_hdd_cdd(df)

    print("[E] Calendar encoding...")
    df = _calendar(df)

    print("[F] ENEX DAM market-structure features (current + t-1 lags)...")
    df = _aggrcurves_features(df)

    print("[G] Target variables...")
    df = _targets(df)

    # Drop rows where target or the oldest lag is NaN
    df = df.dropna(subset=["target_1d", "price_lag_14d"])

    out = PROC_DIR / "features_final.csv"
    df.to_csv(out)

    print(f"\nFinal shape : {df.shape}")
    print(f"Date range  : {df.index.min().date()} → {df.index.max().date()}")
    print(f"Saved       → {out.relative_to(out.parents[2])}")
    print(f"\ntarget_1d distribution:\n{df['target_1d'].value_counts().to_string()}")
    return df
