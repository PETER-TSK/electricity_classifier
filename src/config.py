"""
Central configuration for the Greece EPF project.
All paths, hyperparameters, and constants live here.
"""
from pathlib import Path

# ─── Project paths ────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ENEX_DIR   = ROOT / "enex_data"

# ─── Data collection ──────────────────────────────────────────────────────────
COLLECT_START = "2020-01-01"
COLLECT_END   = "2026-03-20"   # extended to include 2026 Q1

# Yahoo Finance tickers
# Note: ECF=F (ICE EUA futures) has only 1 row on Yahoo Finance.
# KRBN (KraneShares Global Carbon ETF) is used as a highly-correlated EUA proxy.
YFINANCE_TICKERS: dict[str, str] = {
    "ttf":     "TTF=F",
    "coal":    "MTF=F",
    "co2_eua": "KRBN",
}

# Open-Meteo archive API — Athens, Greece
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_LAT = 37.98
OPENMETEO_LON = 23.73
OPENMETEO_TZ  = "Europe/Athens"

# data.gov.gr REST API — ADMIE datasets (no auth required)
DATAGOV_BASE = "https://data.gov.gr/api/v1/query"
DATAGOV_DATASETS: dict[str, str] = {
    "res":            "admie_realtimescadares",
    "load":           "admie_realtimescadasystemload",
    "energy_balance": "admie_dailyenergybalanceanalysis",
}
# Greek fuel-type labels used in the energy balance response
NET_IMPORTS_FUEL = "ΚΑΘΑΡΕΣ ΕΙΣΑΓΩΓΕΣ (ΕΙΣΑΓΩΓΕΣ-ΕΞΑΓΩΓΕΣ)"
HYDRO_FUEL       = "ΥΔΡΟΗΛΕΚΤΡΙΚΑ"

# ─── Preprocessing ────────────────────────────────────────────────────────────
COMMODITY_COLS  = ["ttf", "coal", "co2_eua"]  # forward-filled on non-trading days
FFILL_LIMIT     = 3    # max consecutive days to forward-fill
MAX_NAN_ALLOWED = 3    # drop rows that exceed this many missing columns

# ─── Feature engineering ──────────────────────────────────────────────────────
HDD_CDD_BASE  = 18.0               # °C base temperature (Athens standard)
HOLIDAY_YEARS = range(2020, 2027)

# ─── Target variable ─────────────────────────────────────────────────────────
PRICE_BINS   = [0, 80, 150, 250, 9_999]           # EUR/MWh bin edges
PRICE_LABELS = ["Low", "Normal", "High", "Spike"]  # 4 classes

# ─── Temporal train / val / test split ───────────────────────────────────────
# New sliding window: train includes crisis + 1 post-crisis year,
# val is the first full post-crisis year, test is 2025–2026.
TRAIN_END = "2024-01-01"   # train  : [start, 2024-01-01)  — ~3 years incl. crisis + 2023
VAL_END   = "2025-01-01"   # val    : [2024-01-01, 2025-01-01)  — 1 full post-crisis year
                            # test   : [2025-01-01, end]  — 2025 + Q1 2026

# Raw columns dropped before training (either leak the target or are already
# encoded as engineered features)
DROP_FOR_TRAINING = [
    "gr_price",         # raw price → directly encodes target class
    "tmax", "tmin",     # encoded as HDD / CDD
    "t_mean",           # intermediate weather column
    "month",            # encoded as month_sin / month_cos
    "day_of_week",      # encoded as dow_sin / dow_cos
]

# ─── AutoGluon ────────────────────────────────────────────────────────────────
AG_LABEL        = "target_1d"
AG_METRIC       = "balanced_accuracy"
AG_PRESETS      = "best_quality"
AG_STACK_LEVELS = 1    # L2 stacking enabled; requires AG_BAG_FOLDS > 0
AG_BAG_FOLDS    = 5    # 5-fold OOF for ensemble calibration (features pre-computed, no leakage)
AG_BAG_SETS     = 1
AG_TIME_LIMIT   = 7200  # seconds; bagging+stacking needs ~2× time

# ─── Class weights for cost-sensitive training ────────────────────────────────
# High and Spike are rare post-crisis; upweight so the model does not ignore them.
CLASS_WEIGHTS: dict[str, float] = {
    "Low":    2.0,
    "Normal": 1.0,
    "High":   2.0,
    "Spike":  2.0,
}

# ─── Rolling Origin Cross-Validation ─────────────────────────────────────────
# "UP" = prices above Normal threshold (High or Spike in 4-class EPF)
UP_CLASSES: list[str] = ["High", "Spike"]

# (name, train_end, val_end, test_end | None = data end)
# Train: [data_start, train_end)  — expands each origin, never drops old data
# Val:   [train_end,  val_end)    — last 6 months before test (AutoGluon holdout)
# Test:  [val_end,    test_end)   — never seen during training
ROLLING_ORIGINS: list[tuple] = [
    ("2023", "2022-07-01", "2023-01-01", "2024-01-01"),
    ("2024", "2023-07-01", "2024-01-01", "2025-01-01"),
    ("2025", "2024-07-01", "2025-01-01", None),
]

# Threshold sweep: P(UP) = P(High) + P(Spike) >= threshold → predict UP
WF_THRESHOLD_MIN  = 0.40
WF_THRESHOLD_MAX  = 0.75
WF_THRESHOLD_STEP = 0.05
WF_PROD_THRESHOLD = 0.65   # production threshold: predict UP only when confident
