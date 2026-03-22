"""
Phase 4 — Model Training.

Trains AutoGluon stacked ensemble classifiers for D+1 (day-ahead) price regime prediction using a strict
temporal train / validation / test split.
"""
from __future__ import annotations

import logging
import sys

import pandas as pd
from autogluon.tabular import TabularPredictor

from src.config import (
    AG_BAG_FOLDS,
    AG_BAG_SETS,
    AG_LABEL,
    AG_METRIC,
    AG_PRESETS,
    AG_STACK_LEVELS,
    AG_TIME_LIMIT,
    CLASS_WEIGHTS,
    DROP_FOR_TRAINING,
    MODELS_DIR,
    PROC_DIR,
    ROLLING_ORIGINS,
    ROOT,
    TRAIN_END,
    VAL_END,
)

# ─── Logging — stdout + persistent file ──────────────────────────────────────
_LOG_DIR = ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / "training.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_features() -> pd.DataFrame:
    return pd.read_csv(
        PROC_DIR / "features_final.csv",
        index_col=0,
        parse_dates=True,
    )


def _temporal_split(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test on date boundaries defined in config.
    Drops raw columns that would leak the target or are already encoded.
    """
    other_targets = [
        c for c in df.columns
        if c.startswith("target_") and c != target_col
    ]
    cols_to_drop = [c for c in drop_cols + other_targets if c in df.columns]
    clean = df.drop(columns=cols_to_drop).rename(columns={target_col: AG_LABEL})

    train = clean[clean.index <  TRAIN_END]
    val   = clean[(clean.index >= TRAIN_END) & (clean.index < VAL_END)]
    test  = clean[clean.index >= VAL_END]

    log.info("  Train: %d  Val: %d  Test: %d", len(train), len(val), len(test))
    log.info("  Features: %d", len(train.columns) - 1)
    return train, val, test


def _fit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    model_path: str,
) -> TabularPredictor:
    """Fit a TabularPredictor and log the test-set leaderboard."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("  Saving model to: %s", model_path)

    # Cost-sensitive class weights: upweight rare post-crisis High/Spike classes
    train = train.copy()
    val   = val.copy()
    train["sample_weight"] = train[AG_LABEL].map(CLASS_WEIGHTS).fillna(1.0)
    val["sample_weight"]   = val[AG_LABEL].map(CLASS_WEIGHTS).fillna(1.0)

    predictor = TabularPredictor(
        label             = AG_LABEL,
        eval_metric       = AG_METRIC,
        path              = model_path,
        verbosity         = 2,
        sample_weight     = "sample_weight",   # column in train_data
        weight_evaluation = False,             # eval uses balanced_accuracy unweighted
    ).fit(
        train_data       = train,
        tuning_data      = val,
        use_bag_holdout  = True,   # keep val as a true holdout in bagged mode
        presets                = AG_PRESETS,
        num_stack_levels       = AG_STACK_LEVELS,
        num_bag_folds          = AG_BAG_FOLDS,
        num_bag_sets           = AG_BAG_SETS,
        time_limit             = AG_TIME_LIMIT,
        # NNs re-enabled — they provide better Low/High separation; Spike=2.0 prevents Q4 collapse
    )

    log.info("\n--- Leaderboard (test set) ---")
    lb = predictor.leaderboard(test, silent=True)
    log.info("\n%s", lb[["model", "score_test", "score_val", "pred_time_test"]].head(10).to_string())
    return predictor


def _temporal_split_origin(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    train_end: str,
    val_end: str,
    test_end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split for a single rolling origin.
    Train: [data_start, train_end)
    Val:   [train_end,  val_end)
    Test:  [val_end,    test_end)  — or to end of data if test_end is None
    """
    other_targets = [c for c in df.columns if c.startswith("target_") and c != target_col]
    cols_to_drop  = [c for c in drop_cols + other_targets if c in df.columns]
    clean = df.drop(columns=cols_to_drop).rename(columns={target_col: AG_LABEL})

    train = clean[clean.index < train_end]
    val   = clean[(clean.index >= train_end) & (clean.index < val_end)]
    test  = (clean[clean.index >= val_end] if test_end is None
             else clean[(clean.index >= val_end) & (clean.index < test_end)])

    log.info("  Train: %d  Val: %d  Test: %d", len(train), len(val), len(test))
    log.info("  Features: %d", len(train.columns) - 1)
    return train, val, test


# ─── Public API ───────────────────────────────────────────────────────────────

def train_rolling_origins() -> list[dict]:
    """
    Train one AutoGluon model per rolling origin (3 models, expanding window).
    Each model saved to models/day_ahead_{name}/.

    Returns
    -------
    list of dicts: [{"name": str, "predictor": TabularPredictor, "test_df": pd.DataFrame}]
    """
    df = _load_features()
    results = []
    for name, train_end, val_end, test_end in ROLLING_ORIGINS:
        log.info("=" * 60)
        log.info("=== Rolling Origin: %s ===", name)
        log.info("=" * 60)
        train, val, test = _temporal_split_origin(
            df, "target_1d", DROP_FOR_TRAINING, train_end, val_end, test_end
        )
        model_path = str(MODELS_DIR / f"day_ahead_{name}")
        predictor  = _fit(train, val, test, model_path)
        results.append({"name": name, "predictor": predictor, "test_df": test})
    return results


def train_day_ahead() -> TabularPredictor:
    """Train the D+1 (day-ahead) price-regime classifier."""
    log.info("=" * 60)
    log.info("=== Training D+1 model (day-ahead) ===")
    log.info("=" * 60)
    df = _load_features()
    train, val, test = _temporal_split(df, "target_1d", DROP_FOR_TRAINING)
    return _fit(train, val, test, str(MODELS_DIR / "day_ahead"))


