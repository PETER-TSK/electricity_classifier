# Greece EPF — Project TODO

## PHASE 0 — Setup ✅
- [x] Create project directory structure (`data/`, `models/`, `notebooks/`)
- [x] Write `requirements.txt`
- [x] Write extraction scripts (`extract_enex.py`, `extract_aggrcurves.py`)
- [x] Run `extract_enex.py` → `data/raw/gr_dam_price.csv` (1,887 days, Nov 2020–Dec 2025)
- [x] Run `extract_aggrcurves.py` → `data/raw/aggrcurves_dam.csv` (12.9M rows, 1.17 GB)

---

## PHASE 1 — Data Collection ✅
**Notebook**: `notebooks/01_data_collection.ipynb`

### Automatic (run the notebook)

- [x] Download **TTF** futures prices via yfinance → `data/raw/ttf.csv` (1,564 rows)
- [x] Download **Coal** futures prices via yfinance → `data/raw/coal.csv` (1,504 rows)
- [x] Download **CO2 EUA** via yfinance KRBN proxy → `data/raw/co2_eua.csv` (1,416 rows)
- [x] Download **Athens weather** (Tmax/Tmin) via open-meteo → `data/raw/weather_athens.csv` (2,272 days)

### Automatic (data.gov.gr — no token required, ADMIE datasets are public)

- [x] Download **RES generation** (`admie_realtimescadares`) → `data/raw/res.csv` (1,834 days)
- [x] Download **System load** (`admie_realtimescadasystemload`) → `data/raw/load.csv` (1,834 days)
- [x] Download **Energy balance** (`admie_dailyenergybalanceanalysis`) → `data/raw/energy_balance.csv` (1,818 days)

---

## PHASE 2 — Preprocessing ✅
**Notebook**: `notebooks/02_preprocessing.ipynb`
**Requires**: All 8 raw CSVs present in `data/raw/`

- [x] Merge all 8 sources into a single daily DataFrame
- [x] Forward-fill commodity prices (weekends, limit=3 days)
- [x] Drop days with >3 missing values
- [x] Save → `data/processed/merged_daily.csv`
- [x] Verify shape (1,847 rows × 10 columns, 2020-11-29 → 2025-12-31) ✓

---

## PHASE 3 — Feature Engineering ✅
**Notebook**: `notebooks/03_feature_engineering.ipynb`
**Requires**: `data/processed/merged_daily.csv` + `data/raw/aggrcurves_dam.csv`

- [x] **A. Price lags & momentum** — lag_1d, lag_7d, lag_14d, mom_7d
- [x] **B. Commodity features** — ttf_roll7, coal_roll7, ttf×load, co2×load
- [x] **C. Grid balance** — residual_load, res_share, hydro_deficit, import_share
- [x] **D. Weather** — HDD/CDD base 18°C + 7d rolling
- [x] **E. Calendar** — sin/cos encoding, is_holiday (Greek), is_weekend
- [x] **F. Regime flag** — is_energy_crisis (Sep 2021 – Jun 2023)
- [x] **G & H. Targets + ENEX AggrCurves** — target_1d/7d, price_spread_mean, buy/sell volumes, imbalance, steepness, sell price percentiles
- [x] Save → `data/processed/features_final.csv` (1,832 rows × 46 cols)
- [x] Verify: Low=397, Normal=862, High=374, Spike=199 — Spike is ~11%, usable ✓

---

## PHASE 4 — Training ✅
**Notebook**: `notebooks/04_training.ipynb`
**Requires**: `data/processed/features_final.csv`

- [x] Define temporal split: Train 749 / Val 365 / Test 718 rows
- [x] Fit D+1 AutoGluon `TabularPredictor`
  - metric   : `balanced_accuracy`
  - presets  : `best_quality` + Ray parallel folds + FastAI enabled
  - stacking : 3 levels (DyStack), 5-fold bagging, 1 h time limit
  - best model: `NeuralNetFastAI_r111_BAG_L2` — test **0.698**, val 0.679
  - ensemble (`WeightedEnsemble_L4`): val **0.716** (FastAI 63.6% weight)
- [x] Save predictor → `models/day_ahead/`
- [x] Print leaderboard on test set

---

## PHASE 5 — Evaluation ✅
**Notebook**: `notebooks/05_evaluation.ipynb`
**Requires**: `models/day_ahead/` + `data/processed/features_final.csv`

- [x] Classification report per class (Low / Normal / High / Spike)
- [x] Confusion matrix (absolute + row-normalised) → `models/confusion_matrix_d1.png`
- [x] Feature importance (top-20 permutation) → `models/feature_importance_d1.png`
- [x] Leaderboard (top-10 models with test & val scores)
- [x] Analyse **Spike class** recall separately — 0 Spike days in 2024+ (energy crisis ended)
- [x] Temporal drift analysis — 2024: ba=0.561, 2025: ba=0.401 (concept drift confirmed)
- [x] Production readiness verdict — **NOT READY** (High class recall=13.6%, concept drift)

**Key findings**:
- Test set (2024+): 718 days — No Spike days (market regime change post-crisis)
- High class recall: 13.6% (only 8/59 High days correctly predicted — concept drift)
- Model trained on crisis-era data (2020–2022) does not generalise to post-crisis 2024–2025
- **Next step**: Retrain with sliding window including 2024–2025 data

---

## PHASE 6 — Retrain with Fixed Pipeline ⏳
Implements 8 academic fixes identified in Phase 5 evaluation.

**Config changes** (`src/config.py`):
- [x] `COLLECT_END` extended to 2026-03-20
- [x] `TRAIN_END` → 2024-01-01 (sliding window: train includes 2023 post-crisis year)
- [x] `VAL_END`   → 2025-01-01 (val = 2024, test = 2025-2026)
- [x] `AG_BAG_FOLDS` → 0 (disabled to avoid temporal fold leakage)
- [x] `CLASS_WEIGHTS` added (High=3x, Spike=5x for cost-sensitive training)

**Feature changes** (`src/features.py`):
- [x] Removed `is_energy_crisis` static flag (dead feature post-2023)
- [x] Added `ttf_vs_median_12m`, `co2_vs_median_12m`, `price_vs_median_12m` (dynamic regime ratios)
- [x] Added t-1 lags for all 8 AggrCurves features (16 total market-structure features)

**Training changes** (`src/training.py`):
- [x] Added cost-sensitive `sample_weight` column to training set
- [x] `weight_evaluation=True` passed to `predictor.fit()`

**Remaining steps**:
- [ ] Re-run Phase 1 notebook (collect 2026 Q1 data)
- [ ] Re-run Phase 2 notebook (preprocess)
- [ ] Re-run Phase 3 notebook (feature engineering)
- [ ] Recalibrate PRICE_BINS from updated distribution
- [ ] Re-run Phase 4 notebook (retrain)
- [ ] Re-run Phase 5 notebook (evaluate)

---

## Backlog / Future Work
- [ ] Investigate 2021 full-year AggrCurves coverage (currently Nov 2020 start in DAM results)
- [ ] Add SHAP values for interpretability
- [ ] Walk-forward (rolling 30-day window) evaluation in evaluation.py
- [ ] Experiment with `best_quality` vs `high_quality` preset for faster iteration
