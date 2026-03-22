# Greece Day-Ahead Electricity Price Forecasting

**4-class price regime classifier for the Greek Day-Ahead Market (EL-DAM)**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![AutoGluon](https://img.shields.io/badge/AutoGluon-1.5.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Problem

Electricity price forecasting (EPF) for the Greek DAM classifies each trading day into one of four price regimes:

| Class  | EUR/MWh | Interpretation |
|--------|---------|----------------|
| Low    | 0–80    | Surplus renewables / weak demand |
| Normal | 80–150  | Typical baseload dispatch |
| **High**   | **150–250** | **Gas-heavy / stress conditions** |
| **Spike**  | **> 250**   | **Supply scarcity / crisis** |

The primary objective is **Precision on UP** (High + Spike): when the model predicts above-normal prices, it must be right. False alarms carry real trading costs.

---

## Results

Evaluation uses **Expanding-Window Rolling-Origin CV** (the EPF gold standard, Lago 2021) — 3 independent train/test origins, each testing on a full calendar year unseen during training. Threshold set at P(High) + P(Spike) ≥ 0.65.

| Origin | Train window | Test year | Precision(UP) | Recall(UP) | MCC | n(UP predictions) |
|--------|-------------|-----------|:---:|:---:|:---:|:---:|
| 2023 | 2020–2022 | 2023 (recovery) | 0.398 | 0.911 | 0.500 | 128 / 365 days |
| 2024 | 2020–2023 | 2024 (post-crisis) | 1.000 | 0.061 | 0.235 | 2 / 354 days |
| 2025 | 2020–2024 | 2025 (new normal) | 0.450 | 0.346 | 0.354 | 20 / 364 days |
| **Mean** | | | **0.616** | | **0.363** | |

**Verdict: READY (UNSTABLE)** — mean Precision(UP) = 0.62 and MCC = 0.36 meet production thresholds, but high cross-origin variance flags regime sensitivity. A production deployment would require monthly retraining as new market data becomes available.

---

## Methodology

### Model
- **Algorithm**: AutoGluon `TabularPredictor` with `best_quality` preset
- **Ensemble**: 5-fold bagging + L2 stacking (WeightedEnsemble_L2/L3)
- **Metric**: `balanced_accuracy` (training); Precision(UP) + MCC (evaluation)
- **Class weights**: Low=2.0, Normal=1.0, High=2.0, Spike=2.0 (cost-sensitive)
- **Threshold**: P(High) + P(Spike) ≥ 0.65 → predict UP

### Evaluation
Rolling-origin cross-validation with expanding training windows (never shuffles time-series data):

```
Origin 2023:  Train [2020-12 → 2022-06] | Val [2022-07 → 2022-12] | Test 2023
Origin 2024:  Train [2020-12 → 2023-06] | Val [2023-07 → 2023-12] | Test 2024
Origin 2025:  Train [2020-12 → 2024-06] | Val [2024-07 → 2024-12] | Test 2025
```

Per-origin threshold calibration curves (Precision–Recall–MCC vs threshold 0.40→0.75) and monthly breakdowns are available in the evaluation notebook.

---

## Features (49 total)

| Group | Features | Description |
|-------|---------|-------------|
| A. Price history | lag_1d, lag_7d, lag_14d, mom_7d, price regime lags | Autoregressive price signal |
| B. Commodity | ttf, coal, co2_eua + 7-day rolls, ttf×load, co2×load | Fuel cost drivers |
| C. Grid balance | residual_load, res_share, hydro_deficit, import_share | Supply/demand balance |
| D. Weather | HDD, CDD, HDD_7d, CDD_7d, **hdd_x_residual_load** | Demand + cold-snap×grid interaction |
| E. Calendar | month_sin/cos, dow_sin/cos, is_weekend, is_holiday | Seasonality encoding |
| F. Regime | is_energy_crisis (Sep 2021 – Jun 2023) | Structural break flag |
| G. ENEX curves | price_spread, buy/sell volumes, curve_steepness, order book percentiles | Market microstructure |

Key feature: `hdd_x_residual_load = HDD × residual_load / 1e6` captures cold-snap + grid-stress interaction — the primary driver of post-crisis High events (correlation 0.44 with is_high in 2025 test set).

---

## Data Sources

| Source | Data | Period |
|--------|------|--------|
| [ENEX](https://www.enex.gr/) | EL-DAM daily settlement price + order book (AggrCurves) | 2020–2025 |
| [ADMIE / data.gov.gr](https://data.gov.gr/) | RES generation, system load, energy balance | 2020–2025 |
| [Yahoo Finance](https://finance.yahoo.com/) | TTF gas, coal, CO2 EUA (KRBN proxy) | 2020–2025 |
| [Open-Meteo](https://open-meteo.com/) | Athens daily Tmax/Tmin | 2020–2025 |

---

## Project Structure

```
electricity_classifier/
├── src/
│   ├── config.py            # Central config — paths, AutoGluon params, class weights, origins
│   ├── collection.py        # Phase 1: API data collection
│   ├── preprocessing.py     # Phase 2: merge, clean, forward-fill
│   ├── features.py          # Phase 3: 49-feature engineering pipeline
│   ├── training.py          # Phase 4: train_rolling_origins() + train_day_ahead()
│   └── evaluation.py        # Phase 5: evaluate_rolling_origins() + threshold sweep
├── notebooks/
│   └── 05_evaluation.ipynb  # Rolling-origin evaluation with charts (pre-executed)
├── extract_enex.py          # Parse ENEX Results zips → gr_dam_price.csv
├── extract_aggrcurves.py    # Parse ENEX AggrCurves zips → aggrcurves_dam.csv
├── requirements.txt
└── .gitignore
```

> `data/`, `models/`, and `enex_data/` are excluded from the repo (large binary/model files). Run the pipeline to regenerate them.

---

## Setup

```bash
git clone https://github.com/<your-username>/electricity_classifier
cd electricity_classifier
pip install -r requirements.txt
```

---

## Running the Pipeline

```python
# Phase 1 — Collect data
from src.collection import collect_all
collect_all()

# Phase 2 — Preprocess
from src.preprocessing import preprocess
preprocess()

# Phase 3 — Feature engineering
from src.features import build_features
build_features()

# Phase 4 — Train rolling origin models (~2 hours)
from src.training import train_rolling_origins
origins = train_rolling_origins()

# Phase 5 — Evaluate
from src.evaluation import evaluate_rolling_origins
results = evaluate_rolling_origins(origins)
print(results['overall'])
```

Open `notebooks/05_evaluation.ipynb` to see the full evaluation with charts (pre-executed, no training required).

---

## References

- Lago, J. et al. (2021). *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark*. Applied Energy.
- Weron, R. (2014). *Electricity price forecasting: A review of the state-of-the-art with a look into the future*. International Journal of Forecasting.
- Halužan, M., Verbič, M. & Zorić, J. (2020). *Performance of alternative electricity price forecasting methods*. Applied Energy.
- Erickson, N. et al. (2020). *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data*. ICML AutoML Workshop.
