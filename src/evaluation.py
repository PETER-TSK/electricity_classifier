"""
Phase 5 — Evaluation.

Loads the trained D+1 predictor and evaluates it on the 2025 test set.
Produces classification report, confusion matrices, per-class metrics,
Spike deep-dive, monthly temporal breakdown, 30-day walk-forward evaluation,
and a production readiness verdict.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from autogluon.tabular import TabularPredictor

from src.config import (
    AG_LABEL,
    DROP_FOR_TRAINING,
    MODELS_DIR,
    PRICE_LABELS,
    PROC_DIR,
    ROLLING_ORIGINS,
    TRAIN_END,
    UP_CLASSES,
    VAL_END,
    WF_PROD_THRESHOLD,
    WF_THRESHOLD_MAX,
    WF_THRESHOLD_MIN,
    WF_THRESHOLD_STEP,
)


def _load_test_set(target_col: str = "target_1d") -> pd.DataFrame:
    """Load features_final.csv and return the test split."""
    df = pd.read_csv(
        PROC_DIR / "features_final.csv",
        index_col=0,
        parse_dates=True,
    )
    other_targets = [c for c in df.columns if c.startswith("target_") and c != target_col]
    drop = [c for c in DROP_FOR_TRAINING + other_targets if c in df.columns]
    test = df[df.index >= VAL_END].drop(columns=drop)
    return test.rename(columns={target_col: AG_LABEL})


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_confusion_matrix_both(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str] = PRICE_LABELS,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Side-by-side confusion matrices: absolute counts (left) and
    row-normalised recall (right).  Saves to save_path if given.
    """
    cm_abs  = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, cm, fmt, title in zip(
        axes,
        [cm_abs, cm_norm],
        ["d", ".2f"],
        ["Absolute counts", "Row-normalised (recall)"],
    ):
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=(1 if fmt == ".2f" else None),
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — D+1 Greece DAM\n{title}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    return fig


def plot_feature_importance(
    predictor: TabularPredictor,
    test: pd.DataFrame,
    top_n: int = 20,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot top-N permutation feature importances and optionally save."""
    fi = predictor.feature_importance(test)
    fig, ax = plt.subplots(figsize=(10, 7))
    fi.head(top_n)["importance"].plot(kind="barh", ax=ax)
    ax.set_xlabel("Permutation Importance")
    ax.set_title(f"Top-{top_n} Feature Importance — D+1 Model")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    return fig


def plot_per_class_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str] = PRICE_LABELS,
) -> plt.Figure:
    """
    Grouped bar chart: Precision / Recall / F1-score for each class.
    Highlights Spike recall with a dashed threshold line at 0.50.
    """
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width for i in x], p, width, label="Precision", color="#4c72b0")
    ax.bar(list(x),                r, width, label="Recall",    color="#dd8452")
    ax.bar([i + width for i in x], f, width, label="F1",        color="#55a868")

    ax.axhline(0.50, color="red", linestyle="--", linewidth=1.2,
               label="Spike recall target (0.50)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1 — D+1 Model (Test Set)")
    ax.legend()
    plt.tight_layout()
    return fig


# ─── Metric helpers ───────────────────────────────────────────────────────────

def classification_report_df(
    y_true: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    """
    Return classification report as a clean DataFrame.
    Rows: Low, Normal, High, Spike, macro avg, weighted avg.
    Columns: precision, recall, f1-score, support.
    """
    report_dict = classification_report(
        y_true, y_pred,
        labels=PRICE_LABELS,
        target_names=PRICE_LABELS,
        zero_division=0,
        output_dict=True,
    )
    rows = {}
    for cls in PRICE_LABELS:
        rows[cls] = report_dict[cls]
    rows["macro avg"]    = report_dict["macro avg"]
    rows["weighted avg"] = report_dict["weighted avg"]
    df = pd.DataFrame(rows).T
    df["support"] = df["support"].astype(int)
    df[["precision", "recall", "f1-score"]] = df[
        ["precision", "recall", "f1-score"]
    ].round(4)
    return df


def spike_recall_analysis(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Focused Spike-class metrics and missed / false-alarm breakdown.

    Returns dict with: spike_recall, spike_precision, spike_f1,
    n_true_spikes, n_missed_spikes, missed_breakdown,
    false_alarm_count, false_alarm_breakdown.
    """
    labels = PRICE_LABELS
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    idx = labels.index("Spike")

    spike_mask  = y_true == "Spike"
    missed_mask = spike_mask & (y_pred != "Spike")
    fp_mask     = (~spike_mask) & (y_pred == "Spike")

    return {
        "spike_recall":           round(float(r[idx]), 4),
        "spike_precision":        round(float(p[idx]), 4),
        "spike_f1":               round(float(f[idx]), 4),
        "n_true_spikes":          int(spike_mask.sum()),
        "n_missed_spikes":        int(missed_mask.sum()),
        "missed_breakdown":       y_pred[missed_mask].value_counts(),
        "false_alarm_count":      int(fp_mask.sum()),
        "false_alarm_breakdown":  y_true[fp_mask].value_counts(),
    }


def temporal_breakdown(
    predictor: TabularPredictor, test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Per-calendar-month balanced_accuracy and per-class recall on the test set.
    Detects temporal drift month by month. test_df must have AG_LABEL column
    and a DatetimeIndex.
    """
    rows = []
    for period, subset in test_df.groupby(test_df.index.to_period("M")):
        if len(subset) < 10:
            continue
        y_true = subset[AG_LABEL]
        y_pred = predictor.predict(subset.drop(columns=[AG_LABEL]))
        ba = balanced_accuracy_score(y_true, y_pred)
        _, r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=PRICE_LABELS, zero_division=0
        )
        rows.append({
            "month":             str(period),
            "n_days":            len(subset),
            "balanced_accuracy": round(ba, 4),
            **{f"recall_{lbl.lower()}": round(float(r[i]), 4)
               for i, lbl in enumerate(PRICE_LABELS)},
        })
    return pd.DataFrame(rows)


def walk_forward_evaluation(
    predictor: TabularPredictor,
    test_df: pd.DataFrame,
    window_days: int = 30,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Slide a fixed window across the test set, computing balanced_accuracy and
    per-class recall for each window. The EPF gold standard for assessing whether
    model performance is stable or drifting across time.

    Parameters
    ----------
    predictor   : fitted TabularPredictor
    test_df     : test set with AG_LABEL column and DatetimeIndex
    window_days : size of the rolling window in calendar days (default 30)

    Returns
    -------
    wf_df : DataFrame indexed by window start date with columns:
            n_days, balanced_accuracy, recall_low, recall_normal,
            recall_high, recall_spike
    fig   : two-panel matplotlib Figure (BA curve + per-class recall curves)
    """
    dates = test_df.index.sort_values()
    rows = []
    for start in pd.date_range(
        dates[0], dates[-1] - pd.Timedelta(days=window_days - 1), freq="D"
    ):
        end = start + pd.Timedelta(days=window_days - 1)
        subset = test_df[(test_df.index >= start) & (test_df.index <= end)]
        if len(subset) < 15:
            continue
        y_true = subset[AG_LABEL]
        y_pred = predictor.predict(subset.drop(columns=[AG_LABEL]))
        ba = balanced_accuracy_score(y_true, y_pred)
        _, r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=PRICE_LABELS, zero_division=0
        )
        row = {"date": start, "n_days": len(subset), "balanced_accuracy": round(ba, 4)}
        for i, lbl in enumerate(PRICE_LABELS):
            row[f"recall_{lbl.lower()}"] = round(float(r[i]), 4)
        rows.append(row)

    wf_df = pd.DataFrame(rows).set_index("date")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(wf_df.index, wf_df["balanced_accuracy"], color="#2166ac", linewidth=2)
    axes[0].axhline(0.65, color="green",  linestyle="--", linewidth=1.2,
                    label="READY threshold (0.65)")
    axes[0].axhline(0.60, color="orange", linestyle="--", linewidth=1.2,
                    label="BORDERLINE threshold (0.60)")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_title(
        f"Walk-Forward Evaluation — {window_days}-day rolling window (D+1 model, 2025 test set)"
    )
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].plot(wf_df.index, wf_df["recall_high"],   label="High recall",
                 color="#d6604d", linewidth=1.5)
    axes[1].plot(wf_df.index, wf_df["recall_normal"], label="Normal recall",
                 color="#4393c3", linewidth=1.5)
    axes[1].plot(wf_df.index, wf_df["recall_low"],    label="Low recall",
                 color="#878787", linewidth=1.5)
    axes[1].set_ylabel("Per-class Recall")
    axes[1].set_xlabel("Window start date")
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    return wf_df, fig


def production_readiness_check(
    y_true: pd.Series, y_pred: pd.Series, lb: pd.DataFrame
) -> dict:
    """
    Apply production readiness thresholds and return a verdict dict.

    Thresholds (test set, active classes only):
      READY      : balanced_accuracy >= 0.65
                   AND spike_recall >= 0.50 (only if Spike has test samples)
      BORDERLINE : balanced_accuracy >= 0.60
                   AND spike_recall >= 0.40 (only if Spike has test samples)
      NOT_READY  : below BORDERLINE on either active metric

    If Spike class has zero test samples the spike_recall condition is
    skipped and the verdict notes this as a regime change, not a model failure.
    """
    ba  = balanced_accuracy_score(y_true, y_pred)
    n_spikes = int((y_true == "Spike").sum())
    spi = (
        recall_score(
            y_true, y_pred, labels=PRICE_LABELS,
            average=None, zero_division=0,
        )[PRICE_LABELS.index("Spike")]
        if n_spikes > 0 else None
    )

    ens = lb[lb["model"].str.startswith("WeightedEnsemble")]
    val_score = (
        float(ens["score_val"].max()) if not ens.empty
        else float(lb["score_val"].max())
    )

    reasons = []
    spike_na_note = (
        "Spike class: 0 samples in test period (market regime change, not model failure)"
    )
    spike_ok     = (spi is None) or (spi >= 0.50)
    ba_ready     = ba >= 0.65
    ba_border    = ba >= 0.60
    spike_border = (spi is None) or (spi >= 0.40)

    if ba_ready and spike_ok:
        verdict = "READY"
        reasons.append(f"balanced_accuracy {ba:.3f} >= 0.65")
        if spi is not None:
            reasons.append(f"spike_recall {spi:.3f} >= 0.50")
        else:
            reasons.append(spike_na_note)
    elif ba_border and spike_border:
        verdict = "BORDERLINE"
        if ba < 0.65:
            reasons.append(f"balanced_accuracy {ba:.3f} is 0.60–0.65")
        if spi is not None and spi < 0.50:
            reasons.append(f"spike_recall {spi:.3f} is 0.40–0.50")
        elif spi is None:
            reasons.append(spike_na_note)
        reasons.append("Consider retraining with recent data to recalibrate High class")
    else:
        verdict = "NOT READY"
        if ba < 0.60:
            reasons.append(f"balanced_accuracy {ba:.3f} < 0.60 — concept drift detected")
        if spi is not None and spi < 0.40:
            reasons.append(f"spike_recall {spi:.3f} < 0.40 minimum")
        elif spi is None:
            reasons.append(spike_na_note)
        reasons.append("Recommendation: retrain with more recent data")

    return {
        "balanced_accuracy":  round(ba, 4),
        "spike_recall":       round(spi, 4) if spi is not None else None,
        "spike_samples":      n_spikes,
        "ensemble_val_score": round(val_score, 4),
        "verdict":            verdict,
        "reasons":            reasons,
    }


# ─── Rolling Origin CV helpers ────────────────────────────────────────────────

def _binary_up(series: pd.Series) -> pd.Series:
    """Convert 4-class labels to binary: 1=UP (High/Spike), 0=not-UP."""
    return series.isin(UP_CLASSES).astype(int)


def _compute_up_metrics(
    y_true_bin: pd.Series, proba_up: pd.Series, threshold: float
) -> dict:
    """Precision(UP), Recall(UP), MCC at a given predict_proba threshold."""
    y_pred_bin = (proba_up >= threshold).astype(int)
    return {
        "precision_up":   round(float(precision_score(y_true_bin, y_pred_bin, zero_division=0)), 4),
        "recall_up":      round(float(recall_score(y_true_bin, y_pred_bin, zero_division=0)), 4),
        "mcc":            round(float(matthews_corrcoef(y_true_bin, y_pred_bin)), 4),
        "n_predicted_up": int(y_pred_bin.sum()),
        "n_true_up":      int(y_true_bin.sum()),
    }


def _sweep_thresholds(
    y_true_bin: pd.Series, proba_up: pd.Series
) -> pd.DataFrame:
    """
    Sweep thresholds WF_THRESHOLD_MIN → WF_THRESHOLD_MAX.
    Returns DataFrame indexed by threshold with Precision, Recall, MCC columns.
    """
    rows = []
    for thresh in np.arange(WF_THRESHOLD_MIN, WF_THRESHOLD_MAX + 0.001, WF_THRESHOLD_STEP):
        t   = round(float(thresh), 2)
        row = _compute_up_metrics(y_true_bin, proba_up, t)
        row["threshold"] = t
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def _load_test_set_origin(val_end: str, test_end: str | None) -> pd.DataFrame:
    """Load features_final.csv and slice the test window for a rolling origin."""
    df = pd.read_csv(
        PROC_DIR / "features_final.csv",
        index_col=0,
        parse_dates=True,
    )
    other_targets = [c for c in df.columns if c.startswith("target_") and c != "target_1d"]
    drop = [c for c in DROP_FOR_TRAINING + other_targets if c in df.columns]
    test = df[df.index >= val_end] if test_end is None \
           else df[(df.index >= val_end) & (df.index < test_end)]
    return test.drop(columns=drop).rename(columns={"target_1d": AG_LABEL})


def plot_threshold_calibration(
    sweep_df: pd.DataFrame, origin_name: str = ""
) -> plt.Figure:
    """Precision–Recall and MCC curves across thresholds, with production threshold marked."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    suffix = f" — Origin {origin_name}" if origin_name else ""

    axes[0].plot(sweep_df.index, sweep_df["precision_up"],
                 color="#d6604d", linewidth=2, label="Precision(UP)")
    axes[0].plot(sweep_df.index, sweep_df["recall_up"],
                 color="#4393c3", linewidth=2, label="Recall(UP)")
    axes[0].axvline(WF_PROD_THRESHOLD, color="black", linestyle="--",
                    linewidth=1.2, label=f"Prod threshold ({WF_PROD_THRESHOLD})")
    axes[0].axhline(0.60, color="green", linestyle=":", linewidth=1.0,
                    label="Precision target (0.60)")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title(f"Precision & Recall vs Threshold{suffix}")
    axes[0].legend(); axes[0].set_ylim(0, 1)

    axes[1].plot(sweep_df.index, sweep_df["mcc"], color="#55a868", linewidth=2)
    axes[1].axvline(WF_PROD_THRESHOLD, color="black", linestyle="--", linewidth=1.2)
    axes[1].axhline(0.20, color="green", linestyle=":", linewidth=1.0,
                    label="MCC target (0.20)")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("MCC")
    axes[1].set_title(f"MCC vs Threshold{suffix}")
    axes[1].legend()
    plt.tight_layout()
    return fig


def plot_rolling_origin_results(fold_df: pd.DataFrame) -> plt.Figure:
    """Monthly Precision(UP) and MCC across all rolling origins."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    colors = {"2023": "#4393c3", "2024": "#f4a582", "2025": "#d6604d"}

    for origin, grp in fold_df.groupby("origin"):
        axes[0].plot(grp["month"], grp["precision_up"], marker="o",
                     label=f"Origin {origin}", color=colors.get(origin, "gray"))
        axes[1].plot(grp["month"], grp["mcc"], marker="o",
                     label=f"Origin {origin}", color=colors.get(origin, "gray"))

    axes[0].axhline(0.60, color="green", linestyle="--", linewidth=1.2,
                    label="Target Precision(UP) = 0.60")
    axes[0].set_ylabel("Precision(UP)"); axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Monthly Precision(UP) — threshold={WF_PROD_THRESHOLD}")
    axes[0].legend(); axes[0].tick_params(axis="x", rotation=45)

    axes[1].axhline(0.20, color="green", linestyle="--", linewidth=1.2,
                    label="Target MCC = 0.20")
    axes[1].set_ylabel("MCC"); axes[1].set_ylim(-0.5, 1)
    axes[1].set_title("Monthly MCC across Rolling Origins")
    axes[1].legend(); axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


# ─── Rolling Origin public API ────────────────────────────────────────────────

def evaluate_rolling_origins(origins: list[dict]) -> dict:
    """
    Full rolling origin evaluation: Precision(UP), MCC, threshold sweep.

    Parameters
    ----------
    origins : list of dicts — output of train_rolling_origins()
              each dict: {"name": str, "predictor": TabularPredictor, "test_df": pd.DataFrame}

    Returns
    -------
    dict with keys:
      fold_results     : pd.DataFrame — monthly rows per origin
      origin_summary   : pd.DataFrame — per-origin mean/std
      threshold_sweeps : dict[str, pd.DataFrame]
      threshold_figs   : dict[str, plt.Figure]
      results_fig      : plt.Figure
      overall          : dict — aggregated metrics + verdict
    """
    fold_rows  = []
    sweep_dfs  = {}
    sweep_figs = {}

    for origin in origins:
        name      = origin["name"]
        predictor = origin["predictor"]
        test_df   = origin["test_df"]

        proba      = predictor.predict_proba(test_df.drop(columns=[AG_LABEL]))
        up_cols    = [c for c in UP_CLASSES if c in proba.columns]
        proba_up   = proba[up_cols].sum(axis=1)
        y_true_bin = _binary_up(test_df[AG_LABEL])

        sweep_dfs[name]  = _sweep_thresholds(y_true_bin, proba_up)
        sweep_figs[name] = plot_threshold_calibration(sweep_dfs[name], name)

        for period, subset in test_df.groupby(test_df.index.to_period("M")):
            if len(subset) < 10:
                continue
            idx     = subset.index
            metrics = _compute_up_metrics(
                y_true_bin.loc[idx], proba_up.loc[idx], WF_PROD_THRESHOLD
            )
            metrics["origin"] = name
            metrics["month"]  = str(period)
            metrics["n_days"] = len(subset)
            fold_rows.append(metrics)

    fold_df = pd.DataFrame(fold_rows)
    origin_summary = (
        fold_df.groupby("origin")[["precision_up", "recall_up", "mcc"]]
        .agg(["mean", "std"])
        .round(4)
    )

    # Verdict based on per-origin FULL-PERIOD metrics (not per-month average,
    # which is biased by months with zero UP days returning precision=0)
    origin_full_rows = []
    for origin in origins:
        name       = origin["name"]
        predictor  = origin["predictor"]
        test_df    = origin["test_df"]
        proba      = predictor.predict_proba(test_df.drop(columns=[AG_LABEL]))
        up_cols    = [c for c in UP_CLASSES if c in proba.columns]
        proba_up   = proba[up_cols].sum(axis=1)
        y_true_bin = _binary_up(test_df[AG_LABEL])
        m = _compute_up_metrics(y_true_bin, proba_up, WF_PROD_THRESHOLD)
        m["origin"] = name
        origin_full_rows.append(m)
    origin_full_df = pd.DataFrame(origin_full_rows).set_index("origin")

    agg_prec = float(origin_full_df["precision_up"].mean())
    agg_mcc  = float(origin_full_df["mcc"].mean())
    std_prec = float(origin_full_df["precision_up"].std())

    if agg_prec >= 0.60 and agg_mcc >= 0.20:
        verdict = "READY"
    elif agg_prec >= 0.50 and agg_mcc >= 0.10:
        verdict = "BORDERLINE"
    else:
        verdict = "NOT READY"
    if std_prec > 0.15:
        verdict += " (UNSTABLE — high variance across origins)"

    overall = {
        "mean_precision_up": round(agg_prec, 4),
        "mean_mcc":          round(agg_mcc, 4),
        "std_precision_up":  round(std_prec, 4),
        "prod_threshold":    WF_PROD_THRESHOLD,
        "verdict":           verdict,
    }

    print("\n=== Rolling Origin Evaluation ===")
    print(f"Threshold : {WF_PROD_THRESHOLD}")
    print(f"Precision(UP): {agg_prec:.3f}  |  MCC: {agg_mcc:.3f}  |  Std: {std_prec:.3f}")
    print(f"Verdict   : {verdict}\n")
    print(origin_summary.to_string())

    return {
        "fold_results":     fold_df,
        "origin_summary":   origin_summary,
        "threshold_sweeps": sweep_dfs,
        "threshold_figs":   sweep_figs,
        "results_fig":      plot_rolling_origin_results(fold_df),
        "overall":          overall,
    }


# ─── Main orchestrator ────────────────────────────────────────────────────────

def evaluate_day_ahead() -> dict:
    """
    Full evaluation of the D+1 model on the 2025 test set.

    Returns a dict with keys:
      report, report_df, confusion_matrix, confusion_matrix_fig,
      per_class_metrics_fig, feature_importance_fig,
      leaderboard, y_true, y_pred,
      spike_analysis, temporal, walk_forward, walk_forward_fig,
      readiness

    Also saves confusion matrix and feature importance PNGs to models/.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    predictor = TabularPredictor.load(str(MODELS_DIR / "day_ahead"))

    test   = _load_test_set("target_1d")
    X_test = test.drop(columns=[AG_LABEL])
    y_true = test[AG_LABEL]
    y_pred = predictor.predict(X_test)

    print(f"Test set: {len(test)} days\n")
    report = classification_report(
        y_true, y_pred,
        labels=PRICE_LABELS,
        target_names=PRICE_LABELS,
        zero_division=0,
    )
    print("=== D+1 Classification Report ===")
    print(report)

    cm_path = str(MODELS_DIR / "confusion_matrix_d1.png")
    fi_path = str(MODELS_DIR / "feature_importance_d1.png")

    cm_fig  = plot_confusion_matrix_both(y_true, y_pred, save_path=cm_path)
    fi_fig  = plot_feature_importance(predictor, test, save_path=fi_path)
    pcm_fig = plot_per_class_metrics(y_true, y_pred)

    lb = predictor.leaderboard(test, silent=True)
    print("\n--- Leaderboard (top 10, test set) ---")
    print(lb[["model", "score_test", "score_val", "pred_time_test"]].head(10).to_string())

    report_df   = classification_report_df(y_true, y_pred)
    spike_info  = spike_recall_analysis(y_true, y_pred)
    temporal_df = temporal_breakdown(predictor, test)
    wf_df, wf_fig = walk_forward_evaluation(predictor, test)
    readiness   = production_readiness_check(y_true, y_pred, lb)

    return {
        "report":                 report,
        "report_df":              report_df,
        "confusion_matrix":       confusion_matrix(y_true, y_pred, labels=PRICE_LABELS),
        "confusion_matrix_fig":   cm_fig,
        "per_class_metrics_fig":  pcm_fig,
        "feature_importance_fig": fi_fig,
        "leaderboard":            lb,
        "y_true":                 y_true,
        "y_pred":                 y_pred,
        "spike_analysis":         spike_info,
        "temporal":               temporal_df,
        "walk_forward":           wf_df,
        "walk_forward_fig":       wf_fig,
        "readiness":              readiness,
    }
