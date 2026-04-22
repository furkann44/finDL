from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from .config import ALL_SYMBOLS, BACKTESTS_DIR, FIGURES_DIR, METRICS_DIR, MODEL_NAMES, PREDICTIONS_DIR
except ImportError:
    from config import ALL_SYMBOLS, BACKTESTS_DIR, FIGURES_DIR, METRICS_DIR, MODEL_NAMES, PREDICTIONS_DIR


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Beklenen dashboard veri dosyasi bulunamadi: {path}")
    return pd.read_csv(path)


def _read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def _sanitize_symbol(symbol: str) -> str:
    return symbol.lower().replace("/", "_").replace(" ", "")


def load_holdout_summary(include_weighted: bool = True) -> pd.DataFrame:
    frame = _read_csv_or_empty(
        METRICS_DIR / "model_summary.csv",
        [
            "symbol",
            "asset_group",
            "model",
            "split",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "loss",
            "positive_rate",
            "predicted_positive_rate",
        ],
    )
    frame = frame[frame["symbol"].isin(ALL_SYMBOLS)].copy()
    if not include_weighted:
        frame = frame[~frame["model"].str.contains("balanced", na=False)].copy()
    return attach_test_window(frame)


def load_test_window_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prediction_path in sorted(PREDICTIONS_DIR.glob("*_test_predictions.csv")):
        frame = pd.read_csv(prediction_path)
        if frame.empty:
            continue
        symbol = str(frame["symbol"].iloc[0])
        if symbol not in ALL_SYMBOLS:
            continue
        frame = frame.sort_values("datetime").reset_index(drop=True)
        rows.append(
            {
                "symbol": symbol,
                "model": str(frame["model"].iloc[0]),
                "test_start": str(frame["datetime"].iloc[0]),
                "test_end": str(frame["datetime"].iloc[-1]),
                "test_rows": int(len(frame)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["symbol", "model", "test_start", "test_end", "test_rows"])
    return pd.DataFrame(rows)


def attach_test_window(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "model" not in frame.columns:
        return frame

    windows = load_test_window_summary()
    if windows.empty:
        return frame

    merged = frame.merge(windows, on=["symbol", "model"], how="left")
    if "test_coverage" in merged.columns:
        merged["active_trade_days"] = (merged["test_coverage"] * merged["test_rows"]).round().astype("Int64")
        merged["no_trade_days"] = (merged["test_rows"] - merged["active_trade_days"]).astype("Int64")
    if "coverage" in merged.columns and "test_coverage" not in merged.columns:
        merged["active_trade_days"] = (merged["coverage"] * merged["test_rows"]).round().astype("Int64")
    return merged


def ensure_window_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"test_start", "test_end", "test_rows"}
    if required_columns.issubset(frame.columns):
        return frame
    return attach_test_window(frame)


def load_backtest_summary(include_weighted: bool = True) -> pd.DataFrame:
    frame = _read_csv_or_empty(
        BACKTESTS_DIR / "backtest_summary.csv",
        [
            "symbol",
            "model",
            "coverage",
            "total_return",
            "benchmark_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "win_rate",
        ],
    )
    frame = frame[frame["symbol"].isin(ALL_SYMBOLS)].copy()
    if not include_weighted:
        frame = frame[~frame["model"].str.contains("balanced", na=False)].copy()
    return attach_test_window(frame)


def load_rolling_retrain_summary() -> pd.DataFrame:
    path = BACKTESTS_DIR / "rolling_retrain_summary.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "symbol",
                "model",
                "base_model",
                "optimize_for",
                "cycles",
                "signal_rows",
                "test_start",
                "test_end",
                "total_return",
                "benchmark_return",
                "sharpe",
                "max_drawdown",
                "coverage",
                "active_accuracy",
                "active_f1",
                "signals_path",
                "equity_path",
                "chart_path",
            ]
        )
    frame = _read_csv(path)
    return frame[frame["symbol"].isin(ALL_SYMBOLS)].copy()


def load_rolling_retrain_equity(symbol: str, model_name: str) -> pd.DataFrame:
    path = BACKTESTS_DIR / f"{_sanitize_symbol(symbol)}_{model_name}_equity.csv"
    return _read_csv(path)


def load_no_trade_summary(optimize_for: str = "total_return") -> pd.DataFrame:
    file_name = "no_trade_summary.csv" if optimize_for == "active_f1" else f"no_trade_summary_{optimize_for}.csv"
    frame = _read_csv_or_empty(
        METRICS_DIR / file_name,
        [
            "symbol",
            "model",
            "optimize_for",
            "lower_threshold",
            "upper_threshold",
            "validation_coverage",
            "validation_active_f1",
            "validation_active_accuracy",
            "validation_total_return",
            "test_coverage",
            "test_active_f1",
            "test_active_accuracy",
            "test_total_return",
            "test_sharpe",
            "test_max_drawdown",
            "test_win_rate",
            "test_predicted_long_rate",
            "test_predicted_short_rate",
        ],
    )
    frame = frame[frame["symbol"].isin(ALL_SYMBOLS)].copy()
    return attach_test_window(frame)


def load_threshold_tuning_summary() -> pd.DataFrame:
    frame = _read_csv_or_empty(
        METRICS_DIR / "threshold_tuning_summary.csv",
        [
            "symbol",
            "model",
            "best_threshold",
            "validation_f1_default",
            "validation_f1_tuned",
            "test_f1_default",
            "test_f1_tuned",
            "test_f1_gain",
            "test_accuracy_default",
            "test_accuracy_tuned",
            "test_accuracy_gain",
            "test_roc_auc",
            "predicted_positive_rate_default",
            "predicted_positive_rate_tuned",
        ],
    )
    return frame[frame["symbol"].isin(ALL_SYMBOLS)].copy()


def load_walk_forward_baseline_summary() -> pd.DataFrame:
    frame = _read_csv_or_empty(
        METRICS_DIR / "walk_forward_summary.csv",
        [
            "asset_group",
            "symbol",
            "model",
            "validation_scheme",
            "effective_folds",
            "total_test_rows",
            "accuracy",
            "f1",
            "roc_auc",
            "accuracy_mean",
            "accuracy_std",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
        ],
    )
    return frame[frame["symbol"].isin(ALL_SYMBOLS)].copy()


def load_walk_forward_sequence_summary() -> pd.DataFrame:
    frame = _read_csv_or_empty(
        METRICS_DIR / "sequence_walk_forward_summary.csv",
        [
            "asset_group",
            "symbol",
            "model",
            "effective_folds",
            "best_threshold_mean",
            "best_threshold_std",
            "accuracy",
            "f1",
            "roc_auc",
            "accuracy_mean",
            "accuracy_std",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
        ],
    )
    return frame[frame["symbol"].isin(ALL_SYMBOLS)].copy()


def latest_prediction_snapshot(models: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prediction_path in sorted(PREDICTIONS_DIR.glob("*_test_predictions.csv")):
        frame = pd.read_csv(prediction_path)
        if frame.empty:
            continue
        latest = frame.iloc[-1]
        symbol = str(latest["symbol"])
        if symbol not in ALL_SYMBOLS:
            continue
        model_name = str(latest["model"])
        if models and model_name not in models:
            continue
        rows.append(
            {
                "symbol": symbol,
                "model": model_name,
                "datetime": str(latest["datetime"]),
                "probability": float(latest["probability"]),
                "prediction": int(latest["prediction"]),
                "decision_threshold": float(latest.get("decision_threshold", 0.5)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["symbol", "model", "datetime", "probability", "prediction", "decision_threshold"])

    snapshot = pd.DataFrame(rows)
    snapshot["direction"] = snapshot["prediction"].map({1: "Yukari", 0: "Asagi"})
    return snapshot.sort_values(["symbol", "model"]).reset_index(drop=True)


def build_recommendation_table() -> pd.DataFrame:
    holdout_test = load_holdout_summary(include_weighted=False)
    holdout_test = holdout_test[holdout_test["split"] == "test"].copy()
    if holdout_test.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "asset_group",
                "best_holdout_model",
                "holdout_accuracy",
                "holdout_f1",
                "holdout_roc_auc",
                "recommended_model",
                "recommended_lower_threshold",
                "recommended_upper_threshold",
                "recommended_total_return",
                "recommended_sharpe",
                "recommended_coverage",
                "recommended_test_start",
                "recommended_test_end",
                "recommended_test_rows",
                "recommended_active_trade_days",
                "recommended_active_f1",
                "recommended_active_accuracy",
                "best_backtest_model",
                "best_backtest_total_return",
                "best_backtest_sharpe",
                "best_backtest_test_rows",
                "best_rolling_model",
                "best_rolling_base_model",
                "best_rolling_total_return",
                "best_rolling_sharpe",
                "best_rolling_coverage",
                "best_rolling_test_start",
                "best_rolling_test_end",
                "best_rolling_signal_rows",
                "recommendation_alignment",
            ]
        )
    holdout_best = (
        holdout_test.sort_values(["symbol", "roc_auc"], ascending=[True, False])
        .drop_duplicates("symbol")
        .rename(
            columns={
                "model": "best_holdout_model",
                "accuracy": "holdout_accuracy",
                "f1": "holdout_f1",
                "roc_auc": "holdout_roc_auc",
            }
        )
    )[["symbol", "asset_group", "best_holdout_model", "holdout_accuracy", "holdout_f1", "holdout_roc_auc"]]

    no_trade_total_return = load_no_trade_summary(optimize_for="total_return")
    no_trade_best = (
        no_trade_total_return.sort_values(["symbol", "test_total_return"], ascending=[True, False])
        .drop_duplicates("symbol")
        .rename(
            columns={
                "model": "recommended_model",
                "lower_threshold": "recommended_lower_threshold",
                "upper_threshold": "recommended_upper_threshold",
                "test_total_return": "recommended_total_return",
                "test_sharpe": "recommended_sharpe",
                "test_coverage": "recommended_coverage",
                "test_start": "recommended_test_start",
                "test_end": "recommended_test_end",
                "test_rows": "recommended_test_rows",
                "active_trade_days": "recommended_active_trade_days",
                "test_active_f1": "recommended_active_f1",
                "test_active_accuracy": "recommended_active_accuracy",
            }
        )
    )[
        [
            "symbol",
            "recommended_model",
            "recommended_lower_threshold",
            "recommended_upper_threshold",
            "recommended_total_return",
            "recommended_sharpe",
            "recommended_coverage",
            "recommended_test_start",
            "recommended_test_end",
            "recommended_test_rows",
            "recommended_active_trade_days",
            "recommended_active_f1",
            "recommended_active_accuracy",
        ]
    ]

    backtest_summary = load_backtest_summary(include_weighted=False)
    backtest_best = (
        backtest_summary.sort_values(["symbol", "total_return"], ascending=[True, False])
        .drop_duplicates("symbol")
        .rename(
            columns={
                "model": "best_backtest_model",
                "total_return": "best_backtest_total_return",
                "sharpe": "best_backtest_sharpe",
                "test_rows": "best_backtest_test_rows",
            }
        )
    )[["symbol", "best_backtest_model", "best_backtest_total_return", "best_backtest_sharpe", "best_backtest_test_rows"]]

    rolling_summary = load_rolling_retrain_summary()
    rolling_best = (
        rolling_summary.sort_values(["symbol", "total_return"], ascending=[True, False])
        .drop_duplicates("symbol")
        .rename(
            columns={
                "model": "best_rolling_model",
                "base_model": "best_rolling_base_model",
                "total_return": "best_rolling_total_return",
                "sharpe": "best_rolling_sharpe",
                "coverage": "best_rolling_coverage",
                "test_start": "best_rolling_test_start",
                "test_end": "best_rolling_test_end",
                "signal_rows": "best_rolling_signal_rows",
            }
        )
    )[
        [
            "symbol",
            "best_rolling_model",
            "best_rolling_base_model",
            "best_rolling_total_return",
            "best_rolling_sharpe",
            "best_rolling_coverage",
            "best_rolling_test_start",
            "best_rolling_test_end",
            "best_rolling_signal_rows",
        ]
    ]

    recommendations = (
        holdout_best.merge(no_trade_best, on="symbol", how="left")
        .merge(backtest_best, on="symbol", how="left")
        .merge(rolling_best, on="symbol", how="left")
    )

    recommendations["recommendation_alignment"] = recommendations.apply(
        lambda row: (
            "aligned"
            if str(row.get("recommended_model", "")) == str(row.get("best_rolling_base_model", ""))
            else ("mixed" if pd.notna(row.get("best_rolling_base_model")) else "no_rolling")
        ),
        axis=1,
    )
    return recommendations.sort_values(["asset_group", "symbol"]).reset_index(drop=True)


def load_equity_curve(symbol: str, model_name: str) -> pd.DataFrame:
    path = BACKTESTS_DIR / f"{_sanitize_symbol(symbol)}_{model_name}_equity.csv"
    return _read_csv(path)


def diagnostic_image_paths(symbol: str, model_name: str) -> dict[str, Path | None]:
    symbol_tag = _sanitize_symbol(symbol)
    roc_path = FIGURES_DIR / f"{symbol_tag}_{model_name}_test_roc.png"
    confusion_path = FIGURES_DIR / f"{symbol_tag}_{model_name}_test_confusion.png"
    return {
        "roc": roc_path if roc_path.exists() else None,
        "confusion": confusion_path if confusion_path.exists() else None,
    }


def load_holdout_diagnostics_summary() -> pd.DataFrame:
    summary_path = METRICS_DIR.parent / "reports" / "holdout_diagnostics_summary.json"
    if not summary_path.exists():
        return pd.DataFrame(columns=["symbol", "model", "accuracy", "f1", "roc_auc", "roc_curve", "confusion_matrix"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload)


def no_trade_direction(probability: float, lower_threshold: float, upper_threshold: float) -> str:
    if probability >= upper_threshold:
        return "Yukari"
    if probability <= lower_threshold:
        return "Asagi"
    return "No Trade"


def build_asset_detail(symbol: str) -> dict[str, object]:
    recommendations = build_recommendation_table()
    recommendation_row = recommendations[recommendations["symbol"] == symbol]
    holdout_summary = load_holdout_summary(include_weighted=False)
    holdout_summary = holdout_summary[(holdout_summary["symbol"] == symbol) & (holdout_summary["split"] == "test")].copy()
    backtest_summary = load_backtest_summary(include_weighted=False)
    backtest_summary = backtest_summary[backtest_summary["symbol"] == symbol].copy()
    rolling_summary = load_rolling_retrain_summary()
    rolling_summary = rolling_summary[rolling_summary["symbol"] == symbol].copy()
    threshold_summary = load_threshold_tuning_summary()
    threshold_summary = threshold_summary[threshold_summary["symbol"] == symbol].copy()
    no_trade_total = load_no_trade_summary("total_return")
    no_trade_total = no_trade_total[no_trade_total["symbol"] == symbol].copy()
    snapshot = latest_prediction_snapshot(models=None)
    snapshot = snapshot[snapshot["symbol"] == symbol].copy()

    recommendation = recommendation_row.iloc[0].to_dict() if not recommendation_row.empty else None
    recommended_signal = None
    if recommendation is not None and not snapshot.empty:
        recommended_model = recommendation.get("recommended_model")
        recommended_snapshot = snapshot[snapshot["model"] == recommended_model]
        if not recommended_snapshot.empty:
            latest_row = recommended_snapshot.iloc[-1]
            recommended_signal = {
                "model": recommended_model,
                "datetime": latest_row["datetime"],
                "probability": float(latest_row["probability"]),
                "direction": no_trade_direction(
                    float(latest_row["probability"]),
                    float(recommendation["recommended_lower_threshold"]),
                    float(recommendation["recommended_upper_threshold"]),
                ),
                "lower_threshold": float(recommendation["recommended_lower_threshold"]),
                "upper_threshold": float(recommendation["recommended_upper_threshold"]),
            }

    return {
        "recommendation": recommendation,
        "recommended_signal": recommended_signal,
        "holdout_summary": holdout_summary,
        "backtest_summary": backtest_summary,
        "rolling_summary": rolling_summary,
        "threshold_summary": threshold_summary,
        "no_trade_total_summary": no_trade_total,
        "snapshot": snapshot,
    }


def available_assets() -> list[str]:
    holdout_assets = sorted(load_holdout_summary(include_weighted=True)["symbol"].dropna().unique().tolist())
    return holdout_assets or list(ALL_SYMBOLS)


def available_models(include_weighted: bool = True) -> list[str]:
    models = sorted(load_holdout_summary(include_weighted=include_weighted)["model"].dropna().unique().tolist())
    if models:
        return models
    return list(MODEL_NAMES)
