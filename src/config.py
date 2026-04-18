from __future__ import annotations

from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
BACKTESTS_DIR = ARTIFACTS_DIR / "backtests"

DEFAULT_SYMBOL = "BTC/USD"
ALL_SYMBOLS = (
    "BTC/USD",
    "ETH/USD",
    "AAPL",
    "NVDA",
    "XAU/USD",
)
MODEL_NAMES = (
    "baseline",
    "mlp",
    "lstm",
    "gru",
)

ASSET_GROUPS = {
    "BTC/USD": "crypto",
    "ETH/USD": "crypto",
    "AAPL": "equity",
    "NVDA": "equity",
    "XAU/USD": "commodity",
}

INTERVAL = "1day"
TIMEZONE = "UTC"
LOOKBACK = 30
OUTPUT_SIZE = 5000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
LABEL_RETURN_THRESHOLD = 0.0

DATETIME_COLUMN = "datetime"
TARGET_COLUMN = "target"
FUTURE_RETURN_COLUMN = "future_return_1"

FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "hl_spread",
    "co_return",
    "volatility_10",
    "sma_5",
    "sma_10",
    "ema_10",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "close_sma_5_ratio",
    "close_sma_10_ratio",
    "close_ema_10_ratio",
    "trend_strength_5_10",
    "return_1_zscore_10",
    "return_1_zscore_20",
    "volatility_regime_20",
    "breakout_high_20",
    "breakout_low_20",
]


def validate_split_ratios() -> None:
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Train/validation/test oranlari toplami 1.0 olmali.")


def ensure_directories() -> None:
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        METRICS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        PREDICTIONS_DIR,
        BACKTESTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def sanitize_symbol(symbol: str) -> str:
    return symbol.lower().replace("/", "_").replace(" ", "")


def raw_data_path(symbol: str) -> Path:
    return RAW_DATA_DIR / f"{sanitize_symbol(symbol)}_{INTERVAL}.parquet"


def processed_data_path(symbol: str) -> Path:
    return PROCESSED_DATA_DIR / f"{sanitize_symbol(symbol)}_{INTERVAL}_features.parquet"


def baseline_metrics_path(symbol: str) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_baseline_metrics.json"


def model_metrics_path(symbol: str, model_name: str) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_metrics.json"


def lstm_metrics_path(symbol: str) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_lstm_metrics.json"


def lstm_model_path(symbol: str) -> Path:
    return MODELS_DIR / f"{sanitize_symbol(symbol)}_lstm.pt"


def model_artifact_path(symbol: str, model_name: str) -> Path:
    return MODELS_DIR / f"{sanitize_symbol(symbol)}_{model_name}.pt"


def mlp_metrics_path(symbol: str) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_mlp_metrics.json"


def mlp_model_path(symbol: str) -> Path:
    return MODELS_DIR / f"{sanitize_symbol(symbol)}_mlp.pt"


def gru_metrics_path(symbol: str) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_gru_metrics.json"


def gru_model_path(symbol: str) -> Path:
    return MODELS_DIR / f"{sanitize_symbol(symbol)}_gru.pt"


def asset_group_for_symbol(symbol: str) -> str:
    return ASSET_GROUPS.get(symbol, "unknown")


def walk_forward_metrics_path(symbol: str) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_walk_forward_metrics.json"


def walk_forward_summary_path() -> Path:
    return METRICS_DIR / "walk_forward_summary.csv"


def walk_forward_fold_metrics_path() -> Path:
    return METRICS_DIR / "walk_forward_fold_metrics.csv"


def prediction_path(symbol: str, model_name: str, split: str) -> Path:
    return PREDICTIONS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_{split}_predictions.csv"


def tuned_prediction_path(symbol: str, model_name: str, split: str) -> Path:
    return PREDICTIONS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_{split}_tuned_predictions.csv"


def no_trade_prediction_path(symbol: str, model_name: str, split: str) -> Path:
    return PREDICTIONS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_{split}_no_trade_predictions.csv"


def threshold_tuning_metrics_path(symbol: str, model_name: str) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_threshold_tuning.json"


def threshold_tuning_summary_path() -> Path:
    return METRICS_DIR / "threshold_tuning_summary.csv"


def threshold_tuning_report_path() -> Path:
    return REPORTS_DIR / "threshold_tuning_report.md"


def no_trade_summary_path() -> Path:
    return METRICS_DIR / "no_trade_summary.csv"


def no_trade_report_path() -> Path:
    return REPORTS_DIR / "no_trade_report.md"


def sequence_walk_forward_metrics_path(symbol: str, model_name: str) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_sequence_walk_forward.json"


def sequence_walk_forward_summary_path() -> Path:
    return METRICS_DIR / "sequence_walk_forward_summary.csv"


def sequence_walk_forward_fold_metrics_path() -> Path:
    return METRICS_DIR / "sequence_walk_forward_fold_metrics.csv"


def sequence_walk_forward_report_path() -> Path:
    return REPORTS_DIR / "sequence_walk_forward_report.md"


def backtest_summary_path() -> Path:
    return BACKTESTS_DIR / "backtest_summary.csv"


def backtest_equity_path(symbol: str, model_name: str) -> Path:
    return BACKTESTS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_equity.csv"


def backtest_chart_path(symbol: str, model_name: str) -> Path:
    return BACKTESTS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_equity.png"


def rolling_retrain_summary_path() -> Path:
    return BACKTESTS_DIR / "rolling_retrain_summary.csv"


def rolling_retrain_signals_path(symbol: str, model_name: str) -> Path:
    return BACKTESTS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_rolling_signals.csv"


def rolling_retrain_equity_path(symbol: str, model_name: str) -> Path:
    return BACKTESTS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_rolling_equity.csv"


def rolling_retrain_chart_path(symbol: str, model_name: str) -> Path:
    return BACKTESTS_DIR / f"{sanitize_symbol(symbol)}_{model_name}_rolling_equity.png"


def rolling_retrain_report_path() -> Path:
    return REPORTS_DIR / "rolling_retrain_report.md"


def resolve_symbols(symbols: Iterable[str] | None = None, use_all: bool = False) -> list[str]:
    if use_all:
        return list(ALL_SYMBOLS)

    if not symbols:
        return [DEFAULT_SYMBOL]

    requested_symbols = list(symbols)
    unknown = sorted(set(requested_symbols) - set(ALL_SYMBOLS))
    if unknown:
        raise ValueError(f"Bilinmeyen semboller: {unknown}. Gecerli semboller: {list(ALL_SYMBOLS)}")

    return requested_symbols


validate_split_ratios()
