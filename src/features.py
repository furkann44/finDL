from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATETIME_COLUMN, FEATURE_COLUMNS, FUTURE_RETURN_COLUMN, TARGET_COLUMN


def validate_price_frame(frame: pd.DataFrame) -> None:
    required_columns = {DATETIME_COLUMN, "open", "high", "low", "close"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Ham veride eksik kolonlar var: {missing_columns}")


def prepare_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validate_price_frame(frame)
    prepared = frame.copy()
    prepared[DATETIME_COLUMN] = pd.to_datetime(prepared[DATETIME_COLUMN], utc=True)
    prepared = prepared.sort_values(DATETIME_COLUMN).drop_duplicates(subset=[DATETIME_COLUMN]).reset_index(drop=True)
    return prepared


def add_return_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["return_1"] = enriched["close"].pct_change(1)
    enriched["return_5"] = enriched["close"].pct_change(5)
    enriched["hl_spread"] = (enriched["high"] - enriched["low"]) / enriched["close"]
    enriched["co_return"] = (enriched["close"] - enriched["open"]) / enriched["open"]
    return enriched


def add_volatility_feature(frame: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["volatility_10"] = enriched["return_1"].rolling(window=window, min_periods=window).std()
    return enriched


def add_moving_average_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["sma_5"] = enriched["close"].rolling(window=5, min_periods=5).mean()
    enriched["sma_10"] = enriched["close"].rolling(window=10, min_periods=10).mean()
    enriched["ema_10"] = enriched["close"].ewm(span=10, adjust=False).mean()
    return enriched


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi


def add_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["rsi_14"] = compute_rsi(enriched["close"], period=14)

    ema_fast = enriched["close"].ewm(span=12, adjust=False).mean()
    ema_slow = enriched["close"].ewm(span=26, adjust=False).mean()
    enriched["macd"] = ema_fast - ema_slow
    enriched["macd_signal"] = enriched["macd"].ewm(span=9, adjust=False).mean()
    enriched["macd_hist"] = enriched["macd"] - enriched["macd_signal"]
    return enriched


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()

    enriched["close_sma_5_ratio"] = enriched["close"] / enriched["sma_5"] - 1
    enriched["close_sma_10_ratio"] = enriched["close"] / enriched["sma_10"] - 1
    enriched["close_ema_10_ratio"] = enriched["close"] / enriched["ema_10"] - 1
    enriched["trend_strength_5_10"] = enriched["sma_5"] / enriched["sma_10"] - 1

    enriched["return_1_zscore_10"] = rolling_zscore(enriched["return_1"], window=10)
    enriched["return_1_zscore_20"] = rolling_zscore(enriched["return_1"], window=20)

    rolling_volatility_mean = enriched["volatility_10"].rolling(window=20, min_periods=20).mean()
    enriched["volatility_regime_20"] = enriched["volatility_10"] / rolling_volatility_mean - 1

    rolling_high_20 = enriched["high"].rolling(window=20, min_periods=20).max()
    rolling_low_20 = enriched["low"].rolling(window=20, min_periods=20).min()
    enriched["breakout_high_20"] = enriched["close"] / rolling_high_20 - 1
    enriched["breakout_low_20"] = enriched["close"] / rolling_low_20 - 1

    return enriched


def add_target(frame: pd.DataFrame, return_threshold: float = 0.0) -> pd.DataFrame:
    enriched = frame.copy()
    enriched[FUTURE_RETURN_COLUMN] = enriched["close"].shift(-1) / enriched["close"] - 1

    if return_threshold < 0:
        raise ValueError("return_threshold negatif olamaz.")

    if return_threshold == 0:
        enriched[TARGET_COLUMN] = (enriched[FUTURE_RETURN_COLUMN] > 0).astype(int)
    else:
        target = pd.Series(np.nan, index=enriched.index, dtype=float)
        target.loc[enriched[FUTURE_RETURN_COLUMN] > return_threshold] = 1.0
        target.loc[enriched[FUTURE_RETURN_COLUMN] < -return_threshold] = 0.0
        enriched[TARGET_COLUMN] = target

    enriched["label_return_threshold"] = return_threshold
    return enriched


def build_feature_frame(frame: pd.DataFrame, return_threshold: float = 0.0) -> pd.DataFrame:
    enriched = prepare_price_frame(frame)
    enriched = add_return_features(enriched)
    enriched = add_volatility_feature(enriched)
    enriched = add_moving_average_features(enriched)
    enriched = add_momentum_features(enriched)
    enriched = add_context_features(enriched)
    enriched = add_target(enriched, return_threshold=return_threshold)

    required_columns = FEATURE_COLUMNS + [FUTURE_RETURN_COLUMN, TARGET_COLUMN]
    enriched = enriched.dropna(subset=required_columns).reset_index(drop=True)
    enriched[TARGET_COLUMN] = enriched[TARGET_COLUMN].astype(int)

    missing_features = sorted(set(FEATURE_COLUMNS) - set(enriched.columns))
    if missing_features:
        raise ValueError(f"Feature uretimi eksik kolonlarla bitti: {missing_features}")

    if enriched.empty:
        raise ValueError(
            "Feature uretimi sonrasi veri kalmadi. "
            "Muhtemel neden: gerekli feature kolonlarinda yaygin eksik degerler bulunmasi."
        )

    return enriched
