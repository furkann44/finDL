from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import (
    DATETIME_COLUMN,
    FEATURE_COLUMNS,
    FUTURE_RETURN_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    TRAIN_RATIO,
    VAL_RATIO,
    ensure_directories,
    model_metrics_path,
    prediction_path,
    processed_data_path,
    resolve_symbols,
)
from class_weighting import resolve_experiment_model_name, resolve_sklearn_class_weight
from evaluate import build_prediction_frame, compute_classification_metrics, frame_split_summary, save_prediction_frame


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature veri seti uzerinde Logistic Regression baseline egit.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklar icin baseline egit.")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Sinif agirliklandirma stratejisi",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_processed_frame(symbol: str) -> pd.DataFrame:
    input_path = processed_data_path(symbol)
    if not input_path.exists():
        raise FileNotFoundError(f"Islenmis veri bulunamadi: {input_path}")

    frame = pd.read_parquet(input_path).sort_values(DATETIME_COLUMN).reset_index(drop=True)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, DATETIME_COLUMN])
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Model girdi verisinde eksik kolonlar var: {missing_columns}")
    return frame


def time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(frame) < 30:
        raise ValueError("Baseline egitimi icin en az 30 satir gereklidir.")

    train_end = int(len(frame) * TRAIN_RATIO)
    val_end = int(len(frame) * (TRAIN_RATIO + VAL_RATIO))

    train_frame = frame.iloc[:train_end].copy()
    val_frame = frame.iloc[train_end:val_end].copy()
    test_frame = frame.iloc[val_end:].copy()

    if min(len(train_frame), len(val_frame), len(test_frame)) == 0:
        raise ValueError("Train/validation/test bolunmesi bos bir parcaya yol acti.")

    return train_frame, val_frame, test_frame


def prepare_arrays(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()

    x_train = scaler.fit_transform(train_frame[FEATURE_COLUMNS])
    x_val = scaler.transform(val_frame[FEATURE_COLUMNS])
    x_test = scaler.transform(test_frame[FEATURE_COLUMNS])

    y_train = train_frame[TARGET_COLUMN].to_numpy()
    y_val = val_frame[TARGET_COLUMN].to_numpy()
    y_test = test_frame[TARGET_COLUMN].to_numpy()

    if np.unique(y_train).size < 2:
        raise ValueError("Train set tek sinif iceriyor. Model egitimi baslatilamiyor.")

    return x_train, x_val, x_test, y_train, y_val, y_test, scaler


def train_for_symbol(symbol: str, args: argparse.Namespace) -> None:
    frame = load_processed_frame(symbol)
    train_frame, val_frame, test_frame = time_split(frame)
    x_train, x_val, x_test, y_train, y_val, y_test, _ = prepare_arrays(train_frame, val_frame, test_frame)

    experiment_model_name = resolve_experiment_model_name("baseline", args.class_weight)
    sklearn_class_weight = resolve_sklearn_class_weight(args.class_weight)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=sklearn_class_weight)
    model.fit(x_train, y_train)

    val_probabilities = model.predict_proba(x_val)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    validation_prediction_path = prediction_path(symbol, experiment_model_name, "validation")
    test_prediction_path = prediction_path(symbol, experiment_model_name, "test")
    save_prediction_frame(
        build_prediction_frame(
            symbol=symbol,
            model_name=experiment_model_name,
            split_name="validation",
            datetimes=val_frame[DATETIME_COLUMN].astype(str).tolist(),
            y_true=y_val,
            probabilities=val_probabilities,
            future_returns=val_frame[FUTURE_RETURN_COLUMN].to_numpy(),
        ),
        validation_prediction_path,
    )
    save_prediction_frame(
        build_prediction_frame(
            symbol=symbol,
            model_name=experiment_model_name,
            split_name="test",
            datetimes=test_frame[DATETIME_COLUMN].astype(str).tolist(),
            y_true=y_test,
            probabilities=test_probabilities,
            future_returns=test_frame[FUTURE_RETURN_COLUMN].to_numpy(),
        ),
        test_prediction_path,
    )

    result = {
        "symbol": symbol,
        "model_name": experiment_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "training": {
            "class_weight_strategy": args.class_weight,
        },
        "splits": {
            "train": frame_split_summary(train_frame),
            "validation": frame_split_summary(val_frame),
            "test": frame_split_summary(test_frame),
        },
        "metrics": {
            "validation": compute_classification_metrics(y_val, val_probabilities),
            "test": compute_classification_metrics(y_test, test_probabilities),
        },
        "prediction_paths": {
            "validation": str(validation_prediction_path),
            "test": str(test_prediction_path),
        },
    }

    output_path = model_metrics_path(symbol, experiment_model_name)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    LOGGER.info(
        "Baseline tamamlandi | symbol=%s | model=%s | class_weight=%s | metrics_path=%s",
        symbol,
        experiment_model_name,
        args.class_weight,
        output_path,
    )
    LOGGER.info("Validation metrics | %s", result["metrics"]["validation"])
    LOGGER.info("Test metrics | %s", result["metrics"]["test"])


def main() -> None:
    configure_logging()
    ensure_directories()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)

    failures: list[str] = []
    for symbol in symbols:
        try:
            train_for_symbol(symbol, args)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Baseline egitimi basarisiz | symbol=%s | error=%s", symbol, exc)
            failures.append(symbol)

    if failures:
        raise RuntimeError(f"Baseline egitimi bazi semboller icin basarisiz oldu: {failures}")


if __name__ == "__main__":
    main()
