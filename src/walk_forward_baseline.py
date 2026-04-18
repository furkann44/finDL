from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import (
    DATETIME_COLUMN,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    asset_group_for_symbol,
    ensure_directories,
    processed_data_path,
    resolve_symbols,
    walk_forward_fold_metrics_path,
    walk_forward_metrics_path,
    walk_forward_summary_path,
)
from evaluate import compute_classification_metrics, frame_split_summary


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline model icin expanding-window walk-forward validation calistir.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklari calistir.")
    parser.add_argument("--initial-train-ratio", type=float, default=0.5, help="Ilk train orani")
    parser.add_argument("--n-folds", type=int, default=5, help="Hedef fold sayisi")
    parser.add_argument("--min-train-size", type=int, default=252, help="Minimum ilk train boyutu")
    parser.add_argument("--min-test-size", type=int, default=60, help="Minimum fold test boyutu")
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
        raise ValueError(f"Walk-forward girdi verisinde eksik kolonlar var: {missing_columns}")

    return frame


def resolve_fold_sizes(total_size: int, initial_train_size: int, requested_folds: int, min_test_size: int) -> list[int]:
    remaining_size = total_size - initial_train_size
    max_folds = remaining_size // min_test_size
    effective_folds = min(requested_folds, max_folds)

    if effective_folds < 2:
        raise ValueError(
            "Walk-forward icin yeterli veri yok. "
            f"total_size={total_size}, initial_train_size={initial_train_size}, min_test_size={min_test_size}"
        )

    base_size = remaining_size // effective_folds
    remainder = remaining_size % effective_folds
    fold_sizes = [base_size] * effective_folds
    for index in range(remainder):
        fold_sizes[index] += 1

    return fold_sizes


def build_walk_forward_slices(
    frame: pd.DataFrame,
    initial_train_ratio: float,
    n_folds: int,
    min_train_size: int,
    min_test_size: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if not 0 < initial_train_ratio < 1:
        raise ValueError("initial_train_ratio 0 ile 1 arasinda olmali.")

    total_size = len(frame)
    initial_train_size = max(int(total_size * initial_train_ratio), min_train_size)
    if initial_train_size >= total_size:
        raise ValueError("Ilk train boyutu tum veriyi kapliyor. initial_train_ratio veya min_train_size degerini dusurun.")

    fold_sizes = resolve_fold_sizes(total_size, initial_train_size, n_folds, min_test_size)

    slices: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    train_end = initial_train_size
    for fold_size in fold_sizes:
        test_end = train_end + fold_size
        train_frame = frame.iloc[:train_end].copy()
        test_frame = frame.iloc[train_end:test_end].copy()
        slices.append((train_frame, test_frame))
        train_end = test_end

    return slices


def train_and_score_fold(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_frame[FEATURE_COLUMNS])
    x_test = scaler.transform(test_frame[FEATURE_COLUMNS])
    y_train = train_frame[TARGET_COLUMN].to_numpy()
    y_test = test_frame[TARGET_COLUMN].to_numpy()

    if np.unique(y_train).size < 2:
        raise ValueError("Walk-forward train fold tek sinif iceriyor.")

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = compute_classification_metrics(y_test, probabilities)
    return metrics, y_test, probabilities


def summarize_fold_metrics(fold_results: list[dict[str, Any]]) -> dict[str, float | None]:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "positive_rate", "predicted_positive_rate"]
    summary: dict[str, float | None] = {}

    for metric_name in metric_names:
        values = [result[metric_name] for result in fold_results if result.get(metric_name) is not None]
        if not values:
            summary[f"{metric_name}_mean"] = None
            summary[f"{metric_name}_std"] = None
            continue

        summary[f"{metric_name}_mean"] = float(np.mean(values))
        summary[f"{metric_name}_std"] = float(np.std(values))

    return summary


def run_walk_forward_for_symbol(symbol: str, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    frame = load_processed_frame(symbol)
    slices = build_walk_forward_slices(
        frame=frame,
        initial_train_ratio=args.initial_train_ratio,
        n_folds=args.n_folds,
        min_train_size=args.min_train_size,
        min_test_size=args.min_test_size,
    )

    fold_payloads: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for fold_index, (train_frame, test_frame) in enumerate(slices, start=1):
        metrics, y_test, probabilities = train_and_score_fold(train_frame, test_frame)
        all_targets.append(y_test)
        all_probabilities.append(probabilities)

        fold_payload = {
            "fold": fold_index,
            "train": frame_split_summary(train_frame),
            "test": frame_split_summary(test_frame),
            "metrics": metrics,
        }
        fold_payloads.append(fold_payload)

        fold_metric_rows.append(
            {
                "asset_group": asset_group_for_symbol(symbol),
                "symbol": symbol,
                "model": "baseline",
                "fold": fold_index,
                "train_rows": len(train_frame),
                "test_rows": len(test_frame),
                "train_start": str(train_frame.iloc[0][DATETIME_COLUMN]),
                "train_end": str(train_frame.iloc[-1][DATETIME_COLUMN]),
                "test_start": str(test_frame.iloc[0][DATETIME_COLUMN]),
                "test_end": str(test_frame.iloc[-1][DATETIME_COLUMN]),
                **metrics,
            }
        )

        LOGGER.info(
            "Walk-forward fold tamamlandi | symbol=%s | fold=%s/%s | test_rows=%s | accuracy=%.4f | f1=%.4f | roc_auc=%s",
            symbol,
            fold_index,
            len(slices),
            len(test_frame),
            metrics["accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
        )

    concatenated_targets = np.concatenate(all_targets)
    concatenated_probabilities = np.concatenate(all_probabilities)
    out_of_sample_metrics = compute_classification_metrics(concatenated_targets, concatenated_probabilities)
    aggregate_metrics = summarize_fold_metrics([payload["metrics"] for payload in fold_payloads])

    result = {
        "symbol": symbol,
        "asset_group": asset_group_for_symbol(symbol),
        "model": "baseline",
        "validation_scheme": "walk_forward_expanding",
        "parameters": {
            "initial_train_ratio": args.initial_train_ratio,
            "requested_folds": args.n_folds,
            "effective_folds": len(slices),
            "min_train_size": args.min_train_size,
            "min_test_size": args.min_test_size,
        },
        "feature_columns": FEATURE_COLUMNS,
        "folds": fold_payloads,
        "aggregate_fold_metrics": aggregate_metrics,
        "out_of_sample_metrics": out_of_sample_metrics,
        "total_test_rows": int(len(concatenated_targets)),
    }

    output_path = walk_forward_metrics_path(symbol)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("Walk-forward tamamlandi | symbol=%s | folds=%s | metrics_path=%s", symbol, len(slices), output_path)

    summary_row = {
        "asset_group": asset_group_for_symbol(symbol),
        "symbol": symbol,
        "model": "baseline",
        "validation_scheme": "walk_forward_expanding",
        "effective_folds": len(slices),
        "total_test_rows": int(len(concatenated_targets)),
        **out_of_sample_metrics,
        **aggregate_metrics,
    }
    return result, fold_metric_rows, summary_row


def main() -> None:
    configure_logging()
    ensure_directories()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)

    failures: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        try:
            _, symbol_fold_rows, summary_row = run_walk_forward_for_symbol(symbol, args)
            fold_rows.extend(symbol_fold_rows)
            summary_rows.append(summary_row)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Walk-forward basarisiz | symbol=%s | error=%s", symbol, exc)
            failures.append(symbol)

    if summary_rows:
        summary_frame = pd.DataFrame(summary_rows).sort_values(["asset_group", "symbol"]).reset_index(drop=True)
        fold_frame = pd.DataFrame(fold_rows).sort_values(["asset_group", "symbol", "fold"]).reset_index(drop=True)
        summary_frame.to_csv(walk_forward_summary_path(), index=False)
        fold_frame.to_csv(walk_forward_fold_metrics_path(), index=False)
        LOGGER.info("Walk-forward ozet dosyalari kaydedildi | summary=%s | folds=%s", walk_forward_summary_path(), walk_forward_fold_metrics_path())

    if failures:
        raise RuntimeError(f"Walk-forward bazi semboller icin basarisiz oldu: {failures}")


if __name__ == "__main__":
    main()
