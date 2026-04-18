from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import METRICS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR, RANDOM_STATE, ensure_directories, raw_data_path, resolve_symbols, sanitize_symbol
from features import build_feature_frame
from train_baseline import prepare_arrays, time_split
from evaluate import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold-based labeling deneyi calistir.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklarda threshold deneyi yap.")
    parser.add_argument("--return-threshold", type=float, default=0.002, help="Simetrik etiketleme esigi")
    return parser.parse_args()


def threshold_tag(return_threshold: float) -> str:
    return f"thr_{return_threshold:.4f}".replace(".", "p")


def threshold_processed_path(symbol: str, return_threshold: float) -> Path:
    return PROCESSED_DATA_DIR / f"{sanitize_symbol(symbol)}_1day_features_{threshold_tag(return_threshold)}.parquet"


def threshold_metrics_path(symbol: str, return_threshold: float) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_{threshold_tag(return_threshold)}_threshold_experiment.json"


def threshold_summary_path(return_threshold: float) -> Path:
    return METRICS_DIR / f"threshold_experiment_summary_{threshold_tag(return_threshold)}.csv"


def threshold_report_path(return_threshold: float) -> Path:
    return REPORTS_DIR / f"threshold_experiment_report_{threshold_tag(return_threshold)}.md"


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    rounded = frame.copy()
    numeric_columns = rounded.select_dtypes(include="number").columns
    rounded[numeric_columns] = rounded[numeric_columns].round(float_digits)
    headers = list(rounded.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = [
        "| " + " | ".join(str(row[column]) for column in headers) + " |"
        for _, row in rounded.iterrows()
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def run_symbol(symbol: str, return_threshold: float) -> dict[str, object]:
    raw_frame = pd.read_parquet(raw_data_path(symbol))
    processed_frame = build_feature_frame(raw_frame, return_threshold=return_threshold)
    processed_output = threshold_processed_path(symbol, return_threshold)
    processed_frame.to_parquet(processed_output, index=False)

    train_frame, val_frame, test_frame = time_split(processed_frame)
    x_train, x_val, x_test, y_train, y_val, y_test, _ = prepare_arrays(train_frame, val_frame, test_frame)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    val_probabilities = model.predict_proba(x_val)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    result = {
        "symbol": symbol,
        "return_threshold": return_threshold,
        "rows": int(len(processed_frame)),
        "dropped_ratio": float(1 - len(processed_frame) / len(raw_frame)),
        "target_mean": float(processed_frame["target"].mean()),
        "validation_accuracy": compute_classification_metrics(y_val, val_probabilities)["accuracy"],
        "validation_f1": compute_classification_metrics(y_val, val_probabilities)["f1"],
        "validation_roc_auc": compute_classification_metrics(y_val, val_probabilities)["roc_auc"],
        "test_accuracy": compute_classification_metrics(y_test, test_probabilities)["accuracy"],
        "test_f1": compute_classification_metrics(y_test, test_probabilities)["f1"],
        "test_roc_auc": compute_classification_metrics(y_test, test_probabilities)["roc_auc"],
        "processed_path": str(processed_output),
    }
    threshold_metrics_path(symbol, return_threshold).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ensure_directories()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)

    rows = [run_symbol(symbol, args.return_threshold) for symbol in symbols]
    summary_frame = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    summary_frame.to_csv(threshold_summary_path(args.return_threshold), index=False)

    report_lines = [
        "# Threshold Experiment Report",
        "",
        f"Bu rapor, `return_threshold={args.return_threshold}` ile uretilen etiketler uzerinde baseline sonuclarini ozetler.",
        "",
        markdown_table(summary_frame[["symbol", "rows", "dropped_ratio", "target_mean", "test_accuracy", "test_f1", "test_roc_auc"]]),
        "",
        "Not: `dropped_ratio`, esik nedeniyle veri setinden cikarilan ornek oranini gosterir.",
    ]
    threshold_report_path(args.return_threshold).write_text("\n".join(report_lines), encoding="utf-8")

    print(summary_frame.to_string(index=False))
    print(f"\nThreshold raporu kaydedildi: {threshold_report_path(args.return_threshold)}")


if __name__ == "__main__":
    main()
