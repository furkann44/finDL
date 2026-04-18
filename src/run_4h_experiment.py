from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import METRICS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR, RANDOM_STATE, ensure_directories, resolve_symbols, sanitize_symbol
from features import build_feature_frame
from train_baseline import prepare_arrays, time_split
from evaluate import compute_classification_metrics
from twelvedata_client import TwelveDataClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4 saatlik veri icin baseline deney hattisi.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklarda 4h deneyi yap.")
    parser.add_argument("--interval", default="4h", help="Varsayilan 4h")
    return parser.parse_args()


def raw_4h_path(symbol: str, interval: str) -> Path:
    return RAW_DATA_DIR / f"{sanitize_symbol(symbol)}_{interval}.parquet"


def processed_4h_path(symbol: str, interval: str) -> Path:
    return PROCESSED_DATA_DIR / f"{sanitize_symbol(symbol)}_{interval}_features.parquet"


def metrics_4h_path(symbol: str, interval: str) -> Path:
    return METRICS_DIR / f"{sanitize_symbol(symbol)}_{interval}_experiment.json"


def summary_4h_path(interval: str) -> Path:
    return METRICS_DIR / f"intraday_{interval}_summary.csv"


def report_4h_path(interval: str) -> Path:
    return REPORTS_DIR / f"intraday_{interval}_report.md"


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


def run_symbol(symbol: str, interval: str, client: TwelveDataClient) -> dict[str, object]:
    raw_frame = client.fetch_time_series(symbol=symbol, interval=interval)
    raw_output = raw_4h_path(symbol, interval)
    raw_frame.to_parquet(raw_output, index=False)

    processed_frame = build_feature_frame(raw_frame)
    processed_output = processed_4h_path(symbol, interval)
    processed_frame.to_parquet(processed_output, index=False)

    train_frame, val_frame, test_frame = time_split(processed_frame)
    x_train, x_val, x_test, y_train, y_val, y_test, _ = prepare_arrays(train_frame, val_frame, test_frame)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    val_probabilities = model.predict_proba(x_val)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    result = {
        "symbol": symbol,
        "interval": interval,
        "rows": int(len(processed_frame)),
        "target_mean": float(processed_frame["target"].mean()),
        "validation_accuracy": compute_classification_metrics(y_val, val_probabilities)["accuracy"],
        "validation_f1": compute_classification_metrics(y_val, val_probabilities)["f1"],
        "validation_roc_auc": compute_classification_metrics(y_val, val_probabilities)["roc_auc"],
        "test_accuracy": compute_classification_metrics(y_test, test_probabilities)["accuracy"],
        "test_f1": compute_classification_metrics(y_test, test_probabilities)["f1"],
        "test_roc_auc": compute_classification_metrics(y_test, test_probabilities)["roc_auc"],
        "raw_path": str(raw_output),
        "processed_path": str(processed_output),
    }
    metrics_4h_path(symbol, interval).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ensure_directories()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)
    client = TwelveDataClient()

    rows = [run_symbol(symbol, args.interval, client) for symbol in symbols]
    summary_frame = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    summary_frame.to_csv(summary_4h_path(args.interval), index=False)

    report_lines = [
        "# Intraday Experiment Report",
        "",
        f"Bu rapor, `{args.interval}` frekansi ile kurulan baseline hattinin sonuclarini ozetler.",
        "",
        markdown_table(summary_frame[["symbol", "interval", "rows", "target_mean", "test_accuracy", "test_f1", "test_roc_auc"]]),
        "",
        "Not: Bu intraday akisi genisleme amaclidir; gunluk ana pipeline ile ayni leakage kurallarini kullanir.",
    ]
    report_4h_path(args.interval).write_text("\n".join(report_lines), encoding="utf-8")

    print(summary_frame.to_string(index=False))
    print(f"\nIntraday raporu kaydedildi: {report_4h_path(args.interval)}")


if __name__ == "__main__":
    main()
