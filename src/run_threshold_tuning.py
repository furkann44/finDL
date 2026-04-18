from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    FIGURES_DIR,
    MODEL_NAMES,
    ensure_directories,
    prediction_path,
    resolve_symbols,
    threshold_tuning_metrics_path,
    threshold_tuning_report_path,
    threshold_tuning_summary_path,
    tuned_prediction_path,
)
from evaluate import build_prediction_frame, compute_classification_metrics, save_prediction_frame, select_best_threshold, threshold_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation tabanli decision threshold tuning calistir.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklar icin threshold tuning yap.")
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES), help="Varsayilan: tum modeller")
    parser.add_argument("--optimize-for", default="f1", help="f1, accuracy, precision veya recall")
    parser.add_argument("--start", type=float, default=0.30, help="Threshold alt sinir")
    parser.add_argument("--stop", type=float, default=0.70, help="Threshold ust sinir")
    parser.add_argument("--step", type=float, default=0.01, help="Threshold adimi")
    parser.add_argument(
        "--max-rate-gap",
        type=float,
        default=0.20,
        help="Validation predicted positive rate ile gercek positive rate arasindaki maksimum fark.",
    )
    return parser.parse_args()


def resolve_models(requested_models: list[str]) -> list[str]:
    unknown = sorted(set(requested_models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Bilinmeyen modeller: {unknown}. Gecerli modeller: {list(MODEL_NAMES)}")
    return requested_models


def load_prediction_pair(symbol: str, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_path = prediction_path(symbol, model_name, "validation")
    test_path = prediction_path(symbol, model_name, "test")
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Prediction dosyalari eksik | symbol={symbol} | model={model_name}")

    return pd.read_csv(validation_path), pd.read_csv(test_path)


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


def save_gain_chart(summary_frame: pd.DataFrame) -> None:
    chart_frame = summary_frame[["symbol", "model", "test_f1_gain"]].copy()
    pivot = chart_frame.pivot(index="symbol", columns="model", values="test_f1_gain")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Threshold Tuning Test F1 Gain")
    ax.set_xlabel("Symbol")
    ax.set_ylabel("F1 gain vs threshold=0.5")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_tuning_test_f1_gain.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_directories()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)
    models = resolve_models(args.models)
    thresholds = threshold_candidates(start=args.start, stop=args.stop, step=args.step)

    summary_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for symbol in symbols:
        for model_name in models:
            try:
                validation_frame, test_frame = load_prediction_pair(symbol, model_name)
                tuning = select_best_threshold(
                    validation_frame["y_true"].to_numpy(),
                    validation_frame["probability"].to_numpy(),
                    optimize_for=args.optimize_for,
                    thresholds=thresholds,
                    max_rate_gap=args.max_rate_gap,
                )
                best_threshold = float(tuning["best_threshold"])

                default_validation_metrics = compute_classification_metrics(
                    validation_frame["y_true"].to_numpy(),
                    validation_frame["probability"].to_numpy(),
                )
                default_test_metrics = compute_classification_metrics(
                    test_frame["y_true"].to_numpy(),
                    test_frame["probability"].to_numpy(),
                )
                tuned_validation_metrics = compute_classification_metrics(
                    validation_frame["y_true"].to_numpy(),
                    validation_frame["probability"].to_numpy(),
                    threshold=best_threshold,
                )
                tuned_test_metrics = compute_classification_metrics(
                    test_frame["y_true"].to_numpy(),
                    test_frame["probability"].to_numpy(),
                    threshold=best_threshold,
                )

                validation_future_returns = validation_frame["future_return_1"].to_numpy() if "future_return_1" in validation_frame.columns else None
                test_future_returns = test_frame["future_return_1"].to_numpy() if "future_return_1" in test_frame.columns else None

                save_prediction_frame(
                    build_prediction_frame(
                        symbol=symbol,
                        model_name=model_name,
                        split_name="validation",
                        datetimes=validation_frame["datetime"].astype(str).tolist(),
                        y_true=validation_frame["y_true"].to_numpy(),
                        probabilities=validation_frame["probability"].to_numpy(),
                        future_returns=validation_future_returns,
                        threshold=best_threshold,
                    ),
                    tuned_prediction_path(symbol, model_name, "validation"),
                )
                save_prediction_frame(
                    build_prediction_frame(
                        symbol=symbol,
                        model_name=model_name,
                        split_name="test",
                        datetimes=test_frame["datetime"].astype(str).tolist(),
                        y_true=test_frame["y_true"].to_numpy(),
                        probabilities=test_frame["probability"].to_numpy(),
                        future_returns=test_future_returns,
                        threshold=best_threshold,
                    ),
                    tuned_prediction_path(symbol, model_name, "test"),
                )

                payload = {
                    "symbol": symbol,
                    "model": model_name,
                    "optimize_for": args.optimize_for,
                    "best_threshold": best_threshold,
                    "default_metrics": {
                        "validation": default_validation_metrics,
                        "test": default_test_metrics,
                    },
                    "tuned_metrics": {
                        "validation": tuned_validation_metrics,
                        "test": tuned_test_metrics,
                    },
                }
                threshold_tuning_metrics_path(symbol, model_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

                summary_rows.append(
                    {
                        "symbol": symbol,
                        "model": model_name,
                        "best_threshold": best_threshold,
                        "validation_f1_default": default_validation_metrics["f1"],
                        "validation_f1_tuned": tuned_validation_metrics["f1"],
                        "test_f1_default": default_test_metrics["f1"],
                        "test_f1_tuned": tuned_test_metrics["f1"],
                        "test_f1_gain": tuned_test_metrics["f1"] - default_test_metrics["f1"],
                        "test_accuracy_default": default_test_metrics["accuracy"],
                        "test_accuracy_tuned": tuned_test_metrics["accuracy"],
                        "test_accuracy_gain": tuned_test_metrics["accuracy"] - default_test_metrics["accuracy"],
                        "test_roc_auc": tuned_test_metrics["roc_auc"],
                        "predicted_positive_rate_default": default_test_metrics["predicted_positive_rate"],
                        "predicted_positive_rate_tuned": tuned_test_metrics["predicted_positive_rate"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{symbol}-{model_name}: {exc}")

    if not summary_rows:
        raise RuntimeError(f"Threshold tuning icin hic sonuc uretilemedi. Hatalar: {failures}")

    summary_frame = pd.DataFrame(summary_rows).sort_values(["symbol", "model"]).reset_index(drop=True)
    summary_frame.to_csv(threshold_tuning_summary_path(), index=False)
    save_gain_chart(summary_frame)

    report_lines = [
        "# Threshold Tuning Report",
        "",
        "Bu rapor, validation set uzerinde secilen decision threshold degerlerinin test performansina etkisini ozetler.",
        "",
        markdown_table(
            summary_frame[
                [
                    "symbol",
                    "model",
                    "best_threshold",
                    "test_f1_default",
                    "test_f1_tuned",
                    "test_f1_gain",
                    "test_accuracy_default",
                    "test_accuracy_tuned",
                    "test_accuracy_gain",
                    "test_roc_auc",
                ]
            ]
        ),
        "",
        f"- Grafik: `{FIGURES_DIR / 'threshold_tuning_test_f1_gain.png'}`",
    ]
    if failures:
        report_lines.extend(["", "## Warnings", ""] + [f"- {failure}" for failure in failures])

    threshold_tuning_report_path().write_text("\n".join(report_lines), encoding="utf-8")

    print(summary_frame.to_string(index=False))
    print(f"\nThreshold tuning summary kaydedildi: {threshold_tuning_summary_path()}")
    print(f"Threshold tuning report kaydedildi: {threshold_tuning_report_path()}")


if __name__ == "__main__":
    main()
