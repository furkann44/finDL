from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from backtest_predictions import run_backtest
from config import (
    FIGURES_DIR,
    MODEL_NAMES,
    METRICS_DIR,
    REPORTS_DIR,
    ensure_directories,
    no_trade_prediction_path,
    prediction_path,
    resolve_symbols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation tabanli no-trade band tuning calistir.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklar icin no-trade tuning yap.")
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES), help="Varsayilan: tum temel modeller")
    parser.add_argument("--lower-start", type=float, default=0.35, help="Alt threshold baslangici")
    parser.add_argument("--lower-stop", type=float, default=0.49, help="Alt threshold bitisi")
    parser.add_argument("--upper-start", type=float, default=0.51, help="Ust threshold baslangici")
    parser.add_argument("--upper-stop", type=float, default=0.65, help="Ust threshold bitisi")
    parser.add_argument("--step", type=float, default=0.02, help="Threshold adimi")
    parser.add_argument("--min-coverage", type=float, default=0.25, help="Minimum islem kapsami")
    parser.add_argument(
        "--optimize-for",
        choices=["active_f1", "active_accuracy", "total_return", "sharpe"],
        default="active_f1",
        help="Band secimi icin optimizasyon metriği",
    )
    return parser.parse_args()


def resolve_models(requested_models: list[str]) -> list[str]:
    return requested_models


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    if not 0 < start < stop < 1:
        raise ValueError("Threshold grid 0 ile 1 arasinda ve start < stop olmali.")
    count = int(round((stop - start) / step)) + 1
    return [round(value, 6) for value in np.linspace(start, stop, count)]


def no_trade_signal(probabilities: np.ndarray, lower_threshold: float, upper_threshold: float) -> np.ndarray:
    return np.where(probabilities >= upper_threshold, 1, np.where(probabilities <= lower_threshold, -1, 0))


def build_no_trade_frame(
    frame: pd.DataFrame,
    model_name: str,
    split_name: str,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.DataFrame:
    output = frame.copy()
    output["model"] = model_name
    output["split"] = split_name
    output["signal"] = no_trade_signal(output["probability"].to_numpy(), lower_threshold, upper_threshold)
    output["action"] = output["signal"].map({1: "long", -1: "short", 0: "no_trade"})
    output["predicted_class"] = np.where(output["signal"] == 1, 1, np.where(output["signal"] == -1, 0, np.nan))
    output["active_trade"] = (output["signal"] != 0).astype(int)
    output["lower_threshold"] = float(lower_threshold)
    output["upper_threshold"] = float(upper_threshold)
    return output


def compute_no_trade_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    signal = frame["signal"].to_numpy()
    y_true = frame["y_true"].to_numpy()
    probabilities = frame["probability"].to_numpy()
    active_mask = signal != 0
    coverage = float(np.mean(active_mask))
    abstain_rate = 1.0 - coverage

    metrics: dict[str, Any] = {
        "coverage": coverage,
        "abstain_rate": abstain_rate,
        "active_count": int(np.sum(active_mask)),
        "active_accuracy": 0.0,
        "active_precision": 0.0,
        "active_recall": 0.0,
        "active_f1": 0.0,
        "active_roc_auc": None,
        "active_positive_rate": None,
        "predicted_long_rate": float(np.mean(signal == 1)),
        "predicted_short_rate": float(np.mean(signal == -1)),
    }

    if np.any(active_mask):
        active_y_true = y_true[active_mask]
        active_probabilities = probabilities[active_mask]
        active_predicted = np.where(signal[active_mask] == 1, 1, 0)
        metrics["active_accuracy"] = float(accuracy_score(active_y_true, active_predicted))
        metrics["active_precision"] = float(precision_score(active_y_true, active_predicted, zero_division=0))
        metrics["active_recall"] = float(recall_score(active_y_true, active_predicted, zero_division=0))
        metrics["active_f1"] = float(f1_score(active_y_true, active_predicted, zero_division=0))
        metrics["active_positive_rate"] = float(np.mean(active_y_true))
        if np.unique(active_y_true).size > 1:
            metrics["active_roc_auc"] = float(roc_auc_score(active_y_true, active_probabilities))

    backtest_frame, backtest_summary = run_backtest(frame)
    metrics.update(backtest_summary)
    return metrics


def objective_score(metrics: dict[str, Any], optimize_for: str) -> float:
    if optimize_for == "active_f1":
        return float(metrics["active_f1"]) * float(metrics["coverage"])
    if optimize_for == "active_accuracy":
        return float(metrics["active_accuracy"]) * float(metrics["coverage"])
    if optimize_for == "total_return":
        return float(metrics["total_return"])
    if optimize_for == "sharpe":
        return float(metrics["sharpe"])
    raise ValueError(f"Bilinmeyen optimize_for: {optimize_for}")


def objective_tag(optimize_for: str) -> str:
    return optimize_for.lower().replace("/", "_").replace(" ", "_")


def summary_output_path(optimize_for: str):
    if optimize_for == "active_f1":
        return METRICS_DIR / "no_trade_summary.csv"
    return METRICS_DIR / f"no_trade_summary_{objective_tag(optimize_for)}.csv"


def report_output_path(optimize_for: str):
    if optimize_for == "active_f1":
        return REPORTS_DIR / "no_trade_report.md"
    return REPORTS_DIR / f"no_trade_report_{objective_tag(optimize_for)}.md"


def chart_output_path(optimize_for: str):
    if optimize_for == "active_f1":
        return FIGURES_DIR / "no_trade_test_total_return.png"
    return FIGURES_DIR / f"no_trade_test_total_return_{objective_tag(optimize_for)}.png"


def select_best_band(
    validation_frame: pd.DataFrame,
    optimize_for: str,
    min_coverage: float,
    lower_thresholds: list[float],
    upper_thresholds: list[float],
) -> tuple[float, float, dict[str, Any], pd.DataFrame]:
    best_score: float | None = None
    best_lower = 0.4
    best_upper = 0.6
    best_metrics: dict[str, Any] | None = None
    best_frame: pd.DataFrame | None = None

    for lower_threshold in lower_thresholds:
        for upper_threshold in upper_thresholds:
            if lower_threshold >= upper_threshold:
                continue

            candidate_frame = build_no_trade_frame(
                validation_frame,
                model_name=str(validation_frame["model"].iloc[0]),
                split_name="validation",
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
            candidate_metrics = compute_no_trade_metrics(candidate_frame)
            if candidate_metrics["coverage"] < min_coverage:
                continue

            score = objective_score(candidate_metrics, optimize_for)
            if best_score is None or score > best_score:
                best_score = score
                best_lower = lower_threshold
                best_upper = upper_threshold
                best_metrics = candidate_metrics
                best_frame = candidate_frame

    if best_metrics is None or best_frame is None:
        raise RuntimeError("No-trade band secimi basarisiz oldu. Min coverage cok yuksek olabilir.")

    return best_lower, best_upper, best_metrics, best_frame


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


def load_prediction_pair(symbol: str, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_path = prediction_path(symbol, model_name, "validation")
    test_path = prediction_path(symbol, model_name, "test")
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Prediction dosyalari eksik | symbol={symbol} | model={model_name}")
    return pd.read_csv(validation_path), pd.read_csv(test_path)


def save_chart(summary_frame: pd.DataFrame, optimize_for: str) -> None:
    pivot = summary_frame.pivot(index="symbol", columns="model", values="test_total_return")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(f"No-Trade Band Test Total Return ({optimize_for})")
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Total return")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_output_path(optimize_for), dpi=150)
    plt.close()


def main() -> None:
    ensure_directories()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)
    models = resolve_models(args.models)
    lower_thresholds = threshold_grid(args.lower_start, args.lower_stop, args.step)
    upper_thresholds = threshold_grid(args.upper_start, args.upper_stop, args.step)

    summary_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for symbol in symbols:
        for model_name in models:
            try:
                validation_frame, test_frame = load_prediction_pair(symbol, model_name)
                lower_threshold, upper_threshold, validation_metrics, tuned_validation_frame = select_best_band(
                    validation_frame=validation_frame,
                    optimize_for=args.optimize_for,
                    min_coverage=args.min_coverage,
                    lower_thresholds=lower_thresholds,
                    upper_thresholds=upper_thresholds,
                )
                tuned_test_frame = build_no_trade_frame(
                    test_frame,
                    model_name=model_name,
                    split_name="test",
                    lower_threshold=lower_threshold,
                    upper_threshold=upper_threshold,
                )
                test_metrics = compute_no_trade_metrics(tuned_test_frame)

                tuned_validation_path = no_trade_prediction_path(symbol, model_name, "validation")
                tuned_test_path = no_trade_prediction_path(symbol, model_name, "test")
                tuned_validation_frame.to_csv(tuned_validation_path, index=False)
                tuned_test_frame.to_csv(tuned_test_path, index=False)

                summary_rows.append(
                    {
                        "symbol": symbol,
                        "model": model_name,
                        "optimize_for": args.optimize_for,
                        "lower_threshold": lower_threshold,
                        "upper_threshold": upper_threshold,
                        "validation_coverage": validation_metrics["coverage"],
                        "validation_active_f1": validation_metrics["active_f1"],
                        "validation_active_accuracy": validation_metrics["active_accuracy"],
                        "validation_total_return": validation_metrics["total_return"],
                        "test_coverage": test_metrics["coverage"],
                        "test_active_f1": test_metrics["active_f1"],
                        "test_active_accuracy": test_metrics["active_accuracy"],
                        "test_total_return": test_metrics["total_return"],
                        "test_sharpe": test_metrics["sharpe"],
                        "test_max_drawdown": test_metrics["max_drawdown"],
                        "test_win_rate": test_metrics["win_rate"],
                        "test_predicted_long_rate": test_metrics["predicted_long_rate"],
                        "test_predicted_short_rate": test_metrics["predicted_short_rate"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{symbol}-{model_name}: {exc}")

    if not summary_rows:
        raise RuntimeError(f"No-trade tuning icin hic sonuc uretilemedi. Hatalar: {failures}")

    summary_frame = pd.DataFrame(summary_rows).sort_values(["symbol", "model"]).reset_index(drop=True)
    summary_path = summary_output_path(args.optimize_for)
    report_path = report_output_path(args.optimize_for)
    figure_path = chart_output_path(args.optimize_for)
    summary_frame.to_csv(summary_path, index=False)
    save_chart(summary_frame, args.optimize_for)

    report_lines = [
        "# No-Trade Band Report",
        "",
        f"Bu rapor, validation set uzerinde `{args.optimize_for}` hedefi ile secilen lower/upper threshold bandlari icin test performansini ozetler.",
        "",
        markdown_table(
            summary_frame[
                [
                    "symbol",
                    "model",
                    "lower_threshold",
                    "upper_threshold",
                    "test_coverage",
                    "test_active_f1",
                    "test_active_accuracy",
                    "test_total_return",
                    "test_sharpe",
                    "test_max_drawdown",
                ]
            ]
        ),
        "",
        f"- Grafik: `{figure_path}`",
    ]
    if failures:
        report_lines.extend(["", "## Warnings", ""] + [f"- {failure}" for failure in failures])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(summary_frame.to_string(index=False))
    print(f"\nNo-trade summary kaydedildi: {summary_path}")
    print(f"No-trade report kaydedildi: {report_path}")


if __name__ == "__main__":
    main()
