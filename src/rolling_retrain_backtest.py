from __future__ import annotations

import argparse
import copy
import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backtest_predictions import run_backtest, save_equity_chart
from class_weighting import resolve_experiment_model_name, resolve_sklearn_class_weight, resolve_torch_pos_weight
from config import (
    DATETIME_COLUMN,
    FEATURE_COLUMNS,
    FUTURE_RETURN_COLUMN,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_directories,
    resolve_symbols,
    rolling_retrain_chart_path,
    rolling_retrain_equity_path,
    rolling_retrain_report_path,
    rolling_retrain_signals_path,
    rolling_retrain_summary_path,
)
from evaluate import compute_classification_metrics
from models import MLPClassifier
from run_no_trade_tuning import build_no_trade_frame, markdown_table, select_best_band, threshold_grid
from train_baseline import load_processed_frame
from train_mlp import build_loader, evaluate_model as evaluate_mlp_model, set_random_seed, train_one_epoch as train_mlp_one_epoch


LOGGER = logging.getLogger(__name__)
SUPPORTED_MODELS = ("baseline", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling retraining ve historical signal replay backtest calistir.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD NVDA")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklarda calistir")
    parser.add_argument("--models", nargs="*", default=list(SUPPORTED_MODELS), help="Desteklenen modeller: baseline, mlp")
    parser.add_argument("--initial-train-size", type=int, default=1000, help="Ilk egitim penceresindeki toplam gun sayisi")
    parser.add_argument("--validation-size", type=int, default=252, help="Her retrain adiminda gecmis veriden ayrilacak validation gun sayisi")
    parser.add_argument("--retrain-every", type=int, default=21, help="Kac gunde bir yeniden egitilecegi")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Sinif agirliklandirma stratejisi")
    parser.add_argument("--epochs", type=int, default=12, help="MLP icin epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=64, help="MLP batch size")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 32], help="MLP gizli katman boyutlari")
    parser.add_argument("--dropout", type=float, default=0.2, help="MLP dropout")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="MLP ogrenme orani")
    parser.add_argument("--lower-start", type=float, default=0.35, help="No-trade alt threshold baslangici")
    parser.add_argument("--lower-stop", type=float, default=0.49, help="No-trade alt threshold bitisi")
    parser.add_argument("--upper-start", type=float, default=0.51, help="No-trade ust threshold baslangici")
    parser.add_argument("--upper-stop", type=float, default=0.65, help="No-trade ust threshold bitisi")
    parser.add_argument("--step", type=float, default=0.02, help="No-trade threshold adimi")
    parser.add_argument("--min-coverage", type=float, default=0.20, help="Validation tarafinda minimum aktif islem kapsami")
    parser.add_argument(
        "--optimize-for",
        choices=["active_f1", "active_accuracy", "total_return", "sharpe"],
        default="total_return",
        help="Validation tarafinda no-trade bandi secimi icin objective",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def resolve_models(requested_models: list[str]) -> list[str]:
    unknown = sorted(set(requested_models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Desteklenmeyen rolling model secimleri: {unknown}. Gecerli modeller: {list(SUPPORTED_MODELS)}")
    return requested_models


def split_history(history_frame: pd.DataFrame, validation_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(history_frame) <= validation_size:
        raise ValueError("Validation boyutu gecmis pencereyi kapliyor.")

    train_frame = history_frame.iloc[:-validation_size].copy()
    validation_frame = history_frame.iloc[-validation_size:].copy()
    if len(train_frame) < 100:
        raise ValueError("Rolling retraining icin train penceresi cok kucuk kaldi.")
    return train_frame, validation_frame


def build_prediction_frame_for_chunk(
    symbol: str,
    model_name: str,
    split_name: str,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "model": model_name,
            "split": split_name,
            DATETIME_COLUMN: frame[DATETIME_COLUMN].astype(str).tolist(),
            "y_true": frame["target"].to_numpy(),
            "probability": probabilities,
            FUTURE_RETURN_COLUMN: frame[FUTURE_RETURN_COLUMN].to_numpy(),
        }
    )


def fit_baseline_predict(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    class_weight: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_frame[FEATURE_COLUMNS])
    x_validation = scaler.transform(validation_frame[FEATURE_COLUMNS])
    x_future = scaler.transform(future_frame[FEATURE_COLUMNS])

    y_train = train_frame["target"].to_numpy()
    if np.unique(y_train).size < 2:
        raise ValueError("Rolling baseline train penceresi tek sinif iceriyor.")

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight=resolve_sklearn_class_weight(class_weight),
    )
    model.fit(x_train, y_train)
    train_probabilities = model.predict_proba(x_train)[:, 1]
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    future_probabilities = model.predict_proba(x_future)[:, 1]
    return train_probabilities, validation_probabilities, future_probabilities


def fit_mlp_predict(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    class_weight: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_frame[FEATURE_COLUMNS]).astype(np.float32)
    x_validation = scaler.transform(validation_frame[FEATURE_COLUMNS]).astype(np.float32)
    x_future = scaler.transform(future_frame[FEATURE_COLUMNS]).astype(np.float32)
    y_train = train_frame["target"].to_numpy(dtype=np.float32)
    y_validation = validation_frame["target"].to_numpy(dtype=np.float32)
    y_future = future_frame["target"].to_numpy(dtype=np.float32)

    if np.unique(y_train).size < 2:
        raise ValueError("Rolling MLP train penceresi tek sinif iceriyor.")

    train_loader = build_loader(x_train, y_train, batch_size=args.batch_size)
    validation_loader = build_loader(x_validation, y_validation, batch_size=args.batch_size)
    future_loader = build_loader(x_future, y_future, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_sizes=tuple(args.hidden_sizes),
        dropout=args.dropout,
    ).to(device)
    pos_weight_tensor, _ = resolve_torch_pos_weight(y_train, class_weight, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_state: dict[str, Any] | None = None
    best_validation_loss = float("inf")

    for _ in range(args.epochs):
        train_mlp_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, _, _ = evaluate_mlp_model(model, validation_loader, criterion, device)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Rolling MLP egitimi best state uretmedi.")

    model.load_state_dict(best_state)
    _, train_targets, train_probabilities = evaluate_mlp_model(model, train_loader, criterion, device)
    _, validation_targets, validation_probabilities = evaluate_mlp_model(model, validation_loader, criterion, device)
    _, future_targets, future_probabilities = evaluate_mlp_model(model, future_loader, criterion, device)

    if len(train_targets) != len(train_frame) or len(validation_targets) != len(validation_frame) or len(future_targets) != len(future_frame):
        raise RuntimeError("Rolling MLP prediction boyutlari frame boyutlariyla uyusmuyor.")

    return train_probabilities, validation_probabilities, future_probabilities


def fit_model_and_predict(
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    class_weight: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model_name == "baseline":
        return fit_baseline_predict(train_frame, validation_frame, future_frame, class_weight)
    if model_name == "mlp":
        return fit_mlp_predict(train_frame, validation_frame, future_frame, class_weight, args)
    raise ValueError(f"Desteklenmeyen rolling model: {model_name}")


def run_for_symbol_model(symbol: str, model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    frame = load_processed_frame(symbol)
    if len(frame) <= args.initial_train_size + args.retrain_every:
        raise ValueError("Rolling retraining icin veri boyutu secilen pencere parametrelerine gore yetersiz.")

    base_experiment_model_name = resolve_experiment_model_name(model_name, args.class_weight)
    rolling_model_name = f"{base_experiment_model_name}_rolling"
    lower_thresholds = threshold_grid(args.lower_start, args.lower_stop, args.step)
    upper_thresholds = threshold_grid(args.upper_start, args.upper_stop, args.step)

    signal_frames: list[pd.DataFrame] = []
    cycle_summaries: list[dict[str, Any]] = []
    cycle_index = 0

    for anchor in range(args.initial_train_size, len(frame), args.retrain_every):
        future_end = min(anchor + args.retrain_every, len(frame))
        history_frame = frame.iloc[:anchor].copy()
        future_frame = frame.iloc[anchor:future_end].copy()
        if future_frame.empty:
            continue

        train_frame, validation_frame = split_history(history_frame, validation_size=args.validation_size)
        train_probabilities, validation_probabilities, future_probabilities = fit_model_and_predict(
            model_name=model_name,
            train_frame=train_frame,
            validation_frame=validation_frame,
            future_frame=future_frame,
            class_weight=args.class_weight,
            args=args,
        )

        validation_prediction_frame = build_prediction_frame_for_chunk(
            symbol=symbol,
            model_name=rolling_model_name,
            split_name="validation",
            frame=validation_frame,
            probabilities=validation_probabilities,
        )

        try:
            lower_threshold, upper_threshold, validation_band_metrics, _ = select_best_band(
                validation_frame=validation_prediction_frame,
                optimize_for=args.optimize_for,
                min_coverage=args.min_coverage,
                lower_thresholds=lower_thresholds,
                upper_thresholds=upper_thresholds,
            )
        except RuntimeError:
            LOGGER.warning(
                "Rolling no-trade fallback | symbol=%s | model=%s | cycle=%s | min_coverage=%.2f saglanamadi, 0.0 ile tekrar denenecek",
                symbol,
                rolling_model_name,
                cycle_index,
                args.min_coverage,
            )
            lower_threshold, upper_threshold, validation_band_metrics, _ = select_best_band(
                validation_frame=validation_prediction_frame,
                optimize_for=args.optimize_for,
                min_coverage=0.0,
                lower_thresholds=lower_thresholds,
                upper_thresholds=upper_thresholds,
            )

        future_prediction_frame = build_prediction_frame_for_chunk(
            symbol=symbol,
            model_name=rolling_model_name,
            split_name="rolling_test",
            frame=future_frame,
            probabilities=future_probabilities,
        )
        future_no_trade_frame = build_no_trade_frame(
            future_prediction_frame,
            model_name=rolling_model_name,
            split_name="rolling_test",
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        future_no_trade_frame["retrain_cycle"] = cycle_index
        future_no_trade_frame["train_end"] = str(train_frame.iloc[-1][DATETIME_COLUMN])
        future_no_trade_frame["validation_start"] = str(validation_frame.iloc[0][DATETIME_COLUMN])
        future_no_trade_frame["validation_end"] = str(validation_frame.iloc[-1][DATETIME_COLUMN])

        cycle_backtest_frame, cycle_backtest_summary = run_backtest(future_no_trade_frame)
        future_no_trade_frame["strategy_return"] = cycle_backtest_frame["strategy_return"].to_numpy()
        future_no_trade_frame["strategy_equity"] = cycle_backtest_frame["strategy_equity"].to_numpy()
        signal_frames.append(future_no_trade_frame)

        cycle_summaries.append(
            {
                "cycle": cycle_index,
                "symbol": symbol,
                "model": rolling_model_name,
                "train_rows": len(train_frame),
                "validation_rows": len(validation_frame),
                "future_rows": len(future_frame),
                "train_end": str(train_frame.iloc[-1][DATETIME_COLUMN]),
                "validation_end": str(validation_frame.iloc[-1][DATETIME_COLUMN]),
                "future_start": str(future_frame.iloc[0][DATETIME_COLUMN]),
                "future_end": str(future_frame.iloc[-1][DATETIME_COLUMN]),
                "lower_threshold": lower_threshold,
                "upper_threshold": upper_threshold,
                "validation_active_f1": validation_band_metrics["active_f1"],
                "validation_total_return": validation_band_metrics["total_return"],
                "cycle_total_return": cycle_backtest_summary["total_return"],
                "cycle_sharpe": cycle_backtest_summary["sharpe"],
                "cycle_coverage": cycle_backtest_summary["coverage"],
            }
        )

        if cycle_index % 10 == 0 or future_end == len(frame):
            LOGGER.info(
                "Rolling cycle tamamlandi | symbol=%s | model=%s | cycle=%s | future_rows=%s | return=%.4f | coverage=%.4f",
                symbol,
                rolling_model_name,
                cycle_index,
                len(future_frame),
                cycle_backtest_summary["total_return"],
                cycle_backtest_summary["coverage"],
            )
        cycle_index += 1

    if not signal_frames:
        raise RuntimeError("Rolling retraining sonucunda hic signal uretilmedi.")

    signal_log_frame = pd.concat(signal_frames, ignore_index=True).sort_values(DATETIME_COLUMN).reset_index(drop=True)
    rolling_backtest_frame, rolling_backtest_summary = run_backtest(signal_log_frame)
    signal_log_frame["strategy_return"] = rolling_backtest_frame["strategy_return"].to_numpy()
    signal_log_frame["buy_hold_return"] = rolling_backtest_frame["buy_hold_return"].to_numpy()
    signal_log_frame["strategy_equity"] = rolling_backtest_frame["strategy_equity"].to_numpy()
    signal_log_frame["buy_hold_equity"] = rolling_backtest_frame["buy_hold_equity"].to_numpy()

    active_mask = signal_log_frame["signal"] != 0
    active_metrics = {
        "active_accuracy": 0.0,
        "active_f1": 0.0,
        "active_count": int(active_mask.sum()),
    }
    if active_mask.any():
        active_metrics = {
            "active_accuracy": compute_classification_metrics(
                signal_log_frame.loc[active_mask, "y_true"].to_numpy(),
                signal_log_frame.loc[active_mask, "probability"].to_numpy(),
                threshold=0.5,
            )["accuracy"],
            "active_f1": compute_classification_metrics(
                signal_log_frame.loc[active_mask, "y_true"].to_numpy(),
                signal_log_frame.loc[active_mask, "probability"].to_numpy(),
                threshold=0.5,
            )["f1"],
            "active_count": int(active_mask.sum()),
        }

    signals_path = rolling_retrain_signals_path(symbol, base_experiment_model_name)
    equity_path = rolling_retrain_equity_path(symbol, base_experiment_model_name)
    chart_path = rolling_retrain_chart_path(symbol, base_experiment_model_name)
    signal_log_frame.to_csv(signals_path, index=False)
    rolling_backtest_frame.to_csv(equity_path, index=False)
    save_equity_chart(rolling_backtest_frame, chart_path, title=f"{symbol} {rolling_model_name.upper()} Rolling Strategy vs Buy and Hold")

    return {
        "summary": {
            "symbol": symbol,
            "model": rolling_model_name,
            "base_model": model_name,
            "class_weight_strategy": args.class_weight,
            "optimize_for": args.optimize_for,
            "initial_train_size": args.initial_train_size,
            "validation_size": args.validation_size,
            "retrain_every": args.retrain_every,
            "cycles": len(cycle_summaries),
            "signal_rows": len(signal_log_frame),
            "test_start": str(signal_log_frame.iloc[0][DATETIME_COLUMN]),
            "test_end": str(signal_log_frame.iloc[-1][DATETIME_COLUMN]),
            **rolling_backtest_summary,
            **active_metrics,
            "signals_path": str(signals_path),
            "equity_path": str(equity_path),
            "chart_path": str(chart_path),
        },
        "cycles": cycle_summaries,
    }


def save_report(summary_frame: pd.DataFrame) -> None:
    pivot = summary_frame.pivot(index="symbol", columns="model", values="total_return")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Rolling Retraining Total Return")
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Total return")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    chart_path = REPORTS_DIR.parent / "figures" / "rolling_retrain_total_return.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    report_lines = [
        "# Rolling Retraining Report",
        "",
        "Bu rapor, gecmis veriye kadar egitip sonraki donemde sinyal ureten rolling retraining backtest sonuclarini ozetler.",
        "",
        markdown_table(
            summary_frame[
                [
                    "symbol",
                    "model",
                    "optimize_for",
                    "cycles",
                    "signal_rows",
                    "coverage",
                    "total_return",
                    "benchmark_return",
                    "sharpe",
                    "max_drawdown",
                ]
            ]
        ),
        "",
        f"- Grafik: `{chart_path}`",
    ]
    rolling_retrain_report_path().write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    ensure_directories()
    set_random_seed()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)
    models = resolve_models(args.models)

    summary_rows: list[dict[str, Any]] = []
    cycle_payloads: list[dict[str, Any]] = []
    failures: list[str] = []

    for symbol in symbols:
        for model_name in models:
            try:
                result = run_for_symbol_model(symbol, model_name, args)
                summary_rows.append(result["summary"])
                cycle_payloads.extend(result["cycles"])
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Rolling retraining basarisiz | symbol=%s | model=%s | error=%s", symbol, model_name, exc)
                failures.append(f"{symbol}-{model_name}: {exc}")

    if not summary_rows:
        raise RuntimeError(f"Rolling retraining hic sonuc uretmedi. Hatalar: {failures}")

    summary_frame = pd.DataFrame(summary_rows).sort_values(["symbol", "model"]).reset_index(drop=True)
    summary_frame.to_csv(rolling_retrain_summary_path(), index=False)
    save_report(summary_frame)

    if cycle_payloads:
        cycles_path = rolling_retrain_summary_path().with_name("rolling_retrain_cycles.json")
        cycles_path.write_text(json.dumps(cycle_payloads, indent=2), encoding="utf-8")

    print(summary_frame.to_string(index=False))
    print(f"\nRolling retraining summary kaydedildi: {rolling_retrain_summary_path()}")
    print(f"Rolling retraining report kaydedildi: {rolling_retrain_report_path()}")

    if failures:
        print("\nWarnings:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
