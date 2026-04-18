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

from config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    LOOKBACK,
    asset_group_for_symbol,
    ensure_directories,
    resolve_symbols,
    sequence_walk_forward_fold_metrics_path,
    sequence_walk_forward_metrics_path,
    sequence_walk_forward_report_path,
    sequence_walk_forward_summary_path,
)
from evaluate import compute_classification_metrics, select_best_threshold
from models import GRUClassifier, LSTMClassifier
from sequence_training import (
    build_dataloaders_from_frames,
    configure_logging,
    evaluate_model,
    load_processed_frame,
    set_random_seed,
    train_one_epoch,
)
from walk_forward_baseline import build_walk_forward_slices


LOGGER = logging.getLogger(__name__)
SUPPORTED_SEQUENCE_MODELS = ("lstm", "gru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM ve GRU icin expanding-window walk-forward evaluation calistir.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklar icin calistir.")
    parser.add_argument("--models", nargs="*", default=list(SUPPORTED_SEQUENCE_MODELS), help="Varsayilan: lstm gru")
    parser.add_argument("--initial-train-ratio", type=float, default=0.5, help="Ilk train orani")
    parser.add_argument("--n-folds", type=int, default=3, help="Fold sayisi")
    parser.add_argument("--min-train-size", type=int, default=400, help="Minimum ilk train boyutu")
    parser.add_argument("--min-test-size", type=int, default=120, help="Minimum fold test boyutu")
    parser.add_argument("--epochs", type=int, default=8, help="Her fold icin epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=1, help="Katman sayisi")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Ogrenme orani")
    parser.add_argument("--validation-ratio", type=float, default=0.15, help="Fold ici validation orani")
    parser.add_argument(
        "--max-rate-gap",
        type=float,
        default=0.20,
        help="Validation predicted positive rate ile gercek positive rate arasindaki maksimum fark.",
    )
    return parser.parse_args()


def resolve_models(requested_models: list[str]) -> list[str]:
    unknown = sorted(set(requested_models) - set(SUPPORTED_SEQUENCE_MODELS))
    if unknown:
        raise ValueError(
            f"Bilinmeyen sequence modelleri: {unknown}. Gecerli modeller: {list(SUPPORTED_SEQUENCE_MODELS)}"
        )
    return requested_models


def split_train_validation(frame: pd.DataFrame, validation_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio 0 ile 0.5 arasinda olmali.")

    validation_size = max(int(len(frame) * validation_ratio), LOOKBACK + 5)
    if validation_size >= len(frame):
        raise ValueError("Fold ici validation boyutu train frame'i kapliyor.")

    train_frame = frame.iloc[:-validation_size].copy()
    validation_frame = frame.iloc[-validation_size:].copy()
    if len(train_frame) <= LOOKBACK or len(validation_frame) == 0:
        raise ValueError("Fold ici train/validation ayrimi sequence model icin yetersiz.")

    return train_frame, validation_frame


def build_model(model_name: str, hidden_size: int, num_layers: int, dropout: float) -> nn.Module:
    if model_name == "lstm":
        return LSTMClassifier(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_name == "gru":
        return GRUClassifier(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Desteklenmeyen model: {model_name}")


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


def train_fold_model(
    model_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    inner_train_frame, validation_frame = split_train_validation(train_frame, args.validation_ratio)
    train_loader, val_loader, test_loader, split_details = build_dataloaders_from_frames(
        train_frame=inner_train_frame,
        val_frame=validation_frame,
        test_frame=test_frame,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, args.hidden_size, args.num_layers, args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_val, val_probabilities = evaluate_model(model, val_loader, criterion, device)
        val_metrics = compute_classification_metrics(y_val, val_probabilities)
        LOGGER.info(
            "Sequence WF epoch | model=%s | epoch=%s/%s | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | val_roc_auc=%s",
            model_name,
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_metrics["f1"],
            val_metrics["roc_auc"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Fold sequence model durumu kaydedilemedi.")

    model.load_state_dict(best_state)
    val_loss, y_val, val_probabilities = evaluate_model(model, val_loader, criterion, device)
    test_loss, y_test, test_probabilities = evaluate_model(model, test_loader, criterion, device)

    threshold_payload = select_best_threshold(
        y_val,
        val_probabilities,
        optimize_for="f1",
        max_rate_gap=args.max_rate_gap,
    )
    tuned_threshold = float(threshold_payload["best_threshold"])

    return {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "validation_loss": val_loss,
        "test_loss": test_loss,
        "threshold": tuned_threshold,
        "validation_metrics": compute_classification_metrics(y_val, val_probabilities, threshold=tuned_threshold),
        "test_metrics": compute_classification_metrics(y_test, test_probabilities, threshold=tuned_threshold),
        "sequence_counts": split_details["sequence_counts"],
    }


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


def save_report(summary_frame: pd.DataFrame) -> None:
    if summary_frame.empty:
        return

    pivot = summary_frame.pivot(index="symbol", columns="model", values="roc_auc")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Sequence Walk-Forward ROC-AUC")
    ax.set_xlabel("Symbol")
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    chart_path = FIGURES_DIR / "sequence_walk_forward_roc_auc.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    report_lines = [
        "# Sequence Walk-Forward Report",
        "",
        "Bu rapor, LSTM ve GRU modelleri icin expanding-window walk-forward sonuclarini ozetler.",
        "",
        markdown_table(
            summary_frame[
                [
                    "asset_group",
                    "symbol",
                    "model",
                    "effective_folds",
                    "accuracy",
                    "f1",
                    "roc_auc",
                    "best_threshold_mean",
                    "roc_auc_mean",
                    "roc_auc_std",
                ]
            ]
        ),
        "",
        f"- Grafik: `{chart_path}`",
    ]
    sequence_walk_forward_report_path().write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    ensure_directories()
    set_random_seed()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)
    models = resolve_models(args.models)

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for symbol in symbols:
        frame = load_processed_frame(symbol)
        for model_name in models:
            try:
                slices = build_walk_forward_slices(
                    frame=frame,
                    initial_train_ratio=args.initial_train_ratio,
                    n_folds=args.n_folds,
                    min_train_size=args.min_train_size,
                    min_test_size=args.min_test_size,
                )

                fold_payloads: list[dict[str, Any]] = []
                for fold_index, (train_frame, test_frame) in enumerate(slices, start=1):
                    fold_result = train_fold_model(model_name, train_frame, test_frame, args)
                    fold_payloads.append({"fold": fold_index, **fold_result})
                    fold_rows.append(
                        {
                            "asset_group": asset_group_for_symbol(symbol),
                            "symbol": symbol,
                            "model": model_name,
                            "fold": fold_index,
                            "threshold": fold_result["threshold"],
                            "accuracy": fold_result["test_metrics"]["accuracy"],
                            "f1": fold_result["test_metrics"]["f1"],
                            "roc_auc": fold_result["test_metrics"]["roc_auc"],
                            "predicted_positive_rate": fold_result["test_metrics"]["predicted_positive_rate"],
                        }
                    )
                    LOGGER.info(
                        "Sequence WF fold tamamlandi | symbol=%s | model=%s | fold=%s/%s | roc_auc=%s | threshold=%.2f",
                        symbol,
                        model_name,
                        fold_index,
                        len(slices),
                        fold_result["test_metrics"]["roc_auc"],
                        fold_result["threshold"],
                    )

                aggregate_metrics = summarize_fold_metrics([fold["test_metrics"] for fold in fold_payloads])
                threshold_values = [fold["threshold"] for fold in fold_payloads]
                payload = {
                    "symbol": symbol,
                    "asset_group": asset_group_for_symbol(symbol),
                    "model": model_name,
                    "validation_scheme": "walk_forward_expanding",
                    "parameters": {
                        "initial_train_ratio": args.initial_train_ratio,
                        "effective_folds": len(slices),
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "learning_rate": args.learning_rate,
                    },
                    "folds": fold_payloads,
                    "aggregate_test_metrics": aggregate_metrics,
                    "best_threshold_mean": float(np.mean(threshold_values)),
                    "best_threshold_std": float(np.std(threshold_values)),
                }
                sequence_walk_forward_metrics_path(symbol, model_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
                summary_rows.append(
                    {
                        "asset_group": asset_group_for_symbol(symbol),
                        "symbol": symbol,
                        "model": model_name,
                        "effective_folds": len(slices),
                        "best_threshold_mean": float(np.mean(threshold_values)),
                        "best_threshold_std": float(np.std(threshold_values)),
                        "accuracy": aggregate_metrics.get("accuracy_mean"),
                        "f1": aggregate_metrics.get("f1_mean"),
                        "roc_auc": aggregate_metrics.get("roc_auc_mean"),
                        **aggregate_metrics,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{symbol}-{model_name}: {exc}")
                LOGGER.exception("Sequence walk-forward basarisiz | symbol=%s | model=%s | error=%s", symbol, model_name, exc)

    if not summary_rows:
        raise RuntimeError(f"Sequence walk-forward icin hic sonuc uretilemedi. Hatalar: {failures}")

    summary_frame = pd.DataFrame(summary_rows).sort_values(["asset_group", "symbol", "model"]).reset_index(drop=True)
    fold_frame = pd.DataFrame(fold_rows).sort_values(["asset_group", "symbol", "model", "fold"]).reset_index(drop=True)
    summary_frame.to_csv(sequence_walk_forward_summary_path(), index=False)
    fold_frame.to_csv(sequence_walk_forward_fold_metrics_path(), index=False)
    save_report(summary_frame)

    print(summary_frame.to_string(index=False))
    print(f"\nSequence walk-forward summary kaydedildi: {sequence_walk_forward_summary_path()}")
    print(f"Sequence walk-forward report kaydedildi: {sequence_walk_forward_report_path()}")

    if failures:
        print("\nWarnings:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
