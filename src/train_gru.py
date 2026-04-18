from __future__ import annotations

import argparse
import copy
import json
import logging
from typing import Any

import torch
from torch import nn

from class_weighting import resolve_experiment_model_name, resolve_torch_pos_weight
from config import (
    FEATURE_COLUMNS,
    LOOKBACK,
    ensure_directories,
    model_artifact_path,
    model_metrics_path,
    prediction_path,
    resolve_symbols,
)
from dataset import time_split
from evaluate import (
    build_prediction_frame,
    compute_classification_metrics,
    future_returns_for_datetimes,
    save_prediction_frame,
)
from models import GRUClassifier
from sequence_training import (
    build_dataloaders,
    configure_logging,
    evaluate_model,
    load_processed_frame,
    set_random_seed,
    train_one_epoch,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature veri seti uzerinde GRU egit.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklar icin GRU egit.")
    parser.add_argument("--epochs", type=int, default=20, help="Epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=64, help="GRU hidden size")
    parser.add_argument("--num-layers", type=int, default=1, help="GRU katman sayisi")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout orani")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Ogrenme orani")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Sinif agirliklandirma stratejisi",
    )
    return parser.parse_args()


def train_for_symbol(symbol: str, args: argparse.Namespace) -> None:
    frame = load_processed_frame(symbol)
    _, val_frame, test_frame = time_split(frame)
    experiment_model_name = resolve_experiment_model_name("gru", args.class_weight)
    train_loader, val_loader, test_loader, split_details = build_dataloaders(
        frame=frame,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    train_targets = train_loader.dataset.targets.cpu().numpy()
    pos_weight_tensor, pos_weight_value = resolve_torch_pos_weight(train_targets, args.class_weight, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_val, val_probabilities = evaluate_model(model, val_loader, criterion, device)
        val_metrics = compute_classification_metrics(y_val, val_probabilities)

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics["roc_auc"],
        }
        history.append(history_entry)
        LOGGER.info(
            "GRU epoch tamamlandi | symbol=%s | epoch=%s/%s | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | val_roc_auc=%s",
            symbol,
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
        raise RuntimeError("GRU egitimi sirasinda en iyi model durumu kaydedilemedi.")

    model.load_state_dict(best_state)

    val_loss, y_val, val_probabilities = evaluate_model(model, val_loader, criterion, device)
    test_loss, y_test, test_probabilities = evaluate_model(model, test_loader, criterion, device)

    validation_prediction_path = prediction_path(symbol, experiment_model_name, "validation")
    test_prediction_path = prediction_path(symbol, experiment_model_name, "test")
    save_prediction_frame(
        build_prediction_frame(
            symbol=symbol,
            model_name=experiment_model_name,
            split_name="validation",
            datetimes=val_loader.dataset.datetimes,
            y_true=y_val,
            probabilities=val_probabilities,
            future_returns=future_returns_for_datetimes(val_frame, val_loader.dataset.datetimes),
        ),
        validation_prediction_path,
    )
    save_prediction_frame(
        build_prediction_frame(
            symbol=symbol,
            model_name=experiment_model_name,
            split_name="test",
            datetimes=test_loader.dataset.datetimes,
            y_true=y_test,
            probabilities=test_probabilities,
            future_returns=future_returns_for_datetimes(test_frame, test_loader.dataset.datetimes),
        ),
        test_prediction_path,
    )

    model_path = model_artifact_path(symbol, experiment_model_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "gru",
            "model_name": experiment_model_name,
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lookback": LOOKBACK,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )

    result = {
        "symbol": symbol,
        "model_name": experiment_model_name,
        "lookback": LOOKBACK,
        "feature_columns": FEATURE_COLUMNS,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "class_weight_strategy": args.class_weight,
            "pos_weight_value": pos_weight_value,
        },
        "splits": split_details["frame_summaries"],
        "sequence_counts": split_details["sequence_counts"],
        "metrics": {
            "validation": {
                "loss": val_loss,
                **compute_classification_metrics(y_val, val_probabilities),
            },
            "test": {
                "loss": test_loss,
                **compute_classification_metrics(y_test, test_probabilities),
            },
        },
        "history": history,
        "model_path": str(model_path),
        "prediction_paths": {
            "validation": str(validation_prediction_path),
            "test": str(test_prediction_path),
        },
    }

    metrics_path = model_metrics_path(symbol, experiment_model_name)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    LOGGER.info(
        "GRU tamamlandi | symbol=%s | model=%s | class_weight=%s | model_path=%s | metrics_path=%s",
        symbol,
        experiment_model_name,
        args.class_weight,
        model_path,
        metrics_path,
    )
    LOGGER.info("GRU validation metrics | %s", result["metrics"]["validation"])
    LOGGER.info("GRU test metrics | %s", result["metrics"]["test"])


def main() -> None:
    configure_logging()
    ensure_directories()
    set_random_seed()
    args = parse_args()
    symbols = resolve_symbols(symbols=args.symbols, use_all=args.all)

    failures: list[str] = []
    for symbol in symbols:
        try:
            train_for_symbol(symbol, args)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("GRU egitimi basarisiz | symbol=%s | error=%s", symbol, exc)
            failures.append(symbol)

    if failures:
        raise RuntimeError(f"GRU egitimi bazi semboller icin basarisiz oldu: {failures}")


if __name__ == "__main__":
    main()
