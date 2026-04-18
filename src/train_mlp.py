from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from class_weighting import resolve_experiment_model_name, resolve_torch_pos_weight
from config import (
    DATETIME_COLUMN,
    FEATURE_COLUMNS,
    FUTURE_RETURN_COLUMN,
    RANDOM_STATE,
    ensure_directories,
    model_artifact_path,
    model_metrics_path,
    prediction_path,
    resolve_symbols,
)
from evaluate import build_prediction_frame, compute_classification_metrics, save_prediction_frame
from models import MLPClassifier
from train_baseline import load_processed_frame, prepare_arrays, time_split


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature veri seti uzerinde MLP egit.")
    parser.add_argument("--symbols", nargs="*", help="Ornek: --symbols BTC/USD ETH/USD")
    parser.add_argument("--all", action="store_true", help="Tum tanimli varliklar icin MLP egit.")
    parser.add_argument("--epochs", type=int, default=25, help="Epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 32], help="MLP gizli katman boyutlari")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout orani")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Ogrenme orani")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Sinif agirliklandirma stratejisi",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(features: np.ndarray, targets: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(targets)

    return total_loss / len(loader.dataset)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    probability_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * len(targets)
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())
            target_batches.append(targets.cpu().numpy())

    probabilities = np.concatenate(probability_batches)
    y_true = np.concatenate(target_batches)
    return total_loss / len(loader.dataset), y_true, probabilities


def train_for_symbol(symbol: str, args: argparse.Namespace) -> None:
    frame = load_processed_frame(symbol)
    train_frame, val_frame, test_frame = time_split(frame)
    x_train, x_val, x_test, y_train, y_val, y_test, _ = prepare_arrays(train_frame, val_frame, test_frame)
    experiment_model_name = resolve_experiment_model_name("mlp", args.class_weight)

    train_loader = build_loader(x_train, y_train, batch_size=args.batch_size)
    val_loader = build_loader(x_val, y_val, batch_size=args.batch_size)
    test_loader = build_loader(x_test, y_test, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_sizes=tuple(args.hidden_sizes),
        dropout=args.dropout,
    ).to(device)
    pos_weight_tensor, pos_weight_value = resolve_torch_pos_weight(y_train, args.class_weight, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_targets, val_probabilities = evaluate_model(model, val_loader, criterion, device)
        val_metrics = compute_classification_metrics(val_targets, val_probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_f1": val_metrics["f1"],
                "validation_roc_auc": val_metrics["roc_auc"],
            }
        )
        LOGGER.info(
            "MLP epoch tamamlandi | symbol=%s | epoch=%s/%s | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | val_roc_auc=%s",
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
        raise RuntimeError("MLP egitimi sirasinda en iyi model durumu kaydedilemedi.")

    model.load_state_dict(best_state)
    val_loss, val_targets, val_probabilities = evaluate_model(model, val_loader, criterion, device)
    test_loss, test_targets, test_probabilities = evaluate_model(model, test_loader, criterion, device)

    validation_prediction_path = prediction_path(symbol, experiment_model_name, "validation")
    test_prediction_path = prediction_path(symbol, experiment_model_name, "test")
    save_prediction_frame(
        build_prediction_frame(
            symbol=symbol,
            model_name=experiment_model_name,
            split_name="validation",
            datetimes=val_frame[DATETIME_COLUMN].astype(str).tolist(),
            y_true=val_targets,
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
            y_true=test_targets,
            probabilities=test_probabilities,
            future_returns=test_frame[FUTURE_RETURN_COLUMN].to_numpy(),
        ),
        test_prediction_path,
    )

    model_path = model_artifact_path(symbol, experiment_model_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_name": experiment_model_name,
            "input_size": len(FEATURE_COLUMNS),
            "hidden_sizes": args.hidden_sizes,
            "dropout": args.dropout,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )

    result = {
        "symbol": symbol,
        "model_name": experiment_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_sizes": args.hidden_sizes,
            "dropout": args.dropout,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "class_weight_strategy": args.class_weight,
            "pos_weight_value": pos_weight_value,
        },
        "splits": {
            "train": {
                "rows": int(len(train_frame)),
                "start": str(train_frame.iloc[0][DATETIME_COLUMN]),
                "end": str(train_frame.iloc[-1][DATETIME_COLUMN]),
                "target_mean": float(train_frame["target"].mean()),
            },
            "validation": {
                "rows": int(len(val_frame)),
                "start": str(val_frame.iloc[0][DATETIME_COLUMN]),
                "end": str(val_frame.iloc[-1][DATETIME_COLUMN]),
                "target_mean": float(val_frame["target"].mean()),
            },
            "test": {
                "rows": int(len(test_frame)),
                "start": str(test_frame.iloc[0][DATETIME_COLUMN]),
                "end": str(test_frame.iloc[-1][DATETIME_COLUMN]),
                "target_mean": float(test_frame["target"].mean()),
            },
        },
        "metrics": {
            "validation": {"loss": val_loss, **compute_classification_metrics(val_targets, val_probabilities)},
            "test": {"loss": test_loss, **compute_classification_metrics(test_targets, test_probabilities)},
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
        "MLP tamamlandi | symbol=%s | model=%s | class_weight=%s | model_path=%s | metrics_path=%s",
        symbol,
        experiment_model_name,
        args.class_weight,
        model_path,
        metrics_path,
    )
    LOGGER.info("MLP validation metrics | %s", result["metrics"]["validation"])
    LOGGER.info("MLP test metrics | %s", result["metrics"]["test"])


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
            LOGGER.exception("MLP egitimi basarisiz | symbol=%s | error=%s", symbol, exc)
            failures.append(symbol)

    if failures:
        raise RuntimeError(f"MLP egitimi bazi semboller icin basarisiz oldu: {failures}")


if __name__ == "__main__":
    main()
