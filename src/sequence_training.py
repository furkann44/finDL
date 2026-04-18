from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import DATETIME_COLUMN, FEATURE_COLUMNS, LOOKBACK, RANDOM_STATE, TARGET_COLUMN, processed_data_path
from dataset import SequenceDataset, build_sequence_data, build_sequence_data_with_context, fit_feature_scaler, time_split
from evaluate import frame_split_summary


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_frame(symbol: str) -> pd.DataFrame:
    input_path = processed_data_path(symbol)
    if not input_path.exists():
        raise FileNotFoundError(f"Islenmis veri bulunamadi: {input_path}")

    frame = pd.read_parquet(input_path).sort_values(DATETIME_COLUMN).reset_index(drop=True)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, DATETIME_COLUMN])
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Sequence model girdi verisinde eksik kolonlar var: {missing_columns}")

    return frame


def build_dataloaders(
    frame: pd.DataFrame,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    train_frame, val_frame, test_frame = time_split(frame)
    return build_dataloaders_from_frames(
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        batch_size=batch_size,
    )


def build_dataloaders_from_frames(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    scaler = fit_feature_scaler(train_frame)

    train_sequence_data = build_sequence_data(train_frame, scaler=scaler, lookback=LOOKBACK)
    val_sequence_data = build_sequence_data_with_context(train_frame, val_frame, scaler=scaler, lookback=LOOKBACK)
    test_sequence_data = build_sequence_data_with_context(val_frame, test_frame, scaler=scaler, lookback=LOOKBACK)

    train_dataset = SequenceDataset(train_sequence_data)
    val_dataset = SequenceDataset(val_sequence_data)
    test_dataset = SequenceDataset(test_sequence_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    split_details = {
        "frame_summaries": {
            "train": frame_split_summary(train_frame),
            "validation": frame_split_summary(val_frame),
            "test": frame_split_summary(test_frame),
        },
        "sequence_counts": {
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
        },
    }

    return train_loader, val_loader, test_loader, split_details


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
    probabilities: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * len(targets)

            batch_probabilities = torch.sigmoid(logits).cpu().numpy()
            probabilities.append(batch_probabilities)
            targets_list.append(targets.cpu().numpy())

    stacked_targets = np.concatenate(targets_list)
    stacked_probabilities = np.concatenate(probabilities)
    average_loss = total_loss / len(loader.dataset)
    return average_loss, stacked_targets, stacked_probabilities
