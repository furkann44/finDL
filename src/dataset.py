from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config import DATETIME_COLUMN, FEATURE_COLUMNS, LOOKBACK, TARGET_COLUMN, TRAIN_RATIO, VAL_RATIO


@dataclass
class SequenceData:
    sequences: np.ndarray
    targets: np.ndarray
    datetimes: list[str]


class SequenceDataset(Dataset):
    def __init__(self, sequence_data: SequenceData) -> None:
        self.features = torch.tensor(sequence_data.sequences, dtype=torch.float32)
        self.targets = torch.tensor(sequence_data.targets, dtype=torch.float32)
        self.datetimes = sequence_data.datetimes

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(frame) < LOOKBACK + 10:
        raise ValueError(f"Sequence model icin en az {LOOKBACK + 10} satir gereklidir.")

    train_end = int(len(frame) * TRAIN_RATIO)
    val_end = int(len(frame) * (TRAIN_RATIO + VAL_RATIO))

    train_frame = frame.iloc[:train_end].copy()
    val_frame = frame.iloc[train_end:val_end].copy()
    test_frame = frame.iloc[val_end:].copy()

    if min(len(train_frame), len(val_frame), len(test_frame)) <= LOOKBACK:
        raise ValueError("Sequence split sonucunda en az bir parcada yeterli pencere kalmadi.")

    return train_frame, val_frame, test_frame


def fit_feature_scaler(train_frame: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_frame[FEATURE_COLUMNS])
    return scaler


def build_sequence_data(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    lookback: int = LOOKBACK,
) -> SequenceData:
    scaled_features = scaler.transform(frame[FEATURE_COLUMNS]).astype(np.float32)
    targets = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    datetimes = frame[DATETIME_COLUMN].astype(str).tolist()

    sequence_list: list[np.ndarray] = []
    target_list: list[float] = []
    datetime_list: list[str] = []

    for target_index in range(lookback - 1, len(frame)):
        start_index = target_index - lookback + 1
        sequence_list.append(scaled_features[start_index : target_index + 1])
        target_list.append(float(targets[target_index]))
        datetime_list.append(datetimes[target_index])

    if not sequence_list:
        raise ValueError("Sequence veri seti olusturulamadi. Lookback degeri fazla buyuk olabilir.")

    return SequenceData(
        sequences=np.asarray(sequence_list, dtype=np.float32),
        targets=np.asarray(target_list, dtype=np.float32),
        datetimes=datetime_list,
    )


def build_sequence_data_with_context(
    context_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    scaler: StandardScaler,
    lookback: int = LOOKBACK,
) -> SequenceData:
    if len(target_frame) == 0:
        raise ValueError("Target frame bos oldugu icin context ile sequence veri olusturulamadi.")

    context_size = max(lookback - 1, 0)
    history_frame = context_frame.tail(context_size) if context_size > 0 else context_frame.iloc[0:0]
    combined_frame = pd.concat([history_frame, target_frame], axis=0).reset_index(drop=True)

    scaled_features = scaler.transform(combined_frame[FEATURE_COLUMNS]).astype(np.float32)
    combined_targets = combined_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    combined_datetimes = combined_frame[DATETIME_COLUMN].astype(str).tolist()

    sequence_list: list[np.ndarray] = []
    target_list: list[float] = []
    datetime_list: list[str] = []

    first_target_index = len(history_frame)
    for target_index in range(first_target_index, len(combined_frame)):
        start_index = target_index - lookback + 1
        if start_index < 0:
            continue

        sequence_list.append(scaled_features[start_index : target_index + 1])
        target_list.append(float(combined_targets[target_index]))
        datetime_list.append(combined_datetimes[target_index])

    if not sequence_list:
        raise ValueError("Context ile sequence veri seti olusturulamadi. Lookback degeri veya frame boyutlari yetersiz.")

    return SequenceData(
        sequences=np.asarray(sequence_list, dtype=np.float32),
        targets=np.asarray(target_list, dtype=np.float32),
        datetimes=datetime_list,
    )
