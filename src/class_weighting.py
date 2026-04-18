from __future__ import annotations

import numpy as np
import torch


def resolve_experiment_model_name(base_model_name: str, class_weight_strategy: str) -> str:
    if class_weight_strategy == "none":
        return base_model_name
    return f"{base_model_name}_{class_weight_strategy}"


def compute_positive_class_weight(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    positive_count = float(np.sum(labels == 1))
    negative_count = float(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Class weight hesaplamak icin hem pozitif hem negatif ornek gerekir.")
    return negative_count / positive_count


def resolve_sklearn_class_weight(strategy: str) -> str | None:
    if strategy == "none":
        return None
    if strategy == "balanced":
        return "balanced"
    raise ValueError(f"Desteklenmeyen class weight stratejisi: {strategy}")


def resolve_torch_pos_weight(labels: np.ndarray, strategy: str, device: torch.device) -> tuple[torch.Tensor | None, float | None]:
    if strategy == "none":
        return None, None
    if strategy != "balanced":
        raise ValueError(f"Desteklenmeyen class weight stratejisi: {strategy}")

    pos_weight_value = compute_positive_class_weight(labels)
    pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    return pos_weight_tensor, pos_weight_value
