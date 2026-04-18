from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from config import DATETIME_COLUMN, FUTURE_RETURN_COLUMN, TARGET_COLUMN


def compute_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    roc_auc = None
    if np.unique(y_true).size > 1:
        roc_auc = float(roc_auc_score(y_true, probabilities))

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(predictions)),
    }


def threshold_candidates(start: float = 0.30, stop: float = 0.70, step: float = 0.01) -> np.ndarray:
    if not 0 < start < stop < 1:
        raise ValueError("Threshold araligi 0 ile 1 arasinda ve start < stop olacak sekilde verilmelidir.")
    if step <= 0:
        raise ValueError("Threshold step pozitif olmali.")

    total_steps = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, total_steps), 6)


def select_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    optimize_for: str = "f1",
    thresholds: np.ndarray | None = None,
    max_rate_gap: float | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = threshold_candidates()

    if optimize_for not in {"f1", "accuracy", "precision", "recall"}:
        raise ValueError("optimize_for yalnizca f1, accuracy, precision veya recall olabilir.")

    best_payload: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None
    fallback_payload: dict[str, Any] | None = None
    fallback_score: tuple[float, float, float] | None = None

    for threshold in thresholds:
        metrics = compute_classification_metrics(y_true, probabilities, threshold=float(threshold))
        score = (
            float(metrics[optimize_for]),
            float(metrics["accuracy"]),
            -abs(float(metrics["predicted_positive_rate"]) - float(metrics["positive_rate"])),
        )

        candidate_payload = {
            "best_threshold": float(threshold),
            "optimize_for": optimize_for,
            "metrics": metrics,
        }

        if fallback_score is None or score > fallback_score:
            fallback_score = score
            fallback_payload = candidate_payload

        if max_rate_gap is not None:
            rate_gap = abs(float(metrics["predicted_positive_rate"]) - float(metrics["positive_rate"]))
            if rate_gap > max_rate_gap:
                continue

        if best_score is None or score > best_score:
            best_score = score
            best_payload = candidate_payload

    if best_payload is None:
        if fallback_payload is None:
            raise RuntimeError("Threshold secimi basarisiz oldu.")
        return fallback_payload

    return best_payload


def frame_split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "start": str(frame.iloc[0][DATETIME_COLUMN]),
        "end": str(frame.iloc[-1][DATETIME_COLUMN]),
        "target_mean": float(frame[TARGET_COLUMN].mean()),
    }


def future_returns_for_datetimes(frame: pd.DataFrame, datetimes: list[str]) -> np.ndarray:
    lookup = frame.copy()
    lookup[DATETIME_COLUMN] = lookup[DATETIME_COLUMN].astype(str)
    mapped = lookup.set_index(DATETIME_COLUMN)[FUTURE_RETURN_COLUMN]
    return mapped.reindex(datetimes).to_numpy()


def build_prediction_frame(
    symbol: str,
    model_name: str,
    split_name: str,
    datetimes: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    future_returns: np.ndarray | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    prediction_frame = pd.DataFrame(
        {
            "symbol": symbol,
            "model": model_name,
            "split": split_name,
            DATETIME_COLUMN: datetimes,
            "y_true": y_true.astype(int),
            "probability": probabilities.astype(float),
            "prediction": (probabilities >= threshold).astype(int),
            "decision_threshold": float(threshold),
        }
    )

    if future_returns is not None:
        prediction_frame[FUTURE_RETURN_COLUMN] = future_returns.astype(float)

    return prediction_frame


def save_prediction_frame(prediction_frame: pd.DataFrame, output_path: str) -> None:
    prediction_frame.to_csv(output_path, index=False)


def save_roc_curve_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(5, 5))
    if np.unique(y_true).size > 1:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, probabilities):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
    title: str,
    threshold: float = 0.5,
) -> None:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            plt.text(col_index, row_index, int(matrix[row_index, col_index]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
