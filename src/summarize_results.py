from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import ALL_SYMBOLS, METRICS_DIR, asset_group_for_symbol


def infer_model_name(file_path: Path) -> str:
    file_name = file_path.stem
    if file_name.endswith("_baseline_metrics"):
        return "baseline"
    if file_name.endswith("_lstm_metrics"):
        return "lstm"
    if file_name.endswith("_gru_metrics"):
        return "gru"
    if file_name.endswith("_mlp_metrics"):
        return "mlp"
    return "unknown"


def collect_metric_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for metric_path in sorted(METRICS_DIR.glob("*_metrics.json")):
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        symbol = payload.get("symbol", "unknown")
        if symbol not in ALL_SYMBOLS:
            continue
        model_name = payload.get("model_name") or infer_model_name(metric_path)

        for split_name, split_metrics in payload.get("metrics", {}).items():
            rows.append(
                {
                    "symbol": symbol,
                    "asset_group": asset_group_for_symbol(symbol),
                    "model": model_name,
                    "split": split_name,
                    "accuracy": split_metrics.get("accuracy"),
                    "precision": split_metrics.get("precision"),
                    "recall": split_metrics.get("recall"),
                    "f1": split_metrics.get("f1"),
                    "roc_auc": split_metrics.get("roc_auc"),
                    "loss": split_metrics.get("loss"),
                    "positive_rate": split_metrics.get("positive_rate"),
                    "predicted_positive_rate": split_metrics.get("predicted_positive_rate"),
                }
            )

    return rows


def main() -> None:
    rows = collect_metric_rows()
    if not rows:
        raise FileNotFoundError(f"Ozetlenecek metrik dosyasi bulunamadi: {METRICS_DIR}")

    summary_frame = pd.DataFrame(rows).sort_values(["asset_group", "symbol", "model", "split"]).reset_index(drop=True)
    output_path = METRICS_DIR / "model_summary.csv"
    summary_frame.to_csv(output_path, index=False)

    print(summary_frame.to_string(index=False))
    print(f"\nSummary kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
