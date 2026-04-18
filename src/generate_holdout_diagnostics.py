from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import ALL_SYMBOLS, FIGURES_DIR, PREDICTIONS_DIR, REPORTS_DIR, ensure_directories
from evaluate import compute_classification_metrics, save_confusion_matrix_plot, save_roc_curve_plot


def load_prediction_files() -> list[Path]:
    files = sorted(PREDICTIONS_DIR.glob("*_test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"Test prediction dosyasi bulunamadi: {PREDICTIONS_DIR}")
    return files


def build_markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
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


def main() -> None:
    ensure_directories()
    prediction_files = load_prediction_files()
    summary_rows: list[dict[str, object]] = []
    report_sections: list[str] = [
        "# Holdout Diagnostics Report",
        "",
        "Bu rapor, holdout test tahminlerinden ROC curve ve confusion matrix gorselleri uretir.",
        "",
    ]

    for prediction_file in prediction_files:
        frame = pd.read_csv(prediction_file)
        symbol = str(frame["symbol"].iloc[0])
        if symbol not in ALL_SYMBOLS:
            continue
        model_name = str(frame["model"].iloc[0])
        y_true = frame["y_true"].to_numpy()
        probabilities = frame["probability"].to_numpy()
        metrics = compute_classification_metrics(y_true, probabilities)

        roc_path = FIGURES_DIR / f"{symbol.lower().replace('/', '_')}_{model_name}_test_roc.png"
        cm_path = FIGURES_DIR / f"{symbol.lower().replace('/', '_')}_{model_name}_test_confusion.png"
        save_roc_curve_plot(y_true, probabilities, roc_path, title=f"{symbol} {model_name.upper()} Test ROC")
        save_confusion_matrix_plot(y_true, probabilities, cm_path, title=f"{symbol} {model_name.upper()} Test Confusion")

        summary_rows.append(
            {
                "symbol": symbol,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "roc_curve": str(roc_path),
                "confusion_matrix": str(cm_path),
            }
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values(["symbol", "model"]).reset_index(drop=True)
    summary_path = REPORTS_DIR / "holdout_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    report_sections.extend(
        [
            "## Summary",
            "",
            build_markdown_table(summary_frame),
            "",
            "## Notes",
            "",
            "- ROC curve gorselleri sinif ayristirma davranisini gosterir.",
            "- Confusion matrix gorselleri tahmin dengesizligini ve yon biasini gormeyi kolaylastirir.",
        ]
    )

    report_path = REPORTS_DIR / "holdout_diagnostics_report.md"
    report_path.write_text("\n".join(report_sections), encoding="utf-8")

    print(summary_frame.to_string(index=False))
    print(f"\nHoldout diagnostics report kaydedildi: {report_path}")


if __name__ == "__main__":
    main()
