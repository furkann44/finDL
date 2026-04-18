from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, REPORTS_DIR, ensure_directories, walk_forward_fold_metrics_path, walk_forward_summary_path


def load_walk_forward_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = walk_forward_summary_path()
    fold_path = walk_forward_fold_metrics_path()
    if not summary_path.exists() or not fold_path.exists():
        raise FileNotFoundError(
            "Walk-forward ozet dosyalari bulunamadi. Once walk_forward_baseline.py calistirin."
        )

    summary_frame = pd.read_csv(summary_path)
    fold_frame = pd.read_csv(fold_path)
    return summary_frame, fold_frame


def save_bar_chart(frame: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    chart_frame = frame.sort_values("symbol")
    ax = chart_frame.plot(x="symbol", y=metric, kind="bar", legend=False, figsize=(9, 5), color="#2f6fed")
    ax.set_title(title)
    ax.set_xlabel("Symbol")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_fold_roc_auc_chart(fold_frame: pd.DataFrame, output_path: Path) -> None:
    pivot = fold_frame.pivot(index="fold", columns="symbol", values="roc_auc")
    ax = pivot.plot(marker="o", figsize=(9, 5))
    ax.set_title("Walk-Forward Fold ROC-AUC by Symbol")
    ax.set_xlabel("Fold")
    ax.set_ylabel("roc_auc")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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


def build_group_average_table(summary_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_frame.groupby("asset_group", observed=False)[["accuracy", "f1", "roc_auc", "predicted_positive_rate"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("asset_group")
    )


def write_report(summary_frame: pd.DataFrame, fold_frame: pd.DataFrame, figure_paths: dict[str, Path]) -> Path:
    best_row = summary_frame.loc[summary_frame["roc_auc"].idxmax()]
    weakest_row = summary_frame.loc[summary_frame["roc_auc"].idxmin()]
    most_stable_row = summary_frame.loc[summary_frame["roc_auc_std"].idxmin()]
    group_averages = build_group_average_table(summary_frame)

    findings = [
        "Walk-forward test tarafinda en iyi ROC-AUC sonucu "
        f"`{best_row['symbol']}` icin `{best_row['roc_auc']:.4f}` oldu.",
        "En zayif walk-forward ROC-AUC sonucu "
        f"`{weakest_row['symbol']}` icin `{weakest_row['roc_auc']:.4f}` oldu.",
        "Fold bazinda en istikrarli ROC-AUC davranisi "
        f"`{most_stable_row['symbol']}` icin `std={most_stable_row['roc_auc_std']:.4f}` ile goruldu.",
    ]

    report_lines = [
        "# Walk-Forward Report",
        "",
        "Bu rapor, Logistic Regression baseline modeli icin expanding-window walk-forward validation sonuclarini ozetler.",
        "",
        "## Figures",
        "",
        f"- ROC-AUC by asset: `{figure_paths['roc_auc']}`",
        f"- F1 by asset: `{figure_paths['f1']}`",
        f"- Fold ROC-AUC line chart: `{figure_paths['fold_roc_auc']}`",
        "",
        "## Key Findings",
        "",
    ]

    for finding in findings:
        report_lines.append(f"- {finding}")

    report_lines.extend(
        [
            "",
            "## Walk-Forward Summary",
            "",
            markdown_table(
                summary_frame[
                    [
                        "asset_group",
                        "symbol",
                        "effective_folds",
                        "accuracy",
                        "f1",
                        "roc_auc",
                        "roc_auc_mean",
                        "roc_auc_std",
                        "predicted_positive_rate",
                    ]
                ]
            ),
            "",
            "## Asset Group Averages",
            "",
            markdown_table(group_averages),
            "",
            "## Interpretation",
            "",
            "- Bu rapor tek holdout yerine birden fazla zamani disarida birakilan fold ile daha guvenilir bir degerlendirme saglar.",
            "- OOS ROC-AUC degerleri halen sinirliysa sonraki mantikli adim threshold tuning ve sequence modeller icin ayni walk-forward yaklasimini eklemektir.",
        ]
    )

    output_path = REPORTS_DIR / "walk_forward_report.md"
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    return output_path


def main() -> None:
    ensure_directories()
    summary_frame, fold_frame = load_walk_forward_frames()

    roc_auc_path = FIGURES_DIR / "walk_forward_roc_auc.png"
    f1_path = FIGURES_DIR / "walk_forward_f1.png"
    fold_roc_auc_path = FIGURES_DIR / "walk_forward_fold_roc_auc.png"

    save_bar_chart(summary_frame, metric="roc_auc", output_path=roc_auc_path, title="Walk-Forward ROC-AUC by Asset")
    save_bar_chart(summary_frame, metric="f1", output_path=f1_path, title="Walk-Forward F1 by Asset")
    save_fold_roc_auc_chart(fold_frame, output_path=fold_roc_auc_path)

    report_path = write_report(
        summary_frame,
        fold_frame,
        figure_paths={
            "roc_auc": roc_auc_path,
            "f1": f1_path,
            "fold_roc_auc": fold_roc_auc_path,
        },
    )

    print(summary_frame[["asset_group", "symbol", "accuracy", "f1", "roc_auc", "roc_auc_mean", "roc_auc_std"]].to_string(index=False))
    print(f"\nWalk-forward report kaydedildi: {report_path}")


if __name__ == "__main__":
    main()
