from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, METRICS_DIR, REPORTS_DIR, ensure_directories


MODEL_ORDER = ["baseline", "mlp", "gru", "lstm"]


def load_summary_frame() -> pd.DataFrame:
    summary_path = METRICS_DIR / "model_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Model ozet dosyasi bulunamadi: {summary_path}. Once summarize_results.py calistirin."
        )

    frame = pd.read_csv(summary_path)
    frame = frame[frame["model"].isin(MODEL_ORDER)].copy()
    frame["model"] = pd.Categorical(frame["model"], categories=MODEL_ORDER, ordered=True)
    return frame.sort_values(["asset_group", "symbol", "model", "split"]).reset_index(drop=True)


def save_grouped_bar_chart(frame: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    test_frame = frame[frame["split"] == "test"].copy()
    pivot = test_frame.pivot(index="symbol", columns="model", values=metric)
    pivot = pivot.reindex(columns=MODEL_ORDER)

    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("Symbol")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_best_model_table(frame: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    test_frame = frame[frame["split"] == "test"].copy()
    best_indices = test_frame.groupby("symbol")[metric].idxmax()
    best_frame = test_frame.loc[best_indices, ["asset_group", "symbol", "model", "accuracy", "f1", "roc_auc"]]
    return best_frame.sort_values(["asset_group", "symbol"]).reset_index(drop=True)


def build_group_average_table(frame: pd.DataFrame) -> pd.DataFrame:
    test_frame = frame[frame["split"] == "test"].copy()
    grouped = (
        test_frame.groupby(["asset_group", "model"], observed=False)[
            ["accuracy", "f1", "roc_auc", "predicted_positive_rate"]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["asset_group", "model"])
    )
    return grouped


def build_bias_table(frame: pd.DataFrame) -> pd.DataFrame:
    test_frame = frame[frame["split"] == "test"].copy()
    return test_frame[
        [
            "asset_group",
            "symbol",
            "model",
            "accuracy",
            "f1",
            "roc_auc",
            "positive_rate",
            "predicted_positive_rate",
        ]
    ].sort_values(["asset_group", "symbol", "model"]).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    rounded = frame.copy()
    numeric_columns = rounded.select_dtypes(include="number").columns
    rounded[numeric_columns] = rounded[numeric_columns].round(float_digits)

    headers = list(rounded.columns)
    rows = [[str(row[column]) for column in headers] for _, row in rounded.iterrows()]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def build_key_findings(frame: pd.DataFrame) -> list[str]:
    best_models = build_best_model_table(frame)
    group_averages = build_group_average_table(frame)
    findings: list[str] = []

    best_xau = best_models[best_models["symbol"] == "XAU/USD"].iloc[0]
    findings.append(
        "XAU/USD test tarafinda en guclu sonucu verdi; "
        f"en iyi model `{best_xau['model']}` ve test ROC-AUC degeri `{best_xau['roc_auc']:.4f}` oldu."
    )

    best_eth = best_models[best_models["symbol"] == "ETH/USD"].iloc[0]
    findings.append(
        "Kripto tarafinda ETH/USD, BTC/USD'den daha istikrarli davrandi; "
        f"ETH/USD icin en iyi test ROC-AUC `{best_eth['roc_auc']:.4f}` seviyesine ulasti."
    )

    equity_avg = group_averages[group_averages["asset_group"] == "equity"]
    if not equity_avg.empty:
        top_equity = equity_avg.loc[equity_avg["roc_auc"].idxmax()]
        findings.append(
            "Hisse senedi grubunda ROC-AUC halen sinirli; "
            f"en iyi ortalama test ROC-AUC `{top_equity['model']}` ile `{top_equity['roc_auc']:.4f}` oldu."
        )

    high_bias = frame[(frame["split"] == "test") & (frame["predicted_positive_rate"] >= 0.90)]
    if not high_bias.empty:
        biased_pairs = ", ".join(
            f"{row.symbol}-{row.model}" for row in high_bias[["symbol", "model"]].itertuples(index=False)
        )
        findings.append(
            "Bazi modeller sinif dengesine fazla yukari yonlu tepki veriyor; "
            f"testte predicted positive rate `>= 0.90` olan ciftler: {biased_pairs}."
        )

    return findings


def write_phase6_report(frame: pd.DataFrame, figure_paths: dict[str, Path]) -> Path:
    best_models = build_best_model_table(frame)
    group_averages = build_group_average_table(frame)
    bias_table = build_bias_table(frame)
    findings = build_key_findings(frame)

    report_lines = [
        "# Phase 6 Report",
        "",
        "Bu rapor, 5 varlik icin baseline, MLP, LSTM ve GRU modellerinin test/validation sonuclarini asset class bazinda ozetler.",
        "",
        "## Figures",
        "",
        f"- ROC-AUC chart: `{figure_paths['roc_auc']}`",
        f"- F1 chart: `{figure_paths['f1']}`",
        f"- Predicted positive rate chart: `{figure_paths['predicted_positive_rate']}`",
        "",
        "## Key Findings",
        "",
    ]

    for finding in findings:
        report_lines.append(f"- {finding}")

    report_lines.extend(
        [
            "",
            "## Best Model Per Asset (Test ROC-AUC)",
            "",
            markdown_table(best_models),
            "",
            "## Asset Group Average (Test)",
            "",
            markdown_table(group_averages),
            "",
            "## Test Bias Review",
            "",
            markdown_table(bias_table),
            "",
            "## Interpretation",
            "",
            "- ROC-AUC degerleri cogunlukla 0.47-0.56 bandinda kaldigi icin modellerin ayristirma gucu halen sinirli.",
            "- F1 ve accuracy bazi varliklarda yuksek gorunse de predicted positive rate kolonlari, bu skorlarin yukari yon agirlikli tahminlerden etkilendigini gosteriyor.",
            "- Faz 6 sonunda proje, coklu varlik ve coklu model karsilastirmasi yapabilen tekrar uretilebilir bir deney hattina donustu.",
        ]
    )

    output_path = REPORTS_DIR / "phase6_report.md"
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    return output_path


def main() -> None:
    ensure_directories()
    frame = load_summary_frame()

    roc_auc_path = FIGURES_DIR / "test_roc_auc_by_asset.png"
    f1_path = FIGURES_DIR / "test_f1_by_asset.png"
    positive_rate_path = FIGURES_DIR / "test_predicted_positive_rate_by_asset.png"

    save_grouped_bar_chart(frame, metric="roc_auc", output_path=roc_auc_path, title="Test ROC-AUC by Asset and Model")
    save_grouped_bar_chart(frame, metric="f1", output_path=f1_path, title="Test F1 by Asset and Model")
    save_grouped_bar_chart(
        frame,
        metric="predicted_positive_rate",
        output_path=positive_rate_path,
        title="Test Predicted Positive Rate by Asset and Model",
    )

    report_path = write_phase6_report(
        frame,
        figure_paths={
            "roc_auc": roc_auc_path,
            "f1": f1_path,
            "predicted_positive_rate": positive_rate_path,
        },
    )

    print(frame[frame["split"] == "test"][['asset_group', 'symbol', 'model', 'accuracy', 'f1', 'roc_auc']].to_string(index=False))
    print(f"\nPhase 6 report kaydedildi: {report_path}")


if __name__ == "__main__":
    main()
