from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, ensure_directories, sanitize_symbol
from dashboard_data import build_recommendation_table, load_rolling_retrain_summary


def direction_label(action: str) -> str:
    if action == "long":
        return "Yukari"
    if action == "short":
        return "Asagi"
    return "Islem Yok"


def build_recent_signal_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    recommendations = build_recommendation_table()
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    window_end = pd.Timestamp.now(tz="UTC").normalize()
    window_start = window_end - pd.Timedelta(days=10)

    for row in recommendations.itertuples(index=False):
        prediction_path = PREDICTIONS_DIR / f"{sanitize_symbol(row.symbol)}_{row.recommended_model}_test_no_trade_predictions.csv"
        if not prediction_path.exists():
            continue

        prediction_frame = pd.read_csv(prediction_path)
        prediction_frame["datetime"] = pd.to_datetime(prediction_frame["datetime"], utc=True)
        prediction_frame = (
            prediction_frame[
                (prediction_frame["datetime"] >= window_start)
                & (prediction_frame["datetime"] < window_end)
            ]
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        if prediction_frame.empty:
            continue

        prediction_frame["datetime"] = prediction_frame["datetime"].astype(str)
        prediction_frame["signal_direction"] = prediction_frame["action"].map(direction_label)
        prediction_frame["realized_direction"] = prediction_frame["y_true"].map({1: "Yukari", 0: "Asagi"})
        prediction_frame["is_correct"] = prediction_frame.apply(
            lambda record: None if int(record["active_trade"]) == 0 else int(record["predicted_class"] == record["y_true"]),
            axis=1,
        )
        prediction_frame["symbol"] = row.symbol
        prediction_frame["model"] = row.recommended_model
        detail_frames.append(prediction_frame)

        active_frame = prediction_frame[prediction_frame["active_trade"] == 1].copy()
        active_hit_rate = None if active_frame.empty else float(active_frame["is_correct"].mean())
        latest_row = prediction_frame.iloc[-1]

        summary_rows.append(
            {
                "symbol": row.symbol,
                "model": row.recommended_model,
                "window_start": str(window_start)[:10],
                "window_end": str((window_end - pd.Timedelta(days=1)))[:10],
                "observed_start": str(prediction_frame.iloc[0]["datetime"])[:10],
                "observed_end": str(prediction_frame.iloc[-1]["datetime"])[:10],
                "observation_count": int(len(prediction_frame)),
                "active_trade_days": int(active_frame.shape[0]),
                "up_signals": int((prediction_frame["action"] == "long").sum()),
                "down_signals": int((prediction_frame["action"] == "short").sum()),
                "no_trade_signals": int((prediction_frame["action"] == "no_trade").sum()),
                "active_hit_rate": active_hit_rate,
                "latest_signal": latest_row["signal_direction"],
                "latest_probability": float(latest_row["probability"]),
            }
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values("symbol").reset_index(drop=True)
    detail_frame = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary_frame, detail_frame


def save_holdout_chart(recommendations: pd.DataFrame) -> Path:
    chart_frame = recommendations[["symbol", "holdout_roc_auc"]].copy().sort_values("symbol")
    ax = chart_frame.plot(x="symbol", y="holdout_roc_auc", kind="bar", legend=False, figsize=(9, 4.8), color="#3366cc")
    ax.set_title("Varlik Bazinda En Iyi Holdout ROC-AUC")
    ax.set_xlabel("Varlik")
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path = FIGURES_DIR / "final_holdout_best_roc_auc.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_no_trade_return_chart(recommendations: pd.DataFrame) -> Path:
    chart_frame = recommendations[["symbol", "recommended_total_return"]].copy().sort_values("symbol")
    ax = chart_frame.plot(x="symbol", y="recommended_total_return", kind="bar", legend=False, figsize=(9, 4.8), color="#16a34a")
    ax.set_title("Varlik Bazinda Onerilen Model Total Return")
    ax.set_xlabel("Varlik")
    ax.set_ylabel("Total Return")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path = FIGURES_DIR / "final_recommended_total_return.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_recent_signal_distribution_chart(summary_frame: pd.DataFrame) -> Path:
    chart_frame = summary_frame[["symbol", "up_signals", "down_signals", "no_trade_signals"]].copy().set_index("symbol")
    ax = chart_frame.plot(kind="bar", stacked=True, figsize=(9, 4.8), color=["#2563eb", "#ef4444", "#9ca3af"])
    ax.set_title("Son 10 Gunde Sinyal Dagilimi")
    ax.set_xlabel("Varlik")
    ax.set_ylabel("Gun Sayisi")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path = FIGURES_DIR / "final_recent_signal_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_rolling_comparison_chart(recommendations: pd.DataFrame) -> Path:
    frame = recommendations[pd.notna(recommendations["best_rolling_total_return"])].copy()
    if frame.empty:
        output_path = FIGURES_DIR / "final_rolling_vs_recommended.png"
        plt.figure(figsize=(9, 4.8))
        plt.title("Rolling veri bulunamadi")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    frame = frame[["symbol", "recommended_total_return", "best_rolling_total_return"]].sort_values("symbol")
    plot_frame = frame.set_index("symbol")
    ax = plot_frame.plot(kind="bar", figsize=(9, 4.8), color=["#16a34a", "#f59e0b"])
    ax.set_title("Statik Oneri ve Rolling Retraining Karsilastirmasi")
    ax.set_xlabel("Varlik")
    ax.set_ylabel("Total Return")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path = FIGURES_DIR / "final_rolling_vs_recommended.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main() -> None:
    ensure_directories()
    recommendations = build_recommendation_table()
    recommendations.to_csv(METRICS_DIR / "final_presentation_recommendations.csv", index=False)

    recent_summary, recent_detail = build_recent_signal_tables()
    recent_summary.to_csv(METRICS_DIR / "final_recent_signal_summary.csv", index=False)
    if not recent_detail.empty:
        recent_detail.to_csv(METRICS_DIR / "final_recent_signal_examples.csv", index=False)

    save_holdout_chart(recommendations)
    save_no_trade_return_chart(recommendations)
    if not recent_summary.empty:
        save_recent_signal_distribution_chart(recent_summary)
    save_rolling_comparison_chart(recommendations)

    print(recommendations[["symbol", "best_holdout_model", "recommended_model", "recommended_total_return"]].to_string(index=False))
    if not recent_summary.empty:
        print("\nSon 10 gun sinyal ozeti:")
        print(recent_summary.to_string(index=False))


if __name__ == "__main__":
    main()
