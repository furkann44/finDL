from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ALL_SYMBOLS, BACKTESTS_DIR, FIGURES_DIR, PREDICTIONS_DIR, REPORTS_DIR, backtest_chart_path, backtest_equity_path, backtest_summary_path, ensure_directories


TRADING_DAYS = 252


def load_prediction_files(pattern: str = "*_test_predictions.csv") -> list[Path]:
    files = sorted(PREDICTIONS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Backtest icin prediction dosyasi bulunamadi: {PREDICTIONS_DIR} | pattern={pattern}")
    return files


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1
    return float(drawdown.min())


def annualized_return(total_return: float, periods: int) -> float:
    if periods <= 0 or total_return <= -1:
        return 0.0
    return float((1 + total_return) ** (TRADING_DAYS / periods) - 1)


def run_backtest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    backtest_frame = frame.copy().sort_values("datetime").reset_index(drop=True)
    if "signal" in backtest_frame.columns:
        backtest_frame["signal"] = backtest_frame["signal"].astype(float)
    else:
        backtest_frame["signal"] = np.where(backtest_frame["prediction"] == 1, 1.0, -1.0)

    backtest_frame["active_trade"] = (backtest_frame["signal"] != 0).astype(int)
    backtest_frame["strategy_return"] = backtest_frame["signal"] * backtest_frame["future_return_1"]
    backtest_frame["buy_hold_return"] = backtest_frame["future_return_1"]
    backtest_frame["strategy_equity"] = (1 + backtest_frame["strategy_return"]).cumprod()
    backtest_frame["buy_hold_equity"] = (1 + backtest_frame["buy_hold_return"]).cumprod()

    total_return = float(backtest_frame["strategy_equity"].iloc[-1] - 1)
    benchmark_return = float(backtest_frame["buy_hold_equity"].iloc[-1] - 1)
    volatility = float(backtest_frame["strategy_return"].std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = 0.0
    if backtest_frame["strategy_return"].std(ddof=0) > 0:
        sharpe = float(
            backtest_frame["strategy_return"].mean() / backtest_frame["strategy_return"].std(ddof=0) * np.sqrt(TRADING_DAYS)
        )

    summary = {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "annualized_return": annualized_return(total_return, len(backtest_frame)),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": compute_max_drawdown(backtest_frame["strategy_equity"]),
        "win_rate": float((backtest_frame["strategy_return"] > 0).mean()),
        "coverage": float(backtest_frame["active_trade"].mean()),
    }
    return backtest_frame, summary


def save_equity_chart(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    chart_frame = frame.copy()
    chart_frame["datetime"] = pd.to_datetime(chart_frame["datetime"], utc=True, errors="coerce")
    plt.figure(figsize=(9, 4.5))
    plt.plot(chart_frame["datetime"], chart_frame["strategy_equity"], label="strategy")
    plt.plot(chart_frame["datetime"], chart_frame["buy_hold_equity"], label="buy_hold")
    plt.title(title)
    plt.xlabel("datetime")
    plt.ylabel("equity")
    plt.legend()
    plt.grid(alpha=0.3)
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


def main() -> None:
    ensure_directories()
    import argparse

    parser = argparse.ArgumentParser(description="Prediction dosyalari uzerinde basit backtest calistir.")
    parser.add_argument("--pattern", default="*_test_predictions.csv", help="Prediction dosyasi glob paterni")
    args = parser.parse_args()

    prediction_files = load_prediction_files(pattern=args.pattern)
    summary_rows: list[dict[str, object]] = []

    for prediction_file in prediction_files:
        frame = pd.read_csv(prediction_file)
        symbol = str(frame["symbol"].iloc[0])
        if symbol not in ALL_SYMBOLS:
            continue
        model_name = str(frame["model"].iloc[0])
        backtest_frame, summary = run_backtest(frame)

        equity_path = backtest_equity_path(symbol, model_name)
        chart_path = backtest_chart_path(symbol, model_name)
        backtest_frame.to_csv(equity_path, index=False)
        save_equity_chart(backtest_frame, chart_path, title=f"{symbol} {model_name.upper()} Strategy vs Buy and Hold")

        summary_rows.append(
            {
                "symbol": symbol,
                "model": model_name,
                **summary,
                "equity_path": str(equity_path),
                "chart_path": str(chart_path),
            }
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values(["symbol", "model"]).reset_index(drop=True)
    summary_frame.to_csv(backtest_summary_path(), index=False)

    report_lines = [
        "# Backtest Report",
        "",
        "Bu rapor, test prediction dosyalari uzerinde basit long-short yon stratejisinin sonuclarini ozetler.",
        "",
        markdown_table(summary_frame[["symbol", "model", "coverage", "total_return", "benchmark_return", "annualized_return", "sharpe", "max_drawdown", "win_rate"]]),
        "",
        "Not: Bu backtest maliyet, slipaj ve execution etkilerini icermez; yalnizca hizli bir mantik kontroludur.",
    ]
    report_path = REPORTS_DIR / "backtest_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(summary_frame[["symbol", "model", "coverage", "total_return", "benchmark_return", "annualized_return", "sharpe", "max_drawdown", "win_rate"]].to_string(index=False))
    print(f"\nBacktest report kaydedildi: {report_path}")


if __name__ == "__main__":
    main()
