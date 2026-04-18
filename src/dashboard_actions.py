from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .config import ROOT_DIR
except ImportError:
    from config import ROOT_DIR


STEP_LABELS = {
    "fetch_raw": "Ham veriyi cek",
    "build_processed": "Feature veri setini uret",
    "train_baseline": "Baseline egit",
    "train_mlp": "MLP egit",
    "train_lstm": "LSTM egit",
    "train_gru": "GRU egit",
    "summarize_results": "Model ozetini yenile",
    "backtest": "Backtest ozetini yenile",
    "threshold_tuning": "Threshold tuning yenile",
    "no_trade_total_return": "No-trade total return yenile",
    "no_trade_sharpe": "No-trade sharpe yenile",
    "rolling_retrain": "Rolling retraining backtest calistir",
}


def _symbol_args(symbols: list[str], use_all: bool) -> list[str]:
    if use_all:
        return ["--all"]
    if not symbols:
        return []
    return ["--symbols", *symbols]


def prepare_pipeline_commands(
    symbols: list[str],
    use_all: bool,
    steps: list[str],
    lstm_epochs: int = 12,
    gru_epochs: int = 12,
) -> list[dict[str, Any]]:
    symbol_args = _symbol_args(symbols, use_all)
    commands: list[dict[str, Any]] = []

    for step in steps:
        if step == "fetch_raw":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/build_raw_data.py", *symbol_args],
            })
        elif step == "build_processed":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/build_processed_data.py", *symbol_args],
            })
        elif step == "train_baseline":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/train_baseline.py", *symbol_args],
            })
        elif step == "train_mlp":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/train_mlp.py", *symbol_args],
            })
        elif step == "train_lstm":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/train_lstm.py", *symbol_args, "--epochs", str(lstm_epochs)],
            })
        elif step == "train_gru":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/train_gru.py", *symbol_args, "--epochs", str(gru_epochs)],
            })
        elif step == "summarize_results":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/summarize_results.py"],
            })
        elif step == "backtest":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/backtest_predictions.py"],
            })
        elif step == "threshold_tuning":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/run_threshold_tuning.py", *symbol_args],
            })
        elif step == "no_trade_total_return":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/run_no_trade_tuning.py", *symbol_args, "--optimize-for", "total_return"],
            })
        elif step == "no_trade_sharpe":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/run_no_trade_tuning.py", *symbol_args, "--optimize-for", "sharpe"],
            })
        elif step == "rolling_retrain":
            commands.append({
                "step": step,
                "label": STEP_LABELS[step],
                "args": [sys.executable, "src/rolling_retrain_backtest.py", *symbol_args, "--models", "baseline", "mlp", "--optimize-for", "total_return"],
            })
        else:
            raise ValueError(f"Bilinmeyen dashboard pipeline step: {step}")

    return commands


def run_pipeline(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    root_dir = Path(ROOT_DIR)

    for command in commands:
        started_at = time.perf_counter()
        completed = subprocess.run(
            command["args"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        duration_seconds = time.perf_counter() - started_at

        results.append(
            {
                "step": command["step"],
                "label": command["label"],
                "command": " ".join(command["args"]),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "duration_seconds": duration_seconds,
                "success": completed.returncode == 0,
            }
        )

        if completed.returncode != 0:
            break

    return results
