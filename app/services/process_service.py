from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import subprocess
import sys


@dataclass(frozen=True)
class PredictionRunResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    timeout_seconds: int = 0
    error: str = ""


def build_prediction_command(
    ticker: str = "00631L.TW",
    warehouse_file: str = "strategy_warehouse.json",
) -> list[str]:
    cmd = [sys.executable, "predict_tomorrow.py"]
    if ticker:
        cmd.extend(["--ticker", str(ticker)])
    if warehouse_file:
        cmd.extend(["--warehouse-file", str(warehouse_file)])
    return cmd


def run_prediction_job(
    ticker: str = "00631L.TW",
    warehouse_file: str = "strategy_warehouse.json",
    *,
    timeout: int = 120,
    run_func: Callable[..., object] = subprocess.run,
) -> PredictionRunResult:
    cmd = build_prediction_command(ticker=ticker, warehouse_file=warehouse_file)

    try:
        completed = run_func(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return PredictionRunResult(
            returncode=1,
            timed_out=True,
            timeout_seconds=timeout,
            error=f"Prediction timed out after {timeout}s.",
        )
    except Exception as exc:  # pragma: no cover - defensive branch
        return PredictionRunResult(returncode=1, error=str(exc))

    return PredictionRunResult(
        returncode=int(getattr(completed, "returncode", 1)),
        stdout=str(getattr(completed, "stdout", "") or ""),
        stderr=str(getattr(completed, "stderr", "") or ""),
    )
