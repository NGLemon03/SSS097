from __future__ import annotations

import logging
import os


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def resolve_log_level(default: str = "INFO") -> str:
    candidate = (os.getenv("LOG_LEVEL") or os.getenv("SSS_LOG_LEVEL") or default).upper()
    return candidate if candidate in logging._nameToLevel else default.upper()


LOG_LEVEL = resolve_log_level()
PRICE_UPDATE_INTERVAL_MS = _read_int_env("PRICE_UPDATE_INTERVAL_MS", 60 * 60 * 1000)

# Keep compatibility with current UI default behavior.
DEFAULT_TICKERS = (
    "00631L.TW",
    "2330.TW",
    "00663L.TW",
    "00663L.TW",
    "00675L.TW",
    "00685L.TW",
)

# Keep 0050.TW in the unified market-refresh path for Smart Leverage.
MARKET_UPDATE_TICKERS = ("0050.TW", "2330.TW", "2412.TW", "2414.TW", "^TWII")
