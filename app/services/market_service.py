from __future__ import annotations

from datetime import datetime
from typing import Sequence

from analysis.logging_config import get_logger
from app.settings import MARKET_UPDATE_TICKERS
from sss_core.logic import update_prices_if_needed

logger = get_logger("SSS.App")


def refresh_market_data(
    tickers: Sequence[str] | None = None,
    *,
    clear_cache: bool = False,
) -> bool:
    ticker_list = list(dict.fromkeys(tickers or MARKET_UPDATE_TICKERS))
    updated = update_prices_if_needed(ticker_list, clear_cache=clear_cache)
    if updated:
        logger.info("股價資料已更新，將觸發回測重算")
    else:
        logger.debug("股價資料無需更新")
    return updated


def build_price_update_payload(updated: bool) -> dict | None:
    if not updated:
        return None
    return {"updated_at": datetime.now().isoformat()}
