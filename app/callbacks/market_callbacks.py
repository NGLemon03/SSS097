from __future__ import annotations

from dash import Input, Output, no_update

from app.services.market_service import build_price_update_payload, refresh_market_data


def register_market_callbacks(app, *, tickers: list[str]) -> None:
    @app.callback(
        Output("price-update-store", "data"),
        Input("price-update-interval", "n_intervals"),
    )
    def auto_update_prices(_n):
        updated = refresh_market_data(tickers)
        payload = build_price_update_payload(updated)
        if payload:
            return payload
        return no_update
