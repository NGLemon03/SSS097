from __future__ import annotations

from dash import Input, Output, State


def register_ui_callbacks(
    app,
    *,
    toggle_advanced_settings_func,
    update_warehouse_list_func,
) -> None:
    @app.callback(
        Output("collapse-settings", "is_open"),
        [Input("collapse-button", "n_clicks")],
        [State("collapse-settings", "is_open")],
    )
    def _toggle_advanced_settings(n, is_open):
        return toggle_advanced_settings_func(n, is_open)

    @app.callback(
        Output("warehouse-dropdown", "options"),
        Output("warehouse-dropdown", "value"),
        Input("run-btn", "n_clicks"),
    )
    def _update_warehouse_list(n):
        return update_warehouse_list_func(n)
