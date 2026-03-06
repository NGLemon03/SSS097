from __future__ import annotations

from dash import Input, Output


def register_process_callbacks(app, *, run_prediction_script_func) -> None:
    @app.callback(
        Output("prediction-status-msg", "children"),
        Input("btn-run-prediction", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_prediction_script_callback(n_clicks):
        msg, _refresh = run_prediction_script_func(n_clicks)
        return msg
