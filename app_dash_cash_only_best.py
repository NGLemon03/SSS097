from __future__ import annotations

import os
from typing import Any, Iterator

# Must be set before importing app_dash.
os.environ["SSS_DASH_CASH_ONLY_XRAY"] = "1"
os.environ["SSS_DASH_DISABLE_SMAA_HIGHLIGHT"] = "1"

import app_dash as base

CASH_ONLY_PRESET_DEFS = [
    (
        "cash_only_best_original_v1",
        "Cash-only Best (original)",
        "time_only",
    ),
    (
        "cash_only_best_trend_or_crash_off2_v1",
        "Cash-only Best (trend_or_crash_off2)",
        "trend_or_crash_off2",
    ),
    (
        "cash_only_best_crash_off_2d_v1",
        "Cash-only Best (crash_off_2d)",
        "crash_off_2d",
    ),
]
DEFAULT_CASH_ONLY_PRESET_KEY = "cash_only_best_original_v1"


def _ensure_cash_only_preset() -> None:
    source = dict(base.CRASH_OVERLAY_PRESETS.get("best_00631l_v1", {}))
    if not source:
        raise RuntimeError("best_00631l_v1 preset not found in app_dash.CRASH_OVERLAY_PRESETS")

    source["cap"] = 0.0
    source["reentry_min_hold"] = 1
    for preset_key, _, reentry_mode in CASH_ONLY_PRESET_DEFS:
        preset = dict(source)
        preset["reentry_mode"] = reentry_mode
        base.CRASH_OVERLAY_PRESETS[preset_key] = preset

    managed_keys = {preset_key for preset_key, _, _ in CASH_ONLY_PRESET_DEFS}
    options = [
        opt
        for opt in list(base.CRASH_OVERLAY_PRESET_OPTIONS)
        if str(opt.get("value")) not in managed_keys
    ]
    prepend = [{"label": label, "value": preset_key} for preset_key, label, _ in CASH_ONLY_PRESET_DEFS]
    base.CRASH_OVERLAY_PRESET_OPTIONS = prepend + options
    base.DEFAULT_CRASH_OVERLAY_PRESET = DEFAULT_CASH_ONLY_PRESET_KEY


def _iter_components(node: Any) -> Iterator[Any]:
    if node is None:
        return

    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _iter_components(child)
        return

    yield node
    children = getattr(node, "children", None)
    if children is not None:
        yield from _iter_components(children)


def _lock_controls(layout_root: Any) -> None:
    fixed_preset_options = [{"label": label, "value": preset_key} for preset_key, label, _ in CASH_ONLY_PRESET_DEFS]
    fixed_keys = {preset_key for preset_key, _, _ in CASH_ONLY_PRESET_DEFS}

    for comp in _iter_components(layout_root):
        comp_id = getattr(comp, "id", None)
        if comp_id == "crash-overlay-switch":
            setattr(comp, "value", True)
            setattr(comp, "disabled", True)
        elif comp_id == "crash-overlay-preset":
            current_value = getattr(comp, "value", None)
            if current_value not in fixed_keys:
                current_value = DEFAULT_CASH_ONLY_PRESET_KEY
            setattr(comp, "options", fixed_preset_options)
            setattr(comp, "value", current_value)
            setattr(comp, "clearable", False)
            setattr(comp, "disabled", False)
        elif comp_id == "auto-run":
            setattr(comp, "value", True)
            setattr(comp, "disabled", True)


_ensure_cash_only_preset()
_lock_controls(base.app.layout)

app = base.app
server = base.app.server


if __name__ == "__main__":
    base._initialize_app_logging()
    base.safe_startup()
    app.run_server(
        debug=True,
        host="127.0.0.1",
        port=8051,
        threaded=True,
        use_reloader=False,
    )
