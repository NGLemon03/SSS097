from __future__ import annotations

import os

from app.settings import LOG_LEVEL


def configure_runtime_environment() -> None:
    """Normalize env vars used across the app before logger initialization."""
    os.environ.setdefault("SSS_CREATE_LOGS", "1")
    os.environ.setdefault("SSS_LOG_LEVEL", LOG_LEVEL)


def create_app():
    """Backward-compatible app factory."""
    import app_dash

    return app_dash.app
