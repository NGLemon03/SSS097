#!/usr/bin/env python3
"""Deprecated compatibility wrapper.

Use run_full_pipeline_5.py as the official mainline entrypoint.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Deprecated wrapper:\n"
            "  python auto_pipeline.py [args]\n\n"
            "Official entrypoint:\n"
            "  python run_full_pipeline_5.py"
        )
        return 0

    print(
        "[DEPRECATED] auto_pipeline.py is now a compatibility wrapper.\n"
        "Please use: python run_full_pipeline_5.py"
    )
    cmd = [sys.executable, "run_full_pipeline_5.py", *sys.argv[1:]]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
