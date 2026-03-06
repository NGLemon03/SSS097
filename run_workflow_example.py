#!/usr/bin/env python3
"""Deprecated compatibility wrapper.

Use run_enhanced_ensemble.py as the official example workflow entrypoint.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Deprecated wrapper:\n"
            "  python run_workflow_example.py [args]\n\n"
            "Official entrypoint:\n"
            "  python run_enhanced_ensemble.py --method majority --top_k 5"
        )
        return 0

    print(
        "[DEPRECATED] run_workflow_example.py is now a compatibility wrapper.\n"
        "Please use: python run_enhanced_ensemble.py --method majority --top_k 5"
    )
    cmd = [sys.executable, "run_enhanced_ensemble.py", *sys.argv[1:]]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
