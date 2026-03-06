from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(src: Path, cwd: Path) -> Path:
    try:
        return src.resolve().relative_to(cwd.resolve())
    except ValueError:
        drive = src.drive.replace(":", "") if src.drive else "unknown_drive"
        tail = Path(src.as_posix().lstrip("/"))
        return Path("__external__") / drive / tail


def freeze_files(files: list[str], tag: str = "", out_dir: str = "backups") -> dict[str, Any]:
    cwd = Path.cwd().resolve()
    resolved: list[Path] = []
    missing: list[str] = []

    for raw in files:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (cwd / candidate).resolve()
        if not candidate.exists() or not candidate.is_file():
            missing.append(raw)
            continue
        resolved.append(candidate)

    if missing:
        raise FileNotFoundError(f"Missing input files: {', '.join(missing)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_root = (Path(out_dir) / timestamp).resolve()
    snapshot_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for src in resolved:
        rel_path = _safe_relative_path(src, cwd)
        backup_path = snapshot_root / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, backup_path)

        stat = src.stat()
        entries.append(
            {
                "path": rel_path.as_posix(),
                "backup_path": backup_path.relative_to(cwd).as_posix()
                if backup_path.is_relative_to(cwd)
                else backup_path.as_posix(),
                "sha256": _sha256_file(src),
                "size": stat.st_size,
                "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    manifest = {
        "timestamp": timestamp,
        "tag": tag,
        "entries": entries,
    }
    manifest_path = snapshot_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "snapshot_root": snapshot_root,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create timestamped file backups with manifest metadata.")
    parser.add_argument("--files", nargs="+", required=True, help="Target files to back up.")
    parser.add_argument("--tag", default="", help="Optional task tag.")
    parser.add_argument("--out-dir", default="backups", help="Output root directory for snapshots.")
    args = parser.parse_args(argv)

    try:
        result = freeze_files(files=args.files, tag=args.tag, out_dir=args.out_dir)
    except Exception as exc:  # pragma: no cover - exercised by CLI behavior
        print(f"[freeze_snapshot] ERROR: {exc}")
        return 1

    snapshot_root = result["snapshot_root"]
    manifest_path = result["manifest_path"]
    count = len(result["manifest"]["entries"])
    print(f"[freeze_snapshot] snapshot_root={snapshot_root}")
    print(f"[freeze_snapshot] manifest={manifest_path}")
    print(f"[freeze_snapshot] files_backed_up={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
