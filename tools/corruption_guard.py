from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SIGN_BROKEN_ENCODING = "broken_encoding_mojibake"
SIGN_UNTERMINATED = "unterminated_strings_docstrings"
SIGN_COMPILE_SPREAD = "compile_error_spread_risk"
SIGN_UNRELATED_REPLACE = "unrelated_mass_text_replacement_risk"
SIGN_UI_STRING_PATCH = "ui_doc_logger_syntax_patch_pattern"

DEFAULT_BASELINE_ROOTS = [".", ".history", "backups"]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "node_modules"}
GARBLED_GLYPHS = "蝑憭敹鞈頛瘥閮瑼撘摰鈭雿璅皜"


@dataclass
class FileScan:
    path: Path
    text: str
    line_count: int
    replacement_count: int
    double_question_count: int
    garbled_glyph_hits: int
    odd_quote_lines: list[int]
    compile_ok: bool
    compile_error: str | None
    compile_error_line: int | None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


def _find_odd_quote_lines(lines: list[str]) -> list[int]:
    odd_lines: list[int] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if '"""' in line or "'''" in line:
            continue
        double_odd = (line.count('"') % 2) == 1
        single_odd = (line.count("'") % 2) == 1 and '"' not in line
        if double_odd or single_odd:
            odd_lines.append(idx)
    return odd_lines


def _compile_check(path: Path, text: str) -> tuple[bool, str | None, int | None]:
    normalized = text.lstrip("\ufeff")
    try:
        ast.parse(normalized, filename=str(path))
        return True, None, None
    except SyntaxError as exc:
        return False, str(exc.msg), int(exc.lineno) if exc.lineno else None
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}", None


def _scan_file(path: Path) -> FileScan:
    text = _read_text(path)
    lines = text.splitlines()
    compile_ok, compile_error, compile_error_line = _compile_check(path, text) if path.suffix == ".py" else (True, None, None)
    return FileScan(
        path=path,
        text=text,
        line_count=max(1, len(lines)),
        replacement_count=text.count("\ufffd"),
        double_question_count=text.count("??"),
        garbled_glyph_hits=sum(text.count(ch) for ch in GARBLED_GLYPHS),
        odd_quote_lines=_find_odd_quote_lines(lines) if path.suffix == ".py" else [],
        compile_ok=compile_ok,
        compile_error=compile_error,
        compile_error_line=compile_error_line,
    )


def _is_unterminated_error(message: str | None) -> bool:
    if not message:
        return False
    lowered = message.lower()
    markers = [
        "unterminated",
        "eol while scanning string literal",
        "triple-quoted string literal",
    ]
    return any(marker in lowered for marker in markers)


def _is_text_only_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.startswith(('"""', "'''", '"', "'")):
        return True
    if ("logger." in stripped or "st." in stripped or "html." in stripped) and (stripped.count('"') >= 2 or stripped.count("'") >= 2):
        return True
    return False


def _is_ui_logger_string_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    has_token = any(token in stripped for token in ("logger.", "st.", "dbc.", "html.", "docstring", "markdown"))
    has_stringy = ('"""' in stripped or "'''" in stripped or stripped.count('"') >= 2 or stripped.count("'") >= 2)
    return has_token and has_stringy


def _collect_candidate_paths(target: Path, baseline_roots: list[str], max_candidates: int = 40) -> list[Path]:
    target_resolved = target.resolve()
    found: list[Path] = []
    seen: set[Path] = set()

    for root_raw in baseline_roots:
        root = Path(root_raw)
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if target.name not in filenames:
                continue
            candidate = (Path(dirpath) / target.name).resolve()
            if candidate == target_resolved or candidate in seen:
                continue
            seen.add(candidate)
            found.append(candidate)
            if len(found) >= max_candidates:
                return found
    return found


def _candidate_score(target: Path, candidate: Path) -> float:
    try:
        t_stat = target.stat()
        c_stat = candidate.stat()
        size_gap = abs(t_stat.st_size - c_stat.st_size)
        newer_penalty = 0 if c_stat.st_mtime <= t_stat.st_mtime else 100000
        depth_penalty = len(candidate.parts)
        return float(size_gap + newer_penalty + depth_penalty)
    except OSError:
        return float("inf")


def _analyze_diff(current: str, baseline: str) -> dict[str, Any]:
    current_lines = current.splitlines()
    baseline_lines = baseline.splitlines()
    matcher = difflib.SequenceMatcher(None, baseline_lines, current_lines)

    changed_indices: set[int] = set()
    changed_lines: list[str] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_indices.update(range(j1, j2))
        changed_lines.extend(current_lines[j1:j2])

    changed_ratio = len(changed_indices) / max(1, len(current_lines))
    text_only_count = sum(1 for line in changed_lines if _is_text_only_line(line))
    text_only_ratio = text_only_count / max(1, len(changed_lines))
    ui_logger_hits = sum(1 for line in changed_lines if _is_ui_logger_string_line(line))

    return {
        "changed_ratio": changed_ratio,
        "changed_line_count": len(changed_indices),
        "text_only_ratio": text_only_ratio,
        "text_only_count": text_only_count,
        "ui_logger_string_hits": ui_logger_hits,
    }


def _build_report(file_paths: list[Path], threshold: int, baseline_roots: list[str]) -> dict[str, Any]:
    scans = [_scan_file(path) for path in file_paths]

    replacement_total = sum(scan.replacement_count for scan in scans)
    qq_total = sum(scan.double_question_count for scan in scans)
    garbled_total = sum(scan.garbled_glyph_hits for scan in scans)
    line_total = max(1, sum(scan.line_count for scan in scans))
    odd_quote_total = sum(len(scan.odd_quote_lines) for scan in scans)
    syntax_error_count = sum(0 if scan.compile_ok else 1 for scan in scans if scan.path.suffix == ".py")
    unterminated_count = sum(1 for scan in scans if _is_unterminated_error(scan.compile_error))

    sign_scores: dict[str, Any] = {
        SIGN_BROKEN_ENCODING: {
            "replacement_count": replacement_total,
            "double_question_count": qq_total,
            "garbled_glyph_hits": garbled_total,
            "line_total": line_total,
        },
        SIGN_UNTERMINATED: {
            "unterminated_error_count": unterminated_count,
            "odd_quote_line_count": odd_quote_total,
        },
        SIGN_COMPILE_SPREAD: {
            "syntax_error_count": syntax_error_count,
            "odd_quote_line_count": odd_quote_total,
        },
        SIGN_UNRELATED_REPLACE: {
            "triggered_files": [],
        },
        SIGN_UI_STRING_PATCH: {
            "triggered_files": [],
        },
    }

    broken_encoding = (
        replacement_total > 0
        or qq_total >= max(8, line_total // 20)
        or garbled_total >= max(12, line_total // 5)
    )
    unterminated = (unterminated_count > 0 or odd_quote_total >= 5)
    compile_spread = syntax_error_count > 0 and (syntax_error_count >= 2 or odd_quote_total >= 2 or broken_encoding)

    baseline_candidates: list[dict[str, Any]] = []
    reliable_baseline_exists = False
    sign4 = False
    sign5 = False
    drift_reasons: list[str] = []

    for scan in scans:
        candidates = _collect_candidate_paths(scan.path, baseline_roots)
        if not candidates:
            continue

        ordered = sorted(candidates, key=lambda candidate: _candidate_score(scan.path, candidate))
        chosen = ordered[0]
        candidate_text = _read_text(chosen)
        candidate_compile_ok, _, _ = _compile_check(chosen, candidate_text) if chosen.suffix == ".py" else (True, None, None)
        candidate_broken = (
            candidate_text.count("\ufffd") > 0
            or candidate_text.count("??") >= 8
            or sum(candidate_text.count(ch) for ch in GARBLED_GLYPHS) >= 12
        )
        candidate_reliable = candidate_compile_ok and not candidate_broken
        if candidate_reliable:
            reliable_baseline_exists = True

        diff_metrics = _analyze_diff(scan.text, candidate_text)
        this_sign4 = diff_metrics["changed_ratio"] >= 0.35 and diff_metrics["text_only_ratio"] >= 0.50
        this_sign5 = (
            diff_metrics["changed_ratio"] >= 0.25
            and diff_metrics["text_only_ratio"] >= 0.65
            and diff_metrics["ui_logger_string_hits"] >= 5
        )
        sign4 = sign4 or this_sign4
        sign5 = sign5 or this_sign5

        if this_sign4:
            sign_scores[SIGN_UNRELATED_REPLACE]["triggered_files"].append(scan.path.as_posix())
            drift_reasons.append(f"{scan.path.as_posix()}: high changed_ratio with text-only dominance")
        if this_sign5:
            sign_scores[SIGN_UI_STRING_PATCH]["triggered_files"].append(scan.path.as_posix())
            drift_reasons.append(f"{scan.path.as_posix()}: ui/logger/string-only patch pattern")

        baseline_candidates.append(
            {
                "target": scan.path.as_posix(),
                "candidate": chosen.as_posix(),
                "is_reliable": candidate_reliable,
                "compile_ok": candidate_compile_ok,
                "score": _candidate_score(scan.path, chosen),
                "diff_metrics": diff_metrics,
            }
        )

    triggered_signs: list[str] = []
    if broken_encoding:
        triggered_signs.append(SIGN_BROKEN_ENCODING)
    if unterminated:
        triggered_signs.append(SIGN_UNTERMINATED)
    if compile_spread:
        triggered_signs.append(SIGN_COMPILE_SPREAD)
    if sign4:
        triggered_signs.append(SIGN_UNRELATED_REPLACE)
    if sign5:
        triggered_signs.append(SIGN_UI_STRING_PATCH)

    is_contaminated = len(triggered_signs) >= threshold

    if is_contaminated and reliable_baseline_exists:
        recommended_route = "rollback_then_reapply"
    elif is_contaminated and not reliable_baseline_exists and len(triggered_signs) >= 3:
        recommended_route = "rebuild_clean_file"
    else:
        recommended_route = "local_patch"

    compile_checks = [
        {
            "path": scan.path.as_posix(),
            "compile_ok": scan.compile_ok,
            "compile_error": scan.compile_error,
            "compile_error_line": scan.compile_error_line,
            "odd_quote_lines": scan.odd_quote_lines,
        }
        for scan in scans
        if scan.path.suffix == ".py"
    ]

    task_drift_risk = {
        "is_drift_risk": sign4 or sign5,
        "reasons": drift_reasons,
    }

    four_answers = {
        "q1_file_contaminated": is_contaminated,
        "q2_task_drifted": bool(task_drift_risk["is_drift_risk"]),
        "q3_has_reliable_baseline": reliable_baseline_exists,
        "q4_recommended_route": recommended_route,
    }

    return {
        "is_contaminated": is_contaminated,
        "triggered_signs": triggered_signs,
        "sign_scores": sign_scores,
        "compile_checks": compile_checks,
        "baseline_candidates": baseline_candidates,
        "task_drift_risk": task_drift_risk,
        "recommended_route": recommended_route,
        "four_answers": four_answers,
    }


def _render_markdown(report: dict[str, Any], files: list[Path], threshold: int, baseline_roots: list[str]) -> str:
    lines = [
        "# Corruption Guard Report",
        "",
        f"- Files: {', '.join(path.as_posix() for path in files)}",
        f"- Threshold: {threshold}",
        f"- Baseline roots: {', '.join(baseline_roots)}",
        "",
        "## Summary",
        f"- is_contaminated: `{report['is_contaminated']}`",
        f"- triggered_signs: `{', '.join(report['triggered_signs']) if report['triggered_signs'] else 'none'}`",
        f"- recommended_route: `{report['recommended_route']}`",
        "",
        "## Four Answers",
    ]
    for key, value in report["four_answers"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Compile Checks"])
    for item in report["compile_checks"]:
        lines.append(
            f"- {item['path']}: compile_ok={item['compile_ok']}, "
            f"error={item['compile_error']}, line={item['compile_error_line']}, odd_quote_lines={item['odd_quote_lines']}"
        )

    lines.extend(["", "## Baseline Candidates"])
    if not report["baseline_candidates"]:
        lines.append("- none")
    else:
        for item in report["baseline_candidates"]:
            lines.append(
                f"- target={item['target']}, candidate={item['candidate']}, "
                f"is_reliable={item['is_reliable']}, score={item['score']:.2f}"
            )

    return "\n".join(lines) + "\n"


def _write_optional(path_raw: str | None, content: str) -> None:
    if not path_raw:
        return
    path = Path(path_raw)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect likely file corruption and route recovery strategy.")
    parser.add_argument("--files", nargs="+", required=True, help="Target files to analyze.")
    parser.add_argument("--threshold", type=int, default=2, help="Triggered signs needed for contamination.")
    parser.add_argument("--report-md", default=None, help="Optional markdown report path.")
    parser.add_argument("--report-json", default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--baseline-roots",
        nargs="*",
        default=DEFAULT_BASELINE_ROOTS,
        help="Candidate baseline roots, searched in order.",
    )
    args = parser.parse_args(argv)

    file_paths: list[Path] = []
    missing: list[str] = []
    for raw in args.files:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.exists() or not candidate.is_file():
            missing.append(raw)
            continue
        file_paths.append(candidate)

    if missing:
        print(f"[corruption_guard] ERROR: missing files: {', '.join(missing)}")
        return 1

    report = _build_report(file_paths=file_paths, threshold=args.threshold, baseline_roots=args.baseline_roots)
    md = _render_markdown(report, file_paths, args.threshold, args.baseline_roots)
    json_payload = json.dumps(report, indent=2, ensure_ascii=False)

    _write_optional(args.report_md, md)
    _write_optional(args.report_json, json_payload)

    print(f"[corruption_guard] is_contaminated={report['is_contaminated']}")
    print(f"[corruption_guard] triggered_signs={report['triggered_signs']}")
    print(f"[corruption_guard] recommended_route={report['recommended_route']}")
    print(f"[corruption_guard] four_answers={report['four_answers']}")
    return 2 if report["is_contaminated"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
