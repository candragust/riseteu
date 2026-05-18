#!/usr/bin/env python3
"""Move TVT markers in HTML titles to the final suffix form.

Examples:
  EURUSD FLF-BiLSTM TVT Pipeline Summary
  -> EURUSD FLF-BiLSTM Pipeline Summary (TVT)

  EUR/USD H4 TVT v02 FLF-BiLSTM Last3 - OHLC Dot Plot
  -> EUR/USD H4 FLF-BiLSTM Last3 - OHLC Dot Plot (TVT v02)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TVT_RE = re.compile(r"\bTVT(?:[_\s-]*v?0?2)?\b|\bTVTv?0?2\b", flags=re.I)
TVT_V02_RE = re.compile(r"\bTVT(?:[_\s-]*v?0?2)\b|\bTVTv?0?2\b", flags=re.I)


def normalize_title(text: str) -> str:
    if not TVT_RE.search(text):
        return text
    suffix = "(TVT v02)" if TVT_V02_RE.search(text) else "(TVT)"
    normalized = text
    while re.search(r"\s*\((?:TVT(?:[_\s-]*v?0?2)?|TVTv?0?2)\)\s*$", normalized, flags=re.I):
        normalized = re.sub(
            r"\s*\((?:TVT(?:[_\s-]*v?0?2)?|TVTv?0?2)\)\s*$",
            "",
            normalized,
            flags=re.I,
        )
    normalized = re.sub(r"\s+\bTVT[_\s-]*v?0?2\b", " ", normalized, flags=re.I)
    normalized = re.sub(r"\s+\bTVTv?0?2\b", " ", normalized, flags=re.I)
    normalized = re.sub(r"\s+\bTVT\b", " ", normalized, flags=re.I)
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()
    return f"{normalized} {suffix}"


def normalize_html_text_tag(html: str, tag: str) -> tuple[str, int]:
    pattern = re.compile(rf"(<{tag}\b[^>]*>)(.*?)(</{tag}>)", flags=re.I | re.S)
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        content = match.group(2)
        if "<" in content or ">" in content:
            return match.group(0)
        normalized = normalize_title(content.strip())
        if normalized == content.strip():
            return match.group(0)
        count += 1
        leading = content[: len(content) - len(content.lstrip())]
        trailing = content[len(content.rstrip()) :]
        return f"{match.group(1)}{leading}{normalized}{trailing}{match.group(3)}"

    return pattern.sub(repl, html), count


def normalize_plotly_title_text(html: str) -> tuple[str, int]:
    pattern = re.compile(r'("title"\s*:\s*\{\s*"text"\s*:\s*)("(?:\\.|[^"\\])*")')
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        raw = match.group(2)
        try:
            value = json.loads(raw)
        except Exception:
            return match.group(0)
        if not isinstance(value, str):
            return match.group(0)
        normalized = normalize_title(value)
        if normalized == value:
            return match.group(0)
        count += 1
        return f"{match.group(1)}{json.dumps(normalized, ensure_ascii=False)}"

    return pattern.sub(repl, html), count


def discover(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.html") if "tvt_v02" in path.as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize TVT title placement in tvt_v02 HTML reports.")
    parser.add_argument("--root", default=".", help="Project root. Default: cwd.")
    parser.add_argument("--dry-run", action="store_true", help="Show files that would change.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    changed_files = 0
    changed_items = 0

    for path in discover(root):
        old = path.read_text(encoding="utf-8", errors="replace")
        new = old
        file_count = 0
        for tag in ("title", "h1"):
            new, count = normalize_html_text_tag(new, tag)
            file_count += count
        new, count = normalize_plotly_title_text(new)
        file_count += count
        if file_count == 0:
            continue
        changed_files += 1
        changed_items += file_count
        rel = path.relative_to(root).as_posix()
        print(f"{rel}: {file_count} title item(s)")
        if not args.dry_run:
            path.write_text(new, encoding="utf-8")

    print(f"Changed files: {changed_files}; title items: {changed_items}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

