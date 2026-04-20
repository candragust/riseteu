#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "laporan_progres_pipeline_flf_bilstm_eurusd.md"
DEFAULT_OUTPUT = ROOT / "laporan_progres_pipeline_flf_bilstm_eurusd.html"
DEFAULT_TITLE = "Laporan Progres Pipeline FLF-BiLSTM EURUSD H4"


def inline_md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


def extract_title(md_text: str) -> str:
    for raw in md_text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return DEFAULT_TITLE


def convert(md_text: str, title: str) -> str:
    out: list[str] = []
    list_mode: str | None = None
    table_rows: list[list[str]] = []

    def flush_list() -> None:
        nonlocal list_mode
        if list_mode:
            out.append(f"</{list_mode}>")
            list_mode = None

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        out.append("<table>")
        header = table_rows[0]
        out.append("<thead><tr>" + "".join(f"<th>{inline_md(c)}</th>" for c in header) + "</tr></thead>")
        if len(table_rows) > 1:
            out.append("<tbody>")
            for row in table_rows[1:]:
                padded = row + [""] * (len(header) - len(row))
                out.append("<tr>" + "".join(f"<td>{inline_md(c)}</td>" for c in padded[: len(header)]) + "</tr>")
            out.append("</tbody>")
        out.append("</table>")
        table_rows = []

    for raw in md_text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_list()
            flush_table()
            continue

        if stripped.startswith("# "):
            flush_list()
            flush_table()
            out.append(f"<h1>{inline_md(stripped[2:])}</h1>")
            continue
        if stripped.startswith("## "):
            flush_list()
            flush_table()
            out.append(f"<h2>{inline_md(stripped[3:])}</h2>")
            continue
        if stripped.startswith("### "):
            flush_list()
            flush_table()
            out.append(f"<h3>{inline_md(stripped[4:])}</h3>")
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            is_separator = all(re.fullmatch(r"[:\-\s]+", c or "-") for c in cells)
            if is_separator:
                continue
            flush_list()
            table_rows.append(cells)
            continue

        if stripped.startswith("- "):
            flush_table()
            if list_mode != "ul":
                flush_list()
                out.append("<ul>")
                list_mode = "ul"
            out.append(f"<li>{inline_md(stripped[2:])}</li>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_table()
            if list_mode != "ol":
                flush_list()
                out.append("<ol>")
                list_mode = "ol"
            out.append(f"<li>{inline_md(re.sub(r'^\d+\.\s+', '', stripped))}</li>")
            continue

        flush_list()
        flush_table()
        out.append(f"<p>{inline_md(stripped)}</p>")

    flush_list()
    flush_table()

    body = "\n".join(out)
    return f"""<html>
<head>
  <meta charset="UTF-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.6; color: #222; }}
    h1, h2, h3 {{ color: #111; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f0f0; }}
    code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
    ul, ol {{ margin-top: 8px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML report from markdown.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    md_text = args.input.read_text(encoding="utf-8")
    title = args.title or extract_title(md_text)
    args.output.write_text(convert(md_text, title), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
