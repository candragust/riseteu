#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "laporan_progres_pipeline_flf_bilstm_eurusd.md"
DEFAULT_OUTPUT = ROOT / "laporan_progres_pipeline_flf_bilstm_eurusd.docx"
DEFAULT_TEMPLATE = ROOT / "propos tesis - candraAgustinus-1225800007- V14Revis.docx"
DEFAULT_TITLE = "Laporan Progres Implementasi Pipeline FLF-BiLSTM untuk EURUSD H4"


def strip_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.strip()


def make_run(text: str) -> str:
    escaped = html.escape(text)
    return f'<w:r><w:t xml:space="preserve">{escaped}</w:t></w:r>'


def make_paragraph(text: str, style: str | None = None) -> str:
    text = strip_inline_markdown(text)
    if not text:
        return "<w:p/>"
    ppr = ""
    if style:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
    return f"<w:p>{ppr}{make_run(text)}</w:p>"


def make_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    col_width = 2200
    grid = "".join(f'<w:gridCol w:w="{col_width}"/>' for _ in range(col_count))

    def cell_xml(text: str, header: bool = False) -> str:
        text = strip_inline_markdown(text)
        p_style = '<w:pPr><w:jc w:val="center"/></w:pPr>' if header else ""
        run = make_run(text)
        shd = '<w:shd w:val="clear" w:fill="EDEDED"/>' if header else ""
        return (
            f'<w:tc><w:tcPr><w:tcW w:w="{col_width}" w:type="dxa"/>{shd}</w:tcPr>'
            f'<w:p>{p_style}{run}</w:p></w:tc>'
        )

    row_xml = []
    for i, row in enumerate(rows):
        padded = row + [""] * (col_count - len(row))
        cells = "".join(cell_xml(cell, header=(i == 0)) for cell in padded)
        row_xml.append(f"<w:tr>{cells}</w:tr>")

    return (
        '<w:tbl>'
        '<w:tblPr>'
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="auto"/>'
        '</w:tblBorders>'
        '</w:tblPr>'
        f'<w:tblGrid>{grid}</w:tblGrid>'
        + "".join(row_xml)
        + '</w:tbl>'
    )


def markdown_to_paragraphs(markdown_text: str) -> list[str]:
    paragraphs: list[str] = []
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            paragraphs.append(make_table(table_rows))
            paragraphs.append("<w:p/>")
            table_rows = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_table()
            paragraphs.append("<w:p/>")
            continue

        if stripped.startswith("# "):
            flush_table()
            paragraphs.append(make_paragraph(stripped[2:], "Title"))
            continue
        if stripped.startswith("## "):
            flush_table()
            paragraphs.append(make_paragraph(stripped[3:], "Heading1"))
            continue
        if stripped.startswith("### "):
            flush_table()
            paragraphs.append(make_paragraph(stripped[4:], "Heading2"))
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [strip_inline_markdown(cell.strip()) for cell in stripped.strip("|").split("|")]
            is_separator = all(re.fullmatch(r"[:\- ]+", cell or "-") for cell in cells)
            if is_separator:
                continue
            table_rows.append(cells)
            continue

        if stripped.startswith("- "):
            flush_table()
            paragraphs.append(make_paragraph(f"- {stripped[2:]}"))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_table()
            paragraphs.append(make_paragraph(stripped))
            continue

        flush_table()
        paragraphs.append(make_paragraph(stripped))

    flush_table()

    return paragraphs


def build_core_properties(title: str) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{html.escape(title)}</dc:title>
  <dc:subject>Laporan progres tesis</dc:subject>
  <dc:creator>Candra Eko Agustinus</dc:creator>
  <cp:keywords>EURUSD, FLF, BiLSTM, Tesis</cp:keywords>
  <dc:description>{html.escape(title)}</dc:description>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def build_document_xml(template_document_xml: str, paragraphs: list[str]) -> str:
    body_start = template_document_xml.index("<w:body>") + len("<w:body>")
    body_end = template_document_xml.rindex("</w:body>")
    header = template_document_xml[:body_start]
    body = template_document_xml[body_start:body_end]

    sect_match = re.search(r"(<w:sectPr[\s\S]*?</w:sectPr>)\s*$", body)
    if not sect_match:
        raise RuntimeError("Tidak menemukan sectPr pada template DOCX.")

    sect_pr = sect_match.group(1)
    sect_pr = re.sub(r"<w:headerReference[^>]*/>", "", sect_pr)
    sect_pr = re.sub(r"<w:footerReference[^>]*/>", "", sect_pr)

    content = "".join(paragraphs) + sect_pr
    return header + content + "</w:body></w:document>"


def extract_title(markdown_text: str) -> str:
    for raw in markdown_text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("# "):
            return strip_inline_markdown(stripped[2:])
    return DEFAULT_TITLE


def build_docx(input_md: Path, output_docx: Path, template_docx: Path, title: str) -> None:
    markdown_text = input_md.read_text(encoding="utf-8")
    paragraphs = markdown_to_paragraphs(markdown_text)

    with ZipFile(template_docx, "r") as zin:
        template_document_xml = zin.read("word/document.xml").decode("utf-8")
        output_docx.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(output_docx, "w", compression=ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename in {"word/document.xml", "docProps/core.xml"}:
                    continue
                zout.writestr(item, zin.read(item.filename))

            zout.writestr("word/document.xml", build_document_xml(template_document_xml, paragraphs))
            zout.writestr("docProps/core.xml", build_core_properties(title))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DOCX report from thesis progress markdown.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    markdown_text = args.input.read_text(encoding="utf-8")
    title = args.title or extract_title(markdown_text)
    build_docx(args.input, args.output, args.template, title)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
