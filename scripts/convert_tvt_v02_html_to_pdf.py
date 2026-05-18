#!/usr/bin/env python3
"""Convert TVT v02 HTML reports to PDF with print-safe styling.

The converter keeps the source HTML untouched. For each report it creates a
temporary HTML file beside the source, injects print CSS/JS, asks Chrome
headless to print it to PDF, then removes the temporary file.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


DEFAULT_OUTPUT_ROOT = (
    "bukuThesis/penyusunan_buku_tesis_tvt_v02/"
    "04_lampiran_artefak/pdf_tvt_v02"
)
DEFAULT_ACTIVITY_FILE = "agentactivity.md"


@dataclass
class ConversionResult:
    html: Path
    pdf: Path
    status: str
    seconds: float
    bytes: int = 0
    pages: str = ""
    page_size: str = ""
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert every repository HTML path containing tvt_v02 to PDF."
    )
    parser.add_argument("--root", default=".", help="Project root. Default: cwd.")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"PDF output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--activity-file",
        default=DEFAULT_ACTIVITY_FILE,
        help=f"Activity log markdown file. Default: {DEFAULT_ACTIVITY_FILE}",
    )
    parser.add_argument(
        "--chrome-bin",
        default=os.environ.get("CHROME_BIN") or shutil.which("google-chrome") or "google-chrome",
        help="Chrome/Chromium binary. Default: CHROME_BIN or google-chrome.",
    )
    parser.add_argument(
        "--paper",
        default="A3",
        choices=["A2", "A3", "A4", "Letter", "Legal"],
        help="CSS @page paper size. A3 landscape is safer for wide Plotly charts.",
    )
    parser.add_argument(
        "--orientation",
        default="landscape",
        choices=["landscape", "portrait"],
        help="PDF orientation. Default: landscape.",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=4.0,
        help="Print margin in millimeters. Default: 4.",
    )
    parser.add_argument(
        "--plot-width",
        type=int,
        default=1420,
        help="Maximum Plotly relayout width in CSS pixels. Default: 1420.",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=10000,
        help="Chrome virtual-time budget for JS/network rendering. Default: 10000.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-file Chrome timeout in seconds. Default: 180.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="CSV manifest path. Default: <output-root>/pdf_manifest.csv.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Regex filter on relative HTML path. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Regex exclusion filter on relative HTML path. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PDFs.")
    parser.add_argument("--dry-run", action="store_true", help="List files without converting.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed file.")
    return parser.parse_args()


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_activity(activity_file: Path, message: str) -> None:
    try:
        activity_file.parent.mkdir(parents=True, exist_ok=True)
        with activity_file.open("a", encoding="utf-8") as handle:
            handle.write(f"- {now_text()} | {message}\n")
    except Exception:
        pass


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def discover_html(
    root: Path,
    output_root: Path,
    include: list[str],
    exclude: list[str],
) -> list[Path]:
    include_re = [re.compile(pattern) for pattern in include]
    exclude_re = [re.compile(pattern) for pattern in exclude]
    output_root_abs = output_root.resolve()
    files: list[Path] = []

    for path in sorted(root.rglob("*.html")):
        try:
            resolved = path.resolve()
            rel = rel_posix(path, root)
        except Exception:
            continue

        if output_root_abs in [resolved, *resolved.parents]:
            continue
        if ".pdf_render_" in path.name:
            continue
        if "tvt_v02" not in rel:
            continue
        if include_re and not any(regex.search(rel) for regex in include_re):
            continue
        if exclude_re and any(regex.search(rel) for regex in exclude_re):
            continue
        files.append(path)

    return files


def css_block(paper: str, orientation: str, margin_mm: float) -> str:
    return f"""
<style id="tvt-v02-pdf-print-style">
@page {{
  size: {paper} {orientation};
  margin: {margin_mm:g}mm;
}}
html, body {{
  background: #ffffff !important;
  overflow: visible !important;
  -webkit-print-color-adjust: exact !important;
  print-color-adjust: exact !important;
}}
body {{
  max-width: none !important;
}}
* {{
  box-sizing: border-box;
}}
h1, h2, h3, h4 {{
  break-after: avoid;
  page-break-after: avoid;
}}
p, li {{
  orphans: 3;
  widows: 3;
}}
img, canvas, svg.chart {{
  max-width: 100% !important;
  height: auto !important;
  break-inside: avoid;
  page-break-inside: avoid;
}}
.modebar, .modebar-container {{
  display: none !important;
}}
.chart-scroll {{
  overflow: visible !important;
  width: 100% !important;
  max-width: 100% !important;
  break-inside: avoid;
  page-break-inside: avoid;
}}
.chart-scroll > div {{
  width: 100% !important;
  max-width: 100% !important;
}}
.plotly-graph-div,
.js-plotly-plot,
.plot-container,
.svg-container {{
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  break-inside: avoid;
  page-break-inside: avoid;
}}
.plotly-graph-div svg,
.js-plotly-plot svg {{
  max-width: 100% !important;
}}
table {{
  max-width: 100% !important;
  border-collapse: collapse;
  page-break-inside: auto;
}}
thead {{
  display: table-header-group;
}}
tfoot {{
  display: table-footer-group;
}}
tr {{
  break-inside: avoid;
  page-break-inside: avoid;
}}
th, td {{
  word-break: normal;
  overflow-wrap: anywhere;
}}
pre, code {{
  white-space: pre-wrap;
  word-break: break-word;
}}
@media print {{
  .chart-scroll {{
    overflow: visible !important;
  }}
}}
</style>
"""


def js_block(plot_width: int) -> str:
    return f"""
<script id="tvt-v02-pdf-render-script">
(function() {{
  var MAX_PLOT_WIDTH = {plot_width};
  function fitPlots() {{
    document.querySelectorAll('.chart-scroll').forEach(function(el) {{
      el.style.overflow = 'visible';
      el.style.width = '100%';
      el.style.maxWidth = '100%';
    }});
    document.querySelectorAll('.plotly-graph-div').forEach(function(gd) {{
      var parent = gd.closest('.chart-scroll') || gd.parentElement || document.body;
      var parentWidth = parent.clientWidth || document.body.clientWidth || MAX_PLOT_WIDTH;
      var width = Math.max(900, Math.min(MAX_PLOT_WIDTH, parentWidth));
      gd.style.width = '100%';
      gd.style.maxWidth = '100%';
      if (window.Plotly && gd.data) {{
        var currentHeight = parseInt(gd.style.height || gd.getAttribute('height') || '0', 10);
        var layout = {{ autosize: false, width: width }};
        if (currentHeight && currentHeight > 0) {{
          layout.height = Math.min(Math.max(currentHeight, 360), 760);
        }}
        try {{
          var result = window.Plotly.relayout(gd, layout);
          if (result && result.catch) {{
            result.catch(function() {{}});
          }}
        }} catch (err) {{}}
      }}
    }});
  }}
  window.addEventListener('load', function() {{
    setTimeout(fitPlots, 800);
    setTimeout(fitPlots, 2500);
    setTimeout(fitPlots, 5000);
  }});
  setTimeout(fitPlots, 7500);
}})();
</script>
"""


def inject_print_assets(html: str, paper: str, orientation: str, margin_mm: float, plot_width: int) -> str:
    style = css_block(paper, orientation, margin_mm)
    script = js_block(plot_width)
    result = html

    if re.search(r"</head\s*>", result, flags=re.IGNORECASE):
        result = re.sub(r"</head\s*>", style + "\n</head>", result, count=1, flags=re.IGNORECASE)
    else:
        result = style + "\n" + result

    if re.search(r"</body\s*>", result, flags=re.IGNORECASE):
        result = re.sub(r"</body\s*>", script + "\n</body>", result, count=1, flags=re.IGNORECASE)
    else:
        result = result + "\n" + script

    return result


def file_url(path: Path) -> str:
    return "file://" + quote(str(path.resolve()))


def chrome_command(
    chrome_bin: str,
    html_path: Path,
    pdf_path: Path,
    profile_dir: Path,
    settle_ms: int,
) -> list[str]:
    return [
        chrome_bin,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--allow-file-access-from-files",
        "--disable-extensions",
        "--hide-scrollbars",
        "--run-all-compositor-stages-before-draw",
        f"--virtual-time-budget={settle_ms}",
        "--window-size=1600,1200",
        "--print-to-pdf-no-header",
        f"--user-data-dir={profile_dir}",
        f"--print-to-pdf={pdf_path}",
        file_url(html_path),
    ]


def parse_pdfinfo(pdf_path: Path) -> tuple[str, str]:
    pdfinfo_bin = shutil.which("pdfinfo")
    if not pdfinfo_bin:
        return "", ""
    try:
        proc = subprocess.run(
            [pdfinfo_bin, str(pdf_path)],
            check=False,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except Exception:
        return "", ""
    pages = ""
    page_size = ""
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            pages = line.split(":", 1)[1].strip()
        elif line.startswith("Page size:"):
            page_size = line.split(":", 1)[1].strip()
    return pages, page_size


def convert_one(
    html_path: Path,
    root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> ConversionResult:
    start = time.monotonic()
    rel = html_path.relative_to(root)
    pdf_path = output_root / rel.with_suffix(".pdf")

    if pdf_path.exists() and not args.overwrite:
        pages, page_size = parse_pdfinfo(pdf_path)
        return ConversionResult(
            html=html_path,
            pdf=pdf_path,
            status="SKIP",
            seconds=0.0,
            bytes=pdf_path.stat().st_size,
            pages=pages,
            page_size=page_size,
            message="exists",
        )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    original = html_path.read_text(encoding="utf-8", errors="replace")
    rendered = inject_print_assets(
        original,
        paper=args.paper,
        orientation=args.orientation,
        margin_mm=args.margin_mm,
        plot_width=args.plot_width,
    )

    tmp_html = html_path.with_name(f".pdf_render_{html_path.stem}_{os.getpid()}.html")
    profile_dir = Path(tempfile.mkdtemp(prefix="chrome_pdf_", dir=str(output_root)))
    try:
        tmp_html.write_text(rendered, encoding="utf-8")
        cmd = chrome_command(
            chrome_bin=args.chrome_bin,
            html_path=tmp_html,
            pdf_path=pdf_path,
            profile_dir=profile_dir,
            settle_ms=args.settle_ms,
        )
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        seconds = time.monotonic() - start
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")[:500]
            return ConversionResult(html_path, pdf_path, "FAILED", seconds, message=message)
        if not pdf_path.exists():
            return ConversionResult(html_path, pdf_path, "FAILED", seconds, message="pdf not created")
        size = pdf_path.stat().st_size
        if size < 10_000:
            return ConversionResult(html_path, pdf_path, "WARN", seconds, bytes=size, message="pdf too small")
        pages, page_size = parse_pdfinfo(pdf_path)
        return ConversionResult(
            html=html_path,
            pdf=pdf_path,
            status="OK",
            seconds=seconds,
            bytes=size,
            pages=pages,
            page_size=page_size,
            message="",
        )
    except subprocess.TimeoutExpired:
        seconds = time.monotonic() - start
        return ConversionResult(html_path, pdf_path, "FAILED", seconds, message="chrome timeout")
    except Exception as exc:
        seconds = time.monotonic() - start
        return ConversionResult(html_path, pdf_path, "FAILED", seconds, message=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            tmp_html.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            shutil.rmtree(profile_dir, ignore_errors=True)
        except Exception:
            pass


def write_manifest(manifest: Path, root: Path, output_root: Path, results: Iterable[ConversionResult]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "status",
                "html",
                "pdf",
                "bytes",
                "pages",
                "page_size",
                "seconds",
                "message",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "status": result.status,
                    "html": rel_posix(result.html, root),
                    "pdf": result.pdf.relative_to(output_root).as_posix()
                    if output_root in [result.pdf, *result.pdf.parents]
                    else str(result.pdf),
                    "bytes": result.bytes,
                    "pages": result.pages,
                    "page_size": result.page_size,
                    "seconds": f"{result.seconds:.2f}",
                    "message": result.message,
                }
            )


def print_plan(files: list[Path], root: Path, output_root: Path) -> None:
    print(f"Discovered {len(files)} TVT v02 HTML files.")
    for idx, html_path in enumerate(files, 1):
        rel = html_path.relative_to(root)
        pdf = output_root / rel.with_suffix(".pdf")
        print(f"{idx:02d}. {rel.as_posix()} -> {pdf.relative_to(root).as_posix() if root in [pdf, *pdf.parents] else pdf}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_root = (root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    activity_file = (root / args.activity_file).resolve() if not Path(args.activity_file).is_absolute() else Path(args.activity_file)
    manifest = Path(args.manifest) if args.manifest else output_root / "pdf_manifest.csv"
    if not manifest.is_absolute():
        manifest = root / manifest

    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 1
    if not shutil.which(args.chrome_bin) and not Path(args.chrome_bin).exists():
        print(f"Chrome binary not found: {args.chrome_bin}", file=sys.stderr)
        return 1

    files = discover_html(root, output_root, args.include, args.exclude)
    if args.limit > 0:
        files = files[: args.limit]

    print_plan(files, root, output_root)
    if args.dry_run:
        log_activity(activity_file, f"[DRY-RUN] TVT v02 HTML to PDF files={len(files)} output={output_root}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    log_activity(activity_file, f"[START] TVT v02 HTML to PDF files={len(files)} output={output_root}")
    results: list[ConversionResult] = []
    ok_count = 0
    failed_count = 0

    for idx, html_path in enumerate(files, 1):
        rel = rel_posix(html_path, root)
        print(f"[{idx}/{len(files)}] {rel}")
        log_activity(activity_file, f"[PDF {idx}/{len(files)}] START {rel}")
        result = convert_one(html_path, root, output_root, args)
        results.append(result)
        if result.status in {"OK", "SKIP", "WARN"}:
            ok_count += 1
        else:
            failed_count += 1
        print(
            f"  {result.status} pages={result.pages or '-'} "
            f"bytes={result.bytes} seconds={result.seconds:.2f} "
            f"pdf={result.pdf}"
        )
        if result.message:
            print(f"  message: {result.message}")
        log_activity(
            activity_file,
            f"[PDF {idx}/{len(files)}] {result.status} {rel} pages={result.pages or '-'} bytes={result.bytes}",
        )
        write_manifest(manifest, root, output_root, results)
        if result.status == "FAILED" and args.fail_fast:
            break

    write_manifest(manifest, root, output_root, results)
    log_activity(
        activity_file,
        f"[DONE] TVT v02 HTML to PDF ok_or_skipped={ok_count} failed={failed_count} manifest={manifest}",
    )
    print(f"Manifest: {manifest}")
    print(f"Output  : {output_root}")
    print(f"Done. ok_or_skipped={ok_count}, failed={failed_count}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

