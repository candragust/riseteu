import argparse
import html
import json
import math
import re
from pathlib import Path

import pandas as pd


PIP_FACTOR = 10000.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "result"


def discover_summary_files(result_dir: Path) -> list[Path]:
    return sorted(
        path for path in result_dir.rglob("*_summary.json")
        if path.is_file()
    )


def infer_preds_path(summary_path: Path) -> Path | None:
    pred_name = summary_path.name.replace("_summary.json", "_preds.csv")
    pred_path = summary_path.with_name(pred_name)
    return pred_path if pred_path.exists() else None


def read_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def order_to_text(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return f"({value[0]},{value[1]},{value[2]})"
    return html.escape(str(value))


def format_float(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def format_rel(path: Path, base: Path) -> str:
    return html.escape(str(path.relative_to(base)))


def build_metric_table(summary: dict) -> str:
    mae = summary.get("mae_pips", {})
    rows = [
        ("MAE Open (pips)", format_float(mae.get("open"))),
        ("MAE High (pips)", format_float(mae.get("high"))),
        ("MAE Low (pips)", format_float(mae.get("low"))),
        ("MAE Close (pips)", format_float(mae.get("close"))),
        ("MAE Avg (pips)", format_float(mae.get("avg"))),
        ("Split", format_float(summary.get("split"), 6)),
        ("Model", html.escape(str(summary.get("model", "-")))),
        ("Input", html.escape(str(summary.get("input_mode", "-")))),
    ]
    body = "".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>"
        for label, value in rows
    )
    return f"<table class='kv'>{body}</table>"


def build_target_table(summary: dict) -> str:
    targets = summary.get("targets", {})
    rows = []
    for col in ("open", "high", "low", "close"):
        meta = targets.get(col, {})
        search = meta.get("search") or {}
        rows.append(
            {
                "Series": col,
                "Requested": order_to_text(meta.get("requested_order")),
                "Selected": order_to_text(meta.get("selected_order")),
                "Used": order_to_text(meta.get("used_order")),
                "Train": meta.get("train_samples", "-"),
                "Test": meta.get("test_samples", "-"),
                "AIC": format_float(meta.get("aic")),
                "BIC": format_float(meta.get("bic")),
                "Metric": html.escape(str(search.get("selection_metric", "-"))).upper(),
                "Score": format_float(search.get("selection_score")),
            }
        )
    df = pd.DataFrame(rows)
    return df.to_html(index=False, escape=False, classes="data-table")


def svg_line_chart(series_map: dict[str, list[float]], title: str, width: int = 920, height: int = 280) -> str:
    all_values = [v for values in series_map.values() for v in values]
    if not all_values:
        return ""

    left = 52
    right = 14
    top = 18
    bottom = 32
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min = min(all_values)
    y_max = max(all_values)
    if y_min == y_max:
        y_min -= 1e-6
        y_max += 1e-6

    colors = ["#0f172a", "#2563eb", "#dc2626", "#059669"]
    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * frac
        value = y_max - frac * (y_max - y_min)
        grid.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' class='grid' />"
            f"<text x='6' y='{y + 4:.1f}' class='axis-label'>{value:.4f}</text>"
        )

    polylines = []
    legend = []
    for idx, (name, values) in enumerate(series_map.items()):
        if not values:
            continue
        n = max(len(values), 1)
        points = []
        for i, value in enumerate(values):
            x = left if n == 1 else left + (i / (n - 1)) * plot_w
            y = top + (y_max - value) / (y_max - y_min) * plot_h
            points.append(f"{x:.2f},{y:.2f}")
        color = colors[idx % len(colors)]
        polylines.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(points)}' />"
        )
        legend.append(
            f"<span class='legend-item'><span class='swatch' style='background:{color}'></span>{html.escape(name)}</span>"
        )

    return (
        f"<div class='chart-block'><div class='chart-title'>{html.escape(title)}</div>"
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        f"{''.join(grid)}"
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' class='axis' />"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' class='axis' />"
        f"{''.join(polylines)}"
        f"</svg><div class='legend'>{''.join(legend)}</div></div>"
    )


def svg_bar_chart(values: list[float], title: str, width: int = 920, height: int = 260, color: str = "#2563eb") -> str:
    if not values:
        return ""

    left = 52
    right = 14
    top = 18
    bottom = 32
    plot_w = width - left - right
    plot_h = height - top - bottom
    v_max = max(values)
    if v_max <= 0:
        v_max = 1.0

    bars = []
    n = len(values)
    gap = 1 if n > 80 else 2
    bar_w = max(1.0, (plot_w / max(n, 1)) - gap)
    for i, value in enumerate(values):
        x = left + i * (plot_w / max(n, 1))
        h = (value / v_max) * plot_h
        y = top + plot_h - h
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='{color}' opacity='0.82' />"
        )

    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * frac
        value = v_max - frac * v_max
        grid.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' class='grid' />"
            f"<text x='6' y='{y + 4:.1f}' class='axis-label'>{value:.2f}</text>"
        )

    return (
        f"<div class='chart-block'><div class='chart-title'>{html.escape(title)}</div>"
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        f"{''.join(grid)}"
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' class='axis' />"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' class='axis' />"
        f"{''.join(bars)}"
        f"</svg></div>"
    )


def build_tail_table(preds_df: pd.DataFrame, tail: int) -> str:
    tail_df = preds_df.tail(tail).copy()
    tail_df["abs_err_close_pips"] = (tail_df["pred_close"] - tail_df["true_close"]).abs() * PIP_FACTOR
    err_avg = (
        tail_df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
        - tail_df[["true_open", "true_high", "true_low", "true_close"]].values
    )
    tail_df["abs_err_avg_pips"] = pd.DataFrame(err_avg).abs().mean(axis=1).to_numpy() * PIP_FACTOR
    show = tail_df[[
        "pred_open", "pred_high", "pred_low", "pred_close",
        "true_open", "true_high", "true_low", "true_close",
        "abs_err_close_pips", "abs_err_avg_pips",
    ]].copy()
    return show.to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="data-table")


def build_run_section(result_dir: Path, summary_path: Path, tail: int) -> str:
    summary = read_summary(summary_path)
    preds_path = infer_preds_path(summary_path)
    rel_summary = format_rel(summary_path, result_dir)
    rel_preds = format_rel(preds_path, result_dir) if preds_path else "-"
    body = [
        f"<section class='run'>",
        f"<h2>{rel_summary}</h2>",
        "<div class='two-col'>",
        build_metric_table(summary),
        (
            "<div class='source-box'>"
            f"<div><strong>Summary:</strong> <a href='{html.escape(rel_summary)}'>{rel_summary}</a></div>"
            + (
                f"<div><strong>Preds:</strong> <a href='{html.escape(rel_preds)}'>{rel_preds}</a></div>"
                if preds_path else "<div><strong>Preds:</strong> -</div>"
            )
            + "</div>"
        ),
        "</div>",
        "<h3>Selected Orders</h3>",
        build_target_table(summary),
    ]

    if preds_path and preds_path.exists():
        preds_df = pd.read_csv(preds_path)
        true_close = preds_df["true_close"].tolist()
        pred_close = preds_df["pred_close"].tolist()
        err_avg = (
            preds_df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
            - preds_df[["true_open", "true_high", "true_low", "true_close"]].values
        )
        err_avg_pips = pd.DataFrame(err_avg).abs().mean(axis=1).tolist()
        err_avg_pips = [v * PIP_FACTOR for v in err_avg_pips]
        body.extend([
            "<h3>Charts</h3>",
            svg_line_chart({"True Close": true_close, "Pred Close": pred_close}, "True vs Predicted Close"),
            svg_bar_chart(err_avg_pips, "Average Absolute Error per Candle (pips)", color="#dc2626"),
            f"<h3>Tail {min(tail, len(preds_df))} Rows</h3>",
            build_tail_table(preds_df, tail),
        ])

    body.append("</section>")
    return "".join(body)


def load_fold_score(summary_path: Path) -> tuple[int | None, float | None]:
    match = re.search(r"fold(\d+)_summary\.json$", summary_path.name)
    fold = int(match.group(1)) if match else None
    summary = read_summary(summary_path)
    score = summary.get("mae_pips", {}).get("avg")
    return fold, score


def build_rolling_sections(result_dir: Path) -> str:
    sections = []
    for schedule_path in sorted(result_dir.rglob("rolling_fixed_summary.csv")):
        run_dir = schedule_path.parent
        fold_rows = []
        for summary_path in sorted(run_dir.glob("fold*_summary.json")):
            fold, score = load_fold_score(summary_path)
            fold_rows.append({
                "fold": fold,
                "mae_avg_pips": score,
                "summary": summary_path.name,
            })
        if not fold_rows:
            continue

        schedule_df = pd.read_csv(schedule_path)
        fold_df = pd.DataFrame(fold_rows).sort_values("fold")
        merged = fold_df.merge(schedule_df, on="fold", how="left")
        merged["mae_avg_pips"] = merged["mae_avg_pips"].map(lambda x: float(x) if pd.notna(x) else x)
        score_series = [float(v) for v in merged["mae_avg_pips"].dropna().tolist()]

        rel_dir = format_rel(run_dir, result_dir)
        sections.append(
            "<section class='run'>"
            f"<h2>Rolling Summary: {rel_dir}</h2>"
            f"<div class='summary-pill'>Folds: {len(merged)}</div>"
            f"<div class='summary-pill'>Mean MAE Avg: {format_float(merged['mae_avg_pips'].mean())} pips</div>"
            f"{svg_bar_chart(score_series, 'MAE Avg per Fold (pips)', color='#0f766e') if score_series else ''}"
            f"{merged[['fold', 'train_start', 'train_end', 'test_start', 'test_end', 'mae_avg_pips']].to_html(index=False, float_format=lambda x: f'{x:.4f}', classes='data-table')}"
            "</section>"
        )
    return "".join(sections)


def build_html(result_dir: Path, tail: int) -> str:
    summaries = discover_summary_files(result_dir)
    run_sections = [build_run_section(result_dir, path, tail) for path in summaries]
    rolling_sections = build_rolling_sections(result_dir)
    css = """
body { font-family: Arial, sans-serif; margin: 24px; color: #0f172a; background: #f8fafc; }
h1, h2, h3 { margin: 0 0 12px; }
h1 { margin-bottom: 18px; }
p.lead { margin: 0 0 24px; color: #475569; }
section.run { background: #fff; border: 1px solid #dbe4ee; border-radius: 6px; padding: 18px; margin: 0 0 20px; }
.two-col { display: grid; grid-template-columns: minmax(320px, 420px) 1fr; gap: 16px; align-items: start; margin-bottom: 14px; }
.kv { border-collapse: collapse; width: 100%; }
.kv th, .kv td { border: 1px solid #dbe4ee; padding: 8px 10px; }
.kv th { width: 45%; text-align: left; background: #f8fafc; }
.kv td { text-align: right; background: #fff; }
.data-table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
.data-table th, .data-table td { border: 1px solid #dbe4ee; padding: 7px 9px; }
.data-table th { background: #e2e8f0; }
.source-box { border: 1px solid #dbe4ee; border-radius: 6px; padding: 12px; background: #f8fafc; }
.chart-block { margin: 12px 0 18px; }
.chart-title { font-weight: 600; margin: 0 0 8px; }
.chart { width: 100%; height: auto; border: 1px solid #dbe4ee; border-radius: 6px; background: #fff; }
.grid { stroke: #e2e8f0; stroke-width: 1; }
.axis { stroke: #94a3b8; stroke-width: 1.2; }
.axis-label { fill: #475569; font-size: 11px; }
.legend { margin-top: 8px; display: flex; gap: 14px; flex-wrap: wrap; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: #334155; }
.swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.summary-pill { display: inline-block; margin: 0 8px 12px 0; padding: 6px 10px; border-radius: 999px; background: #e0f2fe; color: #075985; font-size: 13px; }
a { color: #1d4ed8; text-decoration: none; }
a:hover { text-decoration: underline; }
"""
    body = [
        "<h1>ARIMA Output Report</h1>",
        f"<p class='lead'>Source folder: {html.escape(str(result_dir))}</p>",
    ]
    if rolling_sections:
        body.append("<h2>Rolling Runs</h2>")
        body.append(rolling_sections)
    if run_sections:
        body.append("<h2>Run Details</h2>")
        body.extend(run_sections)
    else:
        body.append("<p>Tidak ada file *_summary.json yang ditemukan.</p>")

    return (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>ARIMA Output Report</title>"
        f"<style>{css}</style>"
        "</head><body>"
        + "".join(body) +
        "</body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report for ARIMA outputs.")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Directory containing ARIMA result files.")
    parser.add_argument("--out", default=None, help="Output HTML path.")
    parser.add_argument("--tail", type=int, default=12, help="Tail rows shown in each run table.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    out_path = Path(args.out).resolve() if args.out else (result_dir / "index.html")
    html_doc = build_html(result_dir, args.tail)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"ARIMA HTML report written to {out_path}")


if __name__ == "__main__":
    main()
