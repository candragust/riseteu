import argparse
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


PIP_FACTOR = 10000.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "result"


def discover_summary_files(result_dir: Path, recursive: bool = False) -> list[Path]:
    globber = result_dir.rglob if recursive else result_dir.glob
    return sorted(
        path for path in globber("*_summary.json")
        if path.is_file()
    )


def infer_preds_path(summary_path: Path) -> Path | None:
    pred_name = summary_path.name.replace("_summary.json", "_preds.csv")
    pred_path = summary_path.with_name(pred_name)
    return pred_path if pred_path.exists() else None


def infer_data_path(summary_path: Path) -> Path | None:
    data_name = summary_path.name.replace("_summary.json", "_data.csv")
    data_path = summary_path.with_name(data_name)
    return data_path if data_path.exists() else None


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


def squared_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float("nan")
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r**2)


def nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def directional_accuracy(true_open, true_close, pred_open, pred_close):
    true_dir = np.sign(np.asarray(true_close, dtype=float) - np.asarray(true_open, dtype=float))
    pred_dir = np.sign(np.asarray(pred_close, dtype=float) - np.asarray(pred_open, dtype=float))
    if len(true_dir) == 0:
        return float("nan")
    return float((true_dir == pred_dir).mean() * 100.0)


def compute_prediction_metrics(preds_df: pd.DataFrame) -> dict[str, float | int]:
    pred_values = preds_df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
    true_values = preds_df[["true_open", "true_high", "true_low", "true_close"]].values
    diffs = np.abs(pred_values - true_values) * PIP_FACTOR
    mae = diffs.mean(axis=0)
    corr2 = [squared_corr(true_values[:, i], pred_values[:, i]) for i in range(4)]
    true_dir = np.sign(preds_df["true_close"].to_numpy(dtype=float) - preds_df["true_open"].to_numpy(dtype=float))
    pred_dir = np.sign(preds_df["pred_close"].to_numpy(dtype=float) - preds_df["pred_open"].to_numpy(dtype=float))
    return {
        "samples": int(len(preds_df)),
        "mae_open_pips": float(mae[0]),
        "mae_high_pips": float(mae[1]),
        "mae_low_pips": float(mae[2]),
        "mae_close_pips": float(mae[3]),
        "mae_avg_pips": float(mae.mean()),
        "mae_avg_hlc_pips": float((mae[1] + mae[2] + mae[3]) / 3.0),
        "corr2_open": float(corr2[0]),
        "corr2_high": float(corr2[1]),
        "corr2_low": float(corr2[2]),
        "corr2_close": float(corr2[3]),
        "corr2_avg_ohlc": nanmean(corr2),
        "corr2_avg_hlc": nanmean(corr2[1:]),
        "da_body_pct": directional_accuracy(
            preds_df["true_open"],
            preds_df["true_close"],
            preds_df["pred_open"],
            preds_df["pred_close"],
        ),
        "true_bull": int((true_dir > 0).sum()),
        "true_bear": int((true_dir < 0).sum()),
        "pred_bull": int((pred_dir > 0).sum()),
        "pred_bear": int((pred_dir < 0).sum()),
    }


def build_metric_table(summary: dict, preds_df: pd.DataFrame | None = None) -> str:
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
    if preds_df is not None:
        metrics = compute_prediction_metrics(preds_df)
        rows.extend([
            ("corr2 Open", format_float(metrics["corr2_open"])),
            ("corr2 High", format_float(metrics["corr2_high"])),
            ("corr2 Low", format_float(metrics["corr2_low"])),
            ("corr2 Close", format_float(metrics["corr2_close"])),
            ("corr2 Avg OHLC", format_float(metrics["corr2_avg_ohlc"])),
            ("corr2 Avg HLC", format_float(metrics["corr2_avg_hlc"])),
            ("Directional Accuracy body (%)", format_float(metrics["da_body_pct"])),
        ])
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


def svg_line_chart(series_map: dict[str, list[float]], title: str, width: int = 920, height: int = 280, digits: int = 4) -> str:
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
            f"<text x='6' y='{y + 4:.1f}' class='axis-label'>{value:.{digits}f}</text>"
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


def compute_atr_frame(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if not required.issubset(ohlc_df.columns):
        raise ValueError(f"Data OHLC wajib memiliki kolom {sorted(required)} untuk menghitung ATR.")
    df = ohlc_df[["open", "high", "low", "close"]].astype(float).copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr6"] = tr.rolling(6, min_periods=1).mean()
    df["atr12"] = tr.rolling(12, min_periods=1).mean()
    return df


def load_atr_slice(data_path: Path | None, preds_df: pd.DataFrame) -> pd.DataFrame:
    if data_path and data_path.exists():
        data_df = pd.read_csv(data_path)
        if len(data_df) >= len(preds_df):
            return compute_atr_frame(data_df).tail(len(preds_df)).reset_index(drop=True)

    fallback = preds_df[["true_open", "true_high", "true_low", "true_close"]].rename(
        columns={
            "true_open": "open",
            "true_high": "high",
            "true_low": "low",
            "true_close": "close",
        }
    )
    return compute_atr_frame(fallback).reset_index(drop=True)


def svg_candle_chart(preds_df: pd.DataFrame, title: str, width: int = 1000, height: int = 340) -> str:
    if preds_df.empty:
        return ""

    left = 56
    right = 18
    top = 20
    bottom = 34
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(preds_df)
    price_cols = [
        "true_open", "true_high", "true_low", "true_close",
        "pred_open", "pred_high", "pred_low", "pred_close",
    ]
    prices = preds_df[price_cols].to_numpy(dtype=float).reshape(-1)
    y_min = float(np.nanmin(prices))
    y_max = float(np.nanmax(prices))
    pad = max((y_max - y_min) * 0.06, 1e-5)
    y_min -= pad
    y_max += pad

    def ymap(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * frac
        value = y_max - frac * (y_max - y_min)
        grid.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' class='grid' />"
            f"<text x='6' y='{y + 4:.1f}' class='axis-label'>{value:.5f}</text>"
        )

    step = plot_w / max(n, 1)
    true_w = max(2.0, min(7.0, step * 0.62))
    pred_tick = max(2.0, min(5.0, step * 0.36))
    pred_offset = max(1.8, min(4.5, step * 0.26))
    candles = []

    def draw_candle(x, open_, high, low, close, body_w, color, opacity, stroke_width):
        y_high = ymap(high)
        y_low = ymap(low)
        y_open = ymap(open_)
        y_close = ymap(close)
        body_y = min(y_open, y_close)
        body_h = max(abs(y_close - y_open), 1.2)
        return (
            f"<line x1='{x:.2f}' y1='{y_high:.2f}' x2='{x:.2f}' y2='{y_low:.2f}' "
            f"stroke='{color}' stroke-width='{stroke_width}' opacity='{opacity}' />"
            f"<rect x='{x - body_w / 2:.2f}' y='{body_y:.2f}' width='{body_w:.2f}' height='{body_h:.2f}' "
            f"fill='{color}' stroke='{color}' stroke-width='{stroke_width}' opacity='{opacity}' />"
        )

    def draw_ohlc_bar(x, open_, high, low, close, color):
        y_high = ymap(high)
        y_low = ymap(low)
        y_open = ymap(open_)
        y_close = ymap(close)
        return (
            f"<line x1='{x:.2f}' y1='{y_high:.2f}' x2='{x:.2f}' y2='{y_low:.2f}' "
            f"stroke='{color}' stroke-width='1.35' opacity='0.95' />"
            f"<line x1='{x - pred_tick:.2f}' y1='{y_open:.2f}' x2='{x:.2f}' y2='{y_open:.2f}' "
            f"stroke='{color}' stroke-width='1.35' opacity='0.95' />"
            f"<line x1='{x:.2f}' y1='{y_close:.2f}' x2='{x + pred_tick:.2f}' y2='{y_close:.2f}' "
            f"stroke='{color}' stroke-width='1.35' opacity='0.95' />"
        )

    for i, row in preds_df.reset_index(drop=True).iterrows():
        x = left + (i + 0.5) * step
        true_color = "#16a34a" if row["true_close"] >= row["true_open"] else "#dc2626"
        candles.append(
            draw_candle(
                x - pred_offset * 0.35,
                row["true_open"],
                row["true_high"],
                row["true_low"],
                row["true_close"],
                true_w,
                true_color,
                "0.35",
                "1.0",
            )
        )
        candles.append(
            draw_ohlc_bar(
                x + pred_offset,
                row["pred_open"],
                row["pred_high"],
                row["pred_low"],
                row["pred_close"],
                "#16a34a" if row["pred_close"] >= row["pred_open"] else "#b91c1c",
            )
        )

    legend = (
        "<span class='legend-item'><span class='swatch' style='background:#16a34a; opacity:.45'></span>Actual bullish</span>"
        "<span class='legend-item'><span class='swatch' style='background:#dc2626; opacity:.45'></span>Actual bearish</span>"
        "<span class='legend-item'><span class='swatch line-swatch' style='background:#16a34a'></span>Pred bullish OHLC bar</span>"
        "<span class='legend-item'><span class='swatch line-swatch' style='background:#b91c1c'></span>Pred bearish OHLC bar</span>"
    )
    return (
        f"<div class='chart-block'><div class='chart-title'>{html.escape(title)}</div>"
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        f"{''.join(grid)}"
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' class='axis' />"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' class='axis' />"
        f"{''.join(candles)}"
        f"</svg><div class='legend'>{legend}</div></div>"
    )


def svg_error_atr_chart(preds_df: pd.DataFrame, atr_df: pd.DataFrame) -> str:
    err_avg = (
        preds_df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
        - preds_df[["true_open", "true_high", "true_low", "true_close"]].values
    )
    err_avg_pips = (np.abs(err_avg).mean(axis=1) * PIP_FACTOR).tolist()
    atr6_pips = (atr_df["atr6"].to_numpy(dtype=float) * PIP_FACTOR).tolist()
    atr12_pips = (atr_df["atr12"].to_numpy(dtype=float) * PIP_FACTOR).tolist()
    series = [
        ("|Error avg| (pips)", err_avg_pips, "#2563eb", "2", "", True),
        ("ATR6", atr6_pips, "#dc2626", "2", "", False),
        ("ATR12", atr12_pips, "#059669", "2", "6 4", False),
    ]
    width = 1000
    height = 300
    left = 52
    right = 14
    top = 18
    bottom = 32
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for _, vals, *_ in series for v in vals if not math.isnan(v)]
    if not values:
        return ""
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def xmap(idx: int, n: int) -> float:
        return left if n <= 1 else left + (idx / (n - 1)) * plot_w

    def ymap(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * frac
        value = y_max - frac * (y_max - y_min)
        grid.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' class='grid' />"
            f"<text x='6' y='{y + 4:.1f}' class='axis-label'>{value:.2f}</text>"
        )

    lines = []
    legend = []
    for name, vals, color, stroke_width, dash, markers in series:
        points = []
        n = len(vals)
        for idx, value in enumerate(vals):
            if math.isnan(value):
                continue
            points.append((xmap(idx, n), ymap(value)))
        if not points:
            continue
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        lines.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='{stroke_width}'{dash_attr} points='{point_text}' />"
        )
        if markers:
            lines.extend(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='2.4' fill='{color}' opacity='0.9' />"
                for x, y in points
            )
        legend.append(
            f"<span class='legend-item'><span class='swatch' style='background:{color}'></span>{html.escape(name)}</span>"
        )

    return (
        "<div class='chart-block'><div class='chart-title'>Average Absolute Error per Candle (pips) vs ATR</div>"
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        f"{''.join(grid)}"
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' class='axis' />"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' class='axis' />"
        f"{''.join(lines)}"
        f"</svg><div class='legend'>{''.join(legend)}</div></div>"
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
    data_path = infer_data_path(summary_path)
    preds_df = pd.read_csv(preds_path) if preds_path and preds_path.exists() else None
    rel_summary = format_rel(summary_path, result_dir)
    rel_preds = format_rel(preds_path, result_dir) if preds_path else "-"
    rel_data = format_rel(data_path, result_dir) if data_path else "-"
    body = [
        f"<section class='run'>",
        f"<h2>{rel_summary}</h2>",
        "<div class='two-col'>",
        build_metric_table(summary, preds_df),
        (
            "<div class='source-box'>"
            f"<div><strong>Summary:</strong> <a href='{html.escape(rel_summary)}'>{rel_summary}</a></div>"
            + (
                f"<div><strong>Preds:</strong> <a href='{html.escape(rel_preds)}'>{rel_preds}</a></div>"
                if preds_path else "<div><strong>Preds:</strong> -</div>"
            )
            + (
                f"<div><strong>Data:</strong> <a href='{html.escape(rel_data)}'>{rel_data}</a></div>"
                if data_path else "<div><strong>Data:</strong> -</div>"
            )
            + "</div>"
        ),
        "</div>",
        "<h3>Selected Orders</h3>",
        build_target_table(summary),
    ]

    if preds_df is not None:
        atr_df = load_atr_slice(data_path, preds_df)
        body.extend([
            "<h3>Charts</h3>",
            "<p class='metric-note'>corr2 = kuadrat korelasi Pearson. Directional Accuracy body = kecocokan tanda close-open antara candle aktual dan prediksi.</p>",
            svg_candle_chart(preds_df, "Candlestick True vs Predicted OHLC"),
            svg_error_atr_chart(preds_df, atr_df),
        ])

    body.append("</section>")
    return "".join(body)


def load_fold_score(summary_path: Path) -> tuple[int | None, float | None]:
    match = re.search(r"fold(\d+)_summary\.json$", summary_path.name)
    fold = int(match.group(1)) if match else None
    summary = read_summary(summary_path)
    score = summary.get("mae_pips", {}).get("avg")
    return fold, score


def build_rolling_sections(result_dir: Path, recursive: bool = False) -> str:
    sections = []
    globber = result_dir.rglob if recursive else result_dir.glob
    for schedule_path in sorted(globber("rolling_fixed_summary.csv")):
        run_dir = schedule_path.parent
        fold_rows = []
        for summary_path in sorted(run_dir.glob("fold*_summary.json")):
            fold, score = load_fold_score(summary_path)
            row = {
                "fold": fold,
                "mae_avg_pips": score,
                "summary": summary_path.name,
            }
            preds_path = infer_preds_path(summary_path)
            if preds_path and preds_path.exists():
                metrics = compute_prediction_metrics(pd.read_csv(preds_path))
                row.update(
                    {
                        "mae_avg_hlc_pips": metrics["mae_avg_hlc_pips"],
                        "corr2_avg_hlc": metrics["corr2_avg_hlc"],
                        "corr2_avg_ohlc": metrics["corr2_avg_ohlc"],
                        "da_body_pct": metrics["da_body_pct"],
                    }
                )
            fold_rows.append(row)
        if not fold_rows:
            continue

        schedule_df = pd.read_csv(schedule_path)
        fold_df = pd.DataFrame(fold_rows).sort_values("fold")
        merged = fold_df.merge(schedule_df, on="fold", how="left", suffixes=("", "_summary"))
        if "mae_avg_pips" not in merged.columns:
            raise KeyError(
                f"Rolling summary merge did not produce mae_avg_pips for {schedule_path}. "
                f"Columns: {list(merged.columns)}"
            )
        if "mae_avg_pips_summary" in merged.columns:
            merged["mae_avg_pips"] = merged["mae_avg_pips"].fillna(merged["mae_avg_pips_summary"])
        merged["mae_avg_pips"] = merged["mae_avg_pips"].map(lambda x: float(x) if pd.notna(x) else x)
        score_series = [float(v) for v in merged["mae_avg_pips"].dropna().tolist()]
        metric_cols = [
            "fold",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "mae_avg_pips",
            "mae_avg_hlc_pips",
            "corr2_avg_hlc",
            "corr2_avg_ohlc",
            "da_body_pct",
        ]
        metric_cols = [col for col in metric_cols if col in merged.columns]

        rel_dir = format_rel(run_dir, result_dir)
        sections.append(
            "<section class='run'>"
            f"<h2>Rolling Summary: {rel_dir}</h2>"
            f"<div class='summary-pill'>Folds: {len(merged)}</div>"
            f"<div class='summary-pill'>Mean MAE Avg: {format_float(merged['mae_avg_pips'].mean())} pips</div>"
            + (
                f"<div class='summary-pill'>Mean corr2 Avg HLC: {format_float(merged['corr2_avg_hlc'].mean())}</div>"
                if "corr2_avg_hlc" in merged.columns else ""
            )
            + (
                f"<div class='summary-pill'>Mean DA body: {format_float(merged['da_body_pct'].mean(), 2)}%</div>"
                if "da_body_pct" in merged.columns else ""
            )
            + f"{svg_bar_chart(score_series, 'MAE Avg per Fold (pips)', color='#0f766e') if score_series else ''}"
            + f"{merged[metric_cols].to_html(index=False, float_format=lambda x: f'{x:.4f}', classes='data-table')}"
            + "</section>"
        )
    return "".join(sections)


def collect_metrics(result_dir: Path, recursive: bool = False) -> pd.DataFrame:
    rows = []
    for summary_path in discover_summary_files(result_dir, recursive=recursive):
        preds_path = infer_preds_path(summary_path)
        if not preds_path or not preds_path.exists():
            continue
        match = re.search(r"fold(\d+)_summary\.json$", summary_path.name)
        metrics = compute_prediction_metrics(pd.read_csv(preds_path))
        rows.append(
            {
                "run_dir": str(summary_path.parent.relative_to(result_dir)),
                "summary": summary_path.name,
                "fold": int(match.group(1)) if match else np.nan,
                **metrics,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["run_dir", "fold"]).reset_index(drop=True)


def build_html(result_dir: Path, tail: int, recursive: bool = False) -> str:
    summaries = discover_summary_files(result_dir, recursive=recursive)
    run_sections = [build_run_section(result_dir, path, tail) for path in summaries]
    rolling_sections = build_rolling_sections(result_dir, recursive=recursive)
    path_text = str(result_dir).lower()
    if "tvt_v02" in path_text and "d1" in path_text:
        page_title = "ARIMA Output Report - EUR/USD D1 Last3 (TVT v02)"
        lead_html = f"<p class='lead'>Source folder: {html.escape(str(result_dir))} | Scope: EUR/USD D1 OHLC, TVT v02 evaluation folds 19-21</p>"
    elif "d1_ohlc" in path_text or "d1" in result_dir.name.lower():
        page_title = "ARIMA Output Report - EUR/USD 1D WF72m/1m Last5"
        lead_html = f"<p class='lead'>Source folder: {html.escape(str(result_dir))} | Scope: EUR/USD 1D OHLC</p>"
    elif "tvt_v02" in path_text and "h4" in path_text:
        page_title = "ARIMA Output Report - EUR/USD H4 Last3 (TVT v02)"
        lead_html = f"<p class='lead'>Source folder: {html.escape(str(result_dir))} | Scope: EUR/USD H4 OHLC, TVT v02 evaluation folds 19-21</p>"
    else:
        page_title = "ARIMA Output Report"
        lead_html = f"<p class='lead'>Source folder: {html.escape(str(result_dir))}</p>"
    css = """
body { font-family: Arial, sans-serif; margin: 24px; color: #0f172a; background: #f8fafc; }
h1, h2, h3 { margin: 0 0 12px; }
h1 { margin-bottom: 18px; }
p.lead { margin: 0 0 24px; color: #475569; }
p.metric-note { margin: -4px 0 12px; color: #475569; line-height: 1.45; }
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
        f"<h1>{page_title}</h1>",
        lead_html,
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
        f"<title>{page_title}</title>"
        f"<style>{css}</style>"
        "</head><body>"
        + "".join(body) +
        "</body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report for ARIMA outputs.")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Directory containing ARIMA result files.")
    parser.add_argument("--out", default=None, help="Output HTML path.")
    parser.add_argument("--tail", type=int, default=0, help="Deprecated; tail row tables are no longer rendered.")
    parser.add_argument("--recursive", action="store_true", help="Scan nested result folders. By default only the selected result directory is used.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    out_path = Path(args.out).resolve() if args.out else (result_dir / "index.html")
    html_doc = build_html(result_dir, args.tail, recursive=args.recursive)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"ARIMA HTML report written to {out_path}")
    metrics_df = collect_metrics(result_dir, recursive=args.recursive)
    if not metrics_df.empty:
        metrics_path = out_path.with_name(f"{out_path.stem}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"ARIMA metrics CSV written to {metrics_path}")


if __name__ == "__main__":
    main()
