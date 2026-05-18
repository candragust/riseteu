#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from mae_atr_report import (
    PIP_FACTOR,
    build_error_bar,
    build_error_bar_avg,
    build_overlay_visual,
    build_visual,
    compute_mae,
    ensure_atr_columns,
)


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build detailed 1D visual reports and static assets for thesis insertion."
    )
    parser.add_argument("--rolling-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--title-prefix", required=True)
    parser.add_argument("--last-fold", type=int, default=21)
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--out-ohlc-png", required=True)
    parser.add_argument("--out-overlay-png", required=True)
    parser.add_argument("--out-error-png", required=True)
    return parser.parse_args()


def fold_file(base: Path, fold: int, suffix: str) -> Path:
    return base / f"fold{int(fold):02d}_{suffix}.csv"


def compute_fold_metrics(rolling_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(rolling_dir / "rolling_fixed_summary.csv").sort_values("fold").reset_index(drop=True)
    rows: list[dict] = []
    for _, row in summary.iterrows():
        fold = int(row["fold"])
        preds = pd.read_csv(fold_file(rolling_dir, fold, "preds"))
        metrics = compute_mae(preds)
        rows.append(
            {
                "fold": fold,
                "train_start": row["train_start"],
                "train_end": row["train_end"],
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "samples": len(preds),
                "mae_open_pips": metrics["mae_open"] * PIP_FACTOR,
                "mae_high_pips": metrics["mae_high"] * PIP_FACTOR,
                "mae_low_pips": metrics["mae_low"] * PIP_FACTOR,
                "mae_close_pips": metrics["mae_close"] * PIP_FACTOR,
                "mae_avg_pips": metrics["mae_avg"] * PIP_FACTOR,
                "preds_path": str(fold_file(rolling_dir, fold, "preds")),
                "data_path": str(fold_file(rolling_dir, fold, "data")),
                "split_ratio": float(row["split_ratio"]),
            }
        )
    return pd.DataFrame(rows)


def load_test_slice(data_path: Path, split_ratio: float, preds_df: pd.DataFrame) -> pd.DataFrame:
    full = ensure_atr_columns(pd.read_csv(data_path))
    start_val = int(len(full) * split_ratio)
    val_df = full.iloc[start_val:].reset_index(drop=True)
    if len(val_df) != len(preds_df):
        val_df = full.tail(len(preds_df)).reset_index(drop=True)
    return val_df


def compute_fold_detail(preds_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float | int]:
    errs = np.abs(
        preds_df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
        - test_df[["open", "high", "low", "close"]].values
    ) * PIP_FACTOR
    err_avg = errs.mean(axis=1)
    atr6 = test_df["atr6"] * PIP_FACTOR
    atr12 = test_df["atr12"] * PIP_FACTOR
    pred_bull = int((preds_df["pred_close"] > preds_df["pred_open"]).sum())
    pred_bear = int((preds_df["pred_close"] < preds_df["pred_open"]).sum())
    pred_doji = int((preds_df["pred_close"] == preds_df["pred_open"]).sum())
    true_bull = int((test_df["close"] > test_df["open"]).sum())
    true_bear = int((test_df["close"] < test_df["open"]).sum())
    true_doji = int((test_df["close"] == test_df["open"]).sum())
    return {
        "mae_avg_pips": float(err_avg.mean()),
        "mae_open_pips": float(errs[:, 0].mean()),
        "mae_high_pips": float(errs[:, 1].mean()),
        "mae_low_pips": float(errs[:, 2].mean()),
        "mae_close_pips": float(errs[:, 3].mean()),
        "atr6_mean_pips": float(atr6.mean()),
        "atr12_mean_pips": float(atr12.mean()),
        "mae_vs_atr6_pct": float(err_avg.mean() / atr6.mean() * 100.0),
        "mae_vs_atr12_pct": float(err_avg.mean() / atr12.mean() * 100.0),
        "avg_le_atr6_pct": float((err_avg <= atr6).mean() * 100.0),
        "avg_le_atr12_pct": float((err_avg <= atr12).mean() * 100.0),
        "pred_bull": pred_bull,
        "pred_bear": pred_bear,
        "pred_doji": pred_doji,
        "true_bull": true_bull,
        "true_bear": true_bear,
        "true_doji": true_doji,
    }


def build_meta_text(config: dict, fold_row: pd.Series) -> str:
    return (
        f"Train {fold_row['train_start']}→{fold_row['train_end']} | "
        f"Test {fold_row['test_start']}→{fold_row['test_end']} | "
        f"window={config['window']}, units={config['units']}, lr={config['lr']}, "
        f"lambda={config['lambda_coef']}, sigma={config['sigma_coef']}, "
        f"batch={config['batch']}, epochs={config['epochs']}"
    )


def build_detail_html(
    title: str,
    model_label: str,
    metrics_df: pd.DataFrame,
    fold_row: pd.Series,
    fold_detail: dict[str, float | int],
    preds_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_path: Path,
    meta_text: str,
):
    best = metrics_df.loc[metrics_df["mae_avg_pips"].idxmin()]
    worst = metrics_df.loc[metrics_df["mae_avg_pips"].idxmax()]
    summary_table = metrics_df[
        [
            "fold",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "samples",
            "mae_avg_pips",
            "mae_open_pips",
            "mae_high_pips",
            "mae_low_pips",
            "mae_close_pips",
        ]
    ].to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="data-table")

    detail_table = pd.DataFrame(
        [
            {
                "mae_open_pips": fold_detail["mae_open_pips"],
                "mae_high_pips": fold_detail["mae_high_pips"],
                "mae_low_pips": fold_detail["mae_low_pips"],
                "mae_close_pips": fold_detail["mae_close_pips"],
                "mae_avg_pips": fold_detail["mae_avg_pips"],
                "atr6_mean_pips": fold_detail["atr6_mean_pips"],
                "atr12_mean_pips": fold_detail["atr12_mean_pips"],
                "mae_vs_atr6_pct": fold_detail["mae_vs_atr6_pct"],
                "mae_vs_atr12_pct": fold_detail["mae_vs_atr12_pct"],
                "avg_le_atr6_pct": fold_detail["avg_le_atr6_pct"],
                "avg_le_atr12_pct": fold_detail["avg_le_atr12_pct"],
                "pred_bull": fold_detail["pred_bull"],
                "pred_bear": fold_detail["pred_bear"],
                "true_bull": fold_detail["true_bull"],
                "true_bear": fold_detail["true_bear"],
            }
        ]
    ).to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="data-table")

    fig_overlay = build_overlay_visual(test_df, preds_df, "fold 21 full test")
    fig_close = build_visual(test_df, preds_df, "fold 21 full test")
    fig_err_close = build_error_bar(preds_df, test_df, fold_detail["atr6_mean_pips"], fold_detail["atr12_mean_pips"], "fold 21 full test")
    fig_err_avg = build_error_bar_avg(preds_df, test_df, fold_detail["atr6_mean_pips"], fold_detail["atr12_mean_pips"], "fold 21 full test")

    html = f"""
<html>
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
    .meta {{ background: #f6f8fa; border: 1px solid #d0d7de; padding: 12px 14px; border-radius: 8px; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    .data-table th, .data-table td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    .data-table th {{ background: #f6f8fa; }}
    .data-table td:first-child, .data-table th:first-child {{ text-align: left; }}
    h1, h2, h3 {{ color: #183a6b; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">
    <p><strong>Model:</strong> {model_label}</p>
    <p><strong>Konfigurasi Fold 21:</strong> {meta_text}</p>
    <p><strong>Ringkasan Last5:</strong>
      mean MAE avg = <strong>{metrics_df['mae_avg_pips'].mean():.4f} pips</strong>,
      best fold = <strong>{int(best['fold'])}</strong> ({best['mae_avg_pips']:.4f} pips),
      worst fold = <strong>{int(worst['fold'])}</strong> ({worst['mae_avg_pips']:.4f} pips).
    </p>
  </div>
  <h2>MAE per Fold (Last5)</h2>
  {summary_table}
  <h2>Fold 21 Detail: MAE vs ATR dan Arah Candle</h2>
  {detail_table}
  <h2>Charts</h2>
  <h3>Overlay True vs Predicted OHLC</h3>
  {fig_overlay.to_html(full_html=False, include_plotlyjs='cdn')}
  <h3>True vs Predicted Close</h3>
  {fig_close.to_html(full_html=False, include_plotlyjs=False)}
  <h3>Error Close vs ATR</h3>
  {fig_err_close.to_html(full_html=False, include_plotlyjs=False)}
  <h3>Error AVG vs ATR</h3>
  {fig_err_avg.to_html(full_html=False, include_plotlyjs=False)}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def draw_candles(ax, df: pd.DataFrame, *, prefix: str, width: float, alpha: float, linewidth: float, label_prefix: str):
    x = np.arange(len(df))
    for i in range(len(df)):
        o = df[f"{prefix}open"].iloc[i]
        h = df[f"{prefix}high"].iloc[i]
        l = df[f"{prefix}low"].iloc[i]
        c = df[f"{prefix}close"].iloc[i]
        up = c >= o
        color = "#16a34a" if up else "#dc2626"
        ax.vlines(x[i], l, h, color=color, linewidth=linewidth, alpha=alpha)
        lower = min(o, c)
        height = max(abs(c - o), 1e-6)
        rect = Rectangle(
            (x[i] - width / 2.0, lower),
            width,
            height,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_patch(rect)
    ax.set_xlim(-1, len(df))


def save_overlay_png(preds_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    true_df = test_df.rename(columns={"open": "true_open", "high": "true_high", "low": "true_low", "close": "true_close"})
    draw_candles(ax, true_df, prefix="true_", width=0.58, alpha=0.22, linewidth=1.0, label_prefix="True")
    draw_candles(ax, preds_df, prefix="pred_", width=0.24, alpha=0.95, linewidth=1.2, label_prefix="Pred")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Index (fold 21 full test)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    legend_handles = [
        Patch(facecolor="#16a34a", edgecolor="#16a34a", alpha=0.22, label="True Bullish/Bearish Candle"),
        Patch(facecolor="#16a34a", edgecolor="#16a34a", alpha=0.95, label="Predicted Candle Overlay"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_error_png(preds_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path, title: str):
    errs = np.abs(
        preds_df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
        - test_df[["open", "high", "low", "close"]].values
    )
    err_avg = errs.mean(axis=1) * PIP_FACTOR
    atr6 = test_df["atr6"] * PIP_FACTOR
    atr12 = test_df["atr12"] * PIP_FACTOR
    x = np.arange(len(err_avg))

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(x, err_avg, color="#2563eb", marker="o", linewidth=1.8, markersize=4, label="|Error avg| (pips)")
    ax.plot(x, atr6, color="#dc2626", linewidth=1.8, label="ATR6")
    ax.plot(x, atr12, color="#16a34a", linewidth=1.8, linestyle="--", label="ATR12")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Index (fold 21 full test)")
    ax.set_ylabel("Absolute Error (pips)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_ohlc_png(preds_df: pd.DataFrame, out_path: Path, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    pairs = [
        ("open", "Actual Open", "Pred Open", "tab:blue"),
        ("high", "Actual High", "Pred High", "tab:orange"),
        ("low", "Actual Low", "Pred Low", "tab:green"),
        ("close", "Actual Close", "Pred Close", "tab:red"),
    ]
    time_steps = np.arange(len(preds_df))
    for ax, (col, label_true, label_pred, color) in zip(axes.flat, pairs):
        ax.plot(time_steps, preds_df[f"true_{col}"], color="black", linewidth=0.9, label=label_true)
        ax.plot(time_steps, preds_df[f"pred_{col}"], color=color, linestyle="--", linewidth=0.85, label=label_pred)
        ax.set_title(col.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    rolling_dir = Path(args.rolling_dir)
    config = json.loads(Path(args.config).read_text())
    metrics_df = compute_fold_metrics(rolling_dir)
    fold_row = pd.read_csv(rolling_dir / "rolling_fixed_summary.csv").query("fold == @args.last_fold").iloc[0]
    preds_df = pd.read_csv(fold_file(rolling_dir, args.last_fold, "preds"))
    test_df = load_test_slice(fold_file(rolling_dir, args.last_fold, "data"), float(fold_row["split_ratio"]), preds_df)
    fold_detail = compute_fold_detail(preds_df, test_df)

    title = f"{args.title_prefix} - Detail MAE vs ATR WF72m/1m Last5"
    build_detail_html(
        title,
        args.model_label,
        metrics_df,
        fold_row,
        fold_detail,
        preds_df,
        test_df,
        Path(args.out_html),
        build_meta_text(config, fold_row),
    )

    combined_frames = []
    for fold in metrics_df["fold"].tolist():
        combined_frames.append(pd.read_csv(fold_file(rolling_dir, int(fold), "preds")))
    combined_preds = pd.concat(combined_frames, ignore_index=True)
    save_ohlc_png(combined_preds, Path(args.out_ohlc_png), f"Actual vs Predicted OHLC - {args.model_label} (1D last5)")
    save_overlay_png(preds_df, test_df, Path(args.out_overlay_png), f"Overlay True vs Predicted OHLC - {args.model_label} (1D fold 21)")
    save_error_png(preds_df, test_df, Path(args.out_error_png), f"Error AVG vs ATR - {args.model_label} (1D fold 21)")
    print(f"Wrote report: {Path(args.out_html).resolve()}")


if __name__ == "__main__":
    main()
