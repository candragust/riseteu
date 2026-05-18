#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_d1_visual_reports_and_assets_v01 import (
    build_detail_html,
    compute_fold_detail,
    compute_fold_metrics,
    load_test_slice,
    save_error_png,
    save_ohlc_png,
    save_overlay_png,
)
from generate_model_diagnostics_docs import analyze_arima, make_arima_html


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ARIMA 1D residual diagnostics and detailed MAE/ATR report."
    )
    parser.add_argument("--rolling-dir", required=True)
    parser.add_argument("--residual-html", required=True)
    parser.add_argument("--detail-html", required=True)
    parser.add_argument("--out-ohlc-png", required=True)
    parser.add_argument("--out-overlay-png", required=True)
    parser.add_argument("--out-error-png", required=True)
    parser.add_argument("--last-fold", type=int, default=21)
    return parser.parse_args()


def fold_file(base: Path, fold: int, suffix: str) -> Path:
    return base / f"fold{int(fold):02d}_{suffix}.csv"


def build_meta_text(summary: dict, fold_row: pd.Series) -> str:
    orders = summary["targets"]
    return (
        f"Train {fold_row['train_start']}→{fold_row['train_end']} | "
        f"Test {fold_row['test_start']}→{fold_row['test_end']} | "
        f"order search=AIC grid search | "
        f"open={tuple(orders['open']['selected_order'])}, "
        f"high={tuple(orders['high']['selected_order'])}, "
        f"low={tuple(orders['low']['selected_order'])}, "
        f"close={tuple(orders['close']['selected_order'])}"
    )


def main():
    args = parse_args()
    rolling_dir = Path(args.rolling_dir)

    fold_df, order_df, arima_summary, combined_close_resid_pips = analyze_arima(rolling_dir)
    residual_title = "ARIMA Residual Diagnostics - EUR/USD 1D WF72m/1m Last5"
    scope_note = "baseline ARIMA OHLC-only pada EUR/USD 1D, walk-forward fixed 72 bulan train / 1 bulan test, fold 17-21."
    make_arima_html(
        fold_df,
        order_df,
        arima_summary,
        combined_close_resid_pips,
        Path(args.residual_html),
        page_title=residual_title,
        scope_note=scope_note,
    )

    metrics_df = compute_fold_metrics(rolling_dir)
    rolling_summary = pd.read_csv(rolling_dir / "rolling_fixed_summary.csv")
    fold_row = rolling_summary.query("fold == @args.last_fold").iloc[0]
    preds_df = pd.read_csv(fold_file(rolling_dir, args.last_fold, "preds"))
    test_df = load_test_slice(fold_file(rolling_dir, args.last_fold, "data"), float(fold_row["split_ratio"]), preds_df)
    fold_detail = compute_fold_detail(preds_df, test_df)
    summary = json.loads((rolling_dir / f"fold{int(args.last_fold):02d}_summary.json").read_text(encoding="utf-8"))

    detail_title = "EUR/USD 1D ARIMA - Detail MAE vs ATR WF72m/1m Last5"
    build_detail_html(
        detail_title,
        "ARIMA",
        metrics_df,
        fold_row,
        fold_detail,
        preds_df,
        test_df,
        Path(args.detail_html),
        build_meta_text(summary, fold_row),
    )

    combined_frames = [pd.read_csv(fold_file(rolling_dir, int(fold), "preds")) for fold in metrics_df["fold"].tolist()]
    combined_preds = pd.concat(combined_frames, ignore_index=True)
    save_ohlc_png(combined_preds, Path(args.out_ohlc_png), "Actual vs Predicted OHLC - ARIMA (1D last5)")
    save_overlay_png(preds_df, test_df, Path(args.out_overlay_png), "Overlay True vs Predicted OHLC - ARIMA (1D fold 21)")
    save_error_png(preds_df, test_df, Path(args.out_error_png), "Error AVG vs ATR - ARIMA (1D fold 21)")

    print(f"Wrote residual report: {Path(args.residual_html).resolve()}")
    print(f"Wrote detail report: {Path(args.detail_html).resolve()}")


if __name__ == "__main__":
    main()
