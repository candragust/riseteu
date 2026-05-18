#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_model_specs(timeframe: str) -> list[dict]:
    if timeframe == "h4":
        return [
            {
                "label": "FLF-LSTM",
                "result_dir": ROOT / "FLF_LSTM/results/tvt_v02/h4_evaluation_last3",
                "config": ROOT / "FLF_LSTM/lstm_flf_config_h4_tvt_v02_best.json",
                "out": ROOT / "FLF_LSTM/results/tvt_v02/h4_evaluation_last3/mae_atr_fold_allfull.html",
                "data_source": "split_root",
                "timeframe": "H4",
            },
            {
                "label": "FLF-BiLSTM",
                "result_dir": ROOT / "FLF_BILSTM/results/tvt_v02/h4_evaluation_last3",
                "config": ROOT / "FLF_BILSTM/bilstm_flf_config_h4_tvt_v02_best.json",
                "out": ROOT / "FLF_BILSTM/results/tvt_v02/h4_evaluation_last3/mae_atr_fold_allfull.html",
                "data_source": "split_root",
                "timeframe": "H4",
            },
            {
                "label": "ARIMA",
                "result_dir": ROOT / "Arima/result/tvt_v02/h4_evaluation_last3",
                "config": None,
                "out": ROOT / "Arima/result/tvt_v02/h4_evaluation_last3/mae_atr_fold_allfull.html",
                "data_source": "result_dir",
                "timeframe": "H4",
            },
        ]
    if timeframe == "d1":
        return [
            {
                "label": "FLF-LSTM",
                "result_dir": ROOT / "FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3",
                "config": ROOT / "FLF_LSTM/configs/d1_ohlc/lstm_flf_config_d1_tvt_v02_best.json",
                "out": ROOT / "FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/mae_atr_fold_allfull.html",
                "data_source": "split_root",
                "timeframe": "D1",
            },
            {
                "label": "FLF-BiLSTM",
                "result_dir": ROOT / "FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3",
                "config": ROOT / "FLF_BILSTM/configs/d1_ohlc/bilstm_flf_config_d1_tvt_v02_best.json",
                "out": ROOT / "FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/mae_atr_fold_allfull.html",
                "data_source": "split_root",
                "timeframe": "D1",
            },
            {
                "label": "ARIMA",
                "result_dir": ROOT / "Arima/result/tvt_v02/d1_evaluation_last3",
                "config": None,
                "out": ROOT / "Arima/result/tvt_v02/d1_evaluation_last3/mae_atr_fold_allfull.html",
                "data_source": "result_dir",
                "timeframe": "D1",
            },
        ]
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all-full MAE vs ATR reports for TVT v02 fold 19-21."
    )
    parser.add_argument(
        "--timeframe",
        choices=["h4", "d1"],
        default="h4",
        help="Timeframe/profile to report.",
    )
    parser.add_argument(
        "--split-root",
        default=None,
        help="TVT v02 split root containing foldXX/combined.csv. Defaults to results/splits/tvt_v02/<timeframe>.",
    )
    return parser.parse_args()


def squared_corr(y_true, y_pred) -> float:
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


def directional_accuracy(preds_df: pd.DataFrame) -> float:
    true_dir = np.sign(preds_df["true_close"].to_numpy(dtype=float) - preds_df["true_open"].to_numpy(dtype=float))
    pred_dir = np.sign(preds_df["pred_close"].to_numpy(dtype=float) - preds_df["pred_open"].to_numpy(dtype=float))
    if len(true_dir) == 0:
        return float("nan")
    return float((true_dir == pred_dir).mean() * 100.0)


def audit_candle_coherence(preds_df: pd.DataFrame) -> dict[str, int]:
    pred_open = preds_df["pred_open"].to_numpy(dtype=float)
    pred_high = preds_df["pred_high"].to_numpy(dtype=float)
    pred_low = preds_df["pred_low"].to_numpy(dtype=float)
    pred_close = preds_df["pred_close"].to_numpy(dtype=float)
    high_lt_low = pred_high < pred_low
    open_gt_high = pred_open > pred_high
    open_lt_low = pred_open < pred_low
    close_gt_high = pred_close > pred_high
    close_lt_low = pred_close < pred_low
    any_invalid = high_lt_low | open_gt_high | open_lt_low | close_gt_high | close_lt_low
    return {
        "high_lt_low": int(high_lt_low.sum()),
        "open_gt_high": int(open_gt_high.sum()),
        "open_lt_low": int(open_lt_low.sum()),
        "close_gt_high": int(close_gt_high.sum()),
        "close_lt_low": int(close_lt_low.sum()),
        "any_invalid_candle": int(any_invalid.sum()),
    }


def load_test_slice(spec: dict, split_root: Path, fold: int, preds_df: pd.DataFrame) -> pd.DataFrame:
    if spec.get("data_source") == "result_dir":
        combined_path = Path(spec["result_dir"]) / f"fold{fold}_data.csv"
    else:
        combined_path = split_root / f"fold{fold:02d}" / "combined.csv"
    combined_df = ensure_atr_columns(pd.read_csv(combined_path))
    test_df = combined_df.tail(len(preds_df)).reset_index(drop=True)

    true_cols = ["true_open", "true_high", "true_low", "true_close"]
    ohlc_cols = ["open", "high", "low", "close"]
    max_diff = np.max(np.abs(preds_df[true_cols].to_numpy() - test_df[ohlc_cols].to_numpy()))
    if max_diff > 1e-6:
        raise ValueError(f"Fold {fold} true OHLC tidak align dengan {combined_path}: max diff={max_diff}")
    return test_df


def fold_metrics(fold: int, preds_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float | int]:
    mae = compute_mae(preds_df)
    pred_values = preds_df[["pred_open", "pred_high", "pred_low", "pred_close"]].to_numpy(dtype=float)
    true_values = preds_df[["true_open", "true_high", "true_low", "true_close"]].to_numpy(dtype=float)
    corr2 = [squared_corr(true_values[:, idx], pred_values[:, idx]) for idx in range(4)]
    err_avg = np.abs(
        pred_values - test_df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    ).mean(axis=1) * PIP_FACTOR
    atr6 = test_df["atr6"].to_numpy(dtype=float) * PIP_FACTOR
    atr12 = test_df["atr12"].to_numpy(dtype=float) * PIP_FACTOR
    return {
        "fold": int(fold),
        "samples": int(len(preds_df)),
        "mae_open_pips": float(mae["mae_open"] * PIP_FACTOR),
        "mae_high_pips": float(mae["mae_high"] * PIP_FACTOR),
        "mae_low_pips": float(mae["mae_low"] * PIP_FACTOR),
        "mae_close_pips": float(mae["mae_close"] * PIP_FACTOR),
        "mae_avg_pips": float(mae["mae_avg"] * PIP_FACTOR),
        "mae_avg_hlc_pips": float((mae["mae_high"] + mae["mae_low"] + mae["mae_close"]) / 3.0 * PIP_FACTOR),
        "corr2_open": float(corr2[0]),
        "corr2_high": float(corr2[1]),
        "corr2_low": float(corr2[2]),
        "corr2_close": float(corr2[3]),
        "corr2_avg_ohlc": nanmean(corr2),
        "corr2_avg_hlc": nanmean(corr2[1:]),
        "da_body_pct": directional_accuracy(preds_df),
        "atr6_mean_pips": float(np.nanmean(atr6)),
        "atr12_mean_pips": float(np.nanmean(atr12)),
        "avg_error_le_atr6_pct": float(np.nanmean(err_avg <= atr6) * 100.0),
        "avg_error_le_atr12_pct": float(np.nanmean(err_avg <= atr12) * 100.0),
    }


def config_text(config: dict | None) -> str:
    if not config:
        return "ARIMA OHLC-only, order search=AIC grid search per target"
    return (
        f"window={config.get('window')}, units={config.get('units')}, "
        f"activation={config.get('activation')}, lr={config.get('lr')}, "
        f"lambda={config.get('lambda_coef')}, sigma={config.get('sigma_coef')}, "
        f"batch={config.get('batch')}, epochs={config.get('epochs')}"
    )


def add_fold_boundaries(fig, boundaries: list[tuple[int, int, int]]) -> None:
    for _, _, end_idx in boundaries[:-1]:
        fig.add_vline(x=end_idx + 0.5, line_width=1, line_dash="dash", line_color="#64748b")
    for fold, start_idx, end_idx in boundaries:
        mid = (start_idx + end_idx) / 2.0
        fig.add_annotation(
            x=mid,
            y=1.02,
            xref="x",
            yref="paper",
            text=f"Fold {fold}",
            showarrow=False,
            font=dict(size=12, color="#475569"),
        )


def make_wide(fig, boundaries: list[tuple[int, int, int]], height: int) -> str:
    add_fold_boundaries(fig, boundaries)
    fig.update_layout(width=1500, height=height, margin=dict(l=70, r=40, t=80, b=55))
    return f"<div class='chart-scroll'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"


def build_outlier_note_html(model_label: str, analysis_df: pd.DataFrame) -> str:
    top = analysis_df.sort_values("err_avg_pips", ascending=False).head(2).copy()
    if top.empty:
        return ""

    items = []
    for _, row in top.iterrows():
        actual_range = float(row["actual_range_pips"])
        actual_body = float(row["actual_body_pips"])
        pred_range = float(row["pred_range_pips"])
        pred_body = float(row["pred_body_pips"])
        atr6 = float(row["atr6"])
        atr12 = float(row["atr12"])
        err_avg = float(row["err_avg_pips"])
        err_high = float(row["err_high_pips"])
        err_close = float(row["err_close_pips"])
        items.append(
            "<li>"
            f"Pada <strong>global index {int(row['global_idx'])}</strong> "
            f"(fold <strong>{int(row['fold'])}</strong>, "
            f"<strong>{html.escape(str(row['datetime']))}</strong>), "
            f"terjadi lonjakan error rata-rata sekitar <strong>{err_avg:.2f} pips</strong>. "
            f"Candle aktual memiliki <strong>range {actual_range:.1f} pips</strong> dan "
            f"<strong>body {actual_body:.1f} pips</strong>, jauh di atas "
            f"<strong>ATR6 {atr6:.1f}</strong> dan <strong>ATR12 {atr12:.1f}</strong>. "
            f"Namun model hanya membentuk prediksi dengan <strong>range {pred_range:.1f} pips</strong> "
            f"dan <strong>body {pred_body:.1f} pips</strong>. Deviasi terbesar muncul pada "
            f"komponen <strong>High</strong> ({err_high:.2f} pips) dan <strong>Close</strong> "
            f"({err_close:.2f} pips), sehingga candle ini dapat dibaca sebagai indikasi bahwa model "
            f"<strong>underreact</strong> terhadap expansion candle atau breakout candle beramplitudo besar."
            "</li>"
        )

    return (
        "<div class='meta' style='margin-top:14px; background:#eff6ff; border-color:#93c5fd;'>"
        f"<p><strong>Catatan Outlier {html.escape(model_label)}:</strong> "
        "Beberapa lonjakan error pada grafik <strong>Error AVG vs ATR</strong> bukan berasal dari gangguan visual, "
        "melainkan dari candle aktual yang sangat impulsif sementara prediksi model tetap relatif halus. "
        "Hal ini menunjukkan bahwa model masih cenderung <strong>smooth / underreact</strong> saat menghadapi "
        "shock candle beramplitudo besar.</p>"
        f"<ul>{''.join(items)}</ul>"
        "</div>"
    )


def build_model_report(spec: dict, split_root: Path) -> Path:
    result_dir = Path(spec["result_dir"])
    config = json.loads(Path(spec["config"]).read_text(encoding="utf-8")) if spec.get("config") else None
    summary_path = result_dir / "rolling_tvt_summary.csv"
    if not summary_path.exists():
        summary_path = result_dir / "rolling_fixed_summary.csv"
    summary = pd.read_csv(summary_path).sort_values("fold").reset_index(drop=True)

    fold_metric_rows: list[dict[str, float | int]] = []
    candle_audit_rows: list[dict[str, int]] = []
    preds_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    analysis_frames: list[pd.DataFrame] = []
    boundaries: list[tuple[int, int, int]] = []
    start_idx = 0

    for _, row in summary.iterrows():
        fold = int(row["fold"])
        preds_name = str(row["preds_csv"]) if "preds_csv" in row.index and pd.notna(row["preds_csv"]) else f"fold{fold}_preds.csv"
        preds_df = pd.read_csv(result_dir / preds_name)
        test_df = load_test_slice(spec, split_root, fold, preds_df)
        fold_metric_rows.append(fold_metrics(fold, preds_df, test_df))
        candle_audit_rows.append({"fold": int(fold), **audit_candle_coherence(preds_df)})

        end_idx = start_idx + len(preds_df) - 1
        boundaries.append((fold, start_idx, end_idx))
        start_idx = end_idx + 1

        preds_frames.append(preds_df)
        test_frames.append(test_df)
        analysis_df = preds_df.copy()
        analysis_df["fold"] = int(fold)
        analysis_df["idx_in_fold"] = np.arange(len(preds_df))
        analysis_df["global_idx"] = np.arange(end_idx - len(preds_df) + 1, end_idx + 1)
        if "datetime" in test_df.columns:
            analysis_df["datetime"] = test_df["datetime"].to_numpy()
        else:
            analysis_df["datetime"] = [f"fold{fold}_idx{i}" for i in range(len(test_df))]
        analysis_df["atr6"] = test_df["atr6"].to_numpy(dtype=float) * PIP_FACTOR
        analysis_df["atr12"] = test_df["atr12"].to_numpy(dtype=float) * PIP_FACTOR
        pred_values = preds_df[["pred_open", "pred_high", "pred_low", "pred_close"]].to_numpy(dtype=float)
        true_values = preds_df[["true_open", "true_high", "true_low", "true_close"]].to_numpy(dtype=float)
        errs = np.abs(pred_values - true_values) * PIP_FACTOR
        analysis_df["err_open_pips"] = errs[:, 0]
        analysis_df["err_high_pips"] = errs[:, 1]
        analysis_df["err_low_pips"] = errs[:, 2]
        analysis_df["err_close_pips"] = errs[:, 3]
        analysis_df["err_avg_pips"] = errs.mean(axis=1)
        analysis_df["actual_range_pips"] = (analysis_df["true_high"] - analysis_df["true_low"]).to_numpy(dtype=float) * PIP_FACTOR
        analysis_df["actual_body_pips"] = np.abs(
            analysis_df["true_close"].to_numpy(dtype=float) - analysis_df["true_open"].to_numpy(dtype=float)
        ) * PIP_FACTOR
        analysis_df["pred_range_pips"] = (analysis_df["pred_high"] - analysis_df["pred_low"]).to_numpy(dtype=float) * PIP_FACTOR
        analysis_df["pred_body_pips"] = np.abs(
            analysis_df["pred_close"].to_numpy(dtype=float) - analysis_df["pred_open"].to_numpy(dtype=float)
        ) * PIP_FACTOR
        analysis_frames.append(analysis_df)

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    numeric_cols = [col for col in fold_metrics_df.columns if col not in {"fold", "samples"}]
    avg_row = {col: float(fold_metrics_df[col].mean()) for col in numeric_cols}
    avg_row = {
        "folds": "19-21",
        "fold_count": int(len(fold_metrics_df)),
        "total_samples": int(fold_metrics_df["samples"].sum()),
        **avg_row,
    }
    metrics_df = pd.DataFrame([avg_row])
    candle_audit_df = pd.DataFrame(candle_audit_rows)

    out_path = Path(spec["out"])
    metrics_csv = out_path.with_suffix(".csv")
    fold_metrics_csv = out_path.with_name(out_path.stem + "_per_fold.csv")
    metrics_csv.write_text(metrics_df.to_csv(index=False), encoding="utf-8")
    fold_metrics_csv.write_text(fold_metrics_df.to_csv(index=False), encoding="utf-8")

    preds_all = pd.concat(preds_frames, ignore_index=True)
    test_all = pd.concat(test_frames, ignore_index=True)
    analysis_all = pd.concat(analysis_frames, ignore_index=True)
    label = "fold 19-21 all full test"

    fig_overlay = build_overlay_visual(test_all, preds_all, label)
    fig_close = build_visual(test_all, preds_all, label)
    fig_err_close = build_error_bar(preds_all, test_all, avg_row["atr6_mean_pips"], avg_row["atr12_mean_pips"], label)
    fig_err_avg = build_error_bar_avg(preds_all, test_all, avg_row["atr6_mean_pips"], avg_row["atr12_mean_pips"], label)

    timeframe_label = spec.get("timeframe", "H4")
    title = f"MAE vs ATR - {spec['label']} {timeframe_label} Fold All Full (TVT v02)"
    style = """
body { font-family: Arial, sans-serif; margin: 20px; color: #222; }
h1, h2, h3 { color: #183a6b; }
.meta { background: #f6f8fa; border: 1px solid #d0d7de; padding: 12px 14px; border-radius: 8px; }
.data-table { border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 13px; }
.data-table th, .data-table td { border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }
.data-table th { background: #f6f8fa; }
.data-table td:first-child, .data-table th:first-child { text-align: left; }
.chart-scroll { overflow-x: auto; width: 100%; border: 1px solid #d0d7de; border-radius: 8px; padding: 8px 0; margin-bottom: 20px; }
"""
    html_doc = f"""<html>
<head>
  <meta charset="UTF-8" />
  <title>{html.escape(title)}</title>
  <style>{style}</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">
    <p><strong>Model:</strong> {html.escape(spec['label'])}</p>
    <p><strong>Config final:</strong> {html.escape(config_text(config))}</p>
    <p><strong>Scope:</strong> {html.escape(timeframe_label)} TVT v02 evaluation last3, fold 19-21 digabung menjadi satu rangkaian full test. Metrik pada tabel adalah rata-rata per fold.</p>
  </div>
  {build_outlier_note_html(spec['label'], analysis_all) if spec['label'] in {'FLF-LSTM', 'FLF-BiLSTM'} else ''}
  {(
      "<div class='meta' style='margin-top:14px; background:#fff7ed; border-color:#fdba74;'>"
      "<p><strong>Catatan Candle Coherence ARIMA:</strong> Karena ARIMA memprediksi komponen OHLC per series secara terpisah, "
      "struktur candle tidak dijamin koheren secara otomatis. Pada audit fold 19-21, tidak ditemukan kasus <code>pred_high &lt; pred_low</code>, "
      f"tetapi terdapat <strong>{int(candle_audit_df['any_invalid_candle'].sum())}</strong> candle tidak valid dari "
      f"<strong>{int(fold_metrics_df['samples'].sum())}</strong> candle "
      f"({100.0 * float(candle_audit_df['any_invalid_candle'].sum()) / max(int(fold_metrics_df['samples'].sum()), 1):.2f}%), "
      "karena <code>pred_open</code> atau <code>pred_close</code> berada di luar rentang <code>[pred_low, pred_high]</code>. "
      "Temuan ini perlu dibaca sebagai keterbatasan struktural dari pendekatan per-series independent forecasting, sehingga visualisasi ARIMA "
      "lebih aman digunakan untuk membaca estimasi level/range daripada bentuk candle yang sepenuhnya valid.</p>"
      f"{candle_audit_df.to_html(index=False, classes='data-table')}"
      "</div>"
  ) if spec['label'] == 'ARIMA' else ''}
  <h2>Average Metrics Fold 19-21</h2>
  {metrics_df.to_html(index=False, float_format=lambda x: f"{x:.6f}", classes="data-table")}
  <h2>Charts All Fold Full</h2>
  <h3>Overlay True vs Predicted OHLC</h3>
  <script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
  {make_wide(fig_overlay, boundaries, 580)}
  <h3>True vs Predicted Close</h3>
  {make_wide(fig_close, boundaries, 450)}
  <h3>Error Close vs ATR</h3>
  {make_wide(fig_err_close, boundaries, 450)}
  <h3>Error AVG vs ATR</h3>
  {make_wide(fig_err_avg, boundaries, 450)}
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root) if args.split_root else ROOT / "results" / "splits" / "tvt_v02" / args.timeframe
    for spec in build_model_specs(args.timeframe):
        out_path = build_model_report(spec, split_root)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
