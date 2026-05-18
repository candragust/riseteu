import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PIP = 0.0001
PIP_FACTOR = 1 / PIP
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_tvt_title(title: str, protocol_label: str | None = None) -> str:
    """Move TVT markers to a single suffix, e.g. "... (TVT v02)"."""
    marker_source = " ".join(part for part in [title, protocol_label or ""] if part)
    if not re.search(r"\bTVT(?:[_\s-]*v?0?2)?\b|\bTVTv?0?2\b", marker_source, flags=re.I):
        return title

    suffix = "(TVT v02)" if re.search(r"\bTVT(?:[_\s-]*v?0?2)\b|\bTVTv?0?2\b", marker_source, flags=re.I) else "(TVT)"
    normalized = title
    normalized = re.sub(r"\s*\((?:TVT(?:[_\s-]*v?0?2)?|TVTv?0?2)\)\s*$", "", normalized, flags=re.I)
    normalized = re.sub(r"\s+\bTVT[_\s-]*v?0?2\b", " ", normalized, flags=re.I)
    normalized = re.sub(r"\s+\bTVTv?0?2\b", " ", normalized, flags=re.I)
    normalized = re.sub(r"\s+\bTVT\b", " ", normalized, flags=re.I)
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()
    return f"{normalized} {suffix}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build validation + visual reports for FLF-LSTM WF72m/1m results."
    )
    parser.add_argument(
        "--rolling-dir",
        type=str,
        required=True,
        help="Directory containing rolling_fixed_summary.csv and foldXX_preds/history.csv",
    )
    parser.add_argument(
        "--out-validation",
        type=str,
        required=True,
        help="Output HTML path for validation summary report.",
    )
    parser.add_argument(
        "--out-ohlc",
        type=str,
        required=True,
        help="Output HTML path for combined OHLC dot chart (all folds).",
    )
    parser.add_argument(
        "--out-loss",
        type=str,
        required=True,
        help="Output HTML path for loss curves.",
    )
    parser.add_argument(
        "--out-gradient",
        type=str,
        required=True,
        help="Output HTML path for gradient proxy chart.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=30,
        help="Tail length for additional metrics (last fold).",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="EURUSD FLF-LSTM",
        help="Title prefix for reports.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional JSON config path used to show exact hyperparameters in the report header.",
    )
    parser.add_argument(
        "--method-label",
        type=str,
        default=None,
        help="Optional explicit method label for the report header.",
    )
    return parser.parse_args()


def fold_file(base: Path, fold: int, suffix: str) -> Path:
    return base / f"fold{int(fold):02d}_{suffix}.csv"


def compute_mae_pips(df_preds: pd.DataFrame):
    diffs = np.abs(
        df_preds[["pred_open", "pred_high", "pred_low", "pred_close"]].values
        - df_preds[["true_open", "true_high", "true_low", "true_close"]].values
    )
    mae_pips = diffs.mean(axis=0) * PIP_FACTOR
    return {
        "mae_open_pips": float(mae_pips[0]),
        "mae_high_pips": float(mae_pips[1]),
        "mae_low_pips": float(mae_pips[2]),
        "mae_close_pips": float(mae_pips[3]),
        "mae_avg_pips": float(mae_pips.mean()),
        "mae_avg_hlc_pips": float((mae_pips[1] + mae_pips[2] + mae_pips[3]) / 3.0),
    }


def squared_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float("nan")
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r**2)


def directional_accuracy(true_open, true_close, pred_open, pred_close):
    true_dir = np.sign(np.asarray(true_close, dtype=float) - np.asarray(true_open, dtype=float))
    pred_dir = np.sign(np.asarray(pred_close, dtype=float) - np.asarray(pred_open, dtype=float))
    if len(true_dir) == 0:
        return float("nan")
    return float((true_dir == pred_dir).mean() * 100.0)


def nanmean(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def compute_pattern_metrics(df_preds: pd.DataFrame):
    pred_values = df_preds[["pred_open", "pred_high", "pred_low", "pred_close"]].values
    true_values = df_preds[["true_open", "true_high", "true_low", "true_close"]].values
    corr2 = [squared_corr(true_values[:, i], pred_values[:, i]) for i in range(4)]
    true_dir = np.sign(df_preds["true_close"].to_numpy(dtype=float) - df_preds["true_open"].to_numpy(dtype=float))
    pred_dir = np.sign(df_preds["pred_close"].to_numpy(dtype=float) - df_preds["pred_open"].to_numpy(dtype=float))
    return {
        "corr2_open": float(corr2[0]),
        "corr2_high": float(corr2[1]),
        "corr2_low": float(corr2[2]),
        "corr2_close": float(corr2[3]),
        "corr2_avg_ohlc": nanmean(corr2),
        "corr2_avg_hlc": nanmean(corr2[1:]),
        "da_body_pct": directional_accuracy(
            df_preds["true_open"],
            df_preds["true_close"],
            df_preds["pred_open"],
            df_preds["pred_close"],
        ),
        "true_bull": int((true_dir > 0).sum()),
        "true_bear": int((true_dir < 0).sum()),
        "true_neutral": int((true_dir == 0).sum()),
        "pred_bull": int((pred_dir > 0).sum()),
        "pred_bear": int((pred_dir < 0).sum()),
        "pred_neutral": int((pred_dir == 0).sum()),
    }


def row_value(row: pd.Series, *names, default=np.nan):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def row_path(rolling_dir: Path, row: pd.Series, fold: int, column: str, suffix: str) -> Path:
    value = row_value(row, column, default=None)
    if value:
        path = Path(str(value))
        return path if path.is_absolute() else rolling_dir / path
    return fold_file(rolling_dir, fold, suffix)


def build_fold_metrics(rolling_dir: Path):
    summary_path = rolling_dir / "rolling_fixed_summary.csv"
    if not summary_path.exists():
        summary_path = rolling_dir / "rolling_tvt_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    fold_df = pd.read_csv(summary_path)
    rows = []
    missing = []
    for _, row in fold_df.iterrows():
        fold = int(row["fold"])
        preds_path = row_path(rolling_dir, row, fold, "preds_csv", "preds")
        hist_path = row_path(rolling_dir, row, fold, "history_csv", "history")
        if not preds_path.exists():
            missing.append(str(preds_path))
            continue
        preds_df = pd.read_csv(preds_path)
        metrics = compute_mae_pips(preds_df)
        pattern_metrics = compute_pattern_metrics(preds_df)
        history_metrics = {
            "epochs_ran": np.nan,
            "best_epoch": np.nan,
            "best_val_loss": np.nan,
        }
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
            history_metrics["epochs_ran"] = int(len(hist))
            if "val_loss" in hist.columns and len(hist):
                history_metrics["best_epoch"] = int(hist["val_loss"].idxmin() + 1)
                history_metrics["best_val_loss"] = float(hist["val_loss"].min())
        rows.append(
            {
                "profile": row_value(row, "profile", default="-"),
                "fold": fold,
                "train_start": row_value(row, "train_start", "train_core_start"),
                "train_end": row_value(row, "train_end", "train_core_end"),
                "validation_start": row_value(row, "validation_start", default=np.nan),
                "validation_end": row_value(row, "validation_end", default=np.nan),
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "train_samples": int(row_value(row, "train_samples", "train_core_samples")),
                "validation_samples": row_value(row, "validation_samples", default=np.nan),
                "test_samples": int(row["test_samples"]),
                "split_ratio": row_value(row, "split_ratio", default=np.nan),
                "preds": str(preds_path),
                "history": str(hist_path),
                **metrics,
                **pattern_metrics,
                **history_metrics,
            }
        )
    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing))
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def infer_method_and_hyperparams(rolling_dir: Path, title: str, config_path: Path | None = None, method_label: str | None = None):
    path_text = str(rolling_dir)
    title_lower = title.lower()
    if method_label:
        method = method_label
    elif "bilstm" in title_lower or "bilstm" in path_text.lower():
        method = "FLF-BiLSTM"
        best_progression = PROJECT_ROOT / "FLF_BILSTM" / "results" / "eurusd_pipeline" / "best_progression.csv"
    else:
        method = "FLF-LSTM"
        best_progression = PROJECT_ROOT / "FLF_LSTM" / "results" / "lstm_pipeline_wf72_test1" / "best_progression.csv"

    if config_path and config_path.exists():
        cfg = pd.read_json(config_path, typ="series")
        hyperparams = (
            f"window={int(cfg['window'])}, "
            f"units={int(cfg['units'])}, "
            f"activation={cfg['activation']}, "
            f"lr={cfg['lr']}, "
            f"epochs={int(cfg['epochs'])}, "
            f"lambda={cfg['lambda_coef']}, "
            f"sigma={cfg['sigma_coef']}, "
            f"batch={int(cfg['batch'])}"
        )
        return method, hyperparams

    if not best_progression.exists():
        return method, "-"

    best_df = pd.read_csv(best_progression)
    best = best_df.iloc[-1].to_dict()
    hyperparams = (
        f"window={int(best['window'])}, "
        f"units={int(best['units'])}, "
        f"activation={best['activation']}, "
        f"lr={best['lr']}, "
        f"epochs={int(best['epochs'])}, "
        f"lambda={best['lambda_coef']}, "
        f"sigma={best['sigma_coef']}, "
        f"batch={int(best['batch'])}"
    )
    return method, hyperparams


def build_ohlc_header_html(rolling_dir: Path, folds: pd.DataFrame, title: str, config_path: Path | None = None, method_label: str | None = None):
    method, hyperparams = infer_method_and_hyperparams(rolling_dir, title, config_path=config_path, method_label=method_label)
    first_train = folds.iloc[0]
    last_train = folds.iloc[-1]
    has_validation = "validation_start" in folds.columns and not folds["validation_start"].isna().all()
    if has_validation:
        training_period = (
            "TVT train-core + validation; "
            f"train-core {first_train['train_start']} s.d. {first_train['train_end']} "
            f"sampai {last_train['train_start']} s.d. {last_train['train_end']}; "
            f"validation {folds['validation_start'].min()} s.d. {folds['validation_end'].max()}"
        )
    else:
        training_period = (
            "Fixed 72 bulan per fold; "
            f"{first_train['train_start']} s.d. {first_train['train_end']} "
            f"sampai {last_train['train_start']} s.d. {last_train['train_end']}"
        )
    testing_period = (
        f"Fold {int(folds['fold'].min())}-{int(folds['fold'].max())}, 1 bulan per fold; "
        f"{folds['test_start'].min()} s.d. {folds['test_end'].max()}"
    )
    return f"""
<section class="report-header">
  <h1>{title}</h1>
  <div class="meta-grid">
    <div class="meta-item"><span class="meta-label">Periode Training</span><span class="meta-value">{training_period}</span></div>
    <div class="meta-item"><span class="meta-label">Periode Testing</span><span class="meta-value">{testing_period}</span></div>
    <div class="meta-item"><span class="meta-label">Metode</span><span class="meta-value">{method}</span></div>
    <div class="meta-item"><span class="meta-label">Hyperparameter</span><span class="meta-value">{hyperparams}</span></div>
  </div>
</section>
"""


def build_ohlc_all(rolling_dir: Path, folds: pd.DataFrame, out_path: Path, title: str, config_path: Path | None = None, method_label: str | None = None):
    title = normalize_tvt_title(title)
    frames = []
    for _, row in folds.iterrows():
        fold = int(row["fold"])
        preds_df = pd.read_csv(fold_file(rolling_dir, fold, "preds"))
        preds_df["fold"] = fold
        preds_df["idx_in_fold"] = np.arange(len(preds_df))
        frames.append(preds_df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df["global_idx"] = np.arange(len(all_df))

    cols = ["open", "high", "low", "close"]
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=[f"{c.upper()} Actual vs Pred" for c in cols],
        vertical_spacing=0.06,
    )

    for i, col in enumerate(cols):
        r = i + 1
        c = 1
        fig.add_trace(
            go.Scatter(
                x=all_df["global_idx"],
                y=all_df[f"true_{col}"],
                mode="lines",
                name=f"Actual {col.upper()}",
                line=dict(color="#d62728", width=1.2),
                showlegend=(i == 0),
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=all_df["global_idx"],
                y=all_df[f"pred_{col}"],
                mode="markers",
                name=f"Pred {col.upper()}",
                marker=dict(size=2.6, color="#1f77b4", symbol="circle", opacity=0.9),
                showlegend=(i == 0),
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="Index (all test candles)", row=r, col=c)
        fig.update_yaxes(title_text="Price", tickformat=".5f", row=r, col=c)

    fig.update_layout(
        template="plotly_white",
        height=1500,
        margin=dict(t=70, r=150, b=60, l=85),
        legend=dict(orientation="v", y=1.0, x=1.02, xanchor="left", yanchor="top"),
    )
    header_html = build_ohlc_header_html(rolling_dir, folds, title, config_path=config_path, method_label=method_label)
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    html = f"""
<html>
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: Arial, sans-serif;
      background: #ffffff;
      color: #183a6b;
    }}
    .report-header {{
      max-width: 1440px;
      margin: 0 auto 16px auto;
    }}
    .report-header h1 {{
      margin: 0 0 14px 0;
      font-size: 22px;
      line-height: 1.25;
      font-weight: 600;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 18px;
    }}
    .meta-item {{
      border: 1px solid #d9e2f0;
      border-radius: 6px;
      padding: 10px 12px;
      background: #fafcff;
    }}
    .meta-label {{
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
      color: #4f6690;
      text-transform: uppercase;
    }}
    .meta-value {{
      display: block;
      font-size: 14px;
      line-height: 1.45;
      color: #183a6b;
      word-break: break-word;
    }}
    .plot-wrap {{
      max-width: 1440px;
      margin: 0 auto;
    }}
    @media (max-width: 900px) {{
      .meta-grid {{
        grid-template-columns: 1fr;
      }}
      body {{
        padding: 14px;
      }}
    }}
  </style>
</head>
<body>
  {header_html}
  <div class="plot-wrap">{plot_html}</div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def build_loss_report(rolling_dir: Path, folds: pd.DataFrame, out_path: Path, title: str):
    title = normalize_tvt_title(title)
    fig = make_subplots(
        rows=len(folds),
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Fold {int(f)}" for f in folds["fold"]],
        vertical_spacing=0.07,
    )

    for i, (_, row) in enumerate(folds.iterrows(), start=1):
        fold = int(row["fold"])
        hist = pd.read_csv(fold_file(rolling_dir, fold, "history"))
        if "loss" in hist.columns:
            fig.add_trace(
                go.Scatter(y=hist["loss"], mode="lines", name=f"Fold {fold} loss", showlegend=(i == 1)),
                row=i,
                col=1,
            )
        if "val_loss" in hist.columns:
            fig.add_trace(
                go.Scatter(
                    y=hist["val_loss"],
                    mode="lines",
                    line=dict(dash="dash"),
                    name=f"Fold {fold} val_loss",
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )
        fig.update_yaxes(title_text="Loss", row=i, col=1)
        fig.update_xaxes(title_text="Epoch", row=i, col=1)

    fig.update_layout(title=title, template="plotly_white", height=max(420, 300 * len(folds)))
    out_path.write_text(fig.to_html(include_plotlyjs="cdn"), encoding="utf-8")


def build_gradient_report(rolling_dir: Path, folds: pd.DataFrame, out_path: Path, title: str):
    title = normalize_tvt_title(title)
    fig = make_subplots(
        rows=len(folds),
        cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Fold {int(f)}" for f in folds["fold"]],
        vertical_spacing=0.07,
    )

    for i, (_, row) in enumerate(folds.iterrows(), start=1):
        fold = int(row["fold"])
        hist = pd.read_csv(fold_file(rolling_dir, fold, "history"))
        if "loss" not in hist.columns or len(hist["loss"]) < 2:
            continue
        loss = hist["loss"].to_numpy()
        improvement = loss[:-1] - loss[1:]
        fig.add_trace(
            go.Bar(
                x=np.arange(1, len(loss)),
                y=improvement,
                name=f"Fold {fold} dLoss",
                showlegend=(i == 1),
                marker_color="#1f77b4",
            ),
            row=i,
            col=1,
        )
        fig.add_hline(y=0.0, row=i, col=1, line_dash="dash", line_color="gray")
        fig.update_yaxes(title_text="dLoss", row=i, col=1)
        fig.update_xaxes(title_text="Epoch", row=i, col=1)

    fig.update_layout(title=title, template="plotly_white", height=max(420, 300 * len(folds)))
    out_path.write_text(fig.to_html(include_plotlyjs="cdn"), encoding="utf-8")


def write_validation_html(folds: pd.DataFrame, out_path: Path, title: str, tail: int):
    title = normalize_tvt_title(title)
    overall_mean = float(folds["mae_avg_pips"].mean())
    overall_median = float(folds["mae_avg_pips"].median())
    overall_corr2_hlc = float(folds["corr2_avg_hlc"].mean())
    overall_da = float(folds["da_body_pct"].mean())
    best = folds.loc[folds["mae_avg_pips"].idxmin()]
    worst = folds.loc[folds["mae_avg_pips"].idxmax()]

    last_fold = int(folds["fold"].max())
    tail_note = ""
    tail_mae = np.nan
    try:
        last_preds = pd.read_csv(Path(folds.loc[folds["fold"] == last_fold, "preds"].iloc[0]))
        tail_df = last_preds.tail(tail)
        tail_mae = compute_mae_pips(tail_df)
        tail_pattern = compute_pattern_metrics(tail_df)
        tail_note = (
            f"<p>Tail {tail} fold terakhir ({last_fold}): "
            f"MAE avg <strong>{tail_mae['mae_avg_pips']:.2f} pips</strong> | "
            f"corr2 avg HLC <strong>{tail_pattern['corr2_avg_hlc']:.4f}</strong> | "
            f"Directional Accuracy body <strong>{tail_pattern['da_body_pct']:.2f}%</strong></p>"
        )
    except Exception:
        tail_note = f"<p>Tail {tail} metrics (fold terakhir) tidak tersedia.</p>"

    cols = [
        "fold",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
        "train_samples",
        "validation_samples",
        "test_samples",
        "split_ratio",
        "epochs_ran",
        "best_epoch",
        "best_val_loss",
        "mae_avg_hlc_pips",
        "mae_avg_pips",
        "mae_open_pips",
        "mae_high_pips",
        "mae_low_pips",
        "mae_close_pips",
        "corr2_avg_hlc",
        "corr2_avg_ohlc",
        "corr2_open",
        "corr2_high",
        "corr2_low",
        "corr2_close",
        "da_body_pct",
        "true_bull",
        "true_bear",
        "pred_bull",
        "pred_bear",
    ]
    cols = [col for col in cols if col in folds.columns and not folds[col].isna().all()]
    table_html = folds[cols].to_html(index=False, float_format="%.4f")

    body = [
        f"<h1>{title}</h1>",
        (
            "<p>"
            f"Rata-rata MAE avg: <strong>{overall_mean:.2f} pips</strong> | "
            f"Median: <strong>{overall_median:.2f} pips</strong> | "
            f"Mean corr2 avg HLC: <strong>{overall_corr2_hlc:.4f}</strong> | "
            f"Mean Directional Accuracy body: <strong>{overall_da:.2f}%</strong> | "
            f"Fold terbaik: <strong>{int(best['fold'])}</strong> ({best['mae_avg_pips']:.2f} pips) | "
            f"Fold terburuk: <strong>{int(worst['fold'])}</strong> ({worst['mae_avg_pips']:.2f} pips)"
            "</p>"
        ),
        (
            "<p class='note'>corr2 = kuadrat korelasi Pearson antara aktual dan prediksi. "
            "Directional Accuracy body = persentase kecocokan tanda close-open antara candle aktual dan prediksi.</p>"
        ),
        tail_note,
        "<h2>Ringkasan Metrik per Fold</h2>",
        table_html,
    ]

    html = (
        "<html><head><meta charset='UTF-8'>"
        "<style>body{font-family:Arial, sans-serif; margin:20px;} "
        ".note{color:#555; max-width:980px;} "
        "table{border-collapse:collapse;} th,td{border:1px solid #ccc; padding:6px 10px; text-align:right;} "
        "th{background:#f5f5f5;}</style>"
        f"<title>{title}</title></head><body>{''.join(body)}</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    rolling_dir = Path(args.rolling_dir)
    out_validation = Path(args.out_validation)
    out_ohlc = Path(args.out_ohlc)
    out_loss = Path(args.out_loss)
    out_gradient = Path(args.out_gradient)
    config_path = Path(args.config_path) if args.config_path else None

    out_validation.parent.mkdir(parents=True, exist_ok=True)
    out_ohlc.parent.mkdir(parents=True, exist_ok=True)
    out_loss.parent.mkdir(parents=True, exist_ok=True)
    out_gradient.parent.mkdir(parents=True, exist_ok=True)

    folds = build_fold_metrics(rolling_dir)
    protocol_label = "TVT v02" if "tvt_v02" in str(rolling_dir).lower() else "WF72m/1m"
    metrics_csv = out_validation.with_name(f"{out_validation.stem}_metrics.csv")
    folds.to_csv(metrics_csv, index=False)
    write_validation_html(
        folds,
        out_validation,
        f"{args.title_prefix} - Validation Report {protocol_label}",
        args.tail,
    )
    build_ohlc_all(
        rolling_dir,
        folds,
        out_ohlc,
        f"{args.title_prefix} - OHLC Dot Plot (All Fold Tests Combined) ({protocol_label})",
        config_path=config_path,
        method_label=args.method_label,
    )
    build_loss_report(
        rolling_dir,
        folds,
        out_loss,
        f"{args.title_prefix} - Loss Curve per Fold ({protocol_label})",
    )
    build_gradient_report(
        rolling_dir,
        folds,
        out_gradient,
        f"{args.title_prefix} - Gradient Proxy per Fold ({protocol_label})",
    )
    print(f"Wrote: {out_validation.resolve()}")
    print(f"Wrote: {metrics_csv.resolve()}")
    print(f"Wrote: {out_ohlc.resolve()}")
    print(f"Wrote: {out_loss.resolve()}")
    print(f"Wrote: {out_gradient.resolve()}")


if __name__ == "__main__":
    main()
