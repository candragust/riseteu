import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PIP_FACTOR = 10000
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ARIMA, FLF-LSTM, and FLF-BiLSTM on EUR/USD 1D WF72m/1m last5 folds."
    )
    parser.add_argument(
        "--arima-dir",
        default=str(PROJECT_ROOT / "Arima" / "result" / "d1_ohlc" / "arima_wf72_test1_last5_v01"),
        help="Directory containing ARIMA foldXX_preds.csv and foldXX_summary.json files.",
    )
    parser.add_argument(
        "--lstm-dir",
        default=str(PROJECT_ROOT / "FLF_LSTM" / "results" / "d1_ohlc" / "wf72_test1_last5_v01"),
        help="Directory containing FLF-LSTM foldXX_preds.csv files.",
    )
    parser.add_argument(
        "--bilstm-dir",
        default=str(PROJECT_ROOT / "FLF_BILSTM" / "results" / "d1_ohlc" / "wf72_test1_last5_v01"),
        help="Directory containing FLF-BiLSTM foldXX_preds.csv files.",
    )
    parser.add_argument(
        "--lstm-config",
        default=str(PROJECT_ROOT / "FLF_LSTM" / "configs" / "d1_ohlc" / "lstm_flf_config_d1_ohlc_best_v02.json"),
        help="Final LSTM config JSON.",
    )
    parser.add_argument(
        "--bilstm-config",
        default=str(PROJECT_ROOT / "FLF_BILSTM" / "configs" / "d1_ohlc" / "bilstm_flf_config_d1_ohlc_best_v02.json"),
        help="Final BiLSTM config JSON.",
    )
    parser.add_argument(
        "--arima-config",
        default=str(PROJECT_ROOT / "Arima" / "arima_baseline_config_d1_ohlc_v01.json"),
        help="ARIMA config JSON.",
    )
    parser.add_argument(
        "--out-html",
        default=str(PROJECT_ROOT / "comparison" / "d1_ohlc" / "comparison_models_d1_wf72_test1_last5_v01.html"),
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[17, 18, 19, 20, 21],
        help="Fold numbers to compare.",
    )
    return parser.parse_args()


def squared_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return float("nan")
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r**2)


def directional_accuracy(true_open, true_close, pred_open, pred_close):
    true_dir = np.sign(np.asarray(true_close) - np.asarray(true_open))
    pred_dir = np.sign(np.asarray(pred_close) - np.asarray(pred_open))
    return float((true_dir == pred_dir).mean() * 100.0)


def compute_fold_metrics(base_dir: Path, folds):
    rows = []
    for fold in folds:
        preds_path = base_dir / f"fold{fold:02d}_preds.csv"
        hist_path = base_dir / f"fold{fold:02d}_history.csv"
        df = pd.read_csv(preds_path)
        pred_values = df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
        true_values = df[["true_open", "true_high", "true_low", "true_close"]].values

        diffs = np.abs(pred_values - true_values) * PIP_FACTOR
        avg = diffs.mean(axis=0)
        corr2 = [squared_corr(true_values[:, i], pred_values[:, i]) for i in range(4)]
        da_body_pct = directional_accuracy(
            df["true_open"],
            df["true_close"],
            df["pred_open"],
            df["pred_close"],
        )

        row = {
            "fold": fold,
            "samples": int(len(df)),
            "mae_open_pips": float(avg[0]),
            "mae_high_pips": float(avg[1]),
            "mae_low_pips": float(avg[2]),
            "mae_close_pips": float(avg[3]),
            "mae_avg_pips": float(avg.mean()),
            "mae_avg_hlc_pips": float((avg[1] + avg[2] + avg[3]) / 3.0),
            "corr2_open": float(corr2[0]),
            "corr2_high": float(corr2[1]),
            "corr2_low": float(corr2[2]),
            "corr2_close": float(corr2[3]),
            "corr2_avg_ohlc": float(np.nanmean(corr2)),
            "corr2_avg_hlc": float(np.nanmean(corr2[1:])),
            "da_body_pct": da_body_pct,
        }
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
            row["epochs_ran"] = int(len(hist))
            row["best_epoch"] = int(hist["val_loss"].idxmin() + 1)
            row["best_val_loss"] = float(hist["val_loss"].min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def summarize_metrics(df: pd.DataFrame):
    best_row = df.loc[df["mae_avg_pips"].idxmin()]
    worst_row = df.loc[df["mae_avg_pips"].idxmax()]
    out = {
        "mean_mae_avg_pips": float(df["mae_avg_pips"].mean()),
        "median_mae_avg_pips": float(df["mae_avg_pips"].median()),
        "mean_mae_avg_hlc_pips": float(df["mae_avg_hlc_pips"].mean()),
        "mean_corr2_avg_hlc": float(df["corr2_avg_hlc"].mean()),
        "mean_da_body_pct": float(df["da_body_pct"].mean()),
        "best_fold": int(best_row["fold"]),
        "best_fold_mae_avg_pips": float(best_row["mae_avg_pips"]),
        "worst_fold": int(worst_row["fold"]),
        "worst_fold_mae_avg_pips": float(worst_row["mae_avg_pips"]),
    }
    if "epochs_ran" in df.columns:
        out["mean_epochs_ran"] = float(df["epochs_ran"].mean())
        out["mean_best_epoch"] = float(df["best_epoch"].mean())
        out["mean_best_val_loss"] = float(df["best_val_loss"].mean())
    return out


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_lstm_meta(config_path: Path):
    cfg = load_json(config_path)
    return (
        f"window={cfg['window']}, units={cfg['units']}, activation={cfg['activation']}, "
        f"lr={cfg['lr']}, lambda={cfg['lambda_coef']}, sigma={cfg['sigma_coef']}, "
        f"batch={cfg['batch']}, epochs={cfg['epochs']}"
    )


def load_bilstm_meta(config_path: Path):
    cfg = load_json(config_path)
    return (
        f"window={cfg['window']}, units={cfg['units']}, activation={cfg['activation']}, "
        f"lr={cfg['lr']}, lambda={cfg['lambda_coef']}, sigma={cfg['sigma_coef']}, "
        f"batch={cfg['batch']}, epochs={cfg['epochs']}"
    )


def load_arima_meta(config_path: Path, arima_dir: Path, folds):
    cfg = load_json(config_path)
    order_counter = {col: Counter() for col in ("open", "high", "low", "close")}
    for fold in folds:
        summary_path = arima_dir / f"fold{fold:02d}_summary.json"
        data = load_json(summary_path)
        for col in order_counter:
            selected = tuple(data["targets"][col]["selected_order"])
            order_counter[col][selected] += 1
    order_parts = []
    for col, counter in order_counter.items():
        order, count = counter.most_common(1)[0]
        order_parts.append(f"{col}={order} ({count}/{len(folds)} fold)")
    return (
        f"order_search={str(cfg['order_search']).lower()}, requested_order={tuple(cfg['order'])}, "
        f"grid p={tuple(cfg['p_values'])}, d={tuple(cfg['d_values'])}, q={tuple(cfg['q_values'])}, "
        f"selection_metric={cfg['selection_metric']}; dominant selected orders: " + ", ".join(order_parts)
    )


def build_pairwise_table(arima_df, lstm_df, bilstm_df):
    merged = (
        arima_df[["fold", "mae_avg_pips", "corr2_avg_hlc", "da_body_pct"]]
        .rename(
            columns={
                "mae_avg_pips": "arima_mae_avg_pips",
                "corr2_avg_hlc": "arima_corr2_avg_hlc",
                "da_body_pct": "arima_da_body_pct",
            }
        )
        .merge(
            lstm_df[["fold", "mae_avg_pips", "corr2_avg_hlc", "da_body_pct"]].rename(
                columns={
                    "mae_avg_pips": "lstm_mae_avg_pips",
                    "corr2_avg_hlc": "lstm_corr2_avg_hlc",
                    "da_body_pct": "lstm_da_body_pct",
                }
            ),
            on="fold",
            how="inner",
        )
        .merge(
            bilstm_df[["fold", "mae_avg_pips", "corr2_avg_hlc", "da_body_pct"]].rename(
                columns={
                    "mae_avg_pips": "bilstm_mae_avg_pips",
                    "corr2_avg_hlc": "bilstm_corr2_avg_hlc",
                    "da_body_pct": "bilstm_da_body_pct",
                }
            ),
            on="fold",
            how="inner",
        )
    )
    merged["winner_mae"] = merged[["arima_mae_avg_pips", "lstm_mae_avg_pips", "bilstm_mae_avg_pips"]].idxmin(axis=1)
    merged["winner_corr2"] = merged[["arima_corr2_avg_hlc", "lstm_corr2_avg_hlc", "bilstm_corr2_avg_hlc"]].idxmax(axis=1)
    merged["winner_da"] = merged[["arima_da_body_pct", "lstm_da_body_pct", "bilstm_da_body_pct"]].idxmax(axis=1)
    return merged


def format_table(df: pd.DataFrame):
    return df.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x), classes="data-table")


def main():
    args = parse_args()

    arima_dir = Path(args.arima_dir)
    lstm_dir = Path(args.lstm_dir)
    bilstm_dir = Path(args.bilstm_dir)
    lstm_cfg_path = Path(args.lstm_config)
    bilstm_cfg_path = Path(args.bilstm_config)
    arima_cfg_path = Path(args.arima_config)
    out_html = Path(args.out_html)
    out_dir = out_html.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    arima_df = compute_fold_metrics(arima_dir, args.folds)
    lstm_df = compute_fold_metrics(lstm_dir, args.folds)
    bilstm_df = compute_fold_metrics(bilstm_dir, args.folds)

    arima_summary = summarize_metrics(arima_df)
    lstm_summary = summarize_metrics(lstm_df)
    bilstm_summary = summarize_metrics(bilstm_df)
    pairwise_df = build_pairwise_table(arima_df, lstm_df, bilstm_df)

    model_rows = [
        {
            "model": "ARIMA",
            "mean_mae_avg_pips": arima_summary["mean_mae_avg_pips"],
            "mean_corr2_avg_hlc": arima_summary["mean_corr2_avg_hlc"],
            "mean_da_body_pct": arima_summary["mean_da_body_pct"],
            "best_fold": arima_summary["best_fold"],
            "best_fold_mae_avg_pips": arima_summary["best_fold_mae_avg_pips"],
        },
        {
            "model": "FLF-LSTM",
            "mean_mae_avg_pips": lstm_summary["mean_mae_avg_pips"],
            "mean_corr2_avg_hlc": lstm_summary["mean_corr2_avg_hlc"],
            "mean_da_body_pct": lstm_summary["mean_da_body_pct"],
            "best_fold": lstm_summary["best_fold"],
            "best_fold_mae_avg_pips": lstm_summary["best_fold_mae_avg_pips"],
        },
        {
            "model": "FLF-BiLSTM",
            "mean_mae_avg_pips": bilstm_summary["mean_mae_avg_pips"],
            "mean_corr2_avg_hlc": bilstm_summary["mean_corr2_avg_hlc"],
            "mean_da_body_pct": bilstm_summary["mean_da_body_pct"],
            "best_fold": bilstm_summary["best_fold"],
            "best_fold_mae_avg_pips": bilstm_summary["best_fold_mae_avg_pips"],
        },
    ]
    model_df = pd.DataFrame(model_rows)

    base_stem = out_html.with_suffix("")
    arima_csv = base_stem.parent / f"{base_stem.name}_arima_metrics.csv"
    lstm_csv = base_stem.parent / f"{base_stem.name}_lstm_metrics.csv"
    bilstm_csv = base_stem.parent / f"{base_stem.name}_bilstm_metrics.csv"
    merged_csv = base_stem.parent / f"{base_stem.name}_pairwise.csv"
    summary_json = base_stem.parent / f"{base_stem.name}_summary.json"

    arima_df.to_csv(arima_csv, index=False)
    lstm_df.to_csv(lstm_csv, index=False)
    bilstm_df.to_csv(bilstm_csv, index=False)
    pairwise_df.to_csv(merged_csv, index=False)

    payload = {
        "folds": args.folds,
        "arima": arima_summary,
        "lstm": lstm_summary,
        "bilstm": bilstm_summary,
        "artifacts": {
            "arima_metrics_csv": str(arima_csv),
            "lstm_metrics_csv": str(lstm_csv),
            "bilstm_metrics_csv": str(bilstm_csv),
            "pairwise_csv": str(merged_csv),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    arima_meta = load_arima_meta(arima_cfg_path, arima_dir, args.folds)
    lstm_meta = load_lstm_meta(lstm_cfg_path)
    bilstm_meta = load_bilstm_meta(bilstm_cfg_path)

    mae_winner = model_df.loc[model_df["mean_mae_avg_pips"].idxmin(), "model"]
    corr_winner = model_df.loc[model_df["mean_corr2_avg_hlc"].idxmax(), "model"]
    da_winner = model_df.loc[model_df["mean_da_body_pct"].idxmax(), "model"]

    html = f"""
<html>
<head>
  <meta charset="UTF-8" />
  <title>Comparison ARIMA vs FLF-LSTM vs FLF-BiLSTM - EUR/USD 1D WF72m/1m Last5</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    p {{ line-height: 1.5; }}
    .meta {{ background: #f6f8fa; padding: 12px 14px; border: 1px solid #d0d7de; border-radius: 8px; margin-bottom: 16px; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
    .data-table th, .data-table td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    .data-table th {{ background: #f6f8fa; }}
    .data-table td:first-child, .data-table th:first-child {{ text-align: left; }}
    ul {{ margin-top: 6px; }}
    code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>ARIMA vs FLF-LSTM vs FLF-BiLSTM</h1>
  <p>Eksperimen <strong>EUR/USD 1D</strong> dengan protokol <strong>walk-forward 72 bulan train + 1 bulan test</strong> pada <strong>5 fold terakhir (17-21)</strong>.</p>

  <h2>Metode</h2>
  <div class="meta"><strong>ARIMA</strong>: {arima_meta}</div>
  <div class="meta"><strong>FLF-LSTM</strong>: {lstm_meta}</div>
  <div class="meta"><strong>FLF-BiLSTM</strong>: {bilstm_meta}</div>

  <h2>Ringkasan Model</h2>
  {format_table(model_df)}

  <h2>Temuan Utama</h2>
  <ul>
    <li>Pemenang <strong>MAE avg pips</strong>: <strong>{mae_winner}</strong>.</li>
    <li>Pemenang <strong>corr2 avg HLC</strong>: <strong>{corr_winner}</strong>.</li>
    <li>Pemenang <strong>Directional Accuracy</strong>: <strong>{da_winner}</strong>.</li>
    <li>Mean MAE avg pips: ARIMA {arima_summary['mean_mae_avg_pips']:.4f}, FLF-LSTM {lstm_summary['mean_mae_avg_pips']:.4f}, FLF-BiLSTM {bilstm_summary['mean_mae_avg_pips']:.4f}.</li>
    <li>Mean corr2 avg HLC: ARIMA {arima_summary['mean_corr2_avg_hlc']:.4f}, FLF-LSTM {lstm_summary['mean_corr2_avg_hlc']:.4f}, FLF-BiLSTM {bilstm_summary['mean_corr2_avg_hlc']:.4f}.</li>
    <li>Mean Directional Accuracy: ARIMA {arima_summary['mean_da_body_pct']:.2f}%, FLF-LSTM {lstm_summary['mean_da_body_pct']:.2f}%, FLF-BiLSTM {bilstm_summary['mean_da_body_pct']:.2f}%.</li>
  </ul>

  <h2>ARIMA Per Fold</h2>
  {format_table(arima_df)}

  <h2>FLF-LSTM Per Fold</h2>
  {format_table(lstm_df)}

  <h2>FLF-BiLSTM Per Fold</h2>
  {format_table(bilstm_df)}

  <h2>Perbandingan Antar Model Per Fold</h2>
  {format_table(pairwise_df)}

  <h2>Artefak</h2>
  <ul>
    <li>Summary JSON: <code>{summary_json}</code></li>
    <li>ARIMA metrics CSV: <code>{arima_csv}</code></li>
    <li>LSTM metrics CSV: <code>{lstm_csv}</code></li>
    <li>BiLSTM metrics CSV: <code>{bilstm_csv}</code></li>
    <li>Pairwise CSV: <code>{merged_csv}</code></li>
  </ul>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    print(f"Saved HTML report to {out_html}")
    print(f"Saved summary JSON to {summary_json}")


if __name__ == "__main__":
    main()
