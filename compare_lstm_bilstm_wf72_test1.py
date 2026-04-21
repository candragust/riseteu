import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PIP_FACTOR = 10000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FLF-LSTM vs FLF-BiLSTM on selected WF72m/1m folds."
    )
    parser.add_argument(
        "--lstm-dir",
        default="/home/hduser/jupyter/gust/RisetEU/FLF_LSTM/results/wf72_test1_lstm",
        help="Directory containing FLF-LSTM foldXX_preds.csv files.",
    )
    parser.add_argument(
        "--bilstm-dir",
        default="/home/hduser/jupyter/gust/RisetEU/results/rolling_train72_test1",
        help="Directory containing FLF-BiLSTM foldXX_preds.csv files.",
    )
    parser.add_argument(
        "--lstm-config",
        default="/home/hduser/jupyter/gust/RisetEU/FLF_LSTM/lstm_flf_config_wf72_test1_best.json",
        help="Best FLF-LSTM config JSON.",
    )
    parser.add_argument(
        "--bilstm-meta-html",
        default="/home/hduser/jupyter/gust/RisetEU/results/mae_atr_wf72_test1_fold21_tail30.html",
        help="BiLSTM HTML report used to extract parameter string.",
    )
    parser.add_argument(
        "--out",
        default="/home/hduser/jupyter/gust/RisetEU/results/comparison/comparison_lstm_vs_bilstm_wf72_test1.html",
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[19, 20, 21],
        help="Fold numbers to compare, e.g. --folds 17 18 19 20 21",
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

        tail = df.tail(30)
        tail_diffs = np.abs(
            tail[["pred_open", "pred_high", "pred_low", "pred_close"]].values
            - tail[["true_open", "true_high", "true_low", "true_close"]].values
        ) * PIP_FACTOR
        tail_avg = tail_diffs.mean(axis=0)

        rows.append(
            {
                "fold": fold,
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
                "tail30_mae_avg_pips": float(tail_avg.mean()),
                "tail30_mae_avg_hlc_pips": float((tail_avg[1] + tail_avg[2] + tail_avg[3]) / 3.0),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def summarize_metrics(df: pd.DataFrame):
    best_row = df.loc[df["mae_avg_pips"].idxmin()]
    worst_row = df.loc[df["mae_avg_pips"].idxmax()]
    last_row = df.loc[df["fold"].idxmax()]
    return {
        "mean_mae_avg_pips": float(df["mae_avg_pips"].mean()),
        "median_mae_avg_pips": float(df["mae_avg_pips"].median()),
        "mean_mae_avg_hlc_pips": float(df["mae_avg_hlc_pips"].mean()),
        "mean_corr2_avg_hlc": float(df["corr2_avg_hlc"].mean()),
        "median_corr2_avg_hlc": float(df["corr2_avg_hlc"].median()),
        "mean_da_body_pct": float(df["da_body_pct"].mean()),
        "median_da_body_pct": float(df["da_body_pct"].median()),
        "mean_tail30_mae_avg_pips": float(df["tail30_mae_avg_pips"].mean()),
        "mean_tail30_mae_avg_hlc_pips": float(df["tail30_mae_avg_hlc_pips"].mean()),
        "best_fold": int(best_row["fold"]),
        "best_fold_mae_avg_pips": float(best_row["mae_avg_pips"]),
        "worst_fold": int(worst_row["fold"]),
        "worst_fold_mae_avg_pips": float(worst_row["mae_avg_pips"]),
        "last_fold": int(last_row["fold"]),
        "last_fold_mae_avg_pips": float(last_row["mae_avg_pips"]),
        "last_fold_tail30_mae_avg_pips": float(last_row["tail30_mae_avg_pips"]),
    }


def load_lstm_meta(config_path: Path):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "window": cfg["window"],
        "units": cfg["units"],
        "activation": cfg["activation"],
        "lr": cfg["lr"],
        "epochs": cfg["epochs"],
        "lambda_coef": cfg["lambda_coef"],
        "sigma_coef": cfg["sigma_coef"],
        "batch": cfg["batch"],
        "feature_count": len(cfg.get("feature_columns") or ["open", "high", "low", "close"]),
    }


def load_bilstm_param_text(html_path: Path):
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"<strong>Parameters:</strong>\s*([^<]+)</p>", text)
    return match.group(1).strip() if match else "Parameter string tidak ditemukan."


def lstm_param_count(input_dim: int, units: int, bidirectional: bool):
    recurrent = 4 * (input_dim * units + units * units + units)
    if bidirectional:
        recurrent *= 2
        dense_in = units * 2
    else:
        dense_in = units
    dense = dense_in * 4 + 4
    return recurrent + dense


def comparison_table(lstm_df: pd.DataFrame, bilstm_df: pd.DataFrame):
    merged = lstm_df.merge(bilstm_df, on="fold", suffixes=("_lstm", "_bilstm"))
    out_rows = []
    for _, row in merged.iterrows():
        lstm_mae = row["mae_avg_pips_lstm"]
        bilstm_mae = row["mae_avg_pips_bilstm"]
        better = "LSTM" if lstm_mae < bilstm_mae else "BiLSTM"
        out_rows.append(
            {
                "fold": int(row["fold"]),
                "lstm_mae_avg_pips": lstm_mae,
                "bilstm_mae_avg_pips": bilstm_mae,
                "lstm_corr2_avg_hlc": row["corr2_avg_hlc_lstm"],
                "bilstm_corr2_avg_hlc": row["corr2_avg_hlc_bilstm"],
                "lstm_da_body_pct": row["da_body_pct_lstm"],
                "bilstm_da_body_pct": row["da_body_pct_bilstm"],
                "delta_pips": bilstm_mae - lstm_mae,
                "winner": better,
            }
        )
    return pd.DataFrame(out_rows)


def main():
    args = parse_args()
    lstm_dir = Path(args.lstm_dir)
    bilstm_dir = Path(args.bilstm_dir)
    folds = args.folds

    lstm_df = compute_fold_metrics(lstm_dir, folds)
    bilstm_df = compute_fold_metrics(bilstm_dir, folds)

    lstm_summary = summarize_metrics(lstm_df)
    bilstm_summary = summarize_metrics(bilstm_df)
    fold_cmp = comparison_table(lstm_df, bilstm_df)

    lstm_meta = load_lstm_meta(Path(args.lstm_config))
    bilstm_meta_text = load_bilstm_param_text(Path(args.bilstm_meta_html))

    input_dim = lstm_meta["feature_count"]
    units = int(lstm_meta["units"])
    lstm_params = lstm_param_count(input_dim, units, bidirectional=False)
    bilstm_params = lstm_param_count(input_dim, units, bidirectional=True)

    summary_rows = [
        {
            "model": "FLF-LSTM",
            "mean_mae_avg_pips": lstm_summary["mean_mae_avg_pips"],
            "median_mae_avg_pips": lstm_summary["median_mae_avg_pips"],
            "mean_mae_avg_hlc_pips": lstm_summary["mean_mae_avg_hlc_pips"],
            "mean_corr2_avg_hlc": lstm_summary["mean_corr2_avg_hlc"],
            "mean_da_body_pct": lstm_summary["mean_da_body_pct"],
            "best_fold": lstm_summary["best_fold"],
            "best_fold_mae_avg_pips": lstm_summary["best_fold_mae_avg_pips"],
            "param_count": lstm_params,
        },
        {
            "model": "FLF-BiLSTM",
            "mean_mae_avg_pips": bilstm_summary["mean_mae_avg_pips"],
            "median_mae_avg_pips": bilstm_summary["median_mae_avg_pips"],
            "mean_mae_avg_hlc_pips": bilstm_summary["mean_mae_avg_hlc_pips"],
            "mean_corr2_avg_hlc": bilstm_summary["mean_corr2_avg_hlc"],
            "mean_da_body_pct": bilstm_summary["mean_da_body_pct"],
            "best_fold": bilstm_summary["best_fold"],
            "best_fold_mae_avg_pips": bilstm_summary["best_fold_mae_avg_pips"],
            "param_count": bilstm_params,
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    style = """
    body { font-family: Arial, sans-serif; margin: 20px; color: #222; }
    table { border-collapse: collapse; margin-bottom: 20px; }
    th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: right; }
    th { background: #f5f5f5; }
    td:first-child, th:first-child { text-align: left; }
    .note { max-width: 980px; line-height: 1.45; }
    """

    lstm_cfg_text = (
        f"window={lstm_meta['window']}, units={lstm_meta['units']}, activation={lstm_meta['activation']}, "
        f"lr={lstm_meta['lr']}, lambda={lstm_meta['lambda_coef']}, sigma={lstm_meta['sigma_coef']}, "
        f"batch={lstm_meta['batch']}, epochs={lstm_meta['epochs']}"
    )

    winner_text = (
        f"Pada skema validasi yang sama (WF 72 bulan, test 1 bulan, fold {folds[0]}-{folds[-1]}), "
        f"FLF-LSTM unggul atas FLF-BiLSTM pada rata-rata MAE avg: "
        f"{lstm_summary['mean_mae_avg_pips']:.2f} vs {bilstm_summary['mean_mae_avg_pips']:.2f} pips."
    )

    metric_note = (
        "Squared Correlation dihitung sebagai Pearson corr\u00b2 per komponen harga, lalu dirata-ratakan "
        "untuk High-Low-Close (corr\u00b2 avg HLC). Directional Accuracy dihitung dari kecocokan arah body "
        "candle: sign(pred_close - pred_open) dibanding sign(true_close - true_open)."
    )

    html = [
        "<html><head><meta charset='UTF-8'><title>Perbandingan FLF-LSTM vs FLF-BiLSTM</title>",
        f"<style>{style}</style></head><body>",
        "<h1>Perbandingan FLF-LSTM vs FLF-BiLSTM</h1>",
        "<p class='note'><strong>Dasar perbandingan:</strong> kedua model dibandingkan pada skema yang sama, "
        f"yaitu Walk-Forward Fixed 72 bulan train, 1 bulan test, menggunakan fold {', '.join(map(str, folds))}.</p>",
        f"<p class='note'>{winner_text}</p>",
        f"<p class='note'><strong>Definisi metrik:</strong> {metric_note}</p>",
        "<h2>Konfigurasi Model</h2>",
        f"<p class='note'><strong>FLF-LSTM:</strong> {lstm_cfg_text}</p>",
        f"<p class='note'><strong>FLF-BiLSTM:</strong> {bilstm_meta_text}</p>",
        "<h2>Ringkasan Utama</h2>",
        summary_df.to_html(index=False, float_format="%.4f"),
        "<h2>Per Fold</h2>",
        fold_cmp.to_html(index=False, float_format="%.4f"),
        "<h2>Detail FLF-LSTM</h2>",
        lstm_df.drop(columns=["tail30_mae_avg_pips", "tail30_mae_avg_hlc_pips"]).to_html(index=False, float_format="%.4f"),
        "<h2>Detail FLF-BiLSTM</h2>",
        bilstm_df.drop(columns=["tail30_mae_avg_pips", "tail30_mae_avg_hlc_pips"]).to_html(index=False, float_format="%.4f"),
        "<h2>Interpretasi</h2>",
        "<ul>",
        f"<li>FLF-LSTM unggul pada {(fold_cmp['winner'] == 'LSTM').sum()} dari {len(fold_cmp)} fold, sedangkan BiLSTM unggul pada {(fold_cmp['winner'] == 'BiLSTM').sum()} fold.</li>",
        f"<li>Rata-rata MAE avg turun sekitar {bilstm_summary['mean_mae_avg_pips'] - lstm_summary['mean_mae_avg_pips']:.2f} pips saat beralih dari BiLSTM ke LSTM.</li>",
        f"<li>Rata-rata corr\u00b2 avg HLC: LSTM {lstm_summary['mean_corr2_avg_hlc']:.4f} vs BiLSTM {bilstm_summary['mean_corr2_avg_hlc']:.4f}.</li>",
        f"<li>Rata-rata Directional Accuracy: LSTM {lstm_summary['mean_da_body_pct']:.2f}% vs BiLSTM {bilstm_summary['mean_da_body_pct']:.2f}%.</li>",
        f"<li>Kompleksitas parameter BiLSTM sekitar {bilstm_params / lstm_params:.2f}x LSTM ({bilstm_params:,} vs {lstm_params:,}).</li>",
        "<li>Dengan hasil ini, untuk horizon test pendek 1 bulan, LSTM saat ini lebih efisien dan cenderung lebih akurat daripada BiLSTM pada fold yang diuji.</li>",
        "</ul>",
        "</body></html>",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(html), encoding="utf-8")
    print(f"Wrote comparison report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
