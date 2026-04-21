import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PIP_FACTOR = 10000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ARIMA, FLF-LSTM, and FLF-BiLSTM on WF72m/1m last5 folds."
    )
    parser.add_argument(
        "--arima-dir",
        default="/home/hduser/jupyter/gust/RisetEU/Arima/result/arima_wf72_test1_last5",
        help="Directory containing ARIMA foldXX_preds.csv files.",
    )
    parser.add_argument(
        "--lstm-dir",
        default="/home/hduser/jupyter/gust/RisetEU/FLF_LSTM/results/wf72_test1_lstm_last5",
        help="Directory containing FLF-LSTM foldXX_preds.csv files.",
    )
    parser.add_argument(
        "--bilstm-dir",
        default="/home/hduser/jupyter/gust/RisetEU/results/rolling_train72_test1_last5",
        help="Directory containing FLF-BiLSTM foldXX_preds.csv files.",
    )
    parser.add_argument(
        "--out",
        default="/home/hduser/jupyter/gust/RisetEU/results/model_comparison_wf72_test1_last5.html",
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
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def summarize_metrics(df: pd.DataFrame):
    best_row = df.loc[df["mae_avg_pips"].idxmin()]
    worst_row = df.loc[df["mae_avg_pips"].idxmax()]
    return {
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


def load_lstm_meta():
    cfg_path = Path("/home/hduser/jupyter/gust/RisetEU/FLF_LSTM/lstm_flf_config_wf72_test1_best.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return (
        f"window={cfg['window']}, units={cfg['units']}, activation={cfg['activation']}, "
        f"lr={cfg['lr']}, lambda={cfg['lambda_coef']}, sigma={cfg['sigma_coef']}, "
        f"batch={cfg['batch']}, epochs={cfg['epochs']}"
    )


def load_bilstm_meta():
    df = pd.read_csv("/home/hduser/jupyter/gust/RisetEU/results/eurusd_pipeline/best_progression.csv")
    row = df.iloc[-1]
    return (
        f"window={int(row['window'])}, units={int(row['units'])}, activation={row['activation']}, "
        f"lr={row['lr']}, lambda={row['lambda_coef']}, sigma={row['sigma_coef']}, "
        f"batch={int(row['batch'])}, epochs={int(row['epochs'])}"
    )


def load_arima_meta():
    cfg = json.loads(Path("/home/hduser/jupyter/gust/RisetEU/Arima/arima_baseline_config.json").read_text(encoding="utf-8"))
    p_values = ",".join(map(str, cfg["p_values"]))
    d_values = ",".join(map(str, cfg["d_values"]))
    q_values = ",".join(map(str, cfg["q_values"]))
    return (
        "OHLC-only, 4 model ARIMA univariat (Open/High/Low/Close), "
        f"order_search={cfg['order_search']}, p=[{p_values}], d=[{d_values}], q=[{q_values}], "
        f"selection_metric={cfg['selection_metric']}, trend={cfg['trend']}"
    )


def build_fold_comparison(arima_df: pd.DataFrame, lstm_df: pd.DataFrame, bilstm_df: pd.DataFrame):
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
        )
    )

    winners = []
    margins = []
    for _, row in merged.iterrows():
        candidates = {
            "ARIMA": row["arima_mae_avg_pips"],
            "FLF-LSTM": row["lstm_mae_avg_pips"],
            "FLF-BiLSTM": row["bilstm_mae_avg_pips"],
        }
        ordered = sorted(candidates.items(), key=lambda x: x[1])
        winners.append(ordered[0][0])
        margins.append(ordered[1][1] - ordered[0][1])
    merged["winner"] = winners
    merged["margin_to_runner_up_pips"] = margins
    return merged


def model_rank_text(summary_df: pd.DataFrame):
    ordered = summary_df.sort_values("mean_mae_avg_pips").reset_index(drop=True)
    lines = []
    for i, row in ordered.iterrows():
        lines.append(f"{i+1}. {row['model']} ({row['mean_mae_avg_pips']:.4f} pips)")
    return " | ".join(lines)


def main():
    args = parse_args()
    folds = args.folds

    arima_df = compute_fold_metrics(Path(args.arima_dir), folds)
    lstm_df = compute_fold_metrics(Path(args.lstm_dir), folds)
    bilstm_df = compute_fold_metrics(Path(args.bilstm_dir), folds)

    arima_summary = summarize_metrics(arima_df)
    lstm_summary = summarize_metrics(lstm_df)
    bilstm_summary = summarize_metrics(bilstm_df)

    summary_df = pd.DataFrame(
        [
            {
                "model": "ARIMA",
                "mean_mae_avg_pips": arima_summary["mean_mae_avg_pips"],
                "median_mae_avg_pips": arima_summary["median_mae_avg_pips"],
                "mean_mae_avg_hlc_pips": arima_summary["mean_mae_avg_hlc_pips"],
                "mean_corr2_avg_hlc": arima_summary["mean_corr2_avg_hlc"],
                "mean_da_body_pct": arima_summary["mean_da_body_pct"],
                "best_fold": arima_summary["best_fold"],
                "best_fold_mae_avg_pips": arima_summary["best_fold_mae_avg_pips"],
            },
            {
                "model": "FLF-LSTM",
                "mean_mae_avg_pips": lstm_summary["mean_mae_avg_pips"],
                "median_mae_avg_pips": lstm_summary["median_mae_avg_pips"],
                "mean_mae_avg_hlc_pips": lstm_summary["mean_mae_avg_hlc_pips"],
                "mean_corr2_avg_hlc": lstm_summary["mean_corr2_avg_hlc"],
                "mean_da_body_pct": lstm_summary["mean_da_body_pct"],
                "best_fold": lstm_summary["best_fold"],
                "best_fold_mae_avg_pips": lstm_summary["best_fold_mae_avg_pips"],
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
            },
        ]
    )

    fold_cmp = build_fold_comparison(arima_df, lstm_df, bilstm_df)
    winner_counts = fold_cmp["winner"].value_counts().to_dict()

    rank_text = model_rank_text(summary_df)
    lstm_cfg_text = load_lstm_meta()
    bilstm_cfg_text = load_bilstm_meta()
    arima_cfg_text = load_arima_meta()

    style = """
    body { font-family: Arial, sans-serif; margin: 20px; color: #222; }
    table { border-collapse: collapse; margin-bottom: 20px; }
    th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: right; }
    th { background: #f5f5f5; }
    td:first-child, th:first-child { text-align: left; }
    .note { max-width: 1100px; line-height: 1.45; }
    ul { line-height: 1.5; }
    """

    html = [
        "<html><head><meta charset='UTF-8'><title>Comparison ARIMA vs FLF-LSTM vs FLF-BiLSTM</title>",
        f"<style>{style}</style></head><body>",
        "<h1>Comparison ARIMA vs FLF-LSTM vs FLF-BiLSTM</h1>",
        "<p class='note'><strong>Dasar perbandingan:</strong> ketiga model dibandingkan pada skema yang sama, "
        f"yaitu Walk-Forward Fixed 72 bulan train, 1 bulan test, menggunakan fold {', '.join(map(str, folds))}.</p>",
        "<p class='note'><strong>Ringkasan peringkat rata-rata MAE(pips):</strong> "
        f"{rank_text}</p>",
        "<h2>Konfigurasi Model</h2>",
        f"<p class='note'><strong>ARIMA:</strong> {arima_cfg_text}</p>",
        f"<p class='note'><strong>FLF-LSTM:</strong> {lstm_cfg_text}</p>",
        f"<p class='note'><strong>FLF-BiLSTM:</strong> {bilstm_cfg_text}</p>",
        "<h2>Ringkasan Utama</h2>",
        summary_df.to_html(index=False, float_format='%.4f'),
        "<h2>Per Fold</h2>",
        fold_cmp.to_html(index=False, float_format='%.4f'),
        "<h2>Detail ARIMA</h2>",
        arima_df.to_html(index=False, float_format='%.4f'),
        "<h2>Detail FLF-LSTM</h2>",
        lstm_df.to_html(index=False, float_format='%.4f'),
        "<h2>Detail FLF-BiLSTM</h2>",
        bilstm_df.to_html(index=False, float_format='%.4f'),
        "<h2>Interpretasi</h2>",
        "<ul>",
        f"<li>Pada rata-rata MAE avg, model terbaik adalah <strong>{summary_df.sort_values('mean_mae_avg_pips').iloc[0]['model']}</strong> dengan nilai {summary_df['mean_mae_avg_pips'].min():.4f} pips.</li>",
        f"<li>FLF-LSTM menang pada {winner_counts.get('FLF-LSTM', 0)} fold, FLF-BiLSTM pada {winner_counts.get('FLF-BiLSTM', 0)} fold, dan ARIMA pada {winner_counts.get('ARIMA', 0)} fold.</li>",
        f"<li>Rata-rata MAE avg ARIMA = {arima_summary['mean_mae_avg_pips']:.4f} pips, FLF-LSTM = {lstm_summary['mean_mae_avg_pips']:.4f} pips, FLF-BiLSTM = {bilstm_summary['mean_mae_avg_pips']:.4f} pips.</li>",
        f"<li>Rata-rata corr² avg HLC ARIMA = {arima_summary['mean_corr2_avg_hlc']:.4f}, FLF-LSTM = {lstm_summary['mean_corr2_avg_hlc']:.4f}, FLF-BiLSTM = {bilstm_summary['mean_corr2_avg_hlc']:.4f}.</li>",
        f"<li>Rata-rata Directional Accuracy body candle ARIMA = {arima_summary['mean_da_body_pct']:.2f}%, FLF-LSTM = {lstm_summary['mean_da_body_pct']:.2f}%, FLF-BiLSTM = {bilstm_summary['mean_da_body_pct']:.2f}%.</li>",
        "<li>Dengan skenario pengujian utama ini, ARIMA berfungsi sebagai baseline statistik, sedangkan FLF-LSTM dan FLF-BiLSTM menunjukkan apakah pemodelan deep learning memberi peningkatan yang konsisten pada prediksi OHLC.</li>",
        "</ul>",
        "</body></html>",
    ]

    out_path = Path(args.out)
    out_path.write_text("".join(html), encoding="utf-8")
    print(f"Wrote comparison report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
