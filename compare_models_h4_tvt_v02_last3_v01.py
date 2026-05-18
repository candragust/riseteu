import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PIP_FACTOR = 10000.0
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ARIMA, FLF-LSTM, and FLF-BiLSTM on EUR/USD H4 TVT v02 last3 folds."
    )
    parser.add_argument(
        "--arima-dir",
        default=str(PROJECT_ROOT / "Arima" / "result" / "tvt_v02" / "h4_evaluation_last3"),
        help="Directory containing ARIMA H4 TVT v02 last3 outputs.",
    )
    parser.add_argument(
        "--lstm-dir",
        default=str(PROJECT_ROOT / "FLF_LSTM" / "results" / "tvt_v02" / "h4_evaluation_last3"),
        help="Directory containing FLF-LSTM H4 TVT v02 last3 outputs.",
    )
    parser.add_argument(
        "--bilstm-dir",
        default=str(PROJECT_ROOT / "FLF_BILSTM" / "results" / "tvt_v02" / "h4_evaluation_last3"),
        help="Directory containing FLF-BiLSTM H4 TVT v02 last3 outputs.",
    )
    parser.add_argument(
        "--lstm-config",
        default=str(PROJECT_ROOT / "FLF_LSTM" / "lstm_flf_config_h4_tvt_v02_best.json"),
        help="Final LSTM config JSON.",
    )
    parser.add_argument(
        "--bilstm-config",
        default=str(PROJECT_ROOT / "FLF_BILSTM" / "bilstm_flf_config_h4_tvt_v02_best.json"),
        help="Final BiLSTM config JSON.",
    )
    parser.add_argument(
        "--arima-config",
        default=str(PROJECT_ROOT / "Arima" / "arima_baseline_config.json"),
        help="ARIMA config JSON.",
    )
    parser.add_argument(
        "--out-html",
        default=str(PROJECT_ROOT / "comparison" / "tvt_v02" / "h4" / "comparison_models_h4_tvt_v02_last3_v01.html"),
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[19, 20, 21],
        help="Fold numbers to compare.",
    )
    parser.add_argument("--profile-label", default="H4 TVT v02", help="Profile label shown in report.")
    parser.add_argument("--timeframe-label", default="H4", help="Timeframe label shown in report.")
    parser.add_argument("--tuning-folds-label", default="13-18", help="Tuning fold label shown in report.")
    parser.add_argument("--evaluation-folds-label", default="19-21", help="Evaluation fold label shown in report.")
    return parser.parse_args()


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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def format_table(df: pd.DataFrame, digits: int = 4) -> str:
    return df.to_html(
        index=False,
        float_format=lambda x: f"{x:.{digits}f}" if isinstance(x, float) else str(x),
        classes="data-table",
        border=0,
    )


def infer_history_path(base_dir: Path, fold: int) -> Path:
    return base_dir / f"fold{fold:02d}_history.csv"


def infer_preds_path(base_dir: Path, fold: int) -> Path:
    return base_dir / f"fold{fold:02d}_preds.csv"


def infer_arima_summary_path(base_dir: Path, fold: int) -> Path:
    return base_dir / f"fold{fold:02d}_summary.json"


def candle_coherence_stats(df: pd.DataFrame) -> dict:
    pred_open = df["pred_open"].to_numpy(dtype=float)
    pred_high = df["pred_high"].to_numpy(dtype=float)
    pred_low = df["pred_low"].to_numpy(dtype=float)
    pred_close = df["pred_close"].to_numpy(dtype=float)

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
        "invalid_candles": int(any_invalid.sum()),
    }


def compute_fold_metrics(base_dir: Path, folds, model_name: str):
    rows = []
    audit_rows = []
    for fold in folds:
        preds_path = infer_preds_path(base_dir, fold)
        df = pd.read_csv(preds_path)

        pred_values = df[["pred_open", "pred_high", "pred_low", "pred_close"]].to_numpy(dtype=float)
        true_values = df[["true_open", "true_high", "true_low", "true_close"]].to_numpy(dtype=float)
        diffs = np.abs(pred_values - true_values) * PIP_FACTOR
        avg = diffs.mean(axis=0)
        corr2 = [squared_corr(true_values[:, i], pred_values[:, i]) for i in range(4)]
        da_body_pct = directional_accuracy(
            df["true_open"], df["true_close"], df["pred_open"], df["pred_close"]
        )
        true_dir = np.sign(df["true_close"].to_numpy(dtype=float) - df["true_open"].to_numpy(dtype=float))
        pred_dir = np.sign(df["pred_close"].to_numpy(dtype=float) - df["pred_open"].to_numpy(dtype=float))

        row = {
            "fold": int(fold),
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
            "corr2_avg_ohlc": nanmean(corr2),
            "corr2_avg_hlc": nanmean(corr2[1:]),
            "da_body_pct": float(da_body_pct),
            "true_bull": int((true_dir > 0).sum()),
            "true_bear": int((true_dir < 0).sum()),
            "true_neutral": int((true_dir == 0).sum()),
            "pred_bull": int((pred_dir > 0).sum()),
            "pred_bear": int((pred_dir < 0).sum()),
            "pred_neutral": int((pred_dir == 0).sum()),
        }

        history_path = infer_history_path(base_dir, fold)
        if history_path.exists():
            hist = pd.read_csv(history_path)
            if "val_loss" in hist.columns and len(hist):
                row["epochs_ran"] = int(len(hist))
                row["best_epoch"] = int(hist["val_loss"].idxmin() + 1)
                row["best_val_loss"] = float(hist["val_loss"].min())

        if model_name == "ARIMA":
            summary_path = infer_arima_summary_path(base_dir, fold)
            if summary_path.exists():
                summary = load_json(summary_path)
                for target in ("open", "high", "low", "close"):
                    row[f"order_{target}"] = str(tuple(summary["targets"][target]["selected_order"]))
            audit_rows.append({"fold": int(fold), **candle_coherence_stats(df)})

        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    audit_df = pd.DataFrame(audit_rows).sort_values("fold").reset_index(drop=True) if audit_rows else None
    return metrics_df, audit_df


def summarize_metrics(df: pd.DataFrame):
    best_row = df.loc[df["mae_avg_pips"].idxmin()]
    worst_row = df.loc[df["mae_avg_pips"].idxmax()]
    out = {
        "mean_mae_open_pips": float(df["mae_open_pips"].mean()),
        "mean_mae_high_pips": float(df["mae_high_pips"].mean()),
        "mean_mae_low_pips": float(df["mae_low_pips"].mean()),
        "mean_mae_close_pips": float(df["mae_close_pips"].mean()),
        "mean_mae_avg_pips": float(df["mae_avg_pips"].mean()),
        "median_mae_avg_pips": float(df["mae_avg_pips"].median()),
        "mean_mae_avg_hlc_pips": float(df["mae_avg_hlc_pips"].mean()),
        "mean_corr2_open": float(df["corr2_open"].mean()),
        "mean_corr2_high": float(df["corr2_high"].mean()),
        "mean_corr2_low": float(df["corr2_low"].mean()),
        "mean_corr2_close": float(df["corr2_close"].mean()),
        "mean_corr2_avg_ohlc": float(df["corr2_avg_ohlc"].mean()),
        "mean_corr2_avg_hlc": float(df["corr2_avg_hlc"].mean()),
        "mean_da_body_pct": float(df["da_body_pct"].mean()),
        "best_fold": int(best_row["fold"]),
        "best_fold_mae_avg_pips": float(best_row["mae_avg_pips"]),
        "worst_fold": int(worst_row["fold"]),
        "worst_fold_mae_avg_pips": float(worst_row["mae_avg_pips"]),
        "total_samples": int(df["samples"].sum()),
    }
    if "epochs_ran" in df.columns:
        out["mean_epochs_ran"] = float(df["epochs_ran"].mean())
        out["mean_best_epoch"] = float(df["best_epoch"].mean())
        out["mean_best_val_loss"] = float(df["best_val_loss"].mean())
    return out


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
        summary_path = infer_arima_summary_path(arima_dir, fold)
        data = load_json(summary_path)
        for col in order_counter:
            order_counter[col][tuple(data["targets"][col]["selected_order"])] += 1
    order_parts = []
    for col, counter in order_counter.items():
        order, count = counter.most_common(1)[0]
        order_parts.append(f"{col}={order} ({count}/{len(folds)} fold)")
    return (
        f"order_search={str(cfg['order_search']).lower()}, requested_order={tuple(cfg['order'])}, "
        f"grid p={tuple(cfg['p_values'])}, d={tuple(cfg['d_values'])}, q={tuple(cfg['q_values'])}, "
        f"selection_metric={cfg['selection_metric']}; dominant selected orders: " + ", ".join(order_parts)
    )


def load_atr_relation_row(result_dir: Path, model_name: str) -> dict:
    path = result_dir / "mae_atr_fold_allfull.csv"
    df = pd.read_csv(path)
    row = df.iloc[0]
    atr12 = float(row["atr12_mean_pips"])
    mae_avg = float(row["mae_avg_pips"])
    le_atr12 = float(row["avg_error_le_atr12_pct"])
    le_atr6 = float(row["avg_error_le_atr6_pct"])
    return {
        "model": model_name,
        "mean_mae_avg_pips": mae_avg,
        "mean_atr12_pips": atr12,
        "mae_avg_pct_atr12": float(mae_avg / atr12 * 100.0),
        "avg_error_le_atr12_pct": le_atr12,
        "avg_error_gt_atr12_pct": float(100.0 - le_atr12),
        "avg_error_le_atr6_pct": le_atr6,
        "avg_error_gt_atr6_pct": float(100.0 - le_atr6),
    }


def build_model_summary_df(arima_summary, lstm_summary, bilstm_summary):
    rows = [
        {
            "model": "ARIMA",
            "mean_mae_avg_pips": arima_summary["mean_mae_avg_pips"],
            "mean_mae_avg_hlc_pips": arima_summary["mean_mae_avg_hlc_pips"],
            "mean_corr2_avg_ohlc": arima_summary["mean_corr2_avg_ohlc"],
            "mean_corr2_avg_hlc": arima_summary["mean_corr2_avg_hlc"],
            "mean_da_body_pct": arima_summary["mean_da_body_pct"],
            "best_fold": arima_summary["best_fold"],
            "best_fold_mae_avg_pips": arima_summary["best_fold_mae_avg_pips"],
            "total_samples": arima_summary["total_samples"],
        },
        {
            "model": "FLF-LSTM",
            "mean_mae_avg_pips": lstm_summary["mean_mae_avg_pips"],
            "mean_mae_avg_hlc_pips": lstm_summary["mean_mae_avg_hlc_pips"],
            "mean_corr2_avg_ohlc": lstm_summary["mean_corr2_avg_ohlc"],
            "mean_corr2_avg_hlc": lstm_summary["mean_corr2_avg_hlc"],
            "mean_da_body_pct": lstm_summary["mean_da_body_pct"],
            "best_fold": lstm_summary["best_fold"],
            "best_fold_mae_avg_pips": lstm_summary["best_fold_mae_avg_pips"],
            "total_samples": lstm_summary["total_samples"],
        },
        {
            "model": "FLF-BiLSTM",
            "mean_mae_avg_pips": bilstm_summary["mean_mae_avg_pips"],
            "mean_mae_avg_hlc_pips": bilstm_summary["mean_mae_avg_hlc_pips"],
            "mean_corr2_avg_ohlc": bilstm_summary["mean_corr2_avg_ohlc"],
            "mean_corr2_avg_hlc": bilstm_summary["mean_corr2_avg_hlc"],
            "mean_da_body_pct": bilstm_summary["mean_da_body_pct"],
            "best_fold": bilstm_summary["best_fold"],
            "best_fold_mae_avg_pips": bilstm_summary["best_fold_mae_avg_pips"],
            "total_samples": bilstm_summary["total_samples"],
        },
    ]
    return pd.DataFrame(rows)


def build_pairwise_df(arima_df, lstm_df, bilstm_df):
    merged = (
        arima_df[["fold", "samples", "mae_avg_pips", "corr2_avg_hlc", "da_body_pct"]]
        .rename(
            columns={
                "samples": "samples_arima",
                "mae_avg_pips": "arima_mae_avg_pips",
                "corr2_avg_hlc": "arima_corr2_avg_hlc",
                "da_body_pct": "arima_da_body_pct",
            }
        )
        .merge(
            lstm_df[["fold", "samples", "mae_avg_pips", "corr2_avg_hlc", "da_body_pct"]].rename(
                columns={
                    "samples": "samples_lstm",
                    "mae_avg_pips": "lstm_mae_avg_pips",
                    "corr2_avg_hlc": "lstm_corr2_avg_hlc",
                    "da_body_pct": "lstm_da_body_pct",
                }
            ),
            on="fold",
            how="inner",
        )
        .merge(
            bilstm_df[["fold", "samples", "mae_avg_pips", "corr2_avg_hlc", "da_body_pct"]].rename(
                columns={
                    "samples": "samples_bilstm",
                    "mae_avg_pips": "bilstm_mae_avg_pips",
                    "corr2_avg_hlc": "bilstm_corr2_avg_hlc",
                    "da_body_pct": "bilstm_da_body_pct",
                }
            ),
            on="fold",
            how="inner",
        )
    )
    def tied_winner(row, columns, labels, higher_is_better: bool) -> str:
        values = np.asarray([row[col] for col in columns], dtype=float)
        target = np.nanmax(values) if higher_is_better else np.nanmin(values)
        winners = [label for label, value in zip(labels, values) if np.isclose(value, target, rtol=1e-9, atol=1e-9)]
        return "+".join(winners)

    labels = ["arima", "lstm", "bilstm"]
    merged["winner_mae"] = merged.apply(
        tied_winner,
        axis=1,
        columns=["arima_mae_avg_pips", "lstm_mae_avg_pips", "bilstm_mae_avg_pips"],
        labels=labels,
        higher_is_better=False,
    )
    merged["winner_corr2"] = merged.apply(
        tied_winner,
        axis=1,
        columns=["arima_corr2_avg_hlc", "lstm_corr2_avg_hlc", "bilstm_corr2_avg_hlc"],
        labels=labels,
        higher_is_better=True,
    )
    merged["winner_da"] = merged.apply(
        tied_winner,
        axis=1,
        columns=["arima_da_body_pct", "lstm_da_body_pct", "bilstm_da_body_pct"],
        labels=labels,
        higher_is_better=True,
    )
    return merged


def bar_group_chart(x, series_map, title, yaxis_title, colors, y_dtick=None):
    fig = go.Figure()
    for (name, values), color in zip(series_map.items(), colors):
        fig.add_trace(go.Bar(name=name, x=x, y=values, marker_color=color))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title=title,
        yaxis_title=yaxis_title,
        height=430,
        margin=dict(l=60, r=30, t=70, b=60),
        legend=dict(orientation="h", y=1.15, x=0.0),
    )
    if y_dtick is not None:
        fig.update_yaxes(dtick=y_dtick)
    return fig


def line_compare_chart(x, series_map, title, yaxis_title, colors, y_dtick=None):
    fig = go.Figure()
    for (name, values), color in zip(series_map.items(), colors):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color),
            )
        )
    fig.update_layout(
        template="plotly_white",
        title=title,
        yaxis_title=yaxis_title,
        xaxis_title="Fold",
        height=420,
        margin=dict(l=60, r=30, t=70, b=60),
        legend=dict(orientation="h", y=1.15, x=0.0),
    )
    if y_dtick is not None:
        fig.update_yaxes(dtick=y_dtick)
    return fig


def to_html_fragment(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)


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

    arima_df, arima_audit_df = compute_fold_metrics(arima_dir, args.folds, "ARIMA")
    lstm_df, _ = compute_fold_metrics(lstm_dir, args.folds, "FLF-LSTM")
    bilstm_df, _ = compute_fold_metrics(bilstm_dir, args.folds, "FLF-BiLSTM")

    arima_summary = summarize_metrics(arima_df)
    lstm_summary = summarize_metrics(lstm_df)
    bilstm_summary = summarize_metrics(bilstm_df)

    model_df = build_model_summary_df(arima_summary, lstm_summary, bilstm_summary)
    pairwise_df = build_pairwise_df(arima_df, lstm_df, bilstm_df)
    atr_relation_df = pd.DataFrame(
        [
            load_atr_relation_row(arima_dir, "ARIMA"),
            load_atr_relation_row(lstm_dir, "FLF-LSTM"),
            load_atr_relation_row(bilstm_dir, "FLF-BiLSTM"),
        ]
    )

    base_stem = out_html.with_suffix("")
    arima_csv = base_stem.parent / f"{base_stem.name}_arima_metrics.csv"
    lstm_csv = base_stem.parent / f"{base_stem.name}_lstm_metrics.csv"
    bilstm_csv = base_stem.parent / f"{base_stem.name}_bilstm_metrics.csv"
    pairwise_csv = base_stem.parent / f"{base_stem.name}_pairwise.csv"
    model_csv = base_stem.parent / f"{base_stem.name}_summary.csv"
    audit_csv = base_stem.parent / f"{base_stem.name}_arima_candle_audit.csv"
    atr_relation_csv = base_stem.parent / f"{base_stem.name}_atr_relation.csv"
    summary_json = base_stem.parent / f"{base_stem.name}_summary.json"

    arima_df.to_csv(arima_csv, index=False)
    lstm_df.to_csv(lstm_csv, index=False)
    bilstm_df.to_csv(bilstm_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)
    model_df.to_csv(model_csv, index=False)
    atr_relation_df.to_csv(atr_relation_csv, index=False)
    if arima_audit_df is not None:
        arima_audit_df.to_csv(audit_csv, index=False)

    payload = {
        "scope": {
            "profile": args.profile_label,
            "folds": args.folds,
            "tuning_folds": args.tuning_folds_label,
            "evaluation_folds": args.evaluation_folds_label,
        },
        "arima": arima_summary,
        "lstm": lstm_summary,
        "bilstm": bilstm_summary,
        "artifacts": {
            "html": str(out_html),
            "summary_csv": str(model_csv),
            "pairwise_csv": str(pairwise_csv),
            "arima_metrics_csv": str(arima_csv),
            "lstm_metrics_csv": str(lstm_csv),
            "bilstm_metrics_csv": str(bilstm_csv),
            "arima_candle_audit_csv": str(audit_csv),
            "atr_relation_csv": str(atr_relation_csv),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    colors = ["#d97706", "#2563eb", "#059669"]
    model_names = model_df["model"].tolist()
    folds = pairwise_df["fold"].tolist()

    mae_component_fig = bar_group_chart(
        ["Open", "High", "Low", "Close", "AVG", "AVG HLC"],
        {
            "ARIMA": [
                arima_summary["mean_mae_open_pips"],
                arima_summary["mean_mae_high_pips"],
                arima_summary["mean_mae_low_pips"],
                arima_summary["mean_mae_close_pips"],
                arima_summary["mean_mae_avg_pips"],
                arima_summary["mean_mae_avg_hlc_pips"],
            ],
            "FLF-LSTM": [
                lstm_summary["mean_mae_open_pips"],
                lstm_summary["mean_mae_high_pips"],
                lstm_summary["mean_mae_low_pips"],
                lstm_summary["mean_mae_close_pips"],
                lstm_summary["mean_mae_avg_pips"],
                lstm_summary["mean_mae_avg_hlc_pips"],
            ],
            "FLF-BiLSTM": [
                bilstm_summary["mean_mae_open_pips"],
                bilstm_summary["mean_mae_high_pips"],
                bilstm_summary["mean_mae_low_pips"],
                bilstm_summary["mean_mae_close_pips"],
                bilstm_summary["mean_mae_avg_pips"],
                bilstm_summary["mean_mae_avg_hlc_pips"],
            ],
        },
        "Perbandingan Mean MAE per Komponen",
        "MAE (pips)",
        colors,
        y_dtick=0.5,
    )

    corr2_component_fig = bar_group_chart(
        ["Open", "High", "Low", "Close", "AVG OHLC", "AVG HLC"],
        {
            "ARIMA": [
                arima_summary["mean_corr2_open"],
                arima_summary["mean_corr2_high"],
                arima_summary["mean_corr2_low"],
                arima_summary["mean_corr2_close"],
                arima_summary["mean_corr2_avg_ohlc"],
                arima_summary["mean_corr2_avg_hlc"],
            ],
            "FLF-LSTM": [
                lstm_summary["mean_corr2_open"],
                lstm_summary["mean_corr2_high"],
                lstm_summary["mean_corr2_low"],
                lstm_summary["mean_corr2_close"],
                lstm_summary["mean_corr2_avg_ohlc"],
                lstm_summary["mean_corr2_avg_hlc"],
            ],
            "FLF-BiLSTM": [
                bilstm_summary["mean_corr2_open"],
                bilstm_summary["mean_corr2_high"],
                bilstm_summary["mean_corr2_low"],
                bilstm_summary["mean_corr2_close"],
                bilstm_summary["mean_corr2_avg_ohlc"],
                bilstm_summary["mean_corr2_avg_hlc"],
            ],
        },
        "Perbandingan Mean corr2 (Squared Correlation)",
        "corr2",
        colors,
    )

    mae_fold_fig = line_compare_chart(
        folds,
        {
            "ARIMA": pairwise_df["arima_mae_avg_pips"].tolist(),
            "FLF-LSTM": pairwise_df["lstm_mae_avg_pips"].tolist(),
            "FLF-BiLSTM": pairwise_df["bilstm_mae_avg_pips"].tolist(),
        },
        "MAE AVG per Fold",
        "MAE AVG (pips)",
        colors,
        y_dtick=0.5,
    )

    corr2_fold_fig = line_compare_chart(
        folds,
        {
            "ARIMA": pairwise_df["arima_corr2_avg_hlc"].tolist(),
            "FLF-LSTM": pairwise_df["lstm_corr2_avg_hlc"].tolist(),
            "FLF-BiLSTM": pairwise_df["bilstm_corr2_avg_hlc"].tolist(),
        },
        "corr2 AVG HLC per Fold",
        "corr2 AVG HLC",
        colors,
    )

    da_fold_fig = line_compare_chart(
        folds,
        {
            "ARIMA": pairwise_df["arima_da_body_pct"].tolist(),
            "FLF-LSTM": pairwise_df["lstm_da_body_pct"].tolist(),
            "FLF-BiLSTM": pairwise_df["bilstm_da_body_pct"].tolist(),
        },
        "Directional Accuracy per Fold",
        "Directional Accuracy (%)",
        colors,
    )

    atr_ratio_fig = bar_group_chart(
        atr_relation_df["model"].tolist(),
        {"MAE AVG sebagai % ATR12": atr_relation_df["mae_avg_pct_atr12"].tolist()},
        "Perbandingan MAE AVG terhadap ATR12",
        "MAE AVG (% dari ATR12)",
        ["#7c3aed"],
        y_dtick=5.0,
    )
    atr12_freq_fig = bar_group_chart(
        atr_relation_df["model"].tolist(),
        {
            "Error AVG <= ATR12 (%)": atr_relation_df["avg_error_le_atr12_pct"].tolist(),
            "Error AVG > ATR12 (%)": atr_relation_df["avg_error_gt_atr12_pct"].tolist(),
        },
        "Frekuensi Average Absolute Error per Candle terhadap ATR12",
        "Persentase Candle (%)",
        ["#2563eb", "#dc2626"],
        y_dtick=5.0,
    )

    mae_winner = model_df.loc[model_df["mean_mae_avg_pips"].idxmin(), "model"]
    corr_winner = model_df.loc[model_df["mean_corr2_avg_hlc"].idxmax(), "model"]
    da_winner = model_df.loc[model_df["mean_da_body_pct"].idxmax(), "model"]

    arima_meta = load_arima_meta(arima_cfg_path, arima_dir, args.folds)
    lstm_meta = load_lstm_meta(lstm_cfg_path)
    bilstm_meta = load_bilstm_meta(bilstm_cfg_path)

    arima_invalid_note = ""
    if arima_audit_df is not None and not arima_audit_df.empty:
        invalid_total = int(arima_audit_df["invalid_candles"].sum())
        sample_total = int(arima_df["samples"].sum())
        invalid_pct = 100.0 * invalid_total / max(sample_total, 1)
        arima_invalid_note = (
            f"ARIMA menunjukkan <strong>{invalid_total}</strong> candle tidak valid dari "
            f"<strong>{sample_total}</strong> candle ({invalid_pct:.2f}%) pada audit struktur OHLC, "
            "meskipun tidak ada kasus <code>pred_high &lt; pred_low</code>. Temuan ini perlu dibaca "
            "sebagai keterbatasan pendekatan per-series independent forecasting."
        )

    lstm_vs_arima_gap = arima_summary["mean_mae_avg_pips"] - lstm_summary["mean_mae_avg_pips"]
    lstm_vs_arima_gap_pct = 100.0 * lstm_vs_arima_gap / max(arima_summary["mean_mae_avg_pips"], 1e-12)
    lstm_vs_bilstm_gap = bilstm_summary["mean_mae_avg_pips"] - lstm_summary["mean_mae_avg_pips"]
    lstm_vs_bilstm_gap_pct = 100.0 * lstm_vs_bilstm_gap / max(bilstm_summary["mean_mae_avg_pips"], 1e-12)
    da_gap_bilstm_vs_lstm = bilstm_summary["mean_da_body_pct"] - lstm_summary["mean_da_body_pct"]
    da_gap_bilstm_vs_arima = bilstm_summary["mean_da_body_pct"] - arima_summary["mean_da_body_pct"]
    winner_mae_counts = pairwise_df["winner_mae"].value_counts().to_dict()
    winner_corr2_counts = pairwise_df["winner_corr2"].value_counts().to_dict()
    winner_da_counts = pairwise_df["winner_da"].value_counts().to_dict()
    atr_best = atr_relation_df.loc[atr_relation_df["mae_avg_pct_atr12"].idxmin()]
    atr_worst = atr_relation_df.loc[atr_relation_df["mae_avg_pct_atr12"].idxmax()]

    summary_by_model = {
        "ARIMA": arima_summary,
        "FLF-LSTM": lstm_summary,
        "FLF-BiLSTM": bilstm_summary,
    }
    pairwise_key = {"ARIMA": "arima", "FLF-LSTM": "lstm", "FLF-BiLSTM": "bilstm"}
    display_by_pairwise_key = {value: key for key, value in pairwise_key.items()}

    def mean_metric_winners(metric_name: str, higher_is_better: bool) -> list[str]:
        values = model_df.set_index("model")[metric_name].astype(float)
        target = values.max() if higher_is_better else values.min()
        return [model for model, value in values.items() if np.isclose(value, target, rtol=1e-9, atol=1e-9)]

    def model_list_text(models: list[str]) -> str:
        if len(models) <= 1:
            return models[0] if models else "-"
        return ", ".join(models[:-1]) + " dan " + models[-1]

    def winner_count_text(counts: dict) -> str:
        if not counts:
            return "-"
        parts = []
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            labels = [display_by_pairwise_key.get(part, part) for part in str(key).split("+")]
            parts.append(f"{model_list_text(labels)} {count}/{len(args.folds)} fold")
        return ", ".join(parts)

    mae_winners = mean_metric_winners("mean_mae_avg_pips", higher_is_better=False)
    corr_winners = mean_metric_winners("mean_corr2_avg_hlc", higher_is_better=True)
    da_winners = mean_metric_winners("mean_da_body_pct", higher_is_better=True)
    mae_winner_label = model_list_text(mae_winners)
    corr_winner_label = model_list_text(corr_winners)
    da_winner_label = model_list_text(da_winners)

    mae_best = float(model_df["mean_mae_avg_pips"].min())
    corr_best = float(model_df["mean_corr2_avg_hlc"].max())
    da_best = float(model_df["mean_da_body_pct"].max())

    primary_model = mae_winners[0]
    mae_gap_parts = []
    for model in model_df["model"]:
        if model == primary_model:
            continue
        other_mae = summary_by_model[model]["mean_mae_avg_pips"]
        gap = other_mae - summary_by_model[primary_model]["mean_mae_avg_pips"]
        gap_pct = 100.0 * gap / max(other_mae, 1e-12)
        mae_gap_parts.append(f"terhadap <strong>{model}</strong>: <strong>{gap:.4f} pips</strong> ({gap_pct:.2f}%)")
    mae_gap_text = "; ".join(mae_gap_parts)

    mae_component_winners = []
    for label, key in [
        ("Open", "mean_mae_open_pips"),
        ("High", "mean_mae_high_pips"),
        ("Low", "mean_mae_low_pips"),
        ("Close", "mean_mae_close_pips"),
    ]:
        winner = min(summary_by_model, key=lambda model: summary_by_model[model][key])
        mae_component_winners.append(f"{label}: <strong>{winner}</strong>")
    corr_component_winners = []
    for label, key in [
        ("Open", "mean_corr2_open"),
        ("High", "mean_corr2_high"),
        ("Low", "mean_corr2_low"),
        ("Close", "mean_corr2_close"),
    ]:
        winner = max(summary_by_model, key=lambda model: summary_by_model[model][key])
        corr_component_winners.append(f"{label}: <strong>{winner}</strong>")

    if set(mae_winners) == set(corr_winners):
        primary_finding = (
            f"<li><strong>{mae_winner_label}</strong> merupakan model terbaik pada dua metrik utama evaluasi, "
            f"yaitu <strong>MAE AVG</strong> dan <strong>corr2 AVG HLC</strong>. Mean <strong>MAE AVG</strong> terbaik "
            f"adalah <strong>{mae_best:.4f} pips</strong>, sedangkan mean <strong>corr2 AVG HLC</strong> terbaik "
            f"adalah <strong>{corr_best:.4f}</strong>. Pemenang per fold untuk MAE: "
            f"<strong>{winner_count_text(winner_mae_counts)}</strong>; pemenang per fold untuk corr2: "
            f"<strong>{winner_count_text(winner_corr2_counts)}</strong>.</li>"
        )
        final_finding = (
            f"<li>Secara keseluruhan, hasil <strong>{html.escape(args.profile_label)}</strong> mendukung pemilihan "
            f"<strong>{mae_winner_label}</strong> sebagai model utama pada skenario ini, dengan model lain tetap "
            "dipertahankan sebagai pembanding untuk membaca trade-off arah, komponen harga, dan baseline klasik.</li>"
        )
    else:
        primary_finding = (
            f"<li>Hasil evaluasi menunjukkan trade-off antar metrik utama: <strong>MAE AVG</strong> terbaik dicapai oleh "
            f"<strong>{mae_winner_label}</strong> dengan <strong>{mae_best:.4f} pips</strong>, sedangkan "
            f"<strong>corr2 AVG HLC</strong> terbaik dicapai oleh <strong>{corr_winner_label}</strong> dengan "
            f"<strong>{corr_best:.4f}</strong>. Karena MAE mengukur deviasi numerik langsung dalam pips, metrik ini "
            "tetap menjadi dasar utama pemeringkatan akurasi.</li>"
        )
        final_finding = (
            f"<li>Secara keseluruhan, hasil <strong>{html.escape(args.profile_label)}</strong> perlu dibaca sebagai "
            "trade-off, bukan kemenangan tunggal mutlak. Pemilihan model utama mengikuti MAE, sedangkan corr2 dan "
            "Directional Accuracy dipakai untuk menjelaskan perilaku pendukung model.</li>"
        )

    main_findings_html = f"""
  <ul>
    {primary_finding}
    <li>Keunggulan <strong>{primary_model}</strong> pada <strong>MAE AVG</strong> bermakna secara praktis: {mae_gap_text}.</li>
    <li>Pemenang per komponen menunjukkan distribusi kekuatan yang lebih rinci. Untuk <strong>MAE</strong>: {', '.join(mae_component_winners)}. Untuk <strong>corr2</strong>: {', '.join(corr_component_winners)}.</li>
    <li><strong>Directional Accuracy</strong> tertinggi dicapai oleh <strong>{da_winner_label}</strong> dengan mean <strong>{da_best:.2f}%</strong>. Pemenang per fold untuk DA adalah <strong>{winner_count_text(winner_da_counts)}</strong>. Karena DA tidak selalu sejalan dengan MAE dan corr2, metrik ini lebih tepat diposisikan sebagai indikator pendukung arah body candle, bukan dasar utama penentuan model final.</li>
    <li>Perbandingan ini menunjukkan bahwa <strong>MAE</strong> paling tepat digunakan untuk menilai akurasi level harga, <strong>corr2</strong> untuk membaca kesesuaian pola, dan <strong>Directional Accuracy</strong> untuk membaca kecenderungan arah body candle. Ketiganya saling melengkapi, tetapi tidak memiliki bobot interpretasi yang sama.</li>
    {final_finding}
  </ul>
"""

    title_profile = re.sub(r"\s*\bTVT\s+v02\b\s*", " ", args.profile_label, flags=re.I).strip()
    title_profile = re.sub(r"\s{2,}", " ", title_profile)
    page_title = f"Comparison ARIMA vs FLF-LSTM vs FLF-BiLSTM - EUR/USD {title_profile} Last3 (TVT v02)"

    html_doc = f"""
<html>
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(page_title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; background: #ffffff; }}
    h1, h2, h3 {{ color: #183a6b; margin-bottom: 8px; }}
    p, li {{ line-height: 1.55; }}
    .meta {{
      background: #f6f8fa;
      padding: 12px 14px;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      margin-bottom: 14px;
    }}
    .note {{
      background: #fff7ed;
      border: 1px solid #fdba74;
      color: #7c2d12;
    }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; font-size: 13px; }}
    .data-table th, .data-table td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    .data-table th {{ background: #f6f8fa; }}
    .data-table td:first-child, .data-table th:first-child {{ text-align: left; }}
    .chart-wrap {{ margin-bottom: 22px; }}
    .artifact-list code {{ background: #f6f8fa; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>ARIMA vs FLF-LSTM vs FLF-BiLSTM</h1>
  <p>Perbandingan akhir <strong>EUR/USD {html.escape(args.profile_label)}</strong> dengan kebijakan <strong>non-overlap 6 fold tuning ({html.escape(args.tuning_folds_label)})</strong> dan <strong>3 fold evaluasi akhir ({html.escape(args.evaluation_folds_label)})</strong>.</p>

  <h2>Metode</h2>
  <div class="meta"><strong>ARIMA</strong>: {arima_meta}</div>
  <div class="meta"><strong>FLF-LSTM</strong>: {lstm_meta}</div>
  <div class="meta"><strong>FLF-BiLSTM</strong>: {bilstm_meta}</div>

  <h2>Ringkasan Hasil</h2>
  {format_table(model_df)}

  <h2>Temuan Utama</h2>
  {main_findings_html}

  <div class="meta note">{arima_invalid_note}</div>

  <h2>Interpretasi Metrik</h2>
  <div class="meta">
    <p><strong>MAE</strong> diposisikan sebagai metrik utama karena langsung mengukur besarnya deviasi numerik prediksi terhadap harga aktual dalam satuan <strong>pips</strong>.</p>
    <p><strong>corr2</strong> digunakan sebagai metrik pendukung untuk membaca tingkat kesesuaian pola antara prediksi dan data aktual, khususnya pada komponen <strong>High-Low-Close</strong>.</p>
    <p><strong>Directional Accuracy</strong> tetap dilaporkan karena memberikan informasi tambahan mengenai kecenderungan arah body candle, tetapi tidak cukup kuat untuk berdiri sendiri sebagai dasar pemilihan model akhir.</p>
  </div>

  <h2>MAE AVG vs ATR12</h2>
  <div class="meta">
    <p>Untuk membaca makna praktis error dalam konteks volatilitas pasar, mean <strong>MAE AVG</strong> dibandingkan dengan mean <strong>ATR12</strong> pada horizon evaluasi yang sama. Pembacaan ini membantu menilai apakah error model masih relatif kecil dibanding amplitudo pergerakan harga yang lazim.</p>
    <p>Pada hasil ini, <strong>{atr_best['model']}</strong> menunjukkan rasio terbaik dengan <strong>{atr_best['mae_avg_pct_atr12']:.2f}%</strong> dari ATR12, sedangkan <strong>{atr_worst['model']}</strong> menunjukkan rasio tertinggi sebesar <strong>{atr_worst['mae_avg_pct_atr12']:.2f}%</strong> dari ATR12.</p>
    <p>Selain itu, frekuensi <strong>Average Absolute Error per Candle</strong> juga dibaca terhadap <strong>ATR12</strong>. Kolom <strong>Error AVG &lt;= ATR12 (%)</strong> menunjukkan persentase candle yang error rata-ratanya masih berada di bawah atau sama dengan ATR12, sedangkan kolom <strong>Error AVG &gt; ATR12 (%)</strong> menunjukkan persentase candle yang melampaui ATR12.</p>
  </div>
  {format_table(atr_relation_df)}
  <div class="chart-wrap">{to_html_fragment(atr_ratio_fig)}</div>
  <div class="chart-wrap">{to_html_fragment(atr12_freq_fig)}</div>

  <h2>Keterbatasan Comparison</h2>
  <div class="meta">
    <p>Comparison ini didasarkan pada <strong>3 fold evaluasi akhir</strong> (<strong>19-21</strong>), sehingga hasilnya kuat sebagai evaluasi final pada skema <strong>TVT v02</strong>, tetapi tetap perlu dibaca dalam konteks horizon evaluasi yang relatif terbatas.</p>
    <p>Selain itu, pada <strong>ARIMA</strong> terdapat keterbatasan struktural karena prediksi <strong>OHLC</strong> dilakukan secara per-series independen. Akibatnya, kualitas estimasi level/range masih dapat dibaca, tetapi koherensi bentuk candle tidak selalu terjaga.</p>
  </div>

  <h2>Grafik Perbandingan</h2>
  <script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
  <div class="chart-wrap">{to_html_fragment(mae_component_fig)}</div>
  <div class="chart-wrap">{to_html_fragment(corr2_component_fig)}</div>
  <div class="chart-wrap">{to_html_fragment(mae_fold_fig)}</div>
  <div class="chart-wrap">{to_html_fragment(corr2_fold_fig)}</div>
  <div class="chart-wrap">{to_html_fragment(da_fold_fig)}</div>

  <h2>Perbandingan Antar Model per Fold</h2>
  {format_table(pairwise_df)}

  <h2>ARIMA Per Fold</h2>
  {format_table(arima_df)}

  <h2>FLF-LSTM Per Fold</h2>
  {format_table(lstm_df)}

  <h2>FLF-BiLSTM Per Fold</h2>
  {format_table(bilstm_df)}

  <h2>Audit Struktur Candle ARIMA</h2>
  {format_table(arima_audit_df) if arima_audit_df is not None else "<p>Tidak ada audit ARIMA.</p>"}

  <h2>Artefak</h2>
  <ul class="artifact-list">
    <li>HTML comparison: <code>{out_html}</code></li>
    <li>Summary CSV: <code>{model_csv}</code></li>
    <li>Pairwise CSV: <code>{pairwise_csv}</code></li>
    <li>ARIMA metrics CSV: <code>{arima_csv}</code></li>
    <li>LSTM metrics CSV: <code>{lstm_csv}</code></li>
    <li>BiLSTM metrics CSV: <code>{bilstm_csv}</code></li>
    <li>ARIMA candle audit CSV: <code>{audit_csv}</code></li>
    <li>ATR relation CSV: <code>{atr_relation_csv}</code></li>
    <li>Summary JSON: <code>{summary_json}</code></li>
  </ul>
</body>
</html>
"""

    out_html.write_text(html_doc, encoding="utf-8")
    print(f"Saved HTML report to {out_html}")
    print(f"Saved summary JSON to {summary_json}")


if __name__ == "__main__":
    main()
