import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PIP = 0.0001
PIP_FACTOR = 1 / PIP


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


def build_fold_metrics(rolling_dir: Path):
    summary_path = rolling_dir / "rolling_fixed_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    fold_df = pd.read_csv(summary_path)
    rows = []
    missing = []
    for _, row in fold_df.iterrows():
        fold = int(row["fold"])
        preds_path = fold_file(rolling_dir, fold, "preds")
        hist_path = fold_file(rolling_dir, fold, "history")
        if not preds_path.exists():
            missing.append(str(preds_path))
            continue
        preds_df = pd.read_csv(preds_path)
        metrics = compute_mae_pips(preds_df)
        rows.append(
            {
                "fold": fold,
                "train_start": row["train_start"],
                "train_end": row["train_end"],
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "train_samples": int(row["train_samples"]),
                "test_samples": int(row["test_samples"]),
                "split_ratio": float(row["split_ratio"]),
                "preds": str(preds_path),
                "history": str(hist_path),
                **metrics,
            }
        )
    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing))
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def build_ohlc_all(rolling_dir: Path, folds: pd.DataFrame, out_path: Path, title: str):
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
        rows=2,
        cols=2,
        subplot_titles=[f"{c.upper()} Actual vs Pred" for c in cols],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for i, col in enumerate(cols):
        r = i // 2 + 1
        c = i % 2 + 1
        fig.add_trace(
            go.Scatter(
                x=all_df["global_idx"],
                y=all_df[f"true_{col}"],
                mode="markers",
                name=f"Actual {col.upper()}",
                marker=dict(size=3, color="#d62728"),
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
                marker=dict(size=3, color="#1f77b4"),
                showlegend=(i == 0),
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="Index (all test candles)", row=r, col=c)
        fig.update_yaxes(title_text="Price", tickformat=".5f", row=r, col=c)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=980,
        legend=dict(orientation="h", y=1.05, x=0),
    )
    out_path.write_text(fig.to_html(include_plotlyjs="cdn"), encoding="utf-8")


def build_loss_report(rolling_dir: Path, folds: pd.DataFrame, out_path: Path, title: str):
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
    overall_mean = float(folds["mae_avg_pips"].mean())
    overall_median = float(folds["mae_avg_pips"].median())
    best = folds.loc[folds["mae_avg_pips"].idxmin()]
    worst = folds.loc[folds["mae_avg_pips"].idxmax()]

    last_fold = int(folds["fold"].max())
    tail_note = ""
    tail_mae = np.nan
    try:
        last_preds = pd.read_csv(Path(folds.loc[folds["fold"] == last_fold, "preds"].iloc[0]))
        tail_df = last_preds.tail(tail)
        tail_mae = compute_mae_pips(tail_df)["mae_avg_pips"]
        tail_note = f"<p>Tail {tail} MAE avg (Fold terakhir = {last_fold}): <strong>{tail_mae:.2f} pips</strong></p>"
    except Exception:
        tail_note = f"<p>Tail {tail} MAE avg (Fold terakhir) tidak tersedia.</p>"

    cols = [
        "fold",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "train_samples",
        "test_samples",
        "split_ratio",
        "mae_avg_hlc_pips",
        "mae_avg_pips",
        "mae_open_pips",
        "mae_high_pips",
        "mae_low_pips",
        "mae_close_pips",
    ]
    table_html = folds[cols].to_html(index=False, float_format="%.4f")

    body = [
        f"<h1>{title}</h1>",
        (
            "<p>"
            f"Rata-rata MAE avg: <strong>{overall_mean:.2f} pips</strong> | "
            f"Median: <strong>{overall_median:.2f} pips</strong> | "
            f"Fold terbaik: <strong>{int(best['fold'])}</strong> ({best['mae_avg_pips']:.2f} pips) | "
            f"Fold terburuk: <strong>{int(worst['fold'])}</strong> ({worst['mae_avg_pips']:.2f} pips)"
            "</p>"
        ),
        tail_note,
        "<h2>Ringkasan MAE per Fold</h2>",
        table_html,
    ]

    html = (
        "<html><head><meta charset='UTF-8'>"
        "<style>body{font-family:Arial, sans-serif; margin:20px;} "
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

    out_validation.parent.mkdir(parents=True, exist_ok=True)
    out_ohlc.parent.mkdir(parents=True, exist_ok=True)
    out_loss.parent.mkdir(parents=True, exist_ok=True)
    out_gradient.parent.mkdir(parents=True, exist_ok=True)

    folds = build_fold_metrics(rolling_dir)
    write_validation_html(
        folds,
        out_validation,
        f"{args.title_prefix} - Validation Report WF72m/1m",
        args.tail,
    )
    build_ohlc_all(
        rolling_dir,
        folds,
        out_ohlc,
        f"{args.title_prefix} - OHLC Dot Plot (All Fold Tests Combined)",
    )
    build_loss_report(
        rolling_dir,
        folds,
        out_loss,
        f"{args.title_prefix} - Loss Curve per Fold (WF72m/1m)",
    )
    build_gradient_report(
        rolling_dir,
        folds,
        out_gradient,
        f"{args.title_prefix} - Gradient Proxy per Fold (WF72m/1m)",
    )
    print(f"Wrote: {out_validation.resolve()}")
    print(f"Wrote: {out_ohlc.resolve()}")
    print(f"Wrote: {out_loss.resolve()}")
    print(f"Wrote: {out_gradient.resolve()}")


if __name__ == "__main__":
    main()
