import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PIP_FACTOR = 10000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build MAE summary HTML for rolling walk-forward outputs."
    )
    parser.add_argument("--rolling-dir", required=True, help="Directory containing rolling_fixed_summary.csv and foldXX_preds.csv")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    parser.add_argument("--title", required=True, help="Report title.")
    parser.add_argument("--tail", type=int, default=30, help="Tail size for last-fold table.")
    parser.add_argument(
        "--atr-note",
        default="ATR tidak digunakan pada baseline ini; report difokuskan pada MAE OHLC.",
        help="Note shown below the title.",
    )
    return parser.parse_args()


def fold_file(base: Path, fold: int, suffix: str) -> Path:
    return base / f"fold{int(fold):02d}_{suffix}.csv"


def compute_mae_pips(preds_df: pd.DataFrame) -> dict:
    diffs = np.abs(
        preds_df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
        - preds_df[["true_open", "true_high", "true_low", "true_close"]].values
    ) * PIP_FACTOR
    avg = diffs.mean(axis=0)
    return {
        "mae_open_pips": float(avg[0]),
        "mae_high_pips": float(avg[1]),
        "mae_low_pips": float(avg[2]),
        "mae_close_pips": float(avg[3]),
        "mae_avg_pips": float(avg.mean()),
        "mae_avg_hlc_pips": float((avg[1] + avg[2] + avg[3]) / 3.0),
    }


def build_rows(rolling_dir: Path) -> pd.DataFrame:
    summary_path = rolling_dir / "rolling_fixed_summary.csv"
    folds_df = pd.read_csv(summary_path)
    rows = []
    for _, row in folds_df.iterrows():
        fold = int(row["fold"])
        preds_path = fold_file(rolling_dir, fold, "preds")
        preds_df = pd.read_csv(preds_path)
        metrics = compute_mae_pips(preds_df)
        rows.append(
            {
                "fold": fold,
                "train_start": row["train_start"],
                "train_end": row["train_end"],
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "samples": int(len(preds_df)),
                **metrics,
                "preds_path": str(preds_path),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def main():
    args = parse_args()
    rolling_dir = Path(args.rolling_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_rows(rolling_dir)
    best = df.loc[df["mae_avg_pips"].idxmin()]
    worst = df.loc[df["mae_avg_pips"].idxmax()]
    last = df.loc[df["fold"].idxmax()]
    last_preds = pd.read_csv(last["preds_path"])
    tail_df = last_preds.tail(min(args.tail, len(last_preds))).copy()
    tail_diffs = np.abs(
        tail_df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
        - tail_df[["true_open", "true_high", "true_low", "true_close"]].values
    ) * PIP_FACTOR
    tail_df["abs_err_avg_pips"] = tail_diffs.mean(axis=1)
    tail_table = tail_df.to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="data-table")
    summary_table = df[
        [
            "fold",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "samples",
            "mae_avg_pips",
            "mae_avg_hlc_pips",
            "mae_open_pips",
            "mae_high_pips",
            "mae_low_pips",
            "mae_close_pips",
        ]
    ].to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="data-table")

    html = f"""
<html>
<head>
  <meta charset="UTF-8" />
  <title>{args.title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
    .note {{ background: #f6f8fa; border: 1px solid #d0d7de; padding: 10px 12px; border-radius: 8px; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    .data-table th, .data-table td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    .data-table th {{ background: #f6f8fa; }}
    .data-table td:first-child, .data-table th:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>{args.title}</h1>
  <div class="note">{args.atr_note}</div>
  <p>
    Mean MAE avg: <strong>{df['mae_avg_pips'].mean():.4f} pips</strong> |
    Median: <strong>{df['mae_avg_pips'].median():.4f} pips</strong> |
    Best fold: <strong>{int(best['fold'])}</strong> ({best['mae_avg_pips']:.4f} pips) |
    Worst fold: <strong>{int(worst['fold'])}</strong> ({worst['mae_avg_pips']:.4f} pips) |
    Last fold: <strong>{int(last['fold'])}</strong> ({last['mae_avg_pips']:.4f} pips)
  </p>
  <h2>MAE per Fold</h2>
  {summary_table}
  <h2>Last Fold Tail {min(args.tail, len(last_preds))}</h2>
  {tail_table}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
