import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def compute_mae(preds_df: pd.DataFrame) -> pd.Series:
    cols_pred = ["pred_open", "pred_high", "pred_low", "pred_close"]
    cols_true = ["true_open", "true_high", "true_low", "true_close"]
    diffs = preds_df[cols_pred].values - preds_df[cols_true].values
    mae = np.mean(np.abs(diffs), axis=0)
    return pd.Series(mae, index=cols_pred)


def load_history(history_path: Path) -> Optional[pd.DataFrame]:
    if history_path.exists():
        return pd.read_csv(history_path)
    return None


def build_html(preds_df: pd.DataFrame, mae: pd.Series, history: Optional[pd.DataFrame]) -> str:
    sections = []

    sections.append("<h1>BiLSTM + FLF Experiment Report</h1>")
    sections.append("<h2>MAE per component</h2>")
    mae_df = mae.reset_index()
    mae_df.columns = ["component", "mae"]
    sections.append(mae_df.to_html(index=False, float_format="%.6f"))

    sections.append("<h2>Sample predictions (head)</h2>")
    sections.append(preds_df.head().to_html(index=False, float_format="%.6f"))

    if history is not None:
        sections.append("<h2>Training history</h2>")
        sections.append(history.to_html(index=False, float_format="%.6f"))
    else:
        sections.append("<p><em>No history file provided.</em></p>")

    body = "\n".join(sections)
    html = f"""
    <html>
      <head>
        <meta charset="UTF-8">
        <title>BiLSTM FLF Experiment Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 20px; }}
          table {{ border-collapse: collapse; margin-bottom: 16px; }}
          th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
          th {{ background: #f0f0f0; }}
          h1, h2 {{ margin-bottom: 8px; }}
        </style>
      </head>
      <body>
        {body}
      </body>
    </html>
    """
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML report for a BiLSTM+FLF experiment (hyperparameter runs)."
    )
    parser.add_argument("--preds", type=str, required=True, help="Path to predictions CSV (with pred_* and true_*).")
    parser.add_argument("--history", type=str, default=None, help="Optional training history CSV.")
    parser.add_argument("--out", type=str, default="results/hp_report.html", help="Output HTML path.")
    args = parser.parse_args()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    preds_df = pd.read_csv(preds_path)
    mae = compute_mae(preds_df)

    history_df = load_history(Path(args.history)) if args.history else None

    html = build_html(preds_df, mae, history_df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path.resolve()}")


if __name__ == "__main__":
    main()

