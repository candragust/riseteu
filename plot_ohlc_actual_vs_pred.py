import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot actual vs predicted OHLC prices.")
    parser.add_argument("--preds", type=str, default="results/final_preds.csv", help="Preds CSV with pred_* and true_* cols.")
    parser.add_argument("--output", type=str, default="results/ohlc_actual_pred.png", help="Output image path.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional starting index within preds.")
    args = parser.parse_args()

    df = pd.read_csv(args.preds).iloc[args.start_index :].reset_index(drop=True)
    time_steps = range(len(df))

    plt.figure(figsize=(10, 8))
    pairs = [
        ("open", "Actual Open", "Pred Open", "tab:blue"),
        ("high", "Actual High", "Pred High", "tab:orange"),
        ("low", "Actual Low", "Pred Low", "tab:green"),
        ("close", "Actual Close", "Pred Close", "tab:red"),
    ]
    for idx, (col, lab_true, lab_pred, color) in enumerate(pairs, start=1):
        ax = plt.subplot(2, 2, idx)
        ax.plot(time_steps, df[f"true_{col}"], color="black", linewidth=0.8, label=lab_true)
        ax.plot(time_steps, df[f"pred_{col}"], color=color, linestyle="--", linewidth=0.7, label=lab_pred)
        ax.set_title(col.capitalize())
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
