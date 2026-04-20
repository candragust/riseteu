import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot actual vs predicted single OHLC price.")
    parser.add_argument("--preds", type=str, default="results/final_preds.csv", help="Preds CSV with pred_* and true_* cols.")
    parser.add_argument("--col", type=str, required=True, choices=["open", "high", "low", "close"], help="Which price to plot.")
    parser.add_argument("--output", type=str, required=True, help="Output image path (png).")
    parser.add_argument("--start-index", type=int, default=0, help="Optional starting index within preds.")
    args = parser.parse_args()

    df = pd.read_csv(args.preds).iloc[args.start_index :].reset_index(drop=True)
    ts = range(len(df))

    plt.figure(figsize=(9, 3.5))
    plt.plot(ts, df[f"true_{args.col}"], color="black", linewidth=0.8, label=f"Actual {args.col.capitalize()}")
    plt.plot(ts, df[f"pred_{args.col}"], color="tab:blue", linestyle="--", linewidth=0.7, label=f"Pred {args.col.capitalize()}")
    plt.title(f"EUR/USD {args.col.capitalize()} price (validation segment)")
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {args.col} plot to {out_path}")


if __name__ == "__main__":
    main()
