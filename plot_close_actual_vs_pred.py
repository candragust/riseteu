import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot actual vs predicted Close price.")
    parser.add_argument("--preds", type=str, default="results/final_preds.csv", help="Preds CSV with pred_* and true_* cols.")
    parser.add_argument("--output", type=str, default="results/close_actual_pred.png", help="Output image path.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional starting index within preds (e.g., to skip early years).")
    args = parser.parse_args()

    df = pd.read_csv(args.preds)
    # Slice from start index if requested
    df = df.iloc[args.start_index :].reset_index(drop=True)

    time_steps = range(len(df))
    plt.figure(figsize=(9, 4.5))
    plt.plot(time_steps, df["true_close"], color="black", linewidth=0.9, label="Actual Close")
    plt.plot(time_steps, df["pred_close"], color="blue", linestyle="--", linewidth=0.8, label="BiLSTM Pred Close")
    plt.title("EUR/USD Close price (validation segment)", fontsize=12)
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
