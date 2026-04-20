#!/usr/bin/env python3
"""
Generate Walk-Forward Fixed folds for Train 72 months / Test 14 days / Step 14 days.
The script targets the EU5 multitask dataset and stores the schedule as CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "results" / "eurusd_H4_multitask_dataset.csv"
DEFAULT_OUT = BASE_DIR / "results" / "wf72_14d_folds.csv"


def month_offset(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=months)


def build_folds(df: pd.DataFrame, train_months: int, test_days: int, shift_days: int):
    folds = []
    cursor = df["datetime"].iloc[0]
    last_timestamp = df["datetime"].iloc[-1]
    fold_idx = 1
    shift_delta = pd.Timedelta(days=shift_days)

    while True:
        train_start = cursor
        train_end = month_offset(train_start, train_months)
        test_start = train_end
        if test_start >= last_timestamp:
            break
        test_end = test_start + pd.Timedelta(days=test_days)

        train_mask = (df["datetime"] >= train_start) & (df["datetime"] < train_end)
        test_mask = (df["datetime"] >= test_start) & (df["datetime"] < test_end)
        train_len = int(train_mask.sum())
        test_len = int(test_mask.sum())
        if train_len == 0 or test_len == 0:
            break

        folds.append(
            {
                "fold": fold_idx,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end_exclusive": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end_exclusive": test_end.strftime("%Y-%m-%d"),
                "train_samples": train_len,
                "test_samples": test_len,
            }
        )

        cursor = cursor + shift_delta
        fold_idx += 1
        if cursor >= last_timestamp:
            break

    return pd.DataFrame(folds)


def main():
    parser = argparse.ArgumentParser(description="Build Walk-Forward Fixed folds for EU5 (72m train, 14d test).")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="CSV path containing datetime column.",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--train-months", type=int, default=72)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--shift-days", type=int, default=14)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    folds_df = build_folds(df, args.train_months, args.test_days, args.shift_days)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(out_path, index=False)
    print(f"Saved {len(folds_df)} folds to {out_path.resolve()}")

    if len(folds_df) >= 3:
        print("Last 3 folds:")
        print(folds_df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
