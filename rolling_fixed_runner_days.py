import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

from rolling_fixed_runner import prepare_dataframe, month_offset


@dataclass
class FoldInfo:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_len: int
    test_len: int
    data_csv: Path
    split_ratio: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run rolling evaluation with train window in months and test window in days."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="EURUSD_H4_25Oct17.csv",
        help="Path to raw EURUSD data with datetime + OHLC columns.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="\t",
        help="Separator hint for raw data (default tab).",
    )
    parser.add_argument(
        "--datetime-col",
        type=str,
        default=None,
        help="Optional datetime column name (defaults to first column).",
    )
    parser.add_argument(
        "--datetime-format",
        type=str,
        default="%Y.%m.%d %H:%M",
        help="Datetime parsing format for pandas.to_datetime.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Explicit OHLC columns order, e.g. 'open,high,low,close'.",
    )
    parser.add_argument("--train-months", type=int, default=72, help="Length of training window in months.")
    parser.add_argument("--test-days", type=int, default=14, help="Length of testing window in days.")
    parser.add_argument(
        "--shift-days",
        type=int,
        default=None,
        help="Advance start cursor by this many days after each fold (defaults to test-days).",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="final_config.json",
        help="Config JSON used as hyperparameter baseline.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/rolling_fixed_days",
        help="Output directory for fold CSV/preds/history.",
    )
    parser.add_argument(
        "--min-test-samples",
        type=int,
        default=80,
        help="Minimum samples required in each test segment (otherwise fold skipped).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print fold schedule without training.",
    )
    return parser.parse_args()


def build_folds(
    df: pd.DataFrame,
    train_months: int,
    test_days: int,
    min_test_samples: int,
    window: int,
    shift_days: int,
):
    folds = []
    cursor = df["datetime"].iloc[0]
    last_timestamp = df["datetime"].iloc[-1]
    fold_idx = 1
    shift_delta = pd.Timedelta(days=shift_days if shift_days else test_days)

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

        if train_len <= window or test_len < max(window + 1, min_test_samples):
            break

        fold_df = pd.concat(
            [
                df.loc[train_mask, ["open", "high", "low", "close"]],
                df.loc[test_mask, ["open", "high", "low", "close"]],
            ],
            ignore_index=True,
        )
        total = len(fold_df)
        split_ratio = train_len / total

        folds.append(
            (
                fold_idx,
                train_start,
                train_end,
                test_start,
                min(test_end, last_timestamp),
                train_len,
                test_len,
                fold_df,
                split_ratio,
            )
        )

        cursor = cursor + shift_delta
        fold_idx += 1

    return folds


def run_fold(fold: FoldInfo, fold_df: pd.DataFrame, base_cfg: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_data = out_dir / f"fold{fold.index:02d}_data.csv"
    preds_path = out_dir / f"fold{fold.index:02d}_preds.csv"
    hist_path = out_dir / f"fold{fold.index:02d}_history.csv"

    fold_df.to_csv(fold_data, index=False)

    cmd = [
        sys.executable,
        "bilstm_flf_experiment.py",
        "--config",
        str(base_cfg),
        "--data",
        str(fold_data),
        "--split",
        f"{fold.split_ratio:.10f}",
        "--out",
        str(preds_path),
        "--history-out",
        str(hist_path),
    ]
    print(
        f"[FOLD {fold.index:02d}] Train {fold.train_start.date()} -> {fold.train_end.date()} | "
        f"Test {fold.test_start.date()} -> {fold.test_end.date()} | "
        f"samples train={fold.train_len} test={fold.test_len}"
    )
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    base_cfg = Path(args.base_config)
    if not base_cfg.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg}")

    cfg = json.loads(base_cfg.read_text(encoding="utf-8"))
    window = int(cfg.get("window", 16))

    df = prepare_dataframe(data_path, args.sep, args.datetime_col, args.datetime_format, args.columns)
    folds_raw = build_folds(
        df,
        args.train_months,
        args.test_days,
        args.min_test_samples,
        window,
        args.shift_days,
    )

    if not folds_raw:
        raise RuntimeError("No valid folds produced. Adjust parameters or lower min-test-samples.")

    out_dir = Path(args.out_dir)
    summary_rows = []
    for fold_tuple in folds_raw:
        (
            idx,
            train_start,
            train_end,
            test_start,
            test_end,
            train_len,
            test_len,
            fold_df,
            ratio,
        ) = fold_tuple
        summary_rows.append(
            {
                "fold": idx,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": (train_end - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": (test_end - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d"),
                "train_samples": train_len,
                "test_samples": test_len,
                "split_ratio": ratio,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "rolling_fixed_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved fold summary to {summary_path.resolve()}")

    if args.dry_run:
        return

    for fold_tuple in folds_raw:
        (
            idx,
            train_start,
            train_end,
            test_start,
            test_end,
            train_len,
            test_len,
            fold_df,
            ratio,
        ) = fold_tuple
        fold = FoldInfo(
            index=idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_len=train_len,
            test_len=test_len,
            data_csv=out_dir / f"fold{idx:02d}_data.csv",
            split_ratio=ratio,
        )
        fold.data_csv.parent.mkdir(parents=True, exist_ok=True)
        run_fold(fold, fold_df, base_cfg, out_dir)


if __name__ == "__main__":
    main()
