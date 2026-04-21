import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eurusd_ohlc_utils import build_folds, prepare_dataframe


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


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


def _resolve_input_path(candidate: str, base_dir: Path) -> Path:
    raw = Path(candidate)
    if raw.exists():
        return raw.resolve()
    if not raw.is_absolute():
        rooted = (base_dir / raw).resolve()
        if rooted.exists():
            return rooted
    raise FileNotFoundError(f"Path not found: {candidate}")


def _resolve_output_path(candidate: str) -> Path:
    raw = Path(candidate)
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ARIMA OHLC baseline with rolling walk-forward evaluation."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "EURUSD_H4_25Oct17.csv"),
        help="Path to raw EURUSD data.",
    )
    parser.add_argument("--sep", type=str, default="\t", help="Separator hint for raw data.")
    parser.add_argument("--datetime-col", type=str, default=None, help="Optional datetime column name.")
    parser.add_argument(
        "--datetime-format",
        type=str,
        default="%Y.%m.%d %H:%M",
        help="Datetime parsing format for pandas.to_datetime.",
    )
    parser.add_argument("--columns", type=str, default=None, help="Explicit OHLC columns order.")
    parser.add_argument("--train-months", type=int, default=72, help="Length of training window in months.")
    parser.add_argument("--test-months", type=int, default=1, help="Length of testing window in months.")
    parser.add_argument(
        "--shift-months",
        type=int,
        default=None,
        help="Advance train window by this many months (defaults to test-months).",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(SCRIPT_DIR / "arima_baseline_config.json"),
        help="Config JSON used as ARIMA baseline.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "Arima" / "result" / "arima_rolling_train72_test1"),
        help="Output directory for fold CSV/preds/summary.",
    )
    parser.add_argument(
        "--min-test-samples",
        type=int,
        default=100,
        help="Minimum samples required in each test segment.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print fold schedule without training.")
    parser.add_argument(
        "--mode",
        choices=["fixed", "expanding"],
        default="fixed",
        help="Rolling mode: fixed (train window shifts) or expanding (train grows over time).",
    )
    parser.add_argument(
        "--last-n-folds",
        type=int,
        default=None,
        help="If set, keep only the last N generated folds.",
    )
    parser.add_argument(
        "--max-test-steps",
        type=int,
        default=None,
        help="Optional limit forwarded to arima_ohlc_experiment.py, useful for smoke tests.",
    )
    return parser.parse_args()


def run_fold(fold: FoldInfo, fold_df: pd.DataFrame, base_cfg: Path, out_dir: Path, max_test_steps=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_data = out_dir / f"fold{fold.index:02d}_data.csv"
    preds_path = out_dir / f"fold{fold.index:02d}_preds.csv"
    summary_path = out_dir / f"fold{fold.index:02d}_summary.json"

    fold_df.to_csv(fold_data, index=False)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "arima_ohlc_experiment.py"),
        "--config",
        str(base_cfg),
        "--data",
        str(fold_data),
        "--split",
        f"{fold.split_ratio:.10f}",
        "--out",
        str(preds_path),
        "--summary-out",
        str(summary_path),
    ]
    if max_test_steps is not None:
        cmd.extend(["--max-test-steps", str(max_test_steps)])

    print(
        f"[ARIMA FOLD {fold.index:02d}] Train {fold.train_start.date()} -> {fold.train_end.date()} | "
        f"Test {fold.test_start.date()} -> {fold.test_end.date()} | "
        f"samples train={fold.train_len} test={fold.test_len}"
    )
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    data_path = _resolve_input_path(args.data, PROJECT_ROOT)
    base_cfg = _resolve_input_path(args.base_config, SCRIPT_DIR)

    _ = json.loads(base_cfg.read_text(encoding="utf-8"))
    min_history = 1

    df = prepare_dataframe(data_path, args.sep, args.datetime_col, args.datetime_format, args.columns)
    folds_raw = build_folds(
        df,
        args.train_months,
        args.test_months,
        args.min_test_samples,
        min_history,
        args.mode,
        args.shift_months,
    )
    if args.last_n_folds:
        folds_raw = folds_raw[-int(args.last_n_folds) :]
    if not folds_raw:
        raise RuntimeError("No valid folds produced. Try reducing min-test-samples or adjusting windows.")

    out_dir = _resolve_output_path(args.out_dir)
    summary_rows = []
    for fold_tuple in folds_raw:
        idx, train_start, train_end, test_start, test_end, train_len, test_len, fold_df, ratio = fold_tuple
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
                "mode": args.mode,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "rolling_fixed_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved fold summary to {summary_path.resolve()}")

    if args.dry_run:
        return

    for fold_tuple in folds_raw:
        idx, train_start, train_end, test_start, test_end, train_len, test_len, fold_df, ratio = fold_tuple
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
        run_fold(fold, fold_df, base_cfg, out_dir, max_test_steps=args.max_test_steps)


if __name__ == "__main__":
    main()
