#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


PROFILE_DEFAULTS = {
    "h4": {
        "data": str(ROOT / "EURUSD_H4_25Oct17.csv"),
        "sep": "\t",
        "datetime_format": "%Y.%m.%d %H:%M",
        "train_core_months": 71,
        "validation_months": 1,
        "test_months": 1,
        "out_dir": str(ROOT / "results" / "splits" / "tvt_v02" / "h4"),
    },
    "d1": {
        "data": str(ROOT / "EURUSD_D1_25Oct17.csv"),
        "sep": "\t",
        "datetime_format": "%Y.%m.%d %H:%M",
        "train_core_months": 69,
        "validation_months": 3,
        "test_months": 1,
        "out_dir": str(ROOT / "results" / "splits" / "tvt_v02" / "d1"),
    },
}

TUNING_FOLD_COUNT = 6
EVALUATION_FOLD_COUNT = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build chronological train_core/validation/test fold splits for EUR/USD experiments."
    )
    parser.add_argument("--profile", choices=["h4", "d1"], required=True, help="Predefined split profile.")
    parser.add_argument("--data", type=str, default=None, help="Raw EUR/USD CSV/TSV path.")
    parser.add_argument("--sep", type=str, default=None, help="Separator hint, e.g. tab, \\t, or comma.")
    parser.add_argument("--datetime-col", type=str, default=None, help="Optional datetime column name.")
    parser.add_argument(
        "--datetime-format",
        type=str,
        default=None,
        help="Datetime parsing format for pandas.to_datetime.",
    )
    parser.add_argument("--columns", type=str, default=None, help="Explicit OHLC columns, e.g. open,high,low,close")
    parser.add_argument("--train-core-months", type=int, default=None, help="Length of train_core block in months.")
    parser.add_argument("--validation-months", type=int, default=None, help="Length of validation block in months.")
    parser.add_argument("--test-months", type=int, default=None, help="Length of test block in months.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to results/splits/tvt_v02/<profile>.",
    )
    parser.add_argument("--last-n-folds", type=int, default=None, help="Optionally keep only last N folds.")
    return parser.parse_args()


def _parse_sep(raw: str) -> str:
    val = raw.strip().lower()
    if val in {"\\t", "tab", "t"}:
        return "\t"
    return raw


def load_data(path: str, sep: str) -> pd.DataFrame:
    candidates = []
    if sep:
        candidates.append(sep)
    candidates.extend([r"[\s\t]+", ",", "\t"])
    last_exc = None
    for cand in candidates:
        try:
            df = pd.read_csv(path, sep=cand, engine="python")
        except Exception as exc:
            last_exc = exc
            continue
        needs_retry = df.shape[1] == 1
        if not needs_retry:
            obj_cols = df.select_dtypes(include=["object", "string"])
            if not obj_cols.empty and obj_cols.astype(str).apply(lambda c: c.str.contains("\t", na=False)).any().any():
                needs_retry = True
        if not needs_retry:
            return df
    if last_exc:
        raise last_exc
    raise ValueError(f"Unable to parse file {path}")


def find_ohlc(df: pd.DataFrame, columns: str | None) -> pd.DataFrame:
    if columns:
        parts = [c.strip() for c in re.split(r"[,;]", columns) if c.strip()]
        if len(parts) != 4:
            raise ValueError("--columns must provide exactly 4 names for OHLC.")
        out = df[parts].astype("float32").copy()
        out.columns = ["open", "high", "low", "close"]
        return out.dropna().reset_index(drop=True)

    norm_cols = [str(c).strip().lower() for c in df.columns]
    df = df.copy()
    df.columns = norm_cols

    def pick_first(match):
        return match[0] if match else None

    open_col = pick_first([c for c in norm_cols if c in {"open", "o"} or "open" in c])
    high_col = pick_first([c for c in norm_cols if c in {"high", "h"} or "high" in c])
    low_col = pick_first([c for c in norm_cols if c in {"low", "l"} or "low" in c])
    close_col = pick_first([c for c in norm_cols if c in {"close", "c"} or "close" in c])

    if None in (open_col, high_col, low_col, close_col):
        numeric_candidates = [c for c in norm_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_candidates) >= 4:
            open_col, high_col, low_col, close_col = numeric_candidates[:4]

    if None in (open_col, high_col, low_col, close_col):
        raise ValueError(f"Unable to infer OHLC columns from headers: {df.columns.tolist()}")

    out = df[[open_col, high_col, low_col, close_col]].astype("float32").copy()
    out.columns = ["open", "high", "low", "close"]
    return out.dropna().reset_index(drop=True)


def prepare_dataframe(data_path: Path, sep: str, datetime_col: str | None, datetime_format: str, columns: str | None):
    df_raw = load_data(str(data_path), sep)
    dt_series = df_raw[datetime_col] if datetime_col and datetime_col in df_raw.columns else df_raw.iloc[:, 0]
    dt_series = pd.to_datetime(dt_series, format=datetime_format, errors="coerce")
    valid_mask = dt_series.notna()
    if not valid_mask.any():
        raise ValueError("No valid datetimes parsed. Check --datetime-col and --datetime-format.")
    df_raw = df_raw.loc[valid_mask].reset_index(drop=True)
    dt_series = dt_series.loc[valid_mask].reset_index(drop=True)
    ohlc_df = find_ohlc(df_raw, columns)
    if len(ohlc_df) != len(dt_series):
        raise ValueError("Sanitized OHLC rows do not match datetime rows.")
    ohlc_df.insert(0, "datetime", dt_series)
    return ohlc_df


def month_offset(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=months)


def actual_range(df: pd.DataFrame):
    if df.empty:
        return None, None
    return df["datetime"].min(), df["datetime"].max()


def build_folds(df: pd.DataFrame, train_core_months: int, validation_months: int, test_months: int):
    folds = []
    cursor = df["datetime"].iloc[0]
    last_timestamp = df["datetime"].iloc[-1]
    fold_idx = 1

    while True:
        train_core_start = cursor
        validation_start = month_offset(train_core_start, train_core_months)
        test_start = month_offset(validation_start, validation_months)
        test_end = month_offset(test_start, test_months)

        if test_start >= last_timestamp:
            break
        if test_end > last_timestamp:
            break

        train_core_mask = (df["datetime"] >= train_core_start) & (df["datetime"] < validation_start)
        validation_mask = (df["datetime"] >= validation_start) & (df["datetime"] < test_start)
        test_mask = (df["datetime"] >= test_start) & (df["datetime"] < test_end)

        train_core_df = df.loc[train_core_mask, ["datetime", "open", "high", "low", "close"]].reset_index(drop=True)
        validation_df = df.loc[validation_mask, ["datetime", "open", "high", "low", "close"]].reset_index(drop=True)
        test_df = df.loc[test_mask, ["datetime", "open", "high", "low", "close"]].reset_index(drop=True)

        if train_core_df.empty or validation_df.empty or test_df.empty:
            break

        combined = pd.concat(
            [
                train_core_df.assign(segment="train_core"),
                validation_df.assign(segment="validation"),
                test_df.assign(segment="test"),
            ],
            ignore_index=True,
        )

        tc_start_actual, tc_end_actual = actual_range(train_core_df)
        val_start_actual, val_end_actual = actual_range(validation_df)
        test_start_actual, test_end_actual = actual_range(test_df)

        folds.append(
            {
                "fold": fold_idx,
                "train_core_start": tc_start_actual,
                "train_core_end": tc_end_actual,
                "validation_start": val_start_actual,
                "validation_end": val_end_actual,
                "test_start": test_start_actual,
                "test_end": test_end_actual,
                "train_core_samples": len(train_core_df),
                "validation_samples": len(validation_df),
                "test_samples": len(test_df),
                "combined_samples": len(combined),
                "train_core_df": train_core_df,
                "validation_df": validation_df,
                "test_df": test_df,
                "combined_df": combined,
            }
        )

        cursor = month_offset(cursor, test_months)
        fold_idx += 1

    return folds


def write_fold_outputs(out_dir: Path, profile: str, folds: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for fold in folds:
        fold_dir = out_dir / f"fold{int(fold['fold']):02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold["train_core_df"].to_csv(fold_dir / "train_core.csv", index=False)
        fold["validation_df"].to_csv(fold_dir / "validation.csv", index=False)
        fold["test_df"].to_csv(fold_dir / "test.csv", index=False)
        fold["combined_df"].to_csv(fold_dir / "combined.csv", index=False)

        meta = {
            "profile": profile,
            "fold": int(fold["fold"]),
            "train_core_start": fold["train_core_start"].strftime("%Y-%m-%d %H:%M:%S"),
            "train_core_end": fold["train_core_end"].strftime("%Y-%m-%d %H:%M:%S"),
            "validation_start": fold["validation_start"].strftime("%Y-%m-%d %H:%M:%S"),
            "validation_end": fold["validation_end"].strftime("%Y-%m-%d %H:%M:%S"),
            "test_start": fold["test_start"].strftime("%Y-%m-%d %H:%M:%S"),
            "test_end": fold["test_end"].strftime("%Y-%m-%d %H:%M:%S"),
            "train_core_samples": int(fold["train_core_samples"]),
            "validation_samples": int(fold["validation_samples"]),
            "test_samples": int(fold["test_samples"]),
            "combined_samples": int(fold["combined_samples"]),
        }
        (fold_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summary_rows.append(meta)

    pd.DataFrame(summary_rows).to_csv(out_dir / "fold_schedule.csv", index=False)
    if summary_rows:
        if len(summary_rows) < (TUNING_FOLD_COUNT + EVALUATION_FOLD_COUNT):
            raise RuntimeError(
                f"Need at least {TUNING_FOLD_COUNT + EVALUATION_FOLD_COUNT} folds for TVT v02 policy, "
                f"but only found {len(summary_rows)}."
            )

        recent_window = summary_rows[-(TUNING_FOLD_COUNT + EVALUATION_FOLD_COUNT) :]
        tuning_rows = recent_window[:TUNING_FOLD_COUNT]
        evaluation_rows = recent_window[TUNING_FOLD_COUNT:]

        pd.DataFrame(recent_window).to_csv(out_dir / "fold_schedule_recent_window.csv", index=False)
        pd.DataFrame(tuning_rows).to_csv(out_dir / "fold_schedule_last6.csv", index=False)
        pd.DataFrame(evaluation_rows).to_csv(out_dir / "fold_schedule_last3.csv", index=False)
        pd.DataFrame(tuning_rows).to_csv(out_dir / "fold_schedule_tuning.csv", index=False)
        pd.DataFrame(evaluation_rows).to_csv(out_dir / "fold_schedule_evaluation.csv", index=False)

        scope = {
            "profile": profile,
            "total_folds": len(summary_rows),
            "tuning_policy": {
                "description": "Use the 6 complete folds immediately preceding the final evaluation window.",
                "folds": [int(row["fold"]) for row in tuning_rows],
            },
            "evaluation_policy": {
                "description": "Use the latest 3 complete folds for final comparative evaluation.",
                "folds": [int(row["fold"]) for row in evaluation_rows],
            },
            "overlap": False,
        }
        (out_dir / "fold_scope_tvt_v02.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    defaults = PROFILE_DEFAULTS[args.profile]

    data_path = Path(args.data or defaults["data"])
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    sep = _parse_sep(args.sep or defaults["sep"])
    datetime_format = args.datetime_format or defaults["datetime_format"]
    train_core_months = int(args.train_core_months or defaults["train_core_months"])
    validation_months = int(args.validation_months or defaults["validation_months"])
    test_months = int(args.test_months or defaults["test_months"])
    out_dir = Path(args.out_dir or defaults["out_dir"])

    df = prepare_dataframe(data_path, sep, args.datetime_col, datetime_format, args.columns)
    folds = build_folds(df, train_core_months, validation_months, test_months)
    if args.last_n_folds is not None and args.last_n_folds > 0:
        folds = folds[-args.last_n_folds :]
    if not folds:
        raise RuntimeError("No valid folds produced for the requested profile.")

    write_fold_outputs(out_dir, args.profile, folds)
    print(f"Wrote {len(folds)} folds to {out_dir.resolve()}")
    print(f"Schedule: {(out_dir / 'fold_schedule.csv').resolve()}")


if __name__ == "__main__":
    main()
