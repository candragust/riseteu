import re
from pathlib import Path
from typing import Optional

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_existing_path(candidate: Optional[str]) -> Optional[str]:
    if candidate in (None, ""):
        return None
    raw = Path(candidate)
    if raw.exists():
        return str(raw.resolve())
    if not raw.is_absolute():
        rooted = PROJECT_ROOT / raw
        if rooted.exists():
            return str(rooted.resolve())
    return None


def resolve_data_path(candidate: Optional[str]) -> str:
    resolved = _resolve_existing_path(candidate)
    if resolved:
        return resolved
    defaults = [
        PROJECT_ROOT / "EURUSD_H4_25Oct17.csv",
        PROJECT_ROOT / "EURUSD_D1_25Oct17.csv",
        PROJECT_ROOT / "risetBiLstmPy" / "EURUSD_H4_25Oct17.csv",
        PROJECT_ROOT / "CodeLstm" / "Data" / "EURUSD_H4.csv",
    ]
    for path in defaults:
        if path.exists():
            return str(path.resolve())
    raise FileNotFoundError("Could not locate EURUSD dataset. Use --data to specify a CSV path.")


def load_data(path: str, sep: str) -> pd.DataFrame:
    candidates = []
    if sep:
        candidates.append(sep)
    else:
        candidates.append(",")
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
            obj_cols = df.select_dtypes(include=["object"])
            if not obj_cols.empty:
                if obj_cols.astype(str).apply(lambda c: c.str.contains("\t", na=False)).any().any():
                    needs_retry = True
        if not needs_retry:
            return df

    if last_exc:
        raise last_exc
    raise ValueError(f"Unable to parse file {path}")


def find_ohlc(df: pd.DataFrame, columns: Optional[str]) -> pd.DataFrame:
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


def prepare_dataframe(data_path: Path, sep: str, datetime_col: str, datetime_format: str, columns: str):
    df_raw = load_data(str(data_path), sep)

    if datetime_col and datetime_col in df_raw.columns:
        dt_series = df_raw[datetime_col]
    else:
        dt_series = df_raw.iloc[:, 0]

    dt_series = pd.to_datetime(dt_series, format=datetime_format, errors="coerce")
    valid_mask = dt_series.notna()
    if not valid_mask.any():
        raise ValueError("No valid datetimes parsed. Check --datetime-col and --datetime-format.")

    df_raw = df_raw.loc[valid_mask].reset_index(drop=True)
    dt_series = dt_series.loc[valid_mask].reset_index(drop=True)

    ohlc_df = find_ohlc(df_raw, columns)
    if len(ohlc_df) != len(dt_series):
        raise ValueError("Sanitized OHLC rows do not match datetime rows; please inspect data quality.")

    ohlc_df.insert(0, "datetime", dt_series)
    return ohlc_df


def month_offset(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=months)


def build_folds(
    df: pd.DataFrame,
    train_months: int,
    test_months: int,
    min_test_samples: int,
    window: int,
    mode: str,
    shift_months: int,
):
    folds = []
    cursor = df["datetime"].iloc[0]
    last_timestamp = df["datetime"].iloc[-1]
    fold_idx = 1
    base_train_months = train_months
    expand_shift = shift_months if shift_months else test_months

    while True:
        if mode == "fixed":
            train_start = cursor
            train_end = month_offset(train_start, train_months)
        else:
            train_start = df["datetime"].iloc[0]
            train_end = month_offset(train_start, base_train_months + (fold_idx - 1) * expand_shift)

        test_start = train_end
        if test_start >= last_timestamp:
            break
        test_end = month_offset(test_start, test_months)

        train_mask = (df["datetime"] >= train_start) & (df["datetime"] < train_end)
        test_mask = (df["datetime"] >= test_start) & (df["datetime"] < test_end)

        train_len = int(train_mask.sum())
        test_len = int(test_mask.sum())

        if train_len <= window or test_len < max(window + 1, min_test_samples):
            break

        fold_df = pd.concat(
            [df.loc[train_mask, ["open", "high", "low", "close"]], df.loc[test_mask, ["open", "high", "low", "close"]]],
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

        if mode == "fixed":
            step = shift_months if shift_months else test_months
            cursor = month_offset(cursor, step)
        fold_idx += 1
        if test_start >= last_timestamp:
            break

    return folds
