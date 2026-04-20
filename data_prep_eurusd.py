import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def detect_and_load(path: Path, sep: Optional[str]) -> pd.DataFrame:
    """
    Try to load CSV/TSV with a user-provided separator first.
    If the result is a single column or still contains tab characters,
    retry with whitespace/tab regex.
    """
    if sep and sep.lower() in {"tab", "\\t", "t"}:
        sep = "\t"
    if sep:
        df = pd.read_csv(path, sep=sep, engine="python")
    else:
        # First attempt: comma
        df = pd.read_csv(path, sep=",", engine="python")

    needs_retry = False
    if df.shape[1] == 1:
        needs_retry = True
    else:
        # Check if object columns still contain tabs
        obj_cols = df.select_dtypes(include=["object"])
        if not obj_cols.empty:
            if obj_cols.astype(str).apply(lambda c: c.str.contains("\t", na=False)).any().any():
                needs_retry = True

    if needs_retry:
        df = pd.read_csv(path, sep=r"[\s\t]+", engine="python")

    return df


def sanitize_ohlc(df: pd.DataFrame, columns: Optional[str]) -> pd.DataFrame:
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip()]
        if len(cols) != 4:
            raise ValueError("Expected 4 column names for --columns (open,high,low,close).")
        out = df[cols].copy()
        out.columns = ["open", "high", "low", "close"]
    else:
        # Auto-detect common names
        norm = [str(c).strip().lower() for c in df.columns]
        df = df.copy()
        df.columns = norm
        def pick(cands):
            return next((c for c in cands if c in norm), None)
        o = pick(["open", "o"])
        h = pick(["high", "h"])
        l = pick(["low", "l"])
        c = pick(["close", "c"])
        if None in (o, h, l, c):
            raise ValueError(f"Could not detect OHLC columns. Headers: {df.columns.tolist()}")
        out = df[[o, h, l, c]].copy()
        out.columns = ["open", "high", "low", "close"]
    # Convert to float
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def add_atr(ohlc: pd.DataFrame, periods=(6, 12)) -> pd.DataFrame:
    """Tambah kolom ATR untuk setiap periode."""
    df = ohlc.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    for p in periods:
        df[f"atr{p}"] = tr.rolling(window=p, min_periods=p).mean()

    df = df.dropna().reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare EURUSD OHLC data: load, sanity check, and save cleaned CSV."
    )
    parser.add_argument("--data", type=str, default="EURUSD_H4_25Oct17.csv", help="Input file path.")
    parser.add_argument("--sep", type=str, default=None, help="Separator hint: ',', 'tab', '\\t'.")
    parser.add_argument(
        "--columns", type=str, default=None, help="Explicit OHLC column names, e.g. 'Open,High,Low,Close'."
    )
    parser.add_argument(
        "--out", type=str, default="results/EURUSD_H4_clean.csv", help="Output cleaned CSV path."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional filter start date (e.g., 2019-01-01) to drop earlier samples.",
    )
    args = parser.parse_args()

    input_path = Path(args.data)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading data from {input_path} (sep hint={args.sep or 'auto'})")
    df_raw = detect_and_load(input_path, args.sep)
    print(f"Loaded shape: {df_raw.shape}")
    print(f"Headers: {df_raw.columns.tolist()}")

    # Optional date filtering before sanitize
    if args.start_date:
        try:
            dt = pd.to_datetime(df_raw.iloc[:, 0], format="%Y.%m.%d %H:%M", errors="coerce")
            mask = dt >= pd.to_datetime(args.start_date)
            before = len(df_raw)
            df_raw = df_raw.loc[mask].reset_index(drop=True)
            print(f"Applied date filter from {args.start_date}: {before} -> {len(df_raw)} rows (kept {mask.sum()})")
        except Exception as exc:
            print(f"Warning: failed to apply date filter; proceeding without filter. ({exc})")

    ohlc = sanitize_ohlc(df_raw, args.columns)
    ohlc = add_atr(ohlc, periods=(6, 12))
    print(f"After sanitize OHLC shape: {ohlc.shape}")
    print("Dtypes:", ohlc.dtypes.to_dict())
    print("Preview:")
    print(ohlc.head())

    desc = ohlc.describe()
    print("Stats (describe):")
    print(desc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ohlc.to_csv(out_path, index=False)
    print(f"Saved cleaned OHLC to {out_path.resolve()}")


if __name__ == "__main__":
    main()
