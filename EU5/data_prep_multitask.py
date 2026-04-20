#!/usr/bin/env python3
"""
Data preparation helper for the EU5 multi-task BiLSTM experiment.

Steps:
1. Load raw EURUSD OHLC data (CSV/TSV).
2. Sanitize numeric OHLC columns and ensure ascending datetime order.
3. Compute ATR features (default ATR6, ATR12, ATR14).
4. Define theta = theta_ratio * ATR14 and derive directional labels
   (bullish, bearish, ranging) per candle.
5. Save enriched dataset to CSV for downstream windowing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _parse_sep(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in {"\\t", "tab", "t"}:
        return "\t"
    return raw


def load_raw(path: Path, sep: Optional[str]) -> pd.DataFrame:
    """
    Load raw CSV with a best-effort separator detection.
    """
    if sep:
        df = pd.read_csv(path, sep=sep, engine="python")
    else:
        try:
            df = pd.read_csv(path, sep=",", engine="python")
        except Exception:
            df = pd.read_csv(path, sep=r"[\s\t]+", engine="python")

    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=r"[\s\t]+", engine="python")
    return df


def sanitize_datetime(df: pd.DataFrame, dt_col: Optional[str]) -> pd.Series:
    """
    Return datetime series. If dt_col is None, try to detect automatically.
    """
    if dt_col:
        return pd.to_datetime(df[dt_col])

    for candidate in df.columns:
        if "date" in candidate.lower() or "time" in candidate.lower():
            try:
                return pd.to_datetime(df[candidate])
            except Exception:
                continue
    # fallback to first column
    return pd.to_datetime(df.iloc[:, 0])


def sanitize_ohlc(df: pd.DataFrame, columns: Optional[str]) -> pd.DataFrame:
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip()]
        if len(cols) != 4:
            raise ValueError("Expected 4 column names for --columns")
        out = df[cols].copy()
    else:
        norm = [str(c).strip().lower() for c in df.columns]
        name_map = dict(zip(norm, df.columns))
        def pick(names: Iterable[str]) -> str:
            for n in names:
                if n in name_map:
                    return name_map[n]
            raise KeyError(f"Missing column in {df.columns}")

        out = df[
            [
                pick(["open", "o"]),
                pick(["high", "h"]),
                pick(["low", "l"]),
                pick(["close", "c"]),
            ]
        ].copy()

    out.columns = ["open", "high", "low", "close"]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def compute_atr(df: pd.DataFrame, periods: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    high = out["high"]
    low = out["low"]
    close = out["close"]
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
        out[f"atr{p}"] = tr.rolling(window=p, min_periods=p).mean()

    return out


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50)
    return rsi


def label_direction(df: pd.DataFrame, atr_col: str, theta_ratio: float) -> pd.DataFrame:
    out = df.copy()
    if atr_col not in out.columns:
        raise KeyError(f"{atr_col} not found. Available: {list(out.columns)}")

    out["delta_close_open"] = out["close"] - out["open"]
    out["theta"] = out[atr_col] * theta_ratio

    cond_bull = out["delta_close_open"] > out["theta"]
    cond_bear = out["delta_close_open"] < -out["theta"]

    out["dir_class"] = 0  # 0 = ranging
    out.loc[cond_bull, "dir_class"] = 1
    out.loc[cond_bear, "dir_class"] = -1

    dir_name_map = {1: "bullish", -1: "bearish", 0: "ranging"}
    out["dir_name"] = out["dir_class"].map(dir_name_map)

    # Binary label for bullish vs bearish (NaN for ranging to allow masking later).
    out["dir_binary"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out.loc[cond_bull, "dir_binary"] = 1
    out.loc[cond_bear, "dir_binary"] = 0

    # Convenience mask (1 if sample is bullish or bearish).
    out["dir_mask"] = (~out["dir_binary"].isna()).astype(int)
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare EURUSD dataset for multi-task BiLSTM.")
    parser.add_argument("--data", type=str, default="EURUSD_H4_25Oct17.csv", help="Input raw file.")
    parser.add_argument("--sep", type=str, default="tab", help="Separator hint: ',', 'tab', '\\t'.")
    parser.add_argument("--datetime-col", type=str, default=None, help="Explicit datetime column name.")
    parser.add_argument("--columns", type=str, default=None, help="OHLC column names, e.g. 'Open,High,Low,Close'.")
    parser.add_argument("--atr-periods", type=str, default="14", help="Comma list of ATR periods to compute.")
    parser.add_argument("--theta-ratio", type=float, default=0.1, help="Theta ratio applied to ATR14 (0.1 -> 10%).")
    parser.add_argument(
        "--out",
        type=str,
        default="results/EU5/eurusd_H4_multitask_dataset.csv",
        help="Output CSV with enriched features.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")

    sep = _parse_sep(args.sep)
    print(f"[1/5] Loading raw OHLC from {data_path} (sep hint={sep or 'auto'})")
    df_raw = load_raw(data_path, sep)
    print(f"         raw shape={df_raw.shape}, columns={list(df_raw.columns)}")

    print("[2/5] Parsing datetime and sorting ascending")
    dt = sanitize_datetime(df_raw, args.datetime_col)
    df_raw = df_raw.assign(datetime=dt).dropna(subset=["datetime"])
    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)

    print("[3/5] Sanitizing OHLC columns")
    ohlc = sanitize_ohlc(df_raw, args.columns)
    ohlc = pd.concat([df_raw[["datetime"]], ohlc], axis=1)

    periods = [int(p.strip()) for p in args.atr_periods.split(",") if p.strip()]
    if 14 not in periods:
        periods.append(14)
    print(f"[4/5] Computing ATR for periods={periods}")
    feat = compute_atr(ohlc, periods=periods)
    feat["rsi14"] = compute_rsi(feat["close"], period=14)
    feat = feat.dropna().reset_index(drop=True)

    print(f"[5/5] Deriving direction labels with theta_ratio={args.theta_ratio}")
    feat = label_direction(feat, atr_col="atr14", theta_ratio=args.theta_ratio)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_path, index=False)

    print(f"Saved enriched dataset to {out_path.resolve()}")
    print("Preview:")
    print(feat.head())
    print("Label distribution (dir_name):")
    print(feat["dir_name"].value_counts())


if __name__ == "__main__":
    main()
