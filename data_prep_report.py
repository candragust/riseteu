import argparse
from pathlib import Path

import pandas as pd


def load_raw(path: Path, sep: str | None):
    # Try provided sep first, fallback to tab/whitespace.
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


def ensure_datetime(df: pd.DataFrame):
    for col in df.columns:
        if "date" in str(col).lower() or "time" in str(col).lower():
            try:
                return pd.to_datetime(df[col])
            except Exception:
                continue
    # fallback: first column
    try:
        return pd.to_datetime(df.iloc[:, 0])
    except Exception:
        return None


def load_clean(path: Path):
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Generate data prep sanity report.")
    parser.add_argument("--raw", type=str, default="EURUSD_H4_25Oct17.csv", help="Path to raw EURUSD file.")
    parser.add_argument("--sep", type=str, default="tab", help="Separator hint for raw (e.g., 'tab', ',', '\\t').")
    parser.add_argument("--clean", type=str, default="results/EURUSD_H4_clean.csv", help="Path to cleaned OHLC+ATR file.")
    parser.add_argument("--out", type=str, default="results/data_prep_report.html", help="Output HTML path.")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    clean_path = Path(args.clean)

    sep_hint = args.sep
    if sep_hint.lower() in {"tab", "\\t", "t"}:
        sep_hint = "\t"

    raw_df = load_raw(raw_path, sep_hint)
    dt = ensure_datetime(raw_df)
    year_counts = None
    if dt is not None:
        year_counts = dt.dt.year.value_counts().sort_index()

    # Basic OHLC stats if columns exist
    ohlc_cols = [c for c in raw_df.columns if str(c).strip().lower() in {"open", "high", "low", "close"}]
    raw_stats = raw_df[ohlc_cols].describe() if ohlc_cols else pd.DataFrame()

    clean_df = load_clean(clean_path)
    clean_stats = clean_df.describe()

    style = (
        "body { font-family: Arial, sans-serif; margin: 20px; } "
        "table { border-collapse: collapse; margin-bottom: 16px; } "
        "th, td { border:1px solid #ccc; padding:6px 10px; text-align:right; } "
        "th { background:#f0f0f0; } "
        "h1, h2 { margin-bottom: 8px; }"
    )

    parts = []
    parts.append("<h1>Data Preparation Report</h1>")

    parts.append(f"<h2>Raw file: {raw_path}</h2>")
    parts.append(f"<p>Shape: {raw_df.shape}, Columns: {list(raw_df.columns)}</p>")
    if year_counts is not None:
        parts.append("<h3>Samples per year (raw)</h3>")
        parts.append(year_counts.reset_index().rename(columns={"index": "year", 0: "count"}).to_html(index=False))
    if not raw_stats.empty:
        parts.append("<h3>Raw OHLC stats</h3>")
        parts.append(raw_stats.to_html(float_format="%.6f"))

    parts.append(f"<h2>Clean file: {clean_path}</h2>")
    parts.append(f"<p>Shape: {clean_df.shape}, Columns: {list(clean_df.columns)}</p>")
    parts.append("<h3>Clean stats</h3>")
    parts.append(clean_stats.to_html(float_format="%.6f"))

    html = f"<html><head><meta charset='UTF-8'><style>{style}</style></head><body>{''.join(parts)}</body></html>"
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path.resolve()}")


if __name__ == "__main__":
    main()

