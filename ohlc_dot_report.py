import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


COLUMNS = ["open", "high", "low", "close"]


def build_chart(df: pd.DataFrame, col: str, actual_size: int, pred_size: int):
    idx = list(range(len(df)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=df[f"true_{col}"],
            mode="lines+markers",
            name=f"Actual {col.capitalize()}",
            line=dict(color="red"),
            marker=dict(size=actual_size, color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=df[f"pred_{col}"],
            mode="lines+markers",
            name=f"Pred {col.capitalize()}",
            line=dict(color="blue", dash="dot"),
            marker=dict(size=pred_size, color="blue"),
        )
    )
    fig.update_layout(
        title=f"{col.capitalize()} Price (tail segment)",
        xaxis_title="Index (tail)",
        yaxis_title="Price",
        yaxis=dict(tickformat=".5f"),
        template="plotly_white",
        height=360,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Build OHLC HTML report with dot charts.")
    parser.add_argument("--preds", required=True, help="CSV with pred_* and true_* columns.")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index (from head).")
    parser.add_argument("--length", type=int, default=30, help="Number of rows to include from start-index.")
    parser.add_argument("--out", default="results/ohlc_dot_report.html", help="Output HTML file.")
    parser.add_argument("--title", default="OHLC Actual vs Pred Report", help="Report title.")
    parser.add_argument("--meta", default="", help="Text describing validation parameters.")
    parser.add_argument("--actual-size", type=int, default=6, help="Marker size for actual series.")
    parser.add_argument("--pred-size", type=int, default=5, help="Marker size for predicted series.")
    args = parser.parse_args()

    df = pd.read_csv(args.preds)
    subset = df.iloc[args.start_index : args.start_index + args.length].reset_index(drop=True)

    charts = [build_chart(subset, col, args.actual_size, args.pred_size) for col in COLUMNS]

    html_parts = [f"<h1>{args.title}</h1>"]
    if args.meta:
        html_parts.append(f"<p><strong>Parameters:</strong> {args.meta}</p>")

    for fig in charts:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html = (
        "<html><head><meta charset='UTF-8'>"
        "<script src='https://cdn.plot.ly/plotly-3.3.0.min.js'></script>"
        "</head><body>"
        f"{''.join(html_parts)}"
        "</body></html>"
    )

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
