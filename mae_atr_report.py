import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PIP = 0.0001
PIP_FACTOR = 1 / PIP


def compute_mae(preds_df: pd.DataFrame):
    cols_pred = ["pred_open", "pred_high", "pred_low", "pred_close"]
    cols_true = ["true_open", "true_high", "true_low", "true_close"]
    diffs = preds_df[cols_pred].values - preds_df[cols_true].values
    mae = np.mean(np.abs(diffs), axis=0)
    return {
        "mae_open": mae[0],
        "mae_high": mae[1],
        "mae_low": mae[2],
        "mae_close": mae[3],
        "mae_avg": mae.mean(),
    }


def load_atr(clean_path: Path, split_ratio: float, tail_len: int):
    df = pd.read_csv(clean_path)
    start_val = int(len(df) * split_ratio)
    val_df = df.iloc[start_val:]
    atr6_mean = val_df["atr6"].mean()
    atr12_mean = val_df["atr12"].mean()
    if tail_len <= 0 or tail_len >= len(val_df):
        sample = val_df.reset_index(drop=True)
        label = "all test candles"
    else:
        sample = val_df.tail(tail_len).reset_index(drop=True)
        label = f"tail {len(sample)}"
    return atr6_mean, atr12_mean, sample, label


def build_visual(tail_ohlc: pd.DataFrame, preds_tail: pd.DataFrame, sample_label: str):
    return go.Figure(
        data=[
            go.Scatter(
                x=list(range(len(tail_ohlc))),
                y=tail_ohlc["close"],
                mode="lines",
                name="True Close",
                line=dict(color="#0f172a", width=2),
            ),
            go.Scatter(
                x=list(range(len(preds_tail))),
                y=preds_tail["pred_close"],
                mode="lines",
                name="Pred Close",
                line=dict(color="#2563eb", width=2),
            ),
        ],
        layout=go.Layout(
            title=f"True vs Predicted Close ({sample_label})",
            xaxis_title="Index (test sample)",
            yaxis_title="Price",
            yaxis=dict(tickformat=".5f"),
            template="plotly_white",
            height=420,
        ),
    )


def build_overlay_visual(tail_ohlc: pd.DataFrame, preds_tail: pd.DataFrame, sample_label: str):
    return go.Figure(
        data=[
            go.Candlestick(
                x=list(range(len(tail_ohlc))),
                open=tail_ohlc["open"],
                high=tail_ohlc["high"],
                low=tail_ohlc["low"],
                close=tail_ohlc["close"],
                name="True OHLC",
                increasing_line=dict(color="rgba(22,163,74,0.42)", width=0.8),
                decreasing_line=dict(color="rgba(220,38,38,0.42)", width=0.8),
                increasing_fillcolor="rgba(34,197,94,0.10)",
                decreasing_fillcolor="rgba(239,68,68,0.10)",
                whiskerwidth=0.16,
                opacity=0.9,
            ),
            go.Ohlc(
                x=list(range(len(preds_tail))),
                open=preds_tail["pred_open"],
                high=preds_tail["pred_high"],
                low=preds_tail["pred_low"],
                close=preds_tail["pred_close"],
                name="Predicted OHLC Bar",
                increasing_line_color="rgba(22,163,74,0.98)",
                decreasing_line_color="rgba(185,28,28,0.98)",
                opacity=1.0,
            ),
        ],
        layout=go.Layout(
            title=f"Overlay True vs Predicted OHLC ({sample_label})",
            xaxis_title="Index (test sample)",
            yaxis_title="Price",
            yaxis=dict(tickformat=".5f"),
            template="plotly_white",
            height=520,
            xaxis=dict(rangeslider=dict(visible=False)),
        ),
    )


def build_error_bar(preds_tail: pd.DataFrame, tail_ohlc: pd.DataFrame, atr6_mean, atr12_mean, sample_label: str):
    errs_pips = np.abs(preds_tail["pred_close"] - tail_ohlc["close"]) * PIP_FACTOR
    atr6_tail = tail_ohlc["atr6"] * PIP_FACTOR
    atr12_tail = tail_ohlc["atr12"] * PIP_FACTOR
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(errs_pips))),
        y=errs_pips,
        mode="lines+markers",
        name="|Error Close| (pips)",
        line=dict(color="#2563eb", width=2),
        marker=dict(color="#2563eb", size=5),
    ))
    # ATR per candle (tail) untuk melihat variasi, plus rata-rata sebagai referensi
    fig.add_trace(go.Scatter(x=list(range(len(errs_pips))), y=atr6_tail,
                             mode="lines", name="ATR6 (tail)", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=list(range(len(errs_pips))), y=atr12_tail,
                             mode="lines", name="ATR12 (tail)", line=dict(color="green", width=2, dash="dash")))
    fig.update_layout(
        title=f"Error Close per Candle ({sample_label}) vs ATR (sample) [pips]",
        xaxis_title="Index (test sample)",
        yaxis_title="Absolute Error (pips)",
        yaxis=dict(tickformat=".1f"),
        template="plotly_white",
        height=420,
    )
    return fig


def build_error_bar_avg(preds_tail: pd.DataFrame, tail_ohlc: pd.DataFrame, atr6_mean, atr12_mean, sample_label: str):
    # MAE_avg per candle = rata-rata abs error 4 komponen per baris
    errs = np.abs(preds_tail[["pred_open", "pred_high", "pred_low", "pred_close"]].values -
                  tail_ohlc[["open", "high", "low", "close"]].values)
    errs_avg_pips = errs.mean(axis=1) * PIP_FACTOR
    atr6_tail = tail_ohlc["atr6"] * PIP_FACTOR
    atr12_tail = tail_ohlc["atr12"] * PIP_FACTOR

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(errs_avg_pips))),
        y=errs_avg_pips,
        mode="lines+markers",
        name="|Error avg| (pips)",
        line=dict(color="#2563eb", width=2),
        marker=dict(color="#2563eb", size=5),
    ))
    fig.add_trace(go.Scatter(x=list(range(len(errs_avg_pips))), y=atr6_tail,
                             mode="lines", name="ATR6 (tail)", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=list(range(len(errs_avg_pips))), y=atr12_tail,
                             mode="lines", name="ATR12 (tail)", line=dict(color="green", width=2, dash="dash")))
    fig.update_layout(
        title=f"Error AVG per Candle ({sample_label}) vs ATR (sample) [pips]",
        xaxis_title="Index (test sample)",
        yaxis_title="Absolute Error (pips)",
        yaxis=dict(tickformat=".1f"),
        template="plotly_white",
        height=420,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate MAE vs ATR comparison report (HTML) for tail samples.")
    parser.add_argument("--preds", required=True, help="Path to preds CSV (with pred_* and true_*).")
    parser.add_argument("--clean", required=True, help="Path to cleaned OHLC CSV with atr6/atr12.")
    parser.add_argument("--split", type=float, default=0.7, help="Train split ratio to align tail validation.")
    parser.add_argument("--out", default="results/mae_atr_report.html", help="Output HTML path.")
    parser.add_argument("--tail", type=int, default=20, help="Number of tail candles to visualize.")
    parser.add_argument("--title", default="MAE vs ATR Report", help="HTML title/header.")
    parser.add_argument("--meta", default="", help="Additional metadata description for the header.")
    args = parser.parse_args()

    preds_df = pd.read_csv(args.preds)
    metrics = compute_mae(preds_df)

    atr6_mean, atr12_mean, tail_clean, sample_label = load_atr(Path(args.clean), args.split, args.tail)
    tail_preds = preds_df.tail(len(tail_clean)).reset_index(drop=True)

    fig_price = build_visual(tail_clean, tail_preds, sample_label)
    fig_overlay = build_overlay_visual(tail_clean, tail_preds, sample_label)
    fig_err = build_error_bar(tail_preds, tail_clean, atr6_mean, atr12_mean, sample_label)
    fig_err_avg = build_error_bar_avg(tail_preds, tail_clean, atr6_mean, atr12_mean, sample_label)

    # Build HTML
    summary_tbl = pd.DataFrame([{
        "mae_open_pips": metrics["mae_open"] * PIP_FACTOR,
        "mae_high_pips": metrics["mae_high"] * PIP_FACTOR,
        "mae_low_pips": metrics["mae_low"] * PIP_FACTOR,
        "mae_close_pips": metrics["mae_close"] * PIP_FACTOR,
        "mae_avg_pips": metrics["mae_avg"] * PIP_FACTOR,
        "visualized_samples": len(tail_clean),
        "test_samples_total": len(preds_df),
    }])

    style = (
        "body { font-family: Arial, sans-serif; margin: 20px; } "
        "table { border-collapse: collapse; } "
        "th, td { border:1px solid #ccc; padding:6px 10px; text-align:right; } "
        "th { background:#f0f0f0; }"
    )

    html_parts = [f"<h1>{args.title}</h1>"]
    if args.meta:
        html_parts.append(f"<p><strong>Parameters:</strong> {args.meta}</p>")
    html_parts.extend([
        summary_tbl.to_html(index=False, float_format="%.6f"),
        "<h2>Charts</h2>",
        "<h3>Overlay True vs Predicted OHLC</h3>",
        fig_overlay.to_html(full_html=False, include_plotlyjs="cdn"),
        "<h3>True vs Predicted Close</h3>",
        fig_price.to_html(full_html=False, include_plotlyjs=False),
        "<h3>Error Close vs ATR</h3>",
        fig_err.to_html(full_html=False, include_plotlyjs=False),
        "<h3>Error AVG vs ATR</h3>",
        fig_err_avg.to_html(full_html=False, include_plotlyjs=False),
    ])
    html = (
        f"<html><head><meta charset='UTF-8'><title>{args.title}</title>"
        f"<style>{style}</style></head><body>"
        f"{''.join(html_parts)}"
        f"</body></html>"
    )

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
