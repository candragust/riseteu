import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    parser = argparse.ArgumentParser(description="Plot loss/val_loss curves from history CSVs.")
    parser.add_argument("--histories", nargs="+", required=True, help="List of history CSV paths.")
    parser.add_argument("--labels", nargs="+", required=False, help="Optional labels for each history.")
    parser.add_argument("--out", type=str, default="results/loss_curves.html", help="Output HTML path.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.histories]
    labels = args.labels if args.labels and len(args.labels) == len(paths) else [p.stem for p in paths]

    fig = make_subplots(rows=len(paths), cols=1, shared_xaxes=False,
                        subplot_titles=[f"{lbl}" for lbl in labels])

    for idx, (path, label) in enumerate(zip(paths, labels), start=1):
        df = pd.read_csv(path)
        if "loss" in df and "val_loss" in df:
            fig.add_trace(go.Scatter(y=df["loss"], mode="lines", name=f"{label} - loss"),
                          row=idx, col=1)
            fig.add_trace(go.Scatter(y=df["val_loss"], mode="lines", name=f"{label} - val_loss", line=dict(dash="dash")),
                          row=idx, col=1)
            fig.update_yaxes(title_text="Loss", row=idx, col=1)
            fig.update_xaxes(title_text="Epoch", row=idx, col=1)
        else:
            print(f"Skipping {path} (missing loss/val_loss).")

    fig.update_layout(title="Loss / Val Loss per Epoch (separated)", template="plotly_white", height=300*len(paths))

    out_path = Path(args.out)
    out_path.write_text(fig.to_html(include_plotlyjs="cdn"), encoding="utf-8")
    print(f"Wrote loss curves to {out_path.resolve()}")


if __name__ == "__main__":
    main()
