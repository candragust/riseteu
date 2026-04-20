import glob
import os
import re
import pandas as pd

PIP = 0.0001
PIP_FACTOR = 1 / PIP


def extract_params(name: str):
    # expected tokens like w32_h200_relu_lr3e4_e20
    params = {
        "window": None,
        "units": None,
        "activation": None,
        "lr": None,
        "epochs": None,
    }
    tokens = name.split("_")
    for t in tokens:
        if t.startswith("w") and t[1:].isdigit():
            params["window"] = int(t[1:])
        elif t.startswith("h") and t[1:].isdigit():
            params["units"] = int(t[1:])
        elif t in ("tanh", "relu", "elu"):
            params["activation"] = t
        elif t.startswith("lr"):
            lr_str = t[2:]
            # handle formats like 1e4 -> 0.0001, 3e4 -> 0.0003, 5e4 -> 0.0005
            m = re.match(r"(\d+)(e)(\d+)", lr_str)
            if m:
                num = float(m.group(1))
                exp = int(m.group(3))
                params["lr"] = num * (10 ** -exp)
            else:
                try:
                    params["lr"] = float(lr_str)
                except ValueError:
                    params["lr"] = None
        elif t.startswith("e") and t[1:].isdigit():
            params["epochs"] = int(t[1:])
    return params


def main():
    rows = []
    for path in glob.glob("results/hp_*_preds.csv"):
        base = os.path.basename(path).replace("_preds.csv", "")
        params = extract_params(base)
        df = pd.read_csv(path)
        mae = (df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
               - df[["true_open", "true_high", "true_low", "true_close"]].values)
        mae = abs(mae).mean(axis=0)
        rows.append({
            "name": base,
            "window": params["window"],
            "H": params["units"],
            "activation": params["activation"],
            "lr": params["lr"],
            "epochs": params["epochs"],
            "batch": 128,
            "mae_open_pips": mae[0] * PIP_FACTOR,
            "mae_high_pips": mae[1] * PIP_FACTOR,
            "mae_low_pips": mae[2] * PIP_FACTOR,
            "mae_close_pips": mae[3] * PIP_FACTOR,
            "mae_avg_pips": mae.mean() * PIP_FACTOR,
            "preds": path,
        })

    if not rows:
        print("No hp_*_preds.csv files found.")
        return

    cols = [
        "name", "window", "H", "activation", "lr", "epochs", "batch",
        "mae_open_pips", "mae_high_pips", "mae_low_pips", "mae_close_pips", "mae_avg_pips", "preds"
    ]
    df = pd.DataFrame(rows)[cols].sort_values(by="mae_avg_pips")

    style = (
        "body { font-family: Arial, sans-serif; margin: 20px; } "
        "table { border-collapse: collapse; } "
        "th, td { border:1px solid #ccc; padding:6px 10px; text-align:right; } "
        "th { background:#f0f0f0; }"
    )
    html = (
        f"<html><head><meta charset='UTF-8'><title>HP Summary</title>"
        f"<style>{style}</style></head><body>"
        f"<h1>Hyperparameter Summary (sorted by MAE_avg_pips)</h1>"
        f"{df.to_html(index=False, float_format='%.4f')}"
        f"</body></html>"
    )

    out_path = "results/hp_summary.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Summary written to {out_path} with {len(df)} rows")


if __name__ == "__main__":
    main()

