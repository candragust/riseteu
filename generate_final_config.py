import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PIP = 0.0001
PIP_FACTOR = 1 / PIP


def mae_avg(path: Path) -> float:
    df = pd.read_csv(path)
    mae = (df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
           - df[["true_open", "true_high", "true_low", "true_close"]].values)
    return abs(mae).mean()


def extract_params(name: str):
    params = {"window": None, "units": None, "activation": None, "lr": None, "epochs": None}
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
        mae = mae_avg(path)
        rows.append({
            "path": path,
            "name": base,
            "window": params["window"],
            "units": params["units"],
            "activation": params["activation"],
            "lr": params["lr"],
            "epochs": params["epochs"],
            "mae_avg_pips": mae * PIP_FACTOR,
        })

    if not rows:
        print("No hp_*_preds.csv found. Abort.")
        return

    best = sorted(rows, key=lambda x: x["mae_avg_pips"])[0]
    print("Best run:", best)

    final_cfg = {
        "data": "results/EURUSD_H4_clean.csv",  # adjust if using filtered variant
        "sep": ",",
        "columns": None,
        "window": best["window"],
        "split": 0.7,
        "units": best["units"],
        "epochs": best["epochs"] if best["epochs"] else 30,
        "batch": 128,
        "lr": best["lr"],
        "beta1": 0.9,
        "beta2": 0.999,
        "schedule_decay": 0.004,
        "seed": 42,
        "activation": best["activation"],
        "recurrent_activation": "sigmoid",
        "output_activation": "",
        "lambda_coef": 0.9,
        "sigma_coef": 0.1,
        "out": "results/final_preds.csv",
        "history_out": "results/final_history.csv"
    }

    out_path = Path("final_config.json")
    out_path.write_text(json.dumps(final_cfg, indent=2), encoding="utf-8")
    print(f"Final config written to {out_path.resolve()}")


if __name__ == "__main__":
    main()

