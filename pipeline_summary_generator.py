import glob
import json
import os
import re
from pathlib import Path

import pandas as pd

PIP = 0.0001
PIP_FACTOR = 1 / PIP


def mae_avg(path: Path) -> float:
    df = pd.read_csv(path)
    mae = (df[["pred_open", "pred_high", "pred_low", "pred_close"]].values
           - df[["true_open", "true_high", "true_low", "true_close"]].values)
    return abs(mae).mean()


def extract_params(name: str):
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


def best_from_runs(filter_fn):
    rows = []
    for path in glob.glob("results/hp_*_preds.csv"):
        base = os.path.basename(path).replace("_preds.csv", "")
        params = extract_params(base)
        if not filter_fn(params):
            continue
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
            "mae_avg_pips": mae.mean() * PIP_FACTOR,
        })
    if not rows:
        return None
    return sorted(rows, key=lambda x: x["mae_avg_pips"])[0]


def main():
    grid = json.loads(Path("experiments/staged_grid.json").read_text())
    stages = []

    def eq_lr(a, b, tol=1e-9):
        return (a == b) or (a is not None and b is not None and abs(a - b) < tol)

    # Stage 1: window sweep
    s1 = grid["stage1_window"]
    best1 = best_from_runs(lambda p: p["activation"] == s1["fixed"]["activation"]
                                     and eq_lr(p["lr"], s1["fixed"]["lr"])
                                     and p["epochs"] == s1["fixed"]["epochs"]
                                     and p["units"] == s1["fixed"]["units"]
                                     and p["window"] in s1["windows"])
    best_window = best1["window"] if best1 else s1["windows"][0]
    stages.append({
        "Tahap": "Window Sweep",
        "Input Window": s1["windows"],
        "Input H": s1["fixed"]["units"],
        "Input Activation": s1["fixed"]["activation"],
        "Input LR": s1["fixed"]["lr"],
        "Input Epoch": s1["fixed"]["epochs"],
        "Window*": best_window,
        "H*": s1["fixed"]["units"],
        "Activation*": s1["fixed"]["activation"],
        "LR*": s1["fixed"]["lr"],
        "Epoch*": s1["fixed"]["epochs"],
        "MAE_avg_pips_best": best1["mae_avg_pips"] if best1 else None,
    })

    # Stage 2: units sweep
    s2 = grid["stage2_units"]
    win2 = s2["windows"][0] if s2["windows"] else best_window
    best2 = best_from_runs(lambda p: p["window"] == win2
                                     and p["activation"] == s2["fixed"]["activation"]
                                     and eq_lr(p["lr"], s2["fixed"]["lr"])
                                     and p["epochs"] == s2["fixed"]["epochs"]
                                     and p["units"] in s2["units"])
    best_units = best2["H"] if best2 else s2["units"][0]
    stages.append({
        "Tahap": "Hidden Units Sweep",
        "Input Window": win2,
        "Input H": s2["units"],
        "Input Activation": s2["fixed"]["activation"],
        "Input LR": s2["fixed"]["lr"],
        "Input Epoch": s2["fixed"]["epochs"],
        "Window*": win2,
        "H*": best_units,
        "Activation*": s2["fixed"]["activation"],
        "LR*": s2["fixed"]["lr"],
        "Epoch*": s2["fixed"]["epochs"],
        "MAE_avg_pips_best": best2["mae_avg_pips"] if best2 else None,
    })

    # Stage 3: activation + LR sweep
    s3 = grid["stage3_activation_lr"]
    win3 = s3["windows"][0] if s3["windows"] else best_window
    h3 = s3["units"][0] if s3["units"] else best_units
    combos = s3["combos"]
    best3 = best_from_runs(lambda p: p["window"] == win3
                                     and p["units"] == h3
                                     and p["epochs"] == s3["fixed"]["epochs"]
                                     and any((p["activation"] == c["activation"] and eq_lr(p["lr"], c["lr"])) for c in combos))
    best_act = best3["activation"] if best3 else (combos[0]["activation"] if combos else None)
    best_lr = best3["lr"] if best3 else (combos[0]["lr"] if combos else None)
    stages.append({
        "Tahap": "Activation + LR Sweep",
        "Input Window": win3,
        "Input H": h3,
        "Input Activation": [c["activation"] for c in combos],
        "Input LR": [c["lr"] for c in combos],
        "Input Epoch": s3["fixed"]["epochs"],
        "Window*": win3,
        "H*": h3,
        "Activation*": best_act,
        "LR*": best_lr,
        "Epoch*": s3["fixed"]["epochs"],
        "MAE_avg_pips_best": best3["mae_avg_pips"] if best3 else None,
    })

    # Stage 4: epoch tuning
    if "stage4_epochs" in grid:
        s4 = grid["stage4_epochs"]
        win4 = s4["windows"][0] if s4["windows"] else best_window
        h4 = s4["units"][0] if s4["units"] else best_units
        act4 = s4["activation"] if s4.get("activation") else best_act
        lr4 = s4["lr"] if s4.get("lr") else best_lr
        epochs_list = s4["epochs"]
        best4 = best_from_runs(lambda p: p["window"] == win4
                                         and p["units"] == h4
                                         and p["activation"] == act4
                                         and eq_lr(p["lr"], lr4)
                                         and p["epochs"] in epochs_list)
        stages.append({
            "Tahap": "Epoch Tuning",
            "Input Window": win4,
            "Input H": h4,
            "Input Activation": act4,
            "Input LR": lr4,
            "Input Epoch": epochs_list,
            "Window*": win4,
            "H*": h4,
            "Activation*": act4,
            "LR*": lr4,
            "Epoch*": best4["epochs"] if best4 else None,
            "MAE_avg_pips_best": best4["mae_avg_pips"] if best4 else None,
        })

    df = pd.DataFrame(stages)
    cols_order = [
        "Tahap", "Input Window", "Input H", "Input Activation", "Input LR", "Input Epoch",
        "Window*", "H*", "Activation*", "LR*", "Epoch*", "MAE_avg_pips_best"
    ]
    df = df[cols_order]

    style = (
        "body { font-family: Arial, sans-serif; margin: 20px; } "
        "table { border-collapse: collapse; } "
        "th, td { border:1px solid #ccc; padding:6px 10px; text-align:left; } "
        "th { background:#f0f0f0; }"
    )
    html = (
        f"<html><head><meta charset='UTF-8'><title>Pipeline Summary</title>"
        f"<style>{style}</style></head><body>"
        f"<h1>Pipeline Summary (Staged HP Sweep)</h1>"
        f"{df.to_html(index=False, escape=False)}"
        f"</body></html>"
    )
    out_path = Path("results/pipeline_summary.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Pipeline summary written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
