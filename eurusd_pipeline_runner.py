import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PIP = 0.0001
PIP_FACTOR = 1 / PIP


PARAM_CLI = {
    "window": "--window",
    "units": "--units",
    "activation": "--activation",
    "lr": "--lr",
    "epochs": "--epochs",
    "lambda_coef": "--lambda-coef",
    "sigma_coef": "--sigma-coef",
    "batch": "--batch",
    "recurrent_activation": "--recurrent-activation",
    "dropout": "--dropout",
    "recurrent_dropout": "--recurrent-dropout",
    "l2_reg": "--l2-reg",
}
PARAM_COLUMNS = ["window", "units", "activation", "lr", "epochs", "lambda_coef", "sigma_coef", "batch"]
PARAM_PREFIX = {
    "window": "w",
    "units": "h",
    "activation": "act",
    "lr": "lr",
    "epochs": "e",
    "lambda_coef": "lam",
    "sigma_coef": "sig",
    "batch": "b",
}


def parse_args():
    parser = argparse.ArgumentParser(description="EURUSD tuning pipeline runner.")
    parser.add_argument("--base-config", default="final_config.json", help="Baseline config JSON.")
    parser.add_argument(
        "--out-dir",
        default="results/eurusd_pipeline",
        help="Where to store pipeline outputs.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only build summary from existing predictions.",
    )
    return parser.parse_args()


def run_job(job_name: str, overrides: Dict, base_cfg: Path, out_dir: Path):
    preds_path = out_dir / f"{job_name}_preds.csv"
    hist_path = out_dir / f"{job_name}_history.csv"
    start_time = time.perf_counter()
    cmd = [
        sys.executable,
        "bilstm_flf_experiment.py",
        "--config",
        str(base_cfg),
        "--out",
        str(preds_path),
        "--history-out",
        str(hist_path),
    ]
    for key, value in overrides.items():
        cli_key = PARAM_CLI.get(key)
        if not cli_key:
            continue
        cmd.extend([cli_key, str(value)])

    print(f"[PIPELINE] Running {job_name}")
    print(" ".join(cmd))
    out_dir.mkdir(parents=True, exist_ok=True)
    from subprocess import run

    run(cmd, check=True)
    return preds_path, time.perf_counter() - start_time


def compute_mae(preds_path: Path):
    df = pd.read_csv(preds_path)
    preds = df[[c for c in df.columns if c.startswith("pred_")]].values
    targets = df[[c for c in df.columns if c.startswith("true_")]].values
    diff = np.abs(preds - targets)
    mae = diff.mean(axis=0)
    return mae * PIP_FACTOR


def format_param_value(value):
    if value is None:
        return None
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.5g}"
    return str(value)


def build_name(base: str, params: Dict):
    parts = []
    for key in PARAM_COLUMNS:
        if key not in params or params[key] is None:
            continue
        val = format_param_value(params[key])
        parts.append(f"{PARAM_PREFIX[key]}{val}")
    suffix = "_".join(parts) if parts else "base"
    return f"{base}_{suffix}"


def run_stages(stages, base_cfg: Path, out_dir: Path, skip_training: bool):
    cfg_data = json.loads(base_cfg.read_text(encoding="utf-8"))
    baseline = {k: cfg_data.get(k) for k in PARAM_COLUMNS if cfg_data.get(k) is not None}
    html_sections = []
    best_history = []

    for stage in stages:
        rows = []
        best_row = None
        best_params = None
        for job in stage["jobs"]:
            params = baseline.copy()
            params.update(job.get("params", {}))

            preds_path = out_dir / f"{job['name']}_preds.csv"
            duration = 0.0
            if not skip_training or not preds_path.exists():
                preds_path, duration = run_job(job["name"], params, base_cfg, out_dir)
            mae_pips = compute_mae(preds_path)

            literal_name = build_name(stage["base"], params)
            row = {
                "name": literal_name,
                "job": job["name"],
                "mae_avg_pips": mae_pips.mean(),
                "mae_open_pips": mae_pips[0],
                "mae_high_pips": mae_pips[1],
                "mae_low_pips": mae_pips[2],
                "mae_close_pips": mae_pips[3],
                "preds": str(preds_path),
                "train_seconds": duration,
            }
            for col in PARAM_COLUMNS:
                row[col] = params.get(col)
            rows.append(row)

            if best_row is None or row["mae_avg_pips"] < best_row["mae_avg_pips"]:
                best_row = row
                best_params = params

        df = pd.DataFrame(rows)
        stage_path = out_dir / f"{stage['id']}_summary.csv"
        df.to_csv(stage_path, index=False)

        html_sections.append(
            f"<h2>{stage['title']}</h2>"
            f"<p>Best run: <strong>{best_row['name']}</strong> "
            f"(MAE_avg_pips={best_row['mae_avg_pips']:.4f})</p>"
            f"{df.fillna('').to_html(index=False, float_format='%.4f')}"
        )
        best_history.append({"stage": stage["title"], "best_name": best_row["name"], **{k: best_run_val for k, best_run_val in best_params.items() if k in PARAM_COLUMNS}})
        baseline = best_params

    best_df = pd.DataFrame(best_history)
    best_df.to_csv(out_dir / "best_progression.csv", index=False)
    return html_sections


def main():
    args = parse_args()
    base_cfg = Path(args.base_config)
    if not base_cfg.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg}")
    out_dir = Path(args.out_dir)

    stages: List[Dict] = [
        {
            "id": "stage1",
            "title": "Stage 1 - Window Sweep",
            "base": "eur_s1",
            "jobs": [
                {"name": "eur_s1_w12", "params": {"window": 12}},
                {"name": "eur_s1_w16", "params": {"window": 16}},
                {"name": "eur_s1_w20", "params": {"window": 20}},
            ],
        },
        {
            "id": "stage2",
            "title": "Stage 2 - Units Sweep",
            "base": "eur_s2",
            "jobs": [
                {"name": "eur_s2_h128", "params": {"units": 128}},
                {"name": "eur_s2_h200", "params": {"units": 200}},
                {"name": "eur_s2_h256", "params": {"units": 256}},
            ],
        },
        {
            "id": "stage3",
            "title": "Stage 3 - Activation + LR",
            "base": "eur_s3",
            "jobs": [
                {"name": "eur_s3_tanh_lr1e4", "params": {"activation": "tanh", "lr": 1e-4}},
                {"name": "eur_s3_tanh_lr3e4", "params": {"activation": "tanh", "lr": 3e-4}},
                {"name": "eur_s3_relu_lr3e4", "params": {"activation": "relu", "lr": 3e-4}},
                {"name": "eur_s3_relu_lr5e4", "params": {"activation": "relu", "lr": 5e-4}},
                {"name": "eur_s3_relu_lr7e4", "params": {"activation": "relu", "lr": 7e-4}},
            ],
        },
        {
            "id": "stage4",
            "title": "Stage 4 - Lambda / Sigma Sweep",
            "base": "eur_s4",
            "jobs": [
                {"name": "eur_s4_lam08_sig015", "params": {"lambda_coef": 0.8, "sigma_coef": 0.15}},
                {"name": "eur_s4_lam09_sig01", "params": {"lambda_coef": 0.9, "sigma_coef": 0.1}},
                {"name": "eur_s4_lam10_sig005", "params": {"lambda_coef": 1.0, "sigma_coef": 0.05}},
            ],
        },
        {
            "id": "stage5",
            "title": "Stage 5 - Batch Size Sweep",
            "base": "eur_s5",
            "jobs": [
                {"name": "eur_s5_batch64", "params": {"batch": 64}},
                {"name": "eur_s5_batch128", "params": {"batch": 128}},
            ],
        },
        {
            "id": "stage6",
            "title": "Stage 6 - Epoch Sweep",
            "base": "eur_s6",
            "jobs": [
                {"name": "eur_s6_e30", "params": {"epochs": 30}},
                {"name": "eur_s6_e39", "params": {"epochs": 39}},
                {"name": "eur_s6_e50", "params": {"epochs": 50}},
            ],
        },
    ]

    html_sections = run_stages(stages, base_cfg, out_dir, args.skip_training)
    html = [
        "<html><head><meta charset='UTF-8'><title>EURUSD Pipeline Summary</title>"
        "<style>body{font-family:Arial;margin:20px;} table{border-collapse:collapse;}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;} th{background:#f0f0f0;}</style>"
        "</head><body><h1>EURUSD Pipeline Summary (Stage 1-6)</h1>"
    ]
    html.extend(html_sections)
    html.append("</body></html>")
    out_file = out_dir / "eurusd_pipeline_summary.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text("".join(html), encoding="utf-8")
    print(f"Pipeline summary written to {out_file.resolve()}")


if __name__ == "__main__":
    main()
