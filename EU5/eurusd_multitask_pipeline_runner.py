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
PARAM_COLUMNS = [
    "window",
    "units",
    "activation",
    "lr",
    "epochs",
    "lambda_coef",
    "sigma_coef",
    "batch",
]
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
    parser = argparse.ArgumentParser(description="EU5 multitask tuning pipeline runner.")
    parser.add_argument(
        "--base-config",
        default=str(Path("RisetEU") / "EU5" / "multitask_config.json"),
        help="Baseline config JSON for the multitask experiment.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("RisetEU") / "EU5" / "results" / "multitask_pipeline_w12"),
        help="Directory where predictions & summaries are stored.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only rebuild summary using existing predictions.",
    )
    return parser.parse_args()


def run_job(job_name: str, overrides: Dict, base_cfg: Path, out_dir: Path):
    preds_path = out_dir / f"{job_name}_preds.csv"
    hist_path = out_dir / f"{job_name}_history.csv"
    start_time = time.perf_counter()
    script = Path(__file__).resolve().parent / "bilstm_multitask_experiment.py"
    cmd = [
        sys.executable,
        str(script),
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
    price_cols = ["open", "high", "low", "close"]
    pred_cols = [f"pred_{c}" for c in price_cols]
    true_cols = [f"true_{c}" for c in price_cols]
    preds = df[pred_cols].values
    targets = df[true_cols].values
    mae = np.abs(preds - targets).mean(axis=0)
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
            mae_hlc = mae_pips[1:].mean()

            literal_name = build_name(stage["base"], params)
            row = {
                "name": literal_name,
                "job": job["name"],
                "mae_avg_hlc_pips": mae_hlc,
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

            if best_row is None or row["mae_avg_hlc_pips"] < best_row["mae_avg_hlc_pips"]:
                best_row = row
                best_params = params

        df = pd.DataFrame(rows)
        stage_path = out_dir / f"{stage['id']}_summary.csv"
        df.to_csv(stage_path, index=False)

        html_sections.append(
            f"<h2>{stage['title']}</h2>"
            f"<p>Best run: <strong>{best_row['name']}</strong> "
            f"(MAE_avg_hlc_pips={best_row['mae_avg_hlc_pips']:.4f})</p>"
            f"{df.fillna('').to_html(index=False, float_format='%.4f')}"
        )
        best_history.append(
            {
                "stage": stage["title"],
                "best_name": best_row["name"],
                **{k: best_val for k, best_val in best_params.items() if k in PARAM_COLUMNS},
            }
        )
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
            "base": "eu5_s1",
            "jobs": [
                {"name": "eu5_s1_base", "params": {}},
                {"name": "eu5_s1_w10", "params": {"window": 10}},
                {"name": "eu5_s1_w12", "params": {"window": 12}},
                {"name": "eu5_s1_w14", "params": {"window": 14}},
                {"name": "eu5_s1_w16", "params": {"window": 16}},
            ],
        },
        {
            "id": "stage2",
            "title": "Stage 2 - Units Sweep",
            "base": "eu5_s2",
            "jobs": [
                {"name": "eu5_s2_h224", "params": {"units": 224}},
                {"name": "eu5_s2_h256", "params": {"units": 256}},
                {"name": "eu5_s2_h320", "params": {"units": 320}},
            ],
        },
        {
            "id": "stage3",
            "title": "Stage 3 - Activation + LR",
            "base": "eu5_s3",
            "jobs": [
                {"name": "eu5_s3_tanh_lr4e4", "params": {"activation": "tanh", "lr": 4e-4}},
                {"name": "eu5_s3_tanh_lr5e4", "params": {"activation": "tanh", "lr": 5e-4}},
                {"name": "eu5_s3_tanh_lr7e4", "params": {"activation": "tanh", "lr": 7e-4}},
                {"name": "eu5_s3_relu_lr5e4", "params": {"activation": "relu", "lr": 5e-4}},
            ],
        },
        {
            "id": "stage4",
            "title": "Stage 4 - Lambda / Sigma Sweep",
            "base": "eu5_s4",
            "jobs": [
                {"name": "eu5_s4_lam09_sig008", "params": {"lambda_coef": 0.9, "sigma_coef": 0.08}},
                {"name": "eu5_s4_lam10_sig01", "params": {"lambda_coef": 1.0, "sigma_coef": 0.1}},
                {"name": "eu5_s4_lam11_sig012", "params": {"lambda_coef": 1.1, "sigma_coef": 0.12}},
            ],
        },
        {
            "id": "stage5",
            "title": "Stage 5 - Batch Size Sweep",
            "base": "eu5_s5",
            "jobs": [
                {"name": "eu5_s5_batch80", "params": {"batch": 80}},
                {"name": "eu5_s5_batch96", "params": {"batch": 96}},
                {"name": "eu5_s5_batch128", "params": {"batch": 128}},
            ],
        },
        {
            "id": "stage6",
            "title": "Stage 6 - Epoch Sweep",
            "base": "eu5_s6",
            "jobs": [
                {"name": "eu5_s6_e40", "params": {"epochs": 40}},
                {"name": "eu5_s6_e50", "params": {"epochs": 50}},
                {"name": "eu5_s6_e60", "params": {"epochs": 60}},
            ],
        },
    ]

    html_sections = run_stages(stages, base_cfg, out_dir, args.skip_training)
    html = [
        "<html><head><meta charset='UTF-8'><title>EU5 Pipeline Summary</title>"
        "<style>body{font-family:Arial;margin:20px;} table{border-collapse:collapse;}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;} th{background:#f0f0f0;}</style>"
        "</head><body><h1>EU5 Pipeline Summary (Stage 1-6)</h1>"
    ]
    html.extend(html_sections)
    html.append("</body></html>")
    out_file = out_dir / "eu5_pipeline_summary.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text("".join(html), encoding="utf-8")
    print(f"Pipeline summary written to {out_file.resolve()}")


if __name__ == "__main__":
    main()
