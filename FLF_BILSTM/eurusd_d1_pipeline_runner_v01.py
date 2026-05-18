import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PIP = 0.0001
PIP_FACTOR = 1 / PIP
ROOT = Path(__file__).resolve().parent


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
SUMMARY_COLUMNS = PARAM_COLUMNS + ["dropout", "recurrent_dropout", "l2_reg"]
DEFAULT_SUMMARY_VALUES = {
    "dropout": 0.0,
    "recurrent_dropout": 0.0,
    "l2_reg": 0.0,
}
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
    parser = argparse.ArgumentParser(description="EUR/USD 1D FLF-BiLSTM tuning pipeline runner.")
    parser.add_argument(
        "--base-config",
        default=str(ROOT / "configs" / "d1_ohlc" / "bilstm_flf_config_d1_ohlc_base_v01.json"),
        help="Baseline config JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "d1_ohlc" / "tuning_v01"),
        help="Where to store pipeline outputs.",
    )
    parser.add_argument("--skip-training", action="store_true", help="Only build summary from existing predictions.")
    parser.add_argument("--max-stage", type=int, default=None, help="Run stages up to this number only.")
    parser.add_argument("--min-stage", type=int, default=1, help="Run stages starting from this number.")
    return parser.parse_args()


def run_job(job_name: str, overrides: Dict, base_cfg: Path, out_dir: Path):
    preds_path = out_dir / f"{job_name}_preds.csv"
    hist_path = out_dir / f"{job_name}_history.csv"
    start_time = time.perf_counter()
    cmd = [
        sys.executable,
        str(ROOT / "bilstm_flf_experiment.py"),
        "--config",
        str(base_cfg),
        "--out",
        str(preds_path),
        "--history-out",
        str(hist_path),
    ]
    for key, value in overrides.items():
        cli_key = PARAM_CLI.get(key)
        if cli_key:
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
    return diff.mean(axis=0) * PIP_FACTOR


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
        if key in params and params[key] is not None:
            parts.append(f"{PARAM_PREFIX[key]}{format_param_value(params[key])}")
    suffix = "_".join(parts) if parts else "base"
    return f"{base}_{suffix}"


def load_resume_baseline(out_dir: Path, min_stage: int, fallback: Dict) -> Dict:
    if min_stage <= 1:
        return fallback.copy()

    progress_path = out_dir / "best_progression.csv"
    if not progress_path.exists():
        return fallback.copy()

    df = pd.read_csv(progress_path)
    target_idx = min_stage - 2
    if target_idx < 0 or target_idx >= len(df):
        return fallback.copy()

    row = df.iloc[target_idx]
    baseline = fallback.copy()
    for key in SUMMARY_COLUMNS:
        if key in row and pd.notna(row[key]):
            value = row[key]
            baseline[key] = value.item() if hasattr(value, "item") else value
    for key, value in DEFAULT_SUMMARY_VALUES.items():
        baseline.setdefault(key, value)
    return baseline


def run_stages(stages, base_cfg: Path, out_dir: Path, skip_training: bool):
    cfg_data = json.loads(base_cfg.read_text(encoding="utf-8"))
    baseline = {k: cfg_data.get(k) for k in SUMMARY_COLUMNS if cfg_data.get(k) is not None}
    for key, value in DEFAULT_SUMMARY_VALUES.items():
        baseline.setdefault(key, value)
    html_sections = []
    best_history = []
    incumbent_row = None
    incumbent_params = baseline.copy()

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

            row = {
                "name": build_name(stage["base"], params),
                "job": job["name"],
                "mae_avg_pips": mae_pips.mean(),
                "mae_open_pips": mae_pips[0],
                "mae_high_pips": mae_pips[1],
                "mae_low_pips": mae_pips[2],
                "mae_close_pips": mae_pips[3],
                "preds": str(preds_path),
                "train_seconds": duration,
            }
            for col in SUMMARY_COLUMNS:
                row[col] = params.get(col, DEFAULT_SUMMARY_VALUES.get(col))
            rows.append(row)

            if best_row is None or row["mae_avg_pips"] < best_row["mae_avg_pips"]:
                best_row = row
                best_params = params

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{stage['id']}_summary.csv", index=False)

        selected_row = best_row
        selected_params = best_params
        selected_from = "stage_best"
        if incumbent_row is not None and best_row["mae_avg_pips"] > incumbent_row["mae_avg_pips"]:
            selected_row = incumbent_row
            selected_params = incumbent_params
            selected_from = "carry_forward"
        else:
            incumbent_row = best_row
            incumbent_params = best_params.copy()

        html_sections.append(
            f"<h2>{stage['title']}</h2>"
            f"<p>Best run: <strong>{best_row['name']}</strong> "
            f"(MAE_avg_pips={best_row['mae_avg_pips']:.4f})</p>"
            f"<p>Selected baseline after stage: <strong>{selected_row['name']}</strong> "
            f"(MAE_avg_pips={selected_row['mae_avg_pips']:.4f}, source={selected_from})</p>"
            f"{df.fillna('').to_html(index=False, float_format='%.4f')}"
        )
        best_history.append(
            {
                "stage": stage["title"],
                "stage_best_name": best_row["name"],
                "stage_best_mae_avg_pips": best_row["mae_avg_pips"],
                "selected_name": selected_row["name"],
                "selected_mae_avg_pips": selected_row["mae_avg_pips"],
                "selected_from": selected_from,
                **{k: selected_params.get(k, DEFAULT_SUMMARY_VALUES.get(k)) for k in SUMMARY_COLUMNS},
            }
        )
        baseline = selected_params.copy()

    pd.DataFrame(best_history).to_csv(out_dir / "best_progression.csv", index=False)
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
            "base": "bilstm_d1_s1",
            "jobs": [
                {"name": "bilstm_d1_s1_w4", "params": {"window": 4}},
                {"name": "bilstm_d1_s1_w6", "params": {"window": 6}},
                {"name": "bilstm_d1_s1_w8", "params": {"window": 8}},
                {"name": "bilstm_d1_s1_w12", "params": {"window": 12}},
            ],
        },
        {
            "id": "stage2",
            "title": "Stage 2 - Units Sweep",
            "base": "bilstm_d1_s2",
            "jobs": [
                {"name": "bilstm_d1_s2_h96", "params": {"units": 96}},
                {"name": "bilstm_d1_s2_h128", "params": {"units": 128}},
                {"name": "bilstm_d1_s2_h192", "params": {"units": 192}},
            ],
        },
        {
            "id": "stage3",
            "title": "Stage 3 - Learning Rate Sweep",
            "base": "bilstm_d1_s3",
            "jobs": [
                {"name": "bilstm_d1_s3_lr3e4", "params": {"lr": 3e-4}},
                {"name": "bilstm_d1_s3_lr5e4", "params": {"lr": 5e-4}},
                {"name": "bilstm_d1_s3_lr7e4", "params": {"lr": 7e-4}},
            ],
        },
        {
            "id": "stage4",
            "title": "Stage 4 - Lambda / Sigma Sweep",
            "base": "bilstm_d1_s4",
            "jobs": [
                {"name": "bilstm_d1_s4_lam08_sig015", "params": {"lambda_coef": 0.8, "sigma_coef": 0.15}},
                {"name": "bilstm_d1_s4_lam09_sig01", "params": {"lambda_coef": 0.9, "sigma_coef": 0.1}},
                {"name": "bilstm_d1_s4_lam10_sig005", "params": {"lambda_coef": 1.0, "sigma_coef": 0.05}},
            ],
        },
        {
            "id": "stage5",
            "title": "Stage 5 - Batch Size Sweep",
            "base": "bilstm_d1_s5",
            "jobs": [
                {"name": "bilstm_d1_s5_b32", "params": {"batch": 32}},
                {"name": "bilstm_d1_s5_b64", "params": {"batch": 64}},
                {"name": "bilstm_d1_s5_b96", "params": {"batch": 96}},
            ],
        },
        {
            "id": "stage6",
            "title": "Stage 6 - Epoch Ceiling Sweep",
            "base": "bilstm_d1_s6",
            "jobs": [
                {"name": "bilstm_d1_s6_e30", "params": {"epochs": 30}},
                {"name": "bilstm_d1_s6_e40", "params": {"epochs": 40}},
                {"name": "bilstm_d1_s6_e50", "params": {"epochs": 50}},
            ],
        },
        {
            "id": "stage7",
            "title": "Stage 7 - Regularization Check",
            "base": "bilstm_d1_s7",
            "jobs": [
                {"name": "bilstm_d1_s7_base", "params": {"dropout": 0.0, "l2_reg": 0.0}},
                {"name": "bilstm_d1_s7_drop005", "params": {"dropout": 0.05, "l2_reg": 0.0}},
                {"name": "bilstm_d1_s7_l2_1e5", "params": {"dropout": 0.0, "l2_reg": 1e-5}},
            ],
        },
    ]

    base_params = {k: json.loads(base_cfg.read_text(encoding="utf-8")).get(k) for k in SUMMARY_COLUMNS}
    base_params = {k: v for k, v in base_params.items() if v is not None}
    for key, value in DEFAULT_SUMMARY_VALUES.items():
        base_params.setdefault(key, value)
    resume_baseline = load_resume_baseline(out_dir, args.min_stage, base_params)

    stages = [stage for idx, stage in enumerate(stages, start=1) if idx >= args.min_stage]
    if args.max_stage is not None:
        stages = stages[: max(0, args.max_stage - args.min_stage + 1)]

    original_cfg = json.loads(base_cfg.read_text(encoding="utf-8"))
    original_cfg.update(resume_baseline)
    temp_cfg = out_dir / f"_resume_config_tmp_{uuid.uuid4().hex}.json"
    temp_cfg.parent.mkdir(parents=True, exist_ok=True)
    temp_cfg.write_text(json.dumps(original_cfg, indent=2), encoding="utf-8")
    html_sections = run_stages(stages, temp_cfg, out_dir, args.skip_training)
    temp_cfg.unlink(missing_ok=True)
    html = [
        "<html><head><meta charset='UTF-8'><title>EUR/USD 1D FLF-BiLSTM Pipeline Summary</title>"
        "<style>body{font-family:Arial;margin:20px;} table{border-collapse:collapse;}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;} th{background:#f0f0f0;}</style>"
        "</head><body><h1>EUR/USD 1D FLF-BiLSTM Pipeline Summary (Stage 1-7)</h1>"
    ]
    html.extend(html_sections)
    html.append("</body></html>")
    out_file = out_dir / "eurusd_bilstm_d1_pipeline_summary_v01.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text("".join(html), encoding="utf-8")
    print(f"Pipeline summary written to {out_file.resolve()}")


if __name__ == "__main__":
    main()
