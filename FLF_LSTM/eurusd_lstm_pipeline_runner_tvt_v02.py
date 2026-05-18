import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ROLLING_RUNNER = ROOT / "rolling_tvt_lstm_runner_v02.py"
DEFAULT_BASE_CONFIG = ROOT / "lstm_flf_config_h4_tvt_v02_base.json"
DEFAULT_SCHEDULE = PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "h4" / "fold_schedule_tuning.csv"
DEFAULT_FOLD_ROOT = PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "h4"
DEFAULT_OUT_DIR = ROOT / "results" / "tvt_v02" / "lstm_pipeline_h4_tuning_last6"
DEFAULT_PYTHON_BIN = "/home/hduser/miniconda3/envs/test/bin/python"

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
    parser = argparse.ArgumentParser(description="FLF-LSTM 4H tuning pipeline on TVT split.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help="Baseline config JSON.")
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE), help="Fold schedule CSV for tuning.")
    parser.add_argument("--fold-root", default=str(DEFAULT_FOLD_ROOT), help="Root directory of foldXX split folders.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Where to store pipeline outputs.")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN, help="Python interpreter for TensorFlow jobs.")
    parser.add_argument("--skip-training", action="store_true", help="Reuse existing per-job summaries.")
    parser.add_argument("--min-stage", type=int, default=1, help="Run starting from this stage number.")
    parser.add_argument("--max-stage", type=int, default=None, help="Run up to this stage number.")
    return parser.parse_args()


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
        parts.append(f"{PARAM_PREFIX[key]}{format_param_value(params[key])}")
    return f"{base}_{'_'.join(parts) if parts else 'base'}"


def write_temp_config(base_cfg: Path, params: Dict, out_dir: Path, job_name: str):
    cfg = json.loads(base_cfg.read_text(encoding="utf-8"))
    cfg.update(params)
    cfg["model_out"] = None
    temp_cfg = out_dir / f"_{job_name}_{uuid.uuid4().hex}.json"
    temp_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return temp_cfg


def load_resume_state(out_dir: Path, min_stage: int, fallback: Dict):
    baseline = fallback.copy()
    incumbent_row = None
    if min_stage <= 1:
        return baseline, incumbent_row

    progress_path = out_dir / "best_progression.csv"
    if not progress_path.exists():
        return baseline, incumbent_row

    df = pd.read_csv(progress_path)
    target_idx = min_stage - 2
    if target_idx < 0 or target_idx >= len(df):
        return baseline, incumbent_row

    row = df.iloc[target_idx]
    for key in PARAM_COLUMNS:
        if key in row and pd.notna(row[key]):
            value = row[key]
            baseline[key] = value.item() if hasattr(value, "item") else value

    selected_name = row["selected_name"] if "selected_name" in row and pd.notna(row["selected_name"]) else row.get("best_name")
    selected_mae = (
        row["selected_mae_avg_pips"]
        if "selected_mae_avg_pips" in row and pd.notna(row["selected_mae_avg_pips"])
        else row.get("stage_best_mae_avg_pips")
    )
    if selected_name is not None and selected_mae is not None and pd.notna(selected_mae):
        incumbent_row = {
            "name": selected_name,
            "mae_avg_pips": float(selected_mae),
        }
    return baseline, incumbent_row


def aggregate_job_summary(summary_path: Path):
    df = pd.read_csv(summary_path)
    row = {
        "fold_count": int(len(df)),
        "mae_avg_pips": float(df["mae_avg_pips"].mean()),
        "mae_open_pips": float(df["mae_open_pips"].mean()),
        "mae_high_pips": float(df["mae_high_pips"].mean()),
        "mae_low_pips": float(df["mae_low_pips"].mean()),
        "mae_close_pips": float(df["mae_close_pips"].mean()),
        "mae_avg_std_pips": float(df["mae_avg_pips"].std(ddof=0)),
    }
    return row


def run_job(job_name: str, params: Dict, base_cfg: Path, schedule: Path, fold_root: Path, out_dir: Path, python_bin: str):
    job_dir = out_dir / job_name
    summary_path = job_dir / "rolling_tvt_summary.csv"
    start_time = time.perf_counter()
    temp_cfg = write_temp_config(base_cfg, params, out_dir, job_name)
    cmd = [
        sys.executable,
        str(ROLLING_RUNNER),
        "--schedule",
        str(schedule),
        "--fold-root",
        str(fold_root),
        "--base-config",
        str(temp_cfg),
        "--python-bin",
        python_bin,
        "--out-dir",
        str(job_dir),
    ]
    print(f"[PIPELINE] Running {job_name}")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    finally:
        temp_cfg.unlink(missing_ok=True)
    return summary_path, time.perf_counter() - start_time


def run_stages(
    stages: List[Dict],
    base_cfg: Path,
    schedule: Path,
    fold_root: Path,
    out_dir: Path,
    python_bin: str,
    skip_training: bool,
    resume_incumbent: Dict | None = None,
):
    cfg_data = json.loads(base_cfg.read_text(encoding="utf-8"))
    baseline = {k: cfg_data.get(k) for k in PARAM_COLUMNS if cfg_data.get(k) is not None}
    incumbent_row = resume_incumbent.copy() if resume_incumbent is not None else None
    incumbent_params = baseline.copy() if incumbent_row is not None else None
    html_sections = []
    best_history = []

    for stage in stages:
        rows = []
        best_row = None
        best_params = None

        for job in stage["jobs"]:
            params = baseline.copy()
            params.update(job.get("params", {}))
            job_dir = out_dir / job["name"]
            summary_path = job_dir / "rolling_tvt_summary.csv"
            duration = 0.0
            if not skip_training or not summary_path.exists():
                summary_path, duration = run_job(job["name"], params, base_cfg, schedule, fold_root, out_dir, python_bin)

            metrics = aggregate_job_summary(summary_path)
            literal_name = build_name(stage["base"], params)
            row = {
                "name": literal_name,
                "job": job["name"],
                **metrics,
                "summary_csv": str(summary_path),
                "train_seconds": duration,
            }
            for col in PARAM_COLUMNS:
                row[col] = params.get(col)
            rows.append(row)

            if best_row is None or row["mae_avg_pips"] < best_row["mae_avg_pips"]:
                best_row = row
                best_params = params.copy()

        df = pd.DataFrame(rows).sort_values("mae_avg_pips").reset_index(drop=True)
        stage_path = out_dir / f"{stage['id']}_summary.csv"
        df.to_csv(stage_path, index=False)

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
            f"(mean MAE_avg_pips={best_row['mae_avg_pips']:.4f}, folds={best_row['fold_count']})</p>"
            f"<p>Selected baseline after stage: <strong>{selected_row['name']}</strong> "
            f"(mean MAE_avg_pips={selected_row['mae_avg_pips']:.4f}, source={selected_from})</p>"
            f"{df.fillna('').to_html(index=False, float_format='%.4f')}"
        )
        best_history.append(
            {
                "stage": stage["title"],
                "best_name": selected_row["name"],
                "stage_best_name": best_row["name"],
                "stage_best_mae_avg_pips": best_row["mae_avg_pips"],
                "selected_name": selected_row["name"],
                "selected_mae_avg_pips": selected_row["mae_avg_pips"],
                "selected_from": selected_from,
                **{k: v for k, v in selected_params.items() if k in PARAM_COLUMNS},
            }
        )
        baseline = selected_params.copy()

    pd.DataFrame(best_history).to_csv(out_dir / "best_progression.csv", index=False)
    return html_sections


def main():
    args = parse_args()
    base_cfg = Path(args.base_config)
    schedule = Path(args.schedule)
    fold_root = Path(args.fold_root)
    out_dir = Path(args.out_dir)

    for path in [base_cfg, schedule, fold_root, ROLLING_RUNNER]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    base_cfg_data = json.loads(base_cfg.read_text(encoding="utf-8"))
    base_params = {k: base_cfg_data.get(k) for k in PARAM_COLUMNS if base_cfg_data.get(k) is not None}
    resume_baseline, resume_incumbent = load_resume_state(out_dir, args.min_stage, base_params)

    stages: List[Dict] = [
        {
            "id": "stage1",
            "title": "Stage 1 - Window Sweep",
            "base": "lstm_tvt_s1",
            "jobs": [
                {"name": "lstm_tvt_s1_w10", "params": {"window": 10}},
                {"name": "lstm_tvt_s1_w12", "params": {"window": 12}},
                {"name": "lstm_tvt_s1_w14", "params": {"window": 14}},
            ],
        },
        {
            "id": "stage2",
            "title": "Stage 2 - Units Sweep",
            "base": "lstm_tvt_s2",
            "jobs": [
                {"name": "lstm_tvt_s2_h224", "params": {"units": 224}},
                {"name": "lstm_tvt_s2_h256", "params": {"units": 256}},
            ],
        },
        {
            "id": "stage3",
            "title": "Stage 3 - Learning Rate Sweep",
            "base": "lstm_tvt_s3",
            "jobs": [
                {"name": "lstm_tvt_s3_relu_lr5e4", "params": {"activation": "relu", "lr": 5e-4}},
                {"name": "lstm_tvt_s3_relu_lr7e4", "params": {"activation": "relu", "lr": 7e-4}},
                {"name": "lstm_tvt_s3_relu_lr9e4", "params": {"activation": "relu", "lr": 9e-4}},
            ],
        },
        {
            "id": "stage4",
            "title": "Stage 4 - Lambda / Sigma Sweep",
            "base": "lstm_tvt_s4",
            "jobs": [
                {"name": "lstm_tvt_s4_lam08_sig015", "params": {"lambda_coef": 0.8, "sigma_coef": 0.15}},
                {"name": "lstm_tvt_s4_lam09_sig01", "params": {"lambda_coef": 0.9, "sigma_coef": 0.1}},
                {"name": "lstm_tvt_s4_lam08_sig01", "params": {"lambda_coef": 0.8, "sigma_coef": 0.1}},
            ],
        },
        {
            "id": "stage5",
            "title": "Stage 5 - Batch Size Sweep",
            "base": "lstm_tvt_s5",
            "jobs": [
                {"name": "lstm_tvt_s5_batch96", "params": {"batch": 96}},
                {"name": "lstm_tvt_s5_batch128", "params": {"batch": 128}},
            ],
        },
        {
            "id": "stage6",
            "title": "Stage 6 - Epoch Sweep",
            "base": "lstm_tvt_s6",
            "jobs": [
                {"name": "lstm_tvt_s6_e40", "params": {"epochs": 40}},
                {"name": "lstm_tvt_s6_e50", "params": {"epochs": 50}},
                {"name": "lstm_tvt_s6_e60", "params": {"epochs": 60}},
            ],
        },
    ]

    if args.min_stage < 1:
        raise ValueError("--min-stage must be >= 1")
    if args.max_stage is not None and args.max_stage < args.min_stage:
        raise ValueError("--max-stage must be >= --min-stage")

    if args.max_stage is not None:
        stages = stages[: args.max_stage]
    stages = stages[args.min_stage - 1 :]

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_cfg = out_dir / f"_resume_config_tmp_{uuid.uuid4().hex}.json"
    base_cfg_data.update(resume_baseline)
    temp_cfg.write_text(json.dumps(base_cfg_data, indent=2), encoding="utf-8")
    try:
        html_sections = run_stages(
            stages,
            temp_cfg,
            schedule,
            fold_root,
            out_dir,
            args.python_bin,
            args.skip_training,
            resume_incumbent=resume_incumbent,
        )
    finally:
        temp_cfg.unlink(missing_ok=True)
    html = [
        "<html><head><meta charset='UTF-8'><title>EURUSD FLF-LSTM Pipeline Summary (TVT)</title>"
        "<style>body{font-family:Arial;margin:20px;} table{border-collapse:collapse;}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;} th{background:#f0f0f0;}</style>"
        "</head><body><h1>EURUSD FLF-LSTM Pipeline Summary (4H, tuning folds 13-18) (TVT)</h1>"
    ]
    html.extend(html_sections)
    html.append("</body></html>")
    out_file = out_dir / "eurusd_lstm_pipeline_tvt_summary.html"
    out_file.write_text("".join(html), encoding="utf-8")
    print(f"Pipeline summary written to {out_file.resolve()}")


if __name__ == "__main__":
    main()
