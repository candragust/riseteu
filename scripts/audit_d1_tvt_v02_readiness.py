#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/home/hduser/jupyter/gust/RisetEU")
TF_PYTHON = Path("/home/hduser/miniconda3/envs/test/bin/python")
REPORT_PYTHON = "python"
OUT_DIR = PROJECT_ROOT / "results" / "d1_ohlc" / "tvt_v02"


def check_path(path: Path, kind: str = "file") -> dict:
    exists = path.is_dir() if kind == "dir" else path.exists()
    return {"path": str(path), "kind": kind, "exists": bool(exists)}


def check_import(python_bin: str | Path, module: str) -> bool:
    code = f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('{module}') else 1)"
    try:
        result = subprocess.run([str(python_bin), "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def summarize_schedule(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "path": str(path),
        "folds": df["fold"].astype(int).tolist(),
        "rows": int(len(df)),
        "test_samples_total": int(df["test_samples"].sum()),
        "validation_samples_total": int(df["validation_samples"].sum()),
        "train_core_samples_min": int(df["train_core_samples"].min()),
        "train_core_samples_max": int(df["train_core_samples"].max()),
    }


def output_status() -> dict:
    groups = {
        "lstm_tuning": PROJECT_ROOT / "FLF_LSTM/results/tvt_v02/d1_ohlc/lstm_pipeline_d1_tuning_last6",
        "bilstm_tuning": PROJECT_ROOT / "FLF_BILSTM/results/tvt_v02/d1_ohlc/bilstm_pipeline_d1_tuning_last6",
        "lstm_eval": PROJECT_ROOT / "FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3",
        "bilstm_eval": PROJECT_ROOT / "FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3",
        "arima_eval": PROJECT_ROOT / "Arima/result/tvt_v02/d1_evaluation_last3",
        "comparison": PROJECT_ROOT / "comparison/tvt_v02/d1",
    }
    out = {}
    for name, path in groups.items():
        files = sorted(p.name for p in path.glob("*")) if path.exists() else []
        out[name] = {
            "path": str(path),
            "exists": path.exists(),
            "file_count": len(files),
            "key_files": [f for f in files if f in {
                "best_progression.csv",
                "audit_finalize_report.md",
                "rolling_tvt_summary.csv",
                "rolling_fixed_summary.csv",
                "mae_atr_fold_allfull.html",
                "index.html",
                "comparison_models_d1_tvt_v02_last3_v01.html",
            }],
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_root = PROJECT_ROOT / "results/splits/tvt_v02/d1"

    required = [
        check_path(split_root, "dir"),
        check_path(split_root / "fold_schedule_tuning.csv"),
        check_path(split_root / "fold_schedule_evaluation.csv"),
        check_path(PROJECT_ROOT / "FLF_LSTM/eurusd_lstm_d1_pipeline_runner_tvt_v02.py"),
        check_path(PROJECT_ROOT / "FLF_BILSTM/eurusd_bilstm_d1_pipeline_runner_tvt_v02.py"),
        check_path(PROJECT_ROOT / "FLF_LSTM/rolling_tvt_lstm_d1_runner_v02.py"),
        check_path(PROJECT_ROOT / "FLF_BILSTM/rolling_tvt_bilstm_d1_runner_v02.py"),
        check_path(PROJECT_ROOT / "scripts/finalize_tvt_v02.py"),
        check_path(PROJECT_ROOT / "Arima/arima_tvt_runner_v02.py"),
        check_path(PROJECT_ROOT / "scripts/run_d1_tvt_v02_end_to_end.sh"),
        check_path(PROJECT_ROOT / "scripts/launch_d1_tvt_v02_end_to_end_detached.sh"),
        check_path(PROJECT_ROOT / "scripts/generate_d1_tvt_v02_reports.sh"),
        check_path(PROJECT_ROOT / "FLF_LSTM/configs/d1_ohlc/lstm_flf_config_d1_tvt_v02_base.json"),
        check_path(PROJECT_ROOT / "FLF_BILSTM/configs/d1_ohlc/bilstm_flf_config_d1_tvt_v02_base.json"),
        check_path(PROJECT_ROOT / "Arima/arima_baseline_config_d1_ohlc_v01.json"),
    ]

    payload = {
        "scope": "D1 TVT v02 detached readiness",
        "project_root": str(PROJECT_ROOT),
        "commands": {
            "full_detached": "scripts/launch_d1_tvt_v02_end_to_end_detached.sh full 1 7",
            "resume_detached": "scripts/launch_d1_tvt_v02_end_to_end_detached.sh full 4 7",
            "rebuild_from_existing": "scripts/launch_d1_tvt_v02_end_to_end_detached.sh skip 1 7",
            "status": "scripts/status_tvt_v02.sh",
        },
        "dependencies": {
            "screen": bool(shutil.which("screen")),
            "tf_python": str(TF_PYTHON),
            "tf_python_exists": TF_PYTHON.exists(),
            "tf_python_modules": {
                module: check_import(TF_PYTHON, module)
                for module in ["tensorflow", "statsmodels", "pandas", "numpy"]
            },
            "report_python": REPORT_PYTHON,
            "report_python_modules": {
                module: check_import(REPORT_PYTHON, module)
                for module in ["plotly", "pandas", "numpy", "statsmodels"]
            },
        },
        "required_paths": required,
        "schedules": {
            "tuning": summarize_schedule(split_root / "fold_schedule_tuning.csv"),
            "evaluation": summarize_schedule(split_root / "fold_schedule_evaluation.csv"),
        },
        "outputs": output_status(),
    }
    payload["ready_to_launch"] = (
        all(item["exists"] for item in required)
        and payload["dependencies"]["screen"]
        and payload["dependencies"]["tf_python_exists"]
        and all(payload["dependencies"]["tf_python_modules"].values())
        and all(payload["dependencies"]["report_python_modules"].values())
    )
    payload["ready"] = payload["ready_to_launch"]
    payload["results_complete"] = all(
        status["exists"] and status["file_count"] > 0
        for status in payload["outputs"].values()
    )

    json_path = OUT_DIR / "audit_d1_tvt_v02_readiness.json"
    md_path = OUT_DIR / "audit_d1_tvt_v02_readiness.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Audit Readiness D1 TVT v02",
        "",
        f"- Ready to launch detached: `{payload['ready_to_launch']}`",
        f"- Results complete: `{payload['results_complete']}`",
        "- Note: readiness checks environment, split files, runners, launchers, and dependencies. "
        "Output status below shows whether D1 artifacts have already been produced.",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- Full detached: `{payload['commands']['full_detached']}`",
        f"- Rebuild existing: `{payload['commands']['rebuild_from_existing']}`",
        "",
        "## Dependencies",
        "",
        f"- `screen`: `{payload['dependencies']['screen']}`",
        f"- TF python: `{TF_PYTHON}`",
        f"- TF modules: `{payload['dependencies']['tf_python_modules']}`",
        f"- Report python: `{REPORT_PYTHON}`",
        f"- Report modules: `{payload['dependencies']['report_python_modules']}`",
        "",
        "## Schedules",
        "",
        f"- Tuning folds: `{payload['schedules']['tuning']['folds']}`",
        f"- Evaluation folds: `{payload['schedules']['evaluation']['folds']}`",
        "",
        "## Current Output Status",
        "",
        "| Group | Exists | File count | Key files |",
        "| --- | ---: | ---: | --- |",
    ]
    for group, status in payload["outputs"].items():
        lines.append(
            f"| {group} | {status['exists']} | {status['file_count']} | "
            f"{', '.join(status['key_files']) if status['key_files'] else '-'} |"
        )
    lines.extend(["", "## Required Paths", "", "| Path | Exists |", "| --- | ---: |"])
    for item in required:
        lines.append(f"| `{item['path']}` | {item['exists']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Readiness JSON: {json_path}")
    print(f"Readiness MD  : {md_path}")
    print(f"Ready to launch: {payload['ready_to_launch']}")
    print(f"Results complete: {payload['results_complete']}")


if __name__ == "__main__":
    main()
