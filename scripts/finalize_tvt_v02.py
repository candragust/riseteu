#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/home/hduser/jupyter/gust/RisetEU")
DEFAULT_PYTHON_BIN = "/home/hduser/miniconda3/envs/test/bin/python"
PARAM_COLUMNS = ["window", "units", "activation", "lr", "epochs", "lambda_coef", "sigma_coef", "batch"]
OPTIONAL_PARAM_COLUMNS = ["dropout", "recurrent_dropout", "l2_reg"]


def build_spec(timeframe: str, model: str):
    if timeframe == "h4" and model == "lstm":
        return {
            "label": "TVT v02 H4 FLF-LSTM",
            "expected_stages": 6,
            "tuning_out_dir": PROJECT_ROOT / "FLF_LSTM" / "results" / "tvt_v02" / "lstm_pipeline_h4_tuning_last6",
            "base_config": PROJECT_ROOT / "FLF_LSTM" / "lstm_flf_config_h4_tvt_v02_base.json",
            "final_config": PROJECT_ROOT / "FLF_LSTM" / "lstm_flf_config_h4_tvt_v02_best.json",
            "eval_runner": PROJECT_ROOT / "FLF_LSTM" / "rolling_tvt_lstm_runner_v02.py",
            "eval_out_dir": PROJECT_ROOT / "FLF_LSTM" / "results" / "tvt_v02" / "h4_evaluation_last3",
            "schedule_eval": PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "h4" / "fold_schedule_evaluation.csv",
            "fold_root": PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "h4",
        }
    if timeframe == "h4" and model == "bilstm":
        return {
            "label": "TVT v02 H4 FLF-BiLSTM",
            "expected_stages": 6,
            "tuning_out_dir": PROJECT_ROOT / "FLF_BILSTM" / "results" / "tvt_v02" / "bilstm_pipeline_h4_tuning_last6",
            "base_config": PROJECT_ROOT / "FLF_BILSTM" / "bilstm_flf_config_h4_tvt_v02_base.json",
            "final_config": PROJECT_ROOT / "FLF_BILSTM" / "bilstm_flf_config_h4_tvt_v02_best.json",
            "eval_runner": PROJECT_ROOT / "FLF_BILSTM" / "rolling_tvt_bilstm_runner_v02.py",
            "eval_out_dir": PROJECT_ROOT / "FLF_BILSTM" / "results" / "tvt_v02" / "h4_evaluation_last3",
            "schedule_eval": PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "h4" / "fold_schedule_evaluation.csv",
            "fold_root": PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "h4",
        }
    if timeframe == "d1" and model == "lstm":
        return {
            "label": "TVT v02 D1 FLF-LSTM",
            "expected_stages": 7,
            "tuning_out_dir": PROJECT_ROOT / "FLF_LSTM" / "results" / "tvt_v02" / "d1_ohlc" / "lstm_pipeline_d1_tuning_last6",
            "base_config": PROJECT_ROOT / "FLF_LSTM" / "configs" / "d1_ohlc" / "lstm_flf_config_d1_tvt_v02_base.json",
            "final_config": PROJECT_ROOT / "FLF_LSTM" / "configs" / "d1_ohlc" / "lstm_flf_config_d1_tvt_v02_best.json",
            "eval_runner": PROJECT_ROOT / "FLF_LSTM" / "rolling_tvt_lstm_d1_runner_v02.py",
            "eval_out_dir": PROJECT_ROOT / "FLF_LSTM" / "results" / "tvt_v02" / "d1_ohlc" / "d1_evaluation_last3",
            "schedule_eval": PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "d1" / "fold_schedule_evaluation.csv",
            "fold_root": PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "d1",
        }
    if timeframe == "d1" and model == "bilstm":
        return {
            "label": "TVT v02 D1 FLF-BiLSTM",
            "expected_stages": 7,
            "tuning_out_dir": PROJECT_ROOT / "FLF_BILSTM" / "results" / "tvt_v02" / "d1_ohlc" / "bilstm_pipeline_d1_tuning_last6",
            "base_config": PROJECT_ROOT / "FLF_BILSTM" / "configs" / "d1_ohlc" / "bilstm_flf_config_d1_tvt_v02_base.json",
            "final_config": PROJECT_ROOT / "FLF_BILSTM" / "configs" / "d1_ohlc" / "bilstm_flf_config_d1_tvt_v02_best.json",
            "eval_runner": PROJECT_ROOT / "FLF_BILSTM" / "rolling_tvt_bilstm_d1_runner_v02.py",
            "eval_out_dir": PROJECT_ROOT / "FLF_BILSTM" / "results" / "tvt_v02" / "d1_ohlc" / "d1_evaluation_last3",
            "schedule_eval": PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "d1" / "fold_schedule_evaluation.csv",
            "fold_root": PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "d1",
        }
    raise ValueError(f"Unsupported combination: {timeframe} {model}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit TVT v02 tuning results, freeze the best config, and launch final evaluation."
    )
    parser.add_argument("timeframe", choices=["h4", "d1"])
    parser.add_argument("model", choices=["lstm", "bilstm"])
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def coerce_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def audit_and_select(spec):
    tuning_out_dir = spec["tuning_out_dir"]
    expected_stages = spec["expected_stages"]
    if not tuning_out_dir.exists():
        raise FileNotFoundError(f"Tuning output directory not found: {tuning_out_dir}")

    missing = []
    for stage_idx in range(1, expected_stages + 1):
        stage_path = tuning_out_dir / f"stage{stage_idx}_summary.csv"
        if not stage_path.exists():
            missing.append(str(stage_path))
    best_progression_path = tuning_out_dir / "best_progression.csv"
    if not best_progression_path.exists():
        missing.append(str(best_progression_path))
    if missing:
        raise RuntimeError("Missing required audit artifacts:\n" + "\n".join(missing))

    progress_df = pd.read_csv(best_progression_path)
    if len(progress_df) != expected_stages:
        raise RuntimeError(
            f"best_progression.csv length mismatch: expected {expected_stages}, got {len(progress_df)}"
        )

    selected_mae = progress_df["selected_mae_avg_pips"].astype(float)
    if not selected_mae.is_monotonic_decreasing:
        raise RuntimeError("Selected MAE is not monotonic non-increasing across stages.")

    final_row = progress_df.iloc[-1]
    final_params = {}
    for key in PARAM_COLUMNS + OPTIONAL_PARAM_COLUMNS:
        if key in final_row:
            value = coerce_value(final_row[key])
            if value is not None:
                final_params[key] = value

    audit_rows = []
    for stage_idx in range(1, expected_stages + 1):
        stage_path = tuning_out_dir / f"stage{stage_idx}_summary.csv"
        stage_df = pd.read_csv(stage_path).sort_values("mae_avg_pips").reset_index(drop=True)
        if stage_df.empty:
            raise RuntimeError(f"Stage summary is empty: {stage_path}")
        best_row = stage_df.iloc[0]
        audit_rows.append(
            {
                "stage": int(stage_idx),
                "stage_summary": str(stage_path),
                "stage_best_name": str(best_row["name"]),
                "stage_best_mae_avg_pips": float(best_row["mae_avg_pips"]),
                "selected_name_after_stage": str(progress_df.iloc[stage_idx - 1]["selected_name"]),
                "selected_mae_after_stage": float(progress_df.iloc[stage_idx - 1]["selected_mae_avg_pips"]),
                "selected_from": str(progress_df.iloc[stage_idx - 1]["selected_from"]),
            }
        )

    return progress_df, audit_rows, final_row, final_params


def write_audit_artifacts(spec, audit_rows, final_row, final_params):
    tuning_out_dir = spec["tuning_out_dir"]
    audit_json = tuning_out_dir / "audit_finalize_report.json"
    audit_md = tuning_out_dir / "audit_finalize_report.md"

    payload = {
        "label": spec["label"],
        "final_selected_name": str(final_row["selected_name"]),
        "final_selected_mae_avg_pips": float(final_row["selected_mae_avg_pips"]),
        "final_selected_from": str(final_row["selected_from"]),
        "final_params": final_params,
        "stages": audit_rows,
    }
    audit_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Audit Finalisasi {spec['label']}",
        "",
        f"- Final selected: `{final_row['selected_name']}`",
        f"- Final mean `MAE avg`: `{float(final_row['selected_mae_avg_pips']):.6f}` pips",
        f"- Final selected_from: `{final_row['selected_from']}`",
        "",
        "## Parameter Final",
        "",
    ]
    for key in PARAM_COLUMNS + OPTIONAL_PARAM_COLUMNS:
        if key in final_params:
            lines.append(f"- `{key}` = `{final_params[key]}`")
    lines.extend(
        [
            "",
            "## Ringkasan Stage",
            "",
            "| Stage | Stage best MAE | Selected after stage | Source |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in audit_rows:
        lines.append(
            f"| {row['stage']} | {row['stage_best_mae_avg_pips']:.6f} | "
            f"{row['selected_mae_after_stage']:.6f} | {row['selected_from']} |"
        )
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit_json, audit_md


def freeze_config(spec, final_params):
    base_data = json.loads(spec["base_config"].read_text(encoding="utf-8"))
    base_data.update(final_params)
    spec["final_config"].write_text(json.dumps(base_data, indent=2), encoding="utf-8")
    return spec["final_config"]


def run_evaluation(spec, python_bin: str):
    spec["eval_out_dir"].mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(spec["eval_runner"]),
        "--schedule",
        str(spec["schedule_eval"]),
        "--fold-root",
        str(spec["fold_root"]),
        "--base-config",
        str(spec["final_config"]),
        "--out-dir",
        str(spec["eval_out_dir"]),
        "--python-bin",
        python_bin,
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return cmd


def main():
    args = parse_args()
    spec = build_spec(args.timeframe, args.model)
    progress_df, audit_rows, final_row, final_params = audit_and_select(spec)
    audit_json, audit_md = write_audit_artifacts(spec, audit_rows, final_row, final_params)
    final_config = freeze_config(spec, final_params)

    print(f"[AUDIT] OK: {spec['label']}")
    print(f"[AUDIT] Final selected: {final_row['selected_name']}")
    print(f"[AUDIT] Final MAE avg : {float(final_row['selected_mae_avg_pips']):.6f} pips")
    print(f"[AUDIT] Audit JSON    : {audit_json}")
    print(f"[AUDIT] Audit MD      : {audit_md}")
    print(f"[AUDIT] Frozen config : {final_config}")

    if args.skip_eval:
        return

    cmd = run_evaluation(spec, args.python_bin)
    print("[EVAL] Completed.")
    print("[EVAL] Command:", " ".join(cmd))
    print(f"[EVAL] Output dir: {spec['eval_out_dir']}")


if __name__ == "__main__":
    main()
