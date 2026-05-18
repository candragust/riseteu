#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SCHEDULE = PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "d1" / "fold_schedule_evaluation.csv"
DEFAULT_FOLD_ROOT = PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "d1"
DEFAULT_BASE_CONFIG = SCRIPT_DIR / "arima_baseline_config_d1_ohlc_v01.json"
DEFAULT_OUT_DIR = SCRIPT_DIR / "result" / "tvt_v02" / "d1_evaluation_last3"
EXPERIMENT_SCRIPT = SCRIPT_DIR / "arima_ohlc_experiment.py"


@dataclass
class FoldPaths:
    fold: int
    combined_data: Path
    metadata_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ARIMA OHLC baseline on pre-split TVT v02 folds."
    )
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    parser.add_argument("--fold-root", default=str(DEFAULT_FOLD_ROOT))
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--folds", default=None, help="Optional comma-separated fold list, e.g. 19,20,21.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-test-steps", type=int, default=None)
    return parser.parse_args()


def parse_fold_selection(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(token.strip()) for token in raw.split(",") if token.strip()}


def load_selected_schedule(schedule_path: Path, selected_folds: set[int] | None) -> pd.DataFrame:
    df = pd.read_csv(schedule_path)
    if "fold" not in df.columns:
        raise ValueError(f"Schedule file has no 'fold' column: {schedule_path}")
    df["fold"] = df["fold"].astype(int)
    if selected_folds:
        df = df[df["fold"].isin(selected_folds)].copy()
    if df.empty:
        raise RuntimeError("No folds selected from schedule.")
    return df.sort_values("fold").reset_index(drop=True)


def resolve_fold_paths(fold_root: Path, fold: int) -> FoldPaths:
    fold_dir = fold_root / f"fold{fold:02d}"
    combined_data = fold_dir / "combined.csv"
    metadata_path = fold_dir / "metadata.json"
    for path in (combined_data, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing fold artifact: {path}")
    return FoldPaths(fold=fold, combined_data=combined_data, metadata_path=metadata_path)


def load_metadata(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def compute_split_ratio(metadata: dict) -> float:
    train_val = int(metadata["train_core_samples"]) + int(metadata["validation_samples"])
    total = train_val + int(metadata["test_samples"])
    return train_val / total


def compute_mae_metrics(preds_path: Path) -> dict[str, float | int]:
    preds = pd.read_csv(preds_path)
    metrics = {}
    for field in ["open", "high", "low", "close"]:
        mae = (preds[f"pred_{field}"] - preds[f"true_{field}"]).abs().mean() * 10000.0
        metrics[f"mae_{field}_pips"] = float(mae)
    metrics["mae_avg_pips"] = float(
        sum(metrics[f"mae_{field}_pips"] for field in ["open", "high", "low", "close"]) / 4.0
    )
    metrics["test_samples_predicted"] = int(len(preds))
    return metrics


def run_fold(args: argparse.Namespace, fold_paths: FoldPaths, metadata: dict, out_dir: Path) -> tuple[Path, Path, Path]:
    fold_data = out_dir / f"fold{fold_paths.fold:02d}_data.csv"
    preds_path = out_dir / f"fold{fold_paths.fold:02d}_preds.csv"
    summary_path = out_dir / f"fold{fold_paths.fold:02d}_summary.json"
    split_ratio = compute_split_ratio(metadata)

    # Copy the exact TVT combined fold into the ARIMA result folder for auditability.
    fold_df = pd.read_csv(fold_paths.combined_data)
    fold_df.to_csv(fold_data, index=False)
    train_samples = int(metadata["train_core_samples"]) + int(metadata["validation_samples"])

    cmd = [
        args.python_bin,
        str(EXPERIMENT_SCRIPT),
        "--config",
        str(args.base_config),
        "--data",
        str(fold_data),
        "--split",
        f"{split_ratio:.10f}",
        "--train-samples",
        str(train_samples),
        "--out",
        str(preds_path),
        "--summary-out",
        str(summary_path),
    ]
    if args.max_test_steps is not None:
        cmd.extend(["--max-test-steps", str(args.max_test_steps)])

    print(
        f"[ARIMA TVT FOLD {fold_paths.fold:02d}] "
        f"train_core={metadata['train_core_start']} -> {metadata['train_core_end']} | "
        f"validation={metadata['validation_start']} -> {metadata['validation_end']} | "
        f"test={metadata['test_start']} -> {metadata['test_end']} | split={split_ratio:.10f}"
        f" | train_samples={train_samples}"
    )
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return fold_data, preds_path, summary_path


def main() -> None:
    args = parse_args()
    args.schedule = Path(args.schedule)
    args.fold_root = Path(args.fold_root)
    args.base_config = Path(args.base_config)
    out_dir = Path(args.out_dir)

    for path in (args.schedule, args.fold_root, args.base_config, EXPERIMENT_SCRIPT):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    selected_folds = parse_fold_selection(args.folds)
    schedule_df = load_selected_schedule(args.schedule, selected_folds)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schedule": str(args.schedule.resolve()),
        "fold_root": str(args.fold_root.resolve()),
        "base_config": str(args.base_config.resolve()),
        "python_bin": args.python_bin,
        "selected_folds": sorted(selected_folds) if selected_folds else schedule_df["fold"].tolist(),
        "experiment_script": str(EXPERIMENT_SCRIPT.resolve()),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    schedule_df.to_csv(out_dir / "selected_schedule.csv", index=False)

    if args.dry_run:
        print(f"Saved selected schedule to {(out_dir / 'selected_schedule.csv').resolve()}")
        return

    summary_rows = []
    for row in schedule_df.to_dict(orient="records"):
        fold = int(row["fold"])
        fold_paths = resolve_fold_paths(args.fold_root, fold)
        metadata = load_metadata(fold_paths.metadata_path)
        _, preds_path, _ = run_fold(args, fold_paths, metadata, out_dir)
        metrics = compute_mae_metrics(preds_path)
        if int(metrics["test_samples_predicted"]) != int(metadata["test_samples"]):
            raise RuntimeError(
                f"Fold {fold:02d} predicted sample count mismatch: "
                f"{metrics['test_samples_predicted']} != {metadata['test_samples']}"
            )
        summary_rows.append(
            {
                "profile": metadata["profile"],
                "fold": fold,
                "train_start": metadata["train_core_start"],
                "train_end": metadata["validation_end"],
                "test_start": metadata["test_start"],
                "test_end": metadata["test_end"],
                "train_samples": int(metadata["train_core_samples"]) + int(metadata["validation_samples"]),
                "test_samples": metadata["test_samples"],
                "split_ratio": compute_split_ratio(metadata),
                "mode": "tvt_v02",
                **metrics,
                "preds_csv": preds_path.name,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("fold").reset_index(drop=True)
    summary_df.to_csv(out_dir / "rolling_fixed_summary.csv", index=False)
    summary_df.to_csv(out_dir / "rolling_tvt_summary.csv", index=False)
    print(f"Saved summary to {(out_dir / 'rolling_fixed_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
