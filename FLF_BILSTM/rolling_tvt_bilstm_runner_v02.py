import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_SCHEDULE = PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "h4" / "fold_schedule_evaluation.csv"
DEFAULT_FOLD_ROOT = PROJECT_ROOT / "results" / "splits" / "tvt_v02" / "h4"
DEFAULT_BASE_CONFIG = ROOT / "bilstm_flf_config_h4_tvt_v02_base.json"
DEFAULT_OUT_DIR = ROOT / "results" / "tvt_v02" / "h4_evaluation_last3"
EXPERIMENT_SCRIPT = ROOT / "bilstm_flf_experiment_tvt_v01.py"


@dataclass
class FoldPaths:
    fold: int
    profile: str
    train_data: Path
    validation_data: Path
    test_data: Path
    metadata_path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BiLSTM evaluation on pre-split train/validation/test folds."
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=str(DEFAULT_SCHEDULE),
        help="CSV schedule selecting which folds to run.",
    )
    parser.add_argument(
        "--fold-root",
        type=str,
        default=str(DEFAULT_FOLD_ROOT),
        help="Directory containing foldXX subfolders with train_core/validation/test files.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(DEFAULT_BASE_CONFIG),
        help="Base JSON config for the BiLSTM experiment.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for predictions, history, and summaries.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch the experiment script.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Optional comma-separated fold list to run, e.g. '17,18,21'.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=",",
        help="CSV separator for split files.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Optional explicit OHLC column order.",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        default=None,
        help="Optional feature columns passed through to the experiment script.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Also save the trained model artifact per fold.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only materialize the selected fold schedule without training.",
    )
    return parser.parse_args()


def parse_fold_selection(raw: str):
    if not raw:
        return None
    selected = set()
    for token in raw.split(","):
        token = token.strip()
        if token:
            selected.add(int(token))
    return selected


def load_selected_schedule(schedule_path: Path, selected_folds):
    df = pd.read_csv(schedule_path)
    if "fold" not in df.columns:
        raise ValueError(f"Schedule file has no 'fold' column: {schedule_path}")
    df["fold"] = df["fold"].astype(int)
    if selected_folds:
        df = df[df["fold"].isin(selected_folds)].copy()
    if df.empty:
        raise RuntimeError("No folds selected from schedule.")
    return df.sort_values("fold").reset_index(drop=True)


def resolve_fold_paths(fold_root: Path, row: pd.Series) -> FoldPaths:
    fold = int(row["fold"])
    fold_dir = fold_root / f"fold{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    train_data = fold_dir / "train_core.csv"
    validation_data = fold_dir / "validation.csv"
    test_data = fold_dir / "test.csv"
    metadata_path = fold_dir / "metadata.json"

    for path in [train_data, validation_data, test_data, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing fold artifact: {path}")

    return FoldPaths(
        fold=fold,
        profile=str(row.get("profile", "")),
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        metadata_path=metadata_path,
    )


def build_experiment_cmd(args, base_cfg: Path, fold_paths: FoldPaths, out_dir: Path):
    preds_path = out_dir / f"fold{fold_paths.fold:02d}_preds.csv"
    history_path = out_dir / f"fold{fold_paths.fold:02d}_history.csv"

    cmd = [
        args.python_bin,
        str(EXPERIMENT_SCRIPT),
        "--config",
        str(base_cfg),
        "--train-data",
        str(fold_paths.train_data),
        "--validation-data",
        str(fold_paths.validation_data),
        "--test-data",
        str(fold_paths.test_data),
        "--sep",
        args.sep,
        "--out",
        str(preds_path),
        "--history-out",
        str(history_path),
    ]

    if args.columns:
        cmd.extend(["--columns", args.columns])
    if args.feature_columns:
        cmd.extend(["--feature-columns", args.feature_columns])
    if args.save_models:
        model_path = out_dir / f"fold{fold_paths.fold:02d}_model.keras"
        cmd.extend(["--model-out", str(model_path)])
    else:
        cmd.extend(["--model-out", ""])

    return cmd, preds_path, history_path


def load_metadata(metadata_path: Path):
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def compute_mae_metrics(preds_path: Path):
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


def main():
    args = parse_args()
    schedule_path = Path(args.schedule)
    fold_root = Path(args.fold_root)
    base_cfg = Path(args.base_config)
    out_dir = Path(args.out_dir)

    for path in [schedule_path, fold_root, base_cfg, EXPERIMENT_SCRIPT]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    selected_folds = parse_fold_selection(args.folds)
    schedule_df = load_selected_schedule(schedule_path, selected_folds)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "schedule": str(schedule_path.resolve()),
        "fold_root": str(fold_root.resolve()),
        "base_config": str(base_cfg.resolve()),
        "python_bin": str(Path(args.python_bin).resolve()) if Path(args.python_bin).exists() else args.python_bin,
        "selected_folds": sorted(selected_folds) if selected_folds else schedule_df["fold"].tolist(),
        "experiment_script": str(EXPERIMENT_SCRIPT.resolve()),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    schedule_copy_path = out_dir / "selected_schedule.csv"
    schedule_df.to_csv(schedule_copy_path, index=False)
    print(f"Saved selected schedule to {schedule_copy_path.resolve()}")

    if args.dry_run:
        return

    summary_rows = []
    for row in schedule_df.to_dict(orient="records"):
        fold_paths = resolve_fold_paths(fold_root, pd.Series(row))
        metadata = load_metadata(fold_paths.metadata_path)
        cmd, preds_path, history_path = build_experiment_cmd(args, base_cfg, fold_paths, out_dir)

        print(
            f"[FOLD {fold_paths.fold:02d}] "
            f"train_core={metadata['train_core_start']} -> {metadata['train_core_end']} | "
            f"validation={metadata['validation_start']} -> {metadata['validation_end']} | "
            f"test={metadata['test_start']} -> {metadata['test_end']}"
        )
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

        metrics = compute_mae_metrics(preds_path)
        summary_rows.append(
            {
                "profile": metadata["profile"],
                "fold": fold_paths.fold,
                "train_core_start": metadata["train_core_start"],
                "train_core_end": metadata["train_core_end"],
                "validation_start": metadata["validation_start"],
                "validation_end": metadata["validation_end"],
                "test_start": metadata["test_start"],
                "test_end": metadata["test_end"],
                "train_core_samples": metadata["train_core_samples"],
                "validation_samples": metadata["validation_samples"],
                "test_samples": metadata["test_samples"],
                **metrics,
                "preds_csv": preds_path.name,
                "history_csv": history_path.name,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("fold").reset_index(drop=True)
    summary_main = out_dir / "rolling_tvt_summary.csv"
    summary_compat = out_dir / "rolling_fixed_summary.csv"
    summary_df.to_csv(summary_main, index=False)
    summary_df.to_csv(summary_compat, index=False)
    print(f"Saved summary to {summary_main.resolve()}")


if __name__ == "__main__":
    main()
