#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_D1 = PROJECT_ROOT / "EURUSD_D1_25Oct17.csv"
CLEAN_D1 = PROJECT_ROOT / "results" / "EURUSD_D1_clean.csv"

FLF_BILSTM_D1_ROOT = PROJECT_ROOT / "FLF_BILSTM" / "results" / "d1_ohlc"
FLF_LSTM_D1_ROOT = PROJECT_ROOT / "FLF_LSTM" / "results" / "d1_ohlc"
ARIMA_D1_ROOT = PROJECT_ROOT / "Arima" / "result" / "d1_ohlc"
COMPARISON_D1_ROOT = PROJECT_ROOT / "comparison" / "d1_ohlc"
COMPARISON_H4_D1_ROOT = PROJECT_ROOT / "comparison" / "h4_vs_d1"
RESULTS_D1_ROOT = PROJECT_ROOT / "results" / "d1_ohlc"

FLF_BILSTM_FOLD_DIR = FLF_BILSTM_D1_ROOT / "rolling_train72_test1"
FLF_BILSTM_LAST5_FOLD_DIR = FLF_BILSTM_D1_ROOT / "rolling_train72_test1_last5"

FLF_LSTM_CONFIG_DIR = PROJECT_ROOT / "FLF_LSTM" / "configs" / "d1_ohlc"
FLF_BILSTM_CONFIG_DIR = PROJECT_ROOT / "FLF_BILSTM" / "configs" / "d1_ohlc"

LSTM_BASE_4H = PROJECT_ROOT / "FLF_LSTM" / "lstm_flf_config_wf72_test1_best.json"
BILSTM_BASE_4H = PROJECT_ROOT / "FLF_BILSTM" / "final_config.json"

TRAIN_MONTHS = 72
TEST_MONTHS = 1
SHIFT_MONTHS = 1
WINDOW_BASELINE = 12
MIN_TEST_SAMPLES = 15


@dataclass
class FoldInfo:
    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    split_ratio: float


def ensure_dirs() -> None:
    dirs = [
        CLEAN_D1.parent,
        RESULTS_D1_ROOT,
        FLF_BILSTM_FOLD_DIR,
        FLF_BILSTM_LAST5_FOLD_DIR,
        FLF_BILSTM_D1_ROOT / "smoke",
        FLF_BILSTM_D1_ROOT / "tuning_v01",
        FLF_BILSTM_D1_ROOT / "screenshots",
        FLF_LSTM_D1_ROOT / "smoke",
        FLF_LSTM_D1_ROOT / "tuning_v01",
        FLF_LSTM_D1_ROOT / "wf72_test1",
        FLF_LSTM_D1_ROOT / "wf72_test1_last5",
        FLF_LSTM_D1_ROOT / "screenshots",
        ARIMA_D1_ROOT / "arima_wf72_test1",
        ARIMA_D1_ROOT / "arima_wf72_test1_last5",
        COMPARISON_D1_ROOT,
        COMPARISON_H4_D1_ROOT,
        FLF_LSTM_CONFIG_DIR,
        FLF_BILSTM_CONFIG_DIR,
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def load_raw_d1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", engine="python")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=r"\t|,", engine="python")
    if df.shape[1] != 5:
        raise ValueError(f"Unexpected D1 raw shape {df.shape}; expected 5 columns.")
    df.columns = ["datetime", "open", "high", "low", "close"]
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M", errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def month_offset(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=months)


def build_folds(df: pd.DataFrame) -> list[tuple[FoldInfo, pd.DataFrame]]:
    folds: list[tuple[FoldInfo, pd.DataFrame]] = []
    cursor = df["datetime"].iloc[0]
    last_timestamp = df["datetime"].iloc[-1]
    fold_idx = 1

    while True:
        train_start = cursor
        train_end = month_offset(train_start, TRAIN_MONTHS)
        test_start = train_end
        if test_start >= last_timestamp:
            break

        test_end = month_offset(test_start, TEST_MONTHS)
        train_mask = (df["datetime"] >= train_start) & (df["datetime"] < train_end)
        test_mask = (df["datetime"] >= test_start) & (df["datetime"] < test_end)
        train_len = int(train_mask.sum())
        test_len = int(test_mask.sum())

        if train_len <= WINDOW_BASELINE or test_len < max(WINDOW_BASELINE + 1, MIN_TEST_SAMPLES):
            break

        fold_df = pd.concat(
            [
                df.loc[train_mask, ["open", "high", "low", "close"]],
                df.loc[test_mask, ["open", "high", "low", "close"]],
            ],
            ignore_index=True,
        )
        split_ratio = train_len / len(fold_df)
        fold = FoldInfo(
            index=fold_idx,
            train_start=str(train_start.date()),
            train_end=str(train_end.date()),
            test_start=str(test_start.date()),
            test_end=str(min(test_end, last_timestamp).date()),
            train_samples=train_len,
            test_samples=test_len,
            split_ratio=split_ratio,
        )
        folds.append((fold, fold_df))

        cursor = month_offset(cursor, SHIFT_MONTHS)
        fold_idx += 1

    return folds


def write_fold_artifacts(folds: list[tuple[FoldInfo, pd.DataFrame]]) -> None:
    summary_rows = []
    for fold, fold_df in folds:
        fold_csv = FLF_BILSTM_FOLD_DIR / f"fold{fold.index:02d}_data.csv"
        fold_df.to_csv(fold_csv, index=False)
        summary_rows.append(asdict(fold))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(FLF_BILSTM_FOLD_DIR / "rolling_fixed_summary.csv", index=False)

    last5_df = summary_df.tail(5).copy()
    last5_df.to_csv(FLF_BILSTM_LAST5_FOLD_DIR / "rolling_fixed_summary.csv", index=False)
    for fold, fold_df in folds[-5:]:
        fold_csv = FLF_BILSTM_LAST5_FOLD_DIR / f"fold{fold.index:02d}_data.csv"
        fold_df.to_csv(fold_csv, index=False)

    summary_df.to_csv(RESULTS_D1_ROOT / "fold_schedule_wf72_test1_v01.csv", index=False)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def write_configs(last_fold: FoldInfo) -> None:
    lstm_cfg = json.loads(LSTM_BASE_4H.read_text(encoding="utf-8"))
    bilstm_cfg = json.loads(BILSTM_BASE_4H.read_text(encoding="utf-8"))

    fold21_path = FLF_BILSTM_FOLD_DIR / f"fold{last_fold.index:02d}_data.csv"

    lstm_cfg.update(
        {
            "data": rel(fold21_path),
            "sep": ",",
            "columns": None,
            "split": round(last_fold.split_ratio, 10),
            "out": rel(FLF_LSTM_D1_ROOT / "smoke" / "lstm_d1_smoke_preds_v01.csv"),
            "history_out": rel(FLF_LSTM_D1_ROOT / "smoke" / "lstm_d1_smoke_history_v01.csv"),
            "model_out": rel(FLF_LSTM_D1_ROOT / "smoke" / "lstm_d1_smoke_model_v01.keras"),
        }
    )
    bilstm_cfg.update(
        {
            "data": rel(fold21_path),
            "sep": ",",
            "columns": None,
            "split": round(last_fold.split_ratio, 10),
            "out": rel(FLF_BILSTM_D1_ROOT / "smoke" / "bilstm_d1_smoke_preds_v01.csv"),
            "history_out": rel(FLF_BILSTM_D1_ROOT / "smoke" / "bilstm_d1_smoke_history_v01.csv"),
        }
    )

    (FLF_LSTM_CONFIG_DIR / "lstm_flf_config_d1_ohlc_base_v01.json").write_text(
        json.dumps(lstm_cfg, indent=2), encoding="utf-8"
    )
    (FLF_BILSTM_CONFIG_DIR / "bilstm_flf_config_d1_ohlc_base_v01.json").write_text(
        json.dumps(bilstm_cfg, indent=2), encoding="utf-8"
    )


def write_manifest(df: pd.DataFrame, folds: list[tuple[FoldInfo, pd.DataFrame]]) -> None:
    fold_infos = [asdict(fold) for fold, _ in folds]
    manifest = {
        "dataset": {
            "raw": rel(RAW_D1),
            "clean": rel(CLEAN_D1),
            "rows": int(len(df)),
            "start": str(df["datetime"].iloc[0].date()),
            "end": str(df["datetime"].iloc[-1].date()),
        },
        "protocol": {
            "train_months": TRAIN_MONTHS,
            "test_months": TEST_MONTHS,
            "shift_months": SHIFT_MONTHS,
            "window_baseline": WINDOW_BASELINE,
            "min_test_samples": MIN_TEST_SAMPLES,
            "fold_count": len(folds),
            "analysis_focus": "last_5_folds",
        },
        "paths": {
            "fold_root": rel(FLF_BILSTM_FOLD_DIR),
            "fold_last5_root": rel(FLF_BILSTM_LAST5_FOLD_DIR),
            "lstm_base_config": rel(FLF_LSTM_CONFIG_DIR / "lstm_flf_config_d1_ohlc_base_v01.json"),
            "bilstm_base_config": rel(FLF_BILSTM_CONFIG_DIR / "bilstm_flf_config_d1_ohlc_base_v01.json"),
        },
        "last_5_folds": fold_infos[-5:],
    }
    (RESULTS_D1_ROOT / "setup_manifest_v01.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_raw_d1(RAW_D1)
    df.to_csv(CLEAN_D1, index=False)

    folds = build_folds(df)
    if not folds:
        raise RuntimeError("No valid D1 folds created for wf72_test1.")

    write_fold_artifacts(folds)
    write_configs(folds[-1][0])
    write_manifest(df, folds)

    print(CLEAN_D1.resolve())
    print((FLF_LSTM_CONFIG_DIR / "lstm_flf_config_d1_ohlc_base_v01.json").resolve())
    print((FLF_BILSTM_CONFIG_DIR / "bilstm_flf_config_d1_ohlc_base_v01.json").resolve())
    print((RESULTS_D1_ROOT / "setup_manifest_v01.json").resolve())


if __name__ == "__main__":
    main()
