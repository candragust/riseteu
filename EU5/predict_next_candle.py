#!/usr/bin/env python3
"""
Train the multitask BiLSTM on the latest dataset and predict the next OHLC candle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model as keras_load_model

from bilstm_multitask_experiment import (
    HyperParams,
    build_model,
    load_dataframe,
    make_dataset,
    make_flf_loss,
    set_seed,
)
COMPONENTS = ["open", "high", "low", "close"]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict next OHLC candle using multitask BiLSTM.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("RisetEU") / "EU5" / "multitask_config.json"),
        help="Baseline config JSON (same format as training runner).",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset CSV to use for training (should include datetime and OHLC+features).",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help="Override split ratio in config (e.g. 0.997 for 72m/14d fold).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path("RisetEU") / "EU5" / "results" / "wf72_14d"),
        help="Directory to store trained model per fold.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold",
        help="Label appended to saved model/prediction file (e.g. fold43).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON path for predicted next candle (defaults to model-dir/next_prediction_<fold>.json).",
    )
    parser.add_argument(
        "--train-existing",
        action="store_true",
        help="Force retraining even if a saved model already exists.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and only load existing model to predict next candle.",
    )
    return parser.parse_args()


def prepare_datasets(features: np.ndarray, prices: np.ndarray, window: int, split_ratio: float):
    X, y = make_dataset(features, prices, window)
    split_index = int(len(X) * split_ratio)
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Invalid split ratio for the supplied dataset.")
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std

    stats = {"mean": mean.tolist(), "std": std.tolist()}
    return (X_train_std, y_train), (X_val_std, y_val), stats


def standardize_window(window: np.ndarray, stats: dict):
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    return (window - mean.squeeze()) / std.squeeze()


def save_metadata(out_dir: Path, stats: dict, hp: HyperParams):
    meta_path = out_dir / "scaler.json"
    payload = {"stats": stats, "lambda_coef": hp.lambda_coef, "sigma_coef": hp.sigma_coef}
    meta_path.write_text(json.dumps(payload), encoding="utf-8")


def load_metadata(out_dir: Path):
    meta_path = out_dir / "scaler.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Scaler metadata not found: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = payload.get("stats")
    if stats is None:
        raise ValueError("Scaler metadata does not contain 'stats'.")
    lambda_coef = payload.get("lambda_coef")
    sigma_coef = payload.get("sigma_coef")
    return stats, lambda_coef, sigma_coef


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    feature_cols = cfg.get("feature_columns", "open,high,low,close,atr14,rsi14")
    feature_cols = [c.strip() for c in feature_cols.split(",") if c.strip()]

    hp = HyperParams(
        window=cfg["window"],
        split=args.split if args.split is not None else cfg["split"],
        units=cfg["units"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        lr=cfg["lr"],
        beta1=cfg["beta1"],
        beta2=cfg["beta2"],
        schedule_decay=cfg["schedule_decay"],
        seed=cfg["seed"],
        activation=cfg["activation"],
        recurrent_activation=cfg["recurrent_activation"],
        output_activation=cfg.get("output_activation"),
        lambda_coef=cfg["lambda_coef"],
        sigma_coef=cfg["sigma_coef"],
        dropout=cfg["dropout"],
        recurrent_dropout=cfg["recurrent_dropout"],
        l2_reg=cfg["l2_reg"],
    )

    data_path = Path(args.data)
    out_dir = Path(args.model_dir) / args.fold
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.out) if args.out else out_dir / f"next_prediction_{args.fold}.json"
    model_path = out_dir / f"model_{args.fold}.keras"

    df = pd.read_csv(data_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    input_df = df.copy()
    features = input_df[feature_cols].astype(np.float32).to_numpy()
    prices = input_df[["open", "high", "low", "close"]].astype(np.float32).to_numpy()

    if args.predict_only and not model_path.exists():
        raise FileNotFoundError(f"No saved model found for predict-only mode: {model_path}")

    stats = None
    if not args.predict_only or args.train_existing or not model_path.exists():
        set_seed(hp.seed)
        (X_train, y_train), (X_val, y_val), stats = prepare_datasets(features, prices, hp.window, hp.split)
        model = build_model(hp, feature_dim=X_train.shape[-1])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=hp.epochs, batch_size=hp.batch, verbose=2)
        model.save(model_path)
        save_metadata(out_dir, stats, hp)
    else:
        stats_data, lambda_coef, sigma_coef = load_metadata(out_dir)
        if lambda_coef is None or sigma_coef is None:
            lambda_coef, sigma_coef = hp.lambda_coef, hp.sigma_coef
        custom_loss = make_flf_loss(lambda_coef, sigma_coef)
        model = keras_load_model(model_path, compile=False, custom_objects={"loss": custom_loss})
        stats = stats_data

    if stats is None:
        stats, _, _ = load_metadata(out_dir)

    last_window = features[-hp.window :]
    last_window_std = standardize_window(last_window, stats).reshape(1, hp.window, -1)
    next_pred = model.predict(last_window_std, verbose=0)[0]

    result = {
        "dataset": str(data_path),
        "fold": args.fold,
        "window": hp.window,
        "pred_open": float(next_pred[0]),
        "pred_high": float(next_pred[1]),
        "pred_low": float(next_pred[2]),
        "pred_close": float(next_pred[3]),
        "last_datetime": df["datetime"].iloc[-1].isoformat(),
    }
    pred_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Next candle prediction saved to {pred_path}")


if __name__ == "__main__":
    main()
