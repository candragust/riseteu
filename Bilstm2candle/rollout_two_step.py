import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def load_config(cfg_path: Path):
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def standardize(train_vals, test_vals):
    mean = train_vals.mean(axis=0, keepdims=True)
    std = train_vals.std(axis=0, keepdims=True) + 1e-8
    return (train_vals - mean) / std, (test_vals - mean) / std, mean, std


def make_rollout_predictions(model, test_std, mean, std, window: int, limit: int | None = None, batch_size: int = 256):
    rows = []
    max_limit = len(test_std) - window - 1
    if limit is None or limit > max_limit:
        limit = max_limit

    # Step 1: predict next candle for all windows in batch
    X1 = np.stack([test_std[i : i + window] for i in range(limit)], axis=0)
    y1_true = np.stack([test_std[i + window] for i in range(limit)], axis=0)
    y2_true = np.stack([test_std[i + window + 1] for i in range(limit)], axis=0)

    pred1_std_all = model.predict(X1, verbose=0, batch_size=batch_size)
    assert pred1_std_all.shape == (limit, 4), f"Unexpected pred1 batch shape {pred1_std_all.shape}"

    # Step 2 windows: slide by dropping first step and appending pred1
    X2 = np.concatenate([X1[:, 1:, :], pred1_std_all[:, None, :]], axis=1)
    pred2_std_all = model.predict(X2, verbose=0, batch_size=batch_size)
    assert pred2_std_all.shape == (limit, 4), f"Unexpected pred2 batch shape {pred2_std_all.shape}"

    # Denorm and collect
    pred1 = pred1_std_all * std + mean
    pred2 = pred2_std_all * std + mean
    t1 = y1_true * std + mean
    t2 = y2_true * std + mean

    for i in range(limit):
        rows.append(
            {
                "idx": i,
                "pred1_open": pred1[i, 0],
                "pred1_high": pred1[i, 1],
                "pred1_low": pred1[i, 2],
                "pred1_close": pred1[i, 3],
                "true1_open": t1[i, 0],
                "true1_high": t1[i, 1],
                "true1_low": t1[i, 2],
                "true1_close": t1[i, 3],
                "pred2_open": pred2[i, 0],
                "pred2_high": pred2[i, 1],
                "pred2_low": pred2[i, 2],
                "pred2_close": pred2[i, 3],
                "true2_open": t2[i, 0],
                "true2_high": t2[i, 1],
                "true2_low": t2[i, 2],
                "true2_close": t2[i, 3],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Rollout 2-step prediction using saved model.")
    parser.add_argument("--config", default="final_config.json", help="Config JSON with data path and window.")
    parser.add_argument("--model", default="results_rollout/finalweightmodel.h5", help="Saved Keras model path.")
    parser.add_argument("--out", default="results_rollout/rollout2_preds.csv", help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on validation samples for speed.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for predict.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_path = cfg.get("data", "results/EURUSD_H4_clean.csv")
    window = int(cfg.get("window", 16))
    split = float(cfg.get("split", 0.7))

    df_raw = pd.read_csv(data_path)
    values = df_raw[["open", "high", "low", "close"]].values.astype("float32")
    split_idx = int(len(values) * split)
    train_vals = values[:split_idx]
    test_vals = values[split_idx - window :]

    train_std, test_std, mean, std = standardize(train_vals, test_vals)
    model = tf.keras.models.load_model(args.model, compile=False)

    df_out = make_rollout_predictions(
        model=model,
        test_std=test_std,
        mean=mean,
        std=std,
        window=window,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Saved rollout 2-step predictions to {out_path} (rows={len(df_out)})")


if __name__ == "__main__":
    main()
