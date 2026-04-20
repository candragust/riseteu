import argparse
import json
import os
import random
import re
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dense, InputLayer, LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam


@dataclass
class InputConfig:
    data: Optional[str]
    sep: str
    columns: Optional[str]
    feature_columns: Optional[str]


@dataclass
class HyperParams:
    window: int
    split: float
    units: int
    epochs: int
    batch: int
    lr: float
    beta1: float
    beta2: float
    schedule_decay: float
    seed: int
    activation: str
    recurrent_activation: str
    output_activation: Optional[str]
    lambda_coef: float
    sigma_coef: float
    dropout: float
    recurrent_dropout: float
    l2_reg: float


@dataclass
class RuntimeConfig:
    out_csv: str
    history_csv: Optional[str]
    model_out: Optional[str]


def _parse_sep(raw: str) -> str:
    val = raw.strip().lower()
    if val in {"\\t", "tab", "t"}:
        return "\t"
    return raw


def parse_args():
    parser = argparse.ArgumentParser(
        description="EURUSD LSTM experiment with Forex Loss Function (FLF)."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config path to override defaults.",
    )

    g_in = parser.add_argument_group("Input")
    g_in.add_argument("--data", type=str, default=None, help="Path to EURUSD OHLC CSV/TSV.")
    g_in.add_argument("--sep", type=str, default=None, help="CSV separator, e.g. ',', 'tab', '\\t'.")
    g_in.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Explicit OHLC columns order, e.g. 'open,high,low,close'.",
    )
    g_in.add_argument(
        "--feature-columns",
        type=str,
        default=None,
        help="Optional extra feature columns (comma-separated), e.g. 'atr14,atr28'.",
    )

    g_hp = parser.add_argument_group("Hyperparameters")
    g_hp.add_argument("--window", type=int, default=None, help="Lookback window length.")
    g_hp.add_argument("--split", type=float, default=None, help="Train split ratio (0-1).")
    g_hp.add_argument("--units", type=int, default=None, help="LSTM units.")
    g_hp.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    g_hp.add_argument("--batch", type=int, default=None, help="Training batch size.")
    g_hp.add_argument("--lr", type=float, default=None, help="Learning rate for Nadam.")
    g_hp.add_argument("--beta1", type=float, default=None, help="Nadam beta_1.")
    g_hp.add_argument("--beta2", type=float, default=None, help="Nadam beta_2.")
    g_hp.add_argument("--schedule-decay", type=float, default=None, help="Nadam schedule decay.")
    g_hp.add_argument("--seed", type=int, default=None, help="Random seed.")
    g_hp.add_argument("--activation", type=str, default=None, help="LSTM activation.")
    g_hp.add_argument(
        "--recurrent-activation",
        type=str,
        default=None,
        help="LSTM recurrent activation.",
    )
    g_hp.add_argument(
        "--output-activation",
        type=str,
        default=None,
        help="Dense output activation (empty means linear).",
    )
    g_hp.add_argument("--lambda-coef", type=float, default=None, help="FLF lambda coefficient.")
    g_hp.add_argument("--sigma-coef", type=float, default=None, help="FLF sigma coefficient.")
    g_hp.add_argument("--dropout", type=float, default=None, help="LSTM dropout rate.")
    g_hp.add_argument("--recurrent-dropout", type=float, default=None, help="LSTM recurrent dropout rate.")
    g_hp.add_argument("--l2-reg", type=float, default=None, help="L2 regularization weight.")

    g_rt = parser.add_argument_group("Runtime")
    g_rt.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to store predictions CSV.",
    )
    g_rt.add_argument(
        "--history-out",
        type=str,
        default=None,
        help="Optional CSV for training history (loss,val_loss).",
    )
    g_rt.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="Optional path to save trained model (e.g., results/final_model.h5).",
    )

    args = parser.parse_args()
    cfg_path = args.config
    cfg_data = {}
    if cfg_path:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

    defaults = {
        "data": None,
        "sep": ",",
        "columns": None,
        "feature_columns": None,
        "window": 20,
        "split": 0.7,
        "units": 200,
        "epochs": 75,
        "batch": 128,
        "lr": 1e-4,
        "beta1": 0.9,
        "beta2": 0.999,
        "schedule_decay": 0.004,
        "seed": 42,
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "output_activation": "",
        "lambda_coef": 0.9,
        "sigma_coef": 0.1,
        "dropout": 0.0,
        "recurrent_dropout": 0.0,
        "l2_reg": 0.0,
        "out": "bilstm_flf_predictions.csv",
        "history_out": None,
        "model_out": None,
    }

    def pick(key):
        # CLI has highest priority, then config, then defaults
        val_cli = getattr(args, key.replace("-", "_"))
        if val_cli is not None:
            return val_cli
        if key in cfg_data and cfg_data[key] is not None:
            return cfg_data[key]
        return defaults[key]

    inp = InputConfig(
        data=pick("data"),
        sep=_parse_sep(pick("sep")),
        columns=pick("columns"),
        feature_columns=pick("feature_columns"),
    )
    hp = HyperParams(
        window=int(pick("window")),
        split=float(pick("split")),
        units=int(pick("units")),
        epochs=int(pick("epochs")),
        batch=int(pick("batch")),
        lr=float(pick("lr")),
        beta1=float(pick("beta1")),
        beta2=float(pick("beta2")),
        schedule_decay=float(pick("schedule_decay")),
        seed=int(pick("seed")),
        activation=str(pick("activation")),
        recurrent_activation=str(pick("recurrent_activation")),
        output_activation=(pick("output_activation") or None),
        lambda_coef=float(pick("lambda_coef")),
        sigma_coef=float(pick("sigma_coef")),
        dropout=float(pick("dropout")),
        recurrent_dropout=float(pick("recurrent_dropout")),
        l2_reg=float(pick("l2_reg")),
    )
    rt = RuntimeConfig(out_csv=str(pick("out")), history_csv=pick("history_out"), model_out=pick("model_out"))
    return inp, hp, rt


def resolve_data_path(candidate: Optional[str]) -> str:
    if candidate and os.path.exists(candidate):
        return candidate
    defaults = [
        "EURUSD_H4_25Oct17.csv",
        "EURUSD_D1_25Oct17.csv",
        os.path.join("risetBiLstmPy", "EURUSD_H4_25Oct17.csv"),
        os.path.join("CodeLstm", "Data", "EURUSD_H4.csv"),
    ]
    for path in defaults:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not locate EURUSD dataset. Use --data to specify a CSV path.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(path: str, sep: str) -> pd.DataFrame:
    # Try user-provided separator first, then fallbacks.
    candidates = []
    if sep:
        candidates.append(sep)
    else:
        candidates.append(",")
    # Add common fallbacks
    candidates.extend([r"[\s\t]+", ",", "\t"])

    last_exc = None
    for cand in candidates:
        try:
            df = pd.read_csv(path, sep=cand, engine="python")
        except Exception as exc:
            last_exc = exc
            continue

        needs_retry = df.shape[1] == 1
        if not needs_retry:
            obj_cols = df.select_dtypes(include=["object"])
            if not obj_cols.empty:
                if obj_cols.astype(str).apply(lambda c: c.str.contains("\t", na=False)).any().any():
                    needs_retry = True
        if not needs_retry:
            return df

    if last_exc:
        raise last_exc
    raise ValueError(f"Unable to parse file {path}")


def find_ohlc(df: pd.DataFrame, columns: Optional[str]) -> pd.DataFrame:
    if columns:
        parts = [c.strip() for c in re.split(r"[,;]", columns) if c.strip()]
        if len(parts) != 4:
            raise ValueError("--columns must provide exactly 4 names for OHLC.")
        out = df[parts].astype("float32").copy()
        out.columns = ["open", "high", "low", "close"]
        return out.dropna().reset_index(drop=True)

    norm_cols = [str(c).strip().lower() for c in df.columns]
    df = df.copy()
    df.columns = norm_cols

    def pick_first(match):
        return match[0] if match else None

    open_col = pick_first([c for c in norm_cols if c in {"open", "o"} or "open" in c])
    high_col = pick_first([c for c in norm_cols if c in {"high", "h"} or "high" in c])
    low_col = pick_first([c for c in norm_cols if c in {"low", "l"} or "low" in c])
    close_col = pick_first([c for c in norm_cols if c in {"close", "c"} or "close" in c])

    if None in (open_col, high_col, low_col, close_col):
        numeric_candidates = [
            c for c in norm_cols if pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric_candidates) >= 4:
            open_col, high_col, low_col, close_col = numeric_candidates[:4]

    if None in (open_col, high_col, low_col, close_col):
        raise ValueError(f"Unable to infer OHLC columns from headers: {df.columns.tolist()}")

    out = df[[open_col, high_col, low_col, close_col]].astype("float32").copy()
    out.columns = ["open", "high", "low", "close"]
    return out.dropna().reset_index(drop=True)


def make_dataset(features: np.ndarray, targets: np.ndarray, window: int):
    x_list, y_list = [], []
    for idx in range(len(features) - window):
        x_list.append(features[idx : idx + window, :])
        y_list.append(targets[idx + window, :])
    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def standardize(train_vals: np.ndarray, test_vals: np.ndarray):
    mean = train_vals.mean(axis=0, keepdims=True)
    std = train_vals.std(axis=0, keepdims=True) + 1e-8
    return (train_vals - mean) / std, (test_vals - mean) / std, mean, std


def build_model(hp: HyperParams, feature_dim: int) -> Sequential:
    model = Sequential(
        [
            InputLayer(input_shape=(hp.window, feature_dim)),
            LSTM(
                units=hp.units,
                activation=hp.activation,
                recurrent_activation=hp.recurrent_activation,
                dropout=hp.dropout,
                recurrent_dropout=hp.recurrent_dropout,
                kernel_regularizer=l2(hp.l2_reg) if hp.l2_reg > 0 else None,
                return_sequences=False,
            ),
            Dense(4, kernel_regularizer=l2(hp.l2_reg) if hp.l2_reg > 0 else None),
        ]
    )
    if hp.output_activation:
        model.add(Activation(hp.output_activation))
    return model


def make_flf_loss(lambda_coef: float, sigma_coef: float):
    def loss_fn(y_true, y_pred):
        y_o, y_h, y_l, y_c = y_true[..., 0], y_true[..., 1], y_true[..., 2], y_true[..., 3]
        p_o, p_h, p_l, p_c = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3]

        alpha_o = lambda_coef * (y_o - p_o)
        alpha_h = lambda_coef * (y_h - p_h)
        alpha_l = lambda_coef * (y_l - p_l)
        alpha_c = lambda_coef * (y_c - p_c)

        avg_true_hl = (y_h + y_l) / 2.0
        avg_pred_hl = (p_h + p_l) / 2.0
        beta = sigma_coef * (avg_true_hl - avg_pred_hl)

        avg_true_oc = (y_o + y_c) / 2.0
        avg_pred_oc = (p_o + p_c) / 2.0
        gamma = sigma_coef * (avg_true_oc - avg_pred_oc)

        comp_o = alpha_o - gamma
        comp_h = alpha_h - beta
        comp_l = alpha_l - beta
        comp_c = alpha_c - gamma

        loss_vec = (
            K.square(comp_o)
            + K.square(comp_h)
            + K.square(comp_l)
            + K.square(comp_c)
        )
        return K.mean(loss_vec / 4.0, axis=-1)

    return loss_fn


def train_and_predict(model: Sequential, X_train, y_train, X_val, y_val, hp: HyperParams):
    # Note: recent Keras Nadam drops schedule_decay; we pass lr, beta1, beta2 only.
    opt = Nadam(
        learning_rate=hp.lr,
        beta_1=hp.beta1,
        beta_2=hp.beta2,
    )
    model.compile(optimizer=opt, loss=make_flf_loss(hp.lambda_coef, hp.sigma_coef))
    callbacks = []
    # EarlyStopping with patience 5, restore best val_loss
    from tensorflow.keras.callbacks import EarlyStopping
    callbacks.append(EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=hp.epochs,
        batch_size=hp.batch,
        verbose=2,
        callbacks=callbacks,
    )
    preds = model.predict(X_val, verbose=0)
    return preds, history.history


def main():
    inp, hp, rt = parse_args()
    set_seed(hp.seed)

    print("=== CONFIG ===")
    print(asdict(inp))
    print(asdict(hp))
    print(asdict(rt))
    print("================")

    data_path = resolve_data_path(inp.data)
    df_raw = load_data(data_path, inp.sep)
    ohlc_df = find_ohlc(df_raw, inp.columns)

    feature_cols = []
    extra_df = None
    if inp.feature_columns:
        feature_cols = [c.strip() for c in re.split(r"[,;]", inp.feature_columns) if c.strip()]
        missing = [c for c in feature_cols if c not in df_raw.columns]
        if missing:
            raise ValueError(f"Missing feature columns in data: {missing}")
        extra_df = df_raw[feature_cols].astype("float32")

    if extra_df is not None:
        combined = pd.concat([ohlc_df, extra_df], axis=1)
    else:
        combined = ohlc_df.copy()
    combined = combined.dropna().reset_index(drop=True)

    base_cols = ["open", "high", "low", "close"]
    feature_names = base_cols + feature_cols
    feature_values = combined[feature_names].values.astype("float32")
    target_values = combined[base_cols].values.astype("float32")
    feature_dim = feature_values.shape[1]

    if len(feature_values) <= hp.window + 1:
        raise ValueError(f"Not enough rows ({len(feature_values)}) for window={hp.window}.")

    split_idx = int(len(feature_values) * hp.split)
    if split_idx <= hp.window:
        raise ValueError("Train split too small relative to window size.")

    train_feat = feature_values[:split_idx]
    test_feat = feature_values[split_idx - hp.window :]
    train_tgt = target_values[:split_idx]
    test_tgt = target_values[split_idx - hp.window :]

    train_feat_std, test_feat_std, feat_mean, feat_std = standardize(train_feat, test_feat)
    train_tgt_std, test_tgt_std, tgt_mean, tgt_std = standardize(train_tgt, test_tgt)
    X_train, y_train = make_dataset(train_feat_std, train_tgt_std, hp.window)
    X_val, y_val = make_dataset(test_feat_std, test_tgt_std, hp.window)
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes  : X={X_val.shape}, y={y_val.shape}")

    model = build_model(hp, feature_dim)
    model.summary(print_fn=lambda line: print("  " + line))

    preds_std, history = train_and_predict(model, X_train, y_train, X_val, y_val, hp)

    preds = preds_std * tgt_std + tgt_mean
    targets = y_val * tgt_std + tgt_mean

    mae = np.mean(np.abs(preds - targets), axis=0)
    print(
        "MAE (open/high/low/close):",
        ", ".join(f"{m:.6f}" for m in mae),
    )

    all_cols = [
        "pred_open",
        "pred_high",
        "pred_low",
        "pred_close",
        "true_open",
        "true_high",
        "true_low",
        "true_close",
    ]
    out_df = pd.DataFrame(
        np.concatenate([preds, targets], axis=1),
        columns=all_cols,
    )
    out_df.to_csv(rt.out_csv, index=False)
    print(f"Saved predictions to {rt.out_csv}")

    if rt.history_csv:
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(rt.history_csv, index=False)
        print(f"Saved training history to {rt.history_csv}")

    if rt.model_out:
        # Save full model (weights + bias)
        model.save(rt.model_out)
        print(f"Saved model to {rt.model_out}")


if __name__ == "__main__":
    main()
