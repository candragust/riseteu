import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2


@dataclass
class InputConfig:
    data: Optional[str]
    sep: str
    feature_columns: List[str]


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


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


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


def parse_args():
    parser = argparse.ArgumentParser(description="EU5 BiLSTM experiment (OHLC regression only).")
    parser.add_argument("--config", type=str, default=None, help="JSON config overriding defaults.")

    g_in = parser.add_argument_group("Input")
    g_in.add_argument("--data", type=str, default=None, help="Path to prepared EU5 CSV dataset.")
    g_in.add_argument("--sep", type=str, default=",", help="CSV separator.")
    g_in.add_argument(
        "--feature-columns",
        type=str,
        default="open,high,low,close,atr14,rsi14",
        help="Comma list of feature columns used for windowing.",
    )

    g_hp = parser.add_argument_group("Hyperparameters")
    g_hp.add_argument("--window", type=int, default=None)
    g_hp.add_argument("--split", type=float, default=None)
    g_hp.add_argument("--units", type=int, default=None)
    g_hp.add_argument("--epochs", type=int, default=None)
    g_hp.add_argument("--batch", type=int, default=None)
    g_hp.add_argument("--lr", type=float, default=None)
    g_hp.add_argument("--beta1", type=float, default=None)
    g_hp.add_argument("--beta2", type=float, default=None)
    g_hp.add_argument("--schedule-decay", type=float, default=None)
    g_hp.add_argument("--seed", type=int, default=None)
    g_hp.add_argument("--activation", type=str, default=None)
    g_hp.add_argument("--recurrent-activation", type=str, default=None)
    g_hp.add_argument("--output-activation", type=str, default=None)
    g_hp.add_argument("--lambda-coef", type=float, default=None)
    g_hp.add_argument("--sigma-coef", type=float, default=None)
    g_hp.add_argument("--dropout", type=float, default=None)
    g_hp.add_argument("--recurrent-dropout", type=float, default=None)
    g_hp.add_argument("--l2-reg", type=float, default=None)

    g_rt = parser.add_argument_group("Output")
    g_rt.add_argument("--out", type=str, default="EU5/results/price_preds.csv")
    g_rt.add_argument("--history-out", type=str, default=None)

    args = parser.parse_args()
    cfg_path = args.config
    cfg_data = {}
    if cfg_path:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

    defaults = {
        "data": None,
        "sep": ",",
        "feature_columns": "open,high,low,close,atr14,rsi14",
        "window": 12,
        "split": 0.7,
        "units": 256,
        "epochs": 50,
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
        "out": "EU5/results/price_preds.csv",
        "history_out": None,
    }

    def pick(key):
        val_cli = getattr(args, key.replace("-", "_"), None)
        if val_cli not in (None, ""):
            return val_cli
        if key in cfg_data and cfg_data[key] is not None:
            return cfg_data[key]
        return defaults[key]

    feature_columns = [c.strip() for c in str(pick("feature_columns")).split(",") if c.strip()]
    inp = InputConfig(
        data=pick("data"),
        sep=str(pick("sep")),
        feature_columns=feature_columns,
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
    rt = RuntimeConfig(out_csv=str(pick("out")), history_csv=pick("history_out"))
    return inp, hp, rt


def load_dataframe(path: str, sep: str, feature_cols: List[str]):
    if not path:
        raise ValueError("--data must be specified.")
    df = pd.read_csv(path, sep=sep)
    needed = set(feature_cols + ["open", "high", "low", "close"])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = df.sort_index().copy()
    df = df.dropna(subset=needed)
    return df


def make_dataset(features: np.ndarray, prices: np.ndarray, window: int):
    xs, ys = [], []
    total = len(features)
    for idx in range(total - window):
        xs.append(features[idx : idx + window])
        ys.append(prices[idx + window])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def standardize(train_vals: np.ndarray, test_vals: np.ndarray):
    mean = train_vals.mean(axis=(0, 1), keepdims=True)
    std = train_vals.std(axis=(0, 1), keepdims=True) + 1e-8
    return (train_vals - mean) / std, (test_vals - mean) / std


def build_model(hp: HyperParams, feature_dim: int) -> Model:
    inputs = Input(shape=(hp.window, feature_dim))
    bilstm = Bidirectional(
        LSTM(
            units=hp.units,
            activation=hp.activation,
            recurrent_activation=hp.recurrent_activation,
            dropout=hp.dropout,
            recurrent_dropout=hp.recurrent_dropout,
            kernel_regularizer=l2(hp.l2_reg) if hp.l2_reg > 0 else None,
            return_sequences=False,
        )
    )(inputs)

    price_out = Dense(4, name="price_head")(bilstm)
    if hp.output_activation:
        price_out = tf.keras.layers.Activation(hp.output_activation)(price_out)

    model = Model(inputs=inputs, outputs=price_out)
    opt = Nadam(learning_rate=hp.lr, beta_1=hp.beta1, beta_2=hp.beta2)
    model.compile(optimizer=opt, loss=make_flf_loss(hp.lambda_coef, hp.sigma_coef))
    return model


def save_history(history, path: Optional[str]):
    if not path:
        return
    df = pd.DataFrame(history.history)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_predictions(out_path: str, y_true_price: np.ndarray, y_pred_price: np.ndarray):
    rows = {
        "true_open": y_true_price[:, 0],
        "true_high": y_true_price[:, 1],
        "true_low": y_true_price[:, 2],
        "true_close": y_true_price[:, 3],
        "pred_open": y_pred_price[:, 0],
        "pred_high": y_pred_price[:, 1],
        "pred_low": y_pred_price[:, 2],
        "pred_close": y_pred_price[:, 3],
    }
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    inp, hp, rt = parse_args()
    set_seed(hp.seed)

    df = load_dataframe(inp.data, inp.sep, inp.feature_columns)
    features = df[inp.feature_columns].astype(np.float32).to_numpy()
    prices = df[["open", "high", "low", "close"]].astype(np.float32).to_numpy()

    X, y = make_dataset(features, prices, hp.window)
    split_index = int(len(X) * hp.split)
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Invalid split. Adjust --split for dataset size.")

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    X_train_std, X_test_std = standardize(X_train, X_test)

    model = build_model(hp, feature_dim=X.shape[-1])
    history = model.fit(
        X_train_std,
        y_train,
        validation_data=(X_test_std, y_test),
        epochs=hp.epochs,
        batch_size=hp.batch,
        verbose=2,
    )

    save_history(history, rt.history_csv)

    price_pred = model.predict(X_test_std, verbose=0)
    save_predictions(rt.out_csv, y_test, price_pred)

    mae = np.abs(price_pred - y_test).mean(axis=0)
    print(
        f"MAE (open/high/low/close): {mae[0]:.6f}, {mae[1]:.6f}, {mae[2]:.6f}, {mae[3]:.6f}"
    )


if __name__ == "__main__":
    main()
