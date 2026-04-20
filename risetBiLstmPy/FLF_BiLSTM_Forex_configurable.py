# FLF_BiLSTM_Forex_configurable.py
# Bidirectional LSTM + Forex Loss Function (FLF) for EURUSD H4
# All important knobs are configurable via ENV variables (with sane defaults).
#
# Quick run (defaults):
#   python FLF_BiLSTM_Forex_configurable.py
#
# Example (custom):
#   EURUSD_TAB_PATH="/data/EURUSD_H4.tsv" SEP="\t" WINDOW=5 \
#   EPOCHS=120 BATCH=128 SPLIT_RATIO=0.65 \
#   UNITS=256 LSTM_ACTIVATION="tanh" RECURRENT_ACTIVATION="sigmoid" \
#   LR=5e-5 BETA_1=0.09 BETA_2=0.0999 SCHEDULE_DECAY=4e-4 \
#   LAMBDA=0.9 SIGMA=0.1 \
#   OUT_PATH="bilstm_flf_preds.csv" \
#   python FLF_BiLSTM_Forex_configurable.py
#
# ENV VARIABLES (name : type [default] — description)
#   EURUSD_TAB_PATH : str ["EURUSD_H4_25Oct17.csv"] — path to tab-delimited (or other) OHLC file
#   SEP             : str ["\t"]                    — delimiter (e.g., "\t", ",", ";")
#   WINDOW          : int [1]                       — input lookback window length (candles)
#   SPLIT_RATIO     : float [0.60]                  — train split ratio (0..1)
#   EPOCHS          : int [50]                      — training epochs
#   BATCH           : int [72]                      — batch size
#   UNITS           : int [200]                     — LSTM units per direction (BiLSTM doubles it)
#   LSTM_ACTIVATION : str ["tanh"]                  — activation for LSTM cell state output
#   RECURRENT_ACTIVATION : str ["sigmoid"]          — activation for gates (keep default unless you know why)
#   OUTPUT_ACTIVATION    : str [""]                 — activation for final Dense; empty means linear
#   LR              : float [1e-5]                  — learning rate for NAdam
#   BETA_1          : float [0.09]
#   BETA_2          : float [0.0999]
#   SCHEDULE_DECAY  : float [4e-4]
#   LAMBDA          : float [0.9]                   — FLF coefficient λ
#   SIGMA           : float [0.1]                   — FLF coefficient σ
#   OUT_PATH        : str ["FLF_BiLSTM_predictions.csv"] — output CSV path for predictions
#
# Notes:
# - Column names are auto-detected case-insensitively (open|o, high|h, low|l, close|c).
# - Keep OUTPUT_ACTIVATION empty for price regression (linear output).

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Activation, InputLayer
from tensorflow.keras.optimizers import Nadam

def getenv_str(name, default):
    v = os.environ.get(name, default)
    return v

def getenv_int(name, default):
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(v)
    except:
        return int(float(v))

def getenv_float(name, default):
    v = os.environ.get(name, None)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except:
        return float(str(v).replace(",", ""))

def getenv_bool(name, default):
    v = os.environ.get(name, None)
    if v is None: return bool(default)
    s = str(v).strip().lower()
    return s in ("1","true","yes","y","on")

EURUSD_TAB_PATH      = getenv_str("EURUSD_TAB_PATH", "EURUSD_H4_25Oct17.csv")
SEP                  = getenv_str("SEP", "\\t")
WINDOW               = getenv_int("WINDOW", 1)
SPLIT_RATIO          = getenv_float("SPLIT_RATIO", 0.60)

EPOCHS               = getenv_int("EPOCHS", 50)
BATCH                = getenv_int("BATCH", 72)

UNITS                = getenv_int("UNITS", 200)
LSTM_ACTIVATION      = getenv_str("LSTM_ACTIVATION", "tanh")
RECURRENT_ACTIVATION = getenv_str("RECURRENT_ACTIVATION", "sigmoid")
OUTPUT_ACTIVATION    = getenv_str("OUTPUT_ACTIVATION", "")

LR                   = getenv_float("LR", 1e-5)
BETA_1               = getenv_float("BETA_1", 0.09)
BETA_2               = getenv_float("BETA_2", 0.0999)
SCHEDULE_DECAY       = getenv_float("SCHEDULE_DECAY", 4e-4)

LAMBDA               = getenv_float("LAMBDA", 0.9)
SIGMA                = getenv_float("SIGMA", 0.1)

OUT_PATH             = getenv_str("OUT_PATH", "FLF_BiLSTM_predictions.csv")

SEED = getenv_int("SEED", 42)
random.seed(SEED)
np.random.seed(SEED)

try:
    tf.random.set_seed(SEED)
except Exception:
    pass

def load_tab_data(path, sep):
    try:
        return pd.read_csv(path, sep=sep, engine="python")
    except Exception:
        df = pd.read_csv(path, engine="python")
        if df.shape[1] == 1:
            col = df.columns[0]
            df = df[col].str.split(sep, expand=True)
        return df

def find_ohlc(df: pd.DataFrame):
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    def pick_first(x): return x[0] if len(x)>0 else None
    open_col  = pick_first([c for c in cols if "open" in c or c == "o"])
    high_col  = pick_first([c for c in cols if "high" in c or c == "h"])
    low_col   = pick_first([c for c in cols if "low"  in c or c == "l"])
    close_col = pick_first([c for c in cols if "close" in c or c == "c"])

    if None in (open_col, high_col, low_col, close_col):
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 4:
            open_col, high_col, low_col, close_col = numeric_cols[:4]

    assert all([open_col, high_col, low_col, close_col]), f"Missing OHLC columns in {df.columns.tolist()}"
    out = df[[open_col, high_col, low_col, close_col]].copy()
    out.columns = ["open","high","low","close"]
    out = out.dropna().reset_index(drop=True)
    return out

def make_dataset(ohlc: pd.DataFrame, window: int):
    vals = ohlc[["open","high","low","close"]].values.astype("float32")
    X_list, y_list = [], []
    for i in range(len(vals)-window):
        X_list.append(vals[i:i+window, :])
        y_list.append(vals[i+window, :])
    return np.array(X_list), np.array(y_list)

def flf_loss(y_true, y_pred):
    y_o, y_h, y_l, y_c   = y_true[..., 0], y_true[..., 1], y_true[..., 2], y_true[..., 3]
    ypo, yph, ypl, ypc   = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3]

    alpha_o = LAMBDA * (y_o - ypo)
    alpha_h = LAMBDA * (y_h - yph)
    alpha_l = LAMBDA * (y_l - ypl)
    alpha_c = LAMBDA * (y_c - ypc)

    avg_true_hl = (y_h + y_l) / 2.0
    avg_pred_hl = (yph + ypl) / 2.0
    beta  = SIGMA * (avg_true_hl - avg_pred_hl)

    avg_true_oc = (y_o + y_c) / 2.0
    avg_pred_oc = (ypo + ypc) / 2.0
    gamma = SIGMA * (avg_true_oc - avg_pred_oc)

    comp_o = alpha_o - gamma
    comp_h = alpha_h - beta
    comp_l = alpha_l - beta
    comp_c = alpha_c - gamma

    loss_vec = K.square(comp_o) + K.square(comp_h) + K.square(comp_l) + K.square(comp_c)
    return K.mean(loss_vec / 4.0, axis=-1)

def build_bilstm_model(units=200, lstm_activation="tanh", recurrent_activation="sigmoid",
                       output_activation=None, window=1):
    model = Sequential()
    model.add(InputLayer(input_shape=(window, 4)))
    model.add(
        Bidirectional(
            LSTM(units=units,
                 activation=lstm_activation,
                 recurrent_activation=recurrent_activation,
                 return_sequences=False)
        )
    )
    model.add(Dense(4))
    if output_activation:
        model.add(Activation(output_activation))
    return model

def main():
    print("=== CONFIG ===")
    print("EURUSD_TAB_PATH      :", EURUSD_TAB_PATH)
    print("SEP                  :", repr(SEP))
    print("WINDOW               :", WINDOW)
    print("SPLIT_RATIO          :", SPLIT_RATIO)
    print("EPOCHS               :", EPOCHS, "BATCH:", BATCH)
    print("UNITS                :", UNITS)
    print("LSTM_ACTIVATION      :", LSTM_ACTIVATION)
    print("RECURRENT_ACTIVATION :", RECURRENT_ACTIVATION)
    print("OUTPUT_ACTIVATION    :", OUTPUT_ACTIVATION or "(linear)")
    print("LR / BETA_1 / BETA_2 / SCHEDULE_DECAY :", LR, BETA_1, BETA_2, SCHEDULE_DECAY)
    print("LAMBDA / SIGMA       :", LAMBDA, SIGMA)
    print("OUT_PATH             :", OUT_PATH)
    print("SEED                 :", {})
    print("================\\n".format({}))

    df = load_tab_data(EURUSD_TAB_PATH, SEP)
    ohlc = find_ohlc(df)
    print("Data (OHLC) shape:", ohlc.shape)

    X, y = make_dataset(ohlc, WINDOW)
    split_idx = int(SPLIT_RATIO * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train/Test samples: {len(X_train)}/{len(X_test)}")

    model = build_bilstm_model(
        units=UNITS,
        lstm_activation=LSTM_ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION,
        output_activation=(OUTPUT_ACTIVATION if OUTPUT_ACTIVATION.strip() else None),
        window=WINDOW
    )
    opt = Nadam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, schedule_decay=SCHEDULE_DECAY)
    model.compile(optimizer=opt, loss=flf_loss)

    t0 = time.time()
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     epochs=EPOCHS, batch_size=BATCH, verbose=1)
    print("Train time: %.2fs" % (time.time() - t0))

    y_pred = model.predict(X_test, verbose=0)

    pred_df = pd.DataFrame(y_pred, columns=["pred_open","pred_high","pred_low","pred_close"])
    true_df = pd.DataFrame(y_test, columns=["true_open","true_high","true_low","true_close"])
    out_df  = pd.concat([true_df, pred_df], axis=1)
    out_df.to_csv(OUT_PATH, index=False)
    print("Saved predictions to:", OUT_PATH)

    def mae(a, b): return float(np.mean(np.abs(a - b)))
    print("Test MAE:",
          {"open": mae(y_test[:,0], y_pred[:,0]),
           "high": mae(y_test[:,1], y_pred[:,1]),
           "low":  mae(y_test[:,2], y_pred[:,2]),
           "close":mae(y_test[:,3], y_pred[:,3])})

    plt.figure(figsize=(8,4))
    plt.plot(hist.history.get("loss", []), label="train_loss")
    if "val_loss" in hist.history:
        plt.plot(hist.history["val_loss"], label="val_loss")
    plt.title("Training vs Validation Loss (FLF)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    N_SHOW = min(300, len(y_pred))
    plt.figure(figsize=(10,4))
    plt.plot(true_df["true_close"].values[:N_SHOW], label="True Close")
    plt.plot(pred_df["pred_close"].values[:N_SHOW], label="Pred Close")
    plt.title(f"Close Price: True vs Predicted (first {N_SHOW} test points)")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
