import os
import re
import argparse
import random
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, InputLayer
from tensorflow.keras.optimizers import Nadam


# =============================
# STEP 1 — KONFIGURASI
# =============================
@dataclass
class InputSettings:
    data: str | None
    sep: str
    columns: str | None  # "open,high,low,close" atau None untuk auto


@dataclass
class HyperParams:
    window: int
    split: float
    units: int
    epochs: int
    batch: int
    lr: float
    seed: int


@dataclass
class RuntimeSettings:
    out: str


def _parse_sep(sep: str) -> str:
    s = sep.strip().lower()
    if s in {"\\t", "tab", "t"}:
        return "\t"
    return sep


def parse_config():
    parser = argparse.ArgumentParser(description="BiLSTM OHLC next-step forecasting (structured)")

    g_in = parser.add_argument_group("Input Settings")
    g_in.add_argument("--data", type=str, default=None,
                      help="Path ke CSV OHLC. Jika kosong, coba deteksi file default di folder.")
    g_in.add_argument("--sep", type=str, default=",",
                      help="Delimiter CSV, contoh: ',' atau 'tab' atau '\\t'.")
    g_in.add_argument("--columns", type=str, default=None,
                      help="Kolom eksplisit untuk OHLC, format: 'open,high,low,close'. Kosong = auto.")

    g_hp = parser.add_argument_group("Hyperparameter Settings")
    g_hp.add_argument("--window", type=int, default=20, help="Panjang lookback window.")
    g_hp.add_argument("--split", type=float, default=0.7, help="Rasio data train (0..1).")
    g_hp.add_argument("--units", type=int, default=128, help="Units LSTM per arah (BiLSTM).")
    g_hp.add_argument("--epochs", type=int, default=50, help="Jumlah epoch training.")
    g_hp.add_argument("--batch", type=int, default=128, help="Ukuran batch training.")
    g_hp.add_argument("--lr", type=float, default=1e-3, help="Learning rate Nadam.")
    g_hp.add_argument("--seed", type=int, default=42, help="Seed random untuk reprodusibilitas.")

    g_rt = parser.add_argument_group("Runtime Output")
    g_rt.add_argument("--out", type=str, default="bilstm_predictions.csv", help="Path output CSV prediksi.")

    args = parser.parse_args()

    inp = InputSettings(data=args.data, sep=_parse_sep(args.sep), columns=args.columns)
    hp = HyperParams(window=args.window, split=args.split, units=args.units,
                     epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)
    rt = RuntimeSettings(out=args.out)

    return inp, hp, rt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =============================
# STEP 2 — INPUT & PERSIAPAN DATA
# =============================
def resolve_data_path(candidate: str | None) -> str:
    if candidate and os.path.exists(candidate):
        return candidate
    defaults = [
        "EURUSD_H4_25Oct17.csv",
        os.path.join("risetBiLstmPy", "EURUSD_H4_25Oct17.csv"),
        "EURUSD_D1_25Oct17.csv",
    ]
    for p in defaults:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("CSV data file tidak ditemukan. Gunakan --data untuk menyetel path.")


def find_ohlc(df: pd.DataFrame, columns: str | None) -> pd.DataFrame:
    if columns:
        parts = [c.strip() for c in re.split(r"[,;]", columns) if c.strip()]
        if len(parts) != 4:
            raise ValueError("--columns harus berisi 4 nama kolom: open,high,low,close")
        out = df[parts].astype("float32").copy()
        out.columns = ["open", "high", "low", "close"]
        out = out.dropna().reset_index(drop=True)
        return out

    cols = [str(c).strip().lower() for c in df.columns]
    df = df.copy()
    df.columns = cols

    def pick_first(x):
        return x[0] if len(x) > 0 else None

    open_col = pick_first([c for c in cols if "open" in c or c == "o"])
    high_col = pick_first([c for c in cols if "high" in c or c == "h"])
    low_col = pick_first([c for c in cols if "low" in c or c == "l"])
    close_col = pick_first([c for c in cols if "close" in c or c == "c"])

    if None in (open_col, high_col, low_col, close_col):
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 4:
            open_col, high_col, low_col, close_col = numeric_cols[:4]

    if None in (open_col, high_col, low_col, close_col):
        raise ValueError(f"Kolom OHLC tidak ditemukan di {df.columns.tolist()}")

    out = df[[open_col, high_col, low_col, close_col]].astype("float32").copy()
    out.columns = ["open", "high", "low", "close"]
    out = out.dropna().reset_index(drop=True)
    return out


def make_dataset(ohlc: np.ndarray, window: int):
    X_list, y_list = [], []
    for i in range(len(ohlc) - window):
        X_list.append(ohlc[i : i + window, :])
        y_list.append(ohlc[i + window, :])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def standardize_train_test(train_arr: np.ndarray, test_arr: np.ndarray):
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True) + 1e-8
    return (train_arr - mean) / std, (test_arr - mean) / std, mean, std


# =============================
# STEP 3 — MODELING
# =============================
def build_model(window: int, units: int) -> Sequential:
    model = Sequential([
        InputLayer(input_shape=(window, 4)),
        Bidirectional(LSTM(units, return_sequences=False)),
        Dense(4),
    ])
    return model


# =============================
# STEP 4 — TRAINING & INFERENSI
# =============================
def train_and_predict(model: Sequential,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      epochs: int, batch: int) -> tuple[np.ndarray, dict]:
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        verbose=2,
    )
    preds = model.predict(X_val, verbose=0)
    return preds, {k: v for k, v in hist.history.items()}


# =============================
# STEP 5 — PIPELINE UTAMA
# =============================
def main():
    # Step 1: Konfigurasi
    inp, hp, rt = parse_config()
    set_seed(hp.seed)
    print("STEP 1 — Konfigurasi")
    print({"input": asdict(inp), "hyperparams": asdict(hp), "runtime": asdict(rt)})

    # Step 2: Load & siapkan data
    print("STEP 2 — Load data")
    data_path = resolve_data_path(inp.data)
    df_raw = pd.read_csv(data_path, sep=inp.sep, engine="python")
    ohlc_df = find_ohlc(df_raw, inp.columns)
    values = ohlc_df[["open", "high", "low", "close"]].values.astype("float32")

    n = len(values)
    if n <= hp.window + 1:
        raise ValueError(f"Baris data kurang ({n}) untuk window={hp.window}")

    split_idx = int(n * hp.split)
    train_vals = values[:split_idx]
    test_vals = values[split_idx - hp.window :]

    train_std, test_std, mean, std = standardize_train_test(train_vals, test_vals)
    X_train, y_train = make_dataset(train_std, hp.window)
    X_val, y_val = make_dataset(test_std, hp.window)
    print(f"Data train: X={X_train.shape}, y={y_train.shape}; Val: X={X_val.shape}, y={y_val.shape}")

    # Step 3: Bangun model
    print("STEP 3 — Bangun model BiLSTM")
    model = build_model(window=hp.window, units=hp.units)
    opt = Nadam(learning_rate=hp.lr)
    model.compile(optimizer=opt, loss="mse")
    model.summary(print_fn=lambda s: print("  "+s))

    # Step 4: Training & inferensi
    print("STEP 4 — Training")
    preds_std, history = train_and_predict(model, X_train, y_train, X_val, y_val, hp.epochs, hp.batch)

    # Step 5: Simpan hasil
    print("STEP 5 — Simpan hasil")
    preds = preds_std * std + mean
    target = y_val * std + mean
    out_df = pd.DataFrame(
        np.concatenate([preds, target], axis=1),
        columns=[
            "pred_open", "pred_high", "pred_low", "pred_close",
            "true_open", "true_high", "true_low", "true_close",
        ],
    )
    out_df.to_csv(rt.out, index=False)
    print(f"Saved predictions to {rt.out}")


if __name__ == "__main__":
    main()
