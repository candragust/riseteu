import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from eurusd_ohlc_utils import find_ohlc, load_data, resolve_data_path


PIP = 0.0001
PIP_FACTOR = 1 / PIP
BASE_COLS = ["open", "high", "low", "close"]
FALLBACK_ORDERS = [(1, 1, 0), (0, 1, 1), (0, 1, 0)]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class InputConfig:
    data: Optional[str]
    sep: str
    columns: Optional[str]


@dataclass
class ArimaConfig:
    split: float
    order: tuple[int, int, int]
    order_by_column: Optional[Dict[str, tuple[int, int, int]]]
    order_search: bool
    p_values: tuple[int, ...]
    d_values: tuple[int, ...]
    q_values: tuple[int, ...]
    selection_metric: str
    trend: str
    enforce_stationarity: bool
    enforce_invertibility: bool
    refit_each_step: bool
    max_test_steps: Optional[int]


@dataclass
class RuntimeConfig:
    out_csv: str
    summary_json: Optional[str]


def _resolve_runtime_path(candidate: Optional[str]) -> Optional[Path]:
    if candidate in (None, ""):
        return None
    raw = Path(candidate)
    if raw.is_absolute():
        return raw
    return PROJECT_ROOT / raw


def _parse_order(raw) -> tuple[int, int, int]:
    if raw is None:
        return (1, 1, 0)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return tuple(int(x) for x in raw)
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("--order must provide exactly three integers, e.g. 1,1,0")
        return tuple(int(x) for x in parts)
    raise ValueError(f"Unsupported order format: {raw!r}")


def _parse_int_values(raw, name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw in (None, ""):
        return default
    if isinstance(raw, (list, tuple)):
        values = [int(x) for x in raw]
    elif isinstance(raw, str):
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        raise ValueError(f"Unsupported {name} format: {raw!r}")
    if not values:
        raise ValueError(f"{name} must not be empty.")
    if any(v < 0 for v in values):
        raise ValueError(f"{name} must contain non-negative integers only.")
    return tuple(sorted(set(values)))


def _parse_order_by_column(raw) -> Optional[Dict[str, tuple[int, int, int]]]:
    if raw in (None, "", {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("order_by_column must be a JSON object keyed by open/high/low/close")

    out = {}
    for key, value in raw.items():
        norm_key = str(key).strip().lower()
        if norm_key not in BASE_COLS:
            raise ValueError(f"Unsupported order_by_column key: {key}")
        out[norm_key] = _parse_order(value)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="ARIMA OHLC baseline experiment (OHLC-only input).")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config path.")

    g_in = parser.add_argument_group("Input")
    g_in.add_argument("--data", type=str, default=None, help="Path to OHLC CSV/TSV.")
    g_in.add_argument("--sep", type=str, default=None, help="CSV separator.")
    g_in.add_argument("--columns", type=str, default=None, help="Explicit OHLC columns order.")

    g_ar = parser.add_argument_group("ARIMA")
    g_ar.add_argument("--split", type=float, default=None, help="Train split ratio (0-1).")
    g_ar.add_argument("--order", type=str, default=None, help="Shared ARIMA order, e.g. 1,1,0.")
    g_ar.add_argument(
        "--order-by-column",
        type=str,
        default=None,
        help='Optional JSON object with per-column orders, e.g. {"open":[1,1,0],...}',
    )
    g_ar.add_argument(
        "--order-search",
        type=str,
        default=None,
        help="Enable order search on the train set only (true/false).",
    )
    g_ar.add_argument(
        "--p-values",
        type=str,
        default=None,
        help="Comma-separated candidate p values for order search, e.g. 0,1,2.",
    )
    g_ar.add_argument(
        "--d-values",
        type=str,
        default=None,
        help="Comma-separated candidate d values for order search, e.g. 0,1.",
    )
    g_ar.add_argument(
        "--q-values",
        type=str,
        default=None,
        help="Comma-separated candidate q values for order search, e.g. 0,1,2.",
    )
    g_ar.add_argument(
        "--selection-metric",
        type=str,
        default=None,
        help="Information criterion for order search: aic or bic.",
    )
    g_ar.add_argument("--trend", type=str, default=None, help="ARIMA trend code, e.g. n or c.")
    g_ar.add_argument(
        "--enforce-stationarity",
        type=str,
        default=None,
        help="Whether to enforce stationarity (true/false).",
    )
    g_ar.add_argument(
        "--enforce-invertibility",
        type=str,
        default=None,
        help="Whether to enforce invertibility (true/false).",
    )
    g_ar.add_argument(
        "--refit-each-step",
        action="store_true",
        help="Refit ARIMA at every test step instead of appending actual values to the fitted state.",
    )
    g_ar.add_argument(
        "--max-test-steps",
        type=int,
        default=None,
        help="Optional limit for number of test steps, useful for smoke tests.",
    )

    g_rt = parser.add_argument_group("Runtime")
    g_rt.add_argument("--out", type=str, default=None, help="Where to store predictions CSV.")
    g_rt.add_argument("--summary-out", type=str, default=None, help="Where to store summary JSON.")

    args = parser.parse_args()
    cfg_data = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

    defaults = {
        "data": None,
        "sep": ",",
        "columns": None,
        "split": 0.7,
        "order": [1, 1, 0],
        "order_by_column": None,
        "order_search": False,
        "p_values": [0, 1, 2],
        "d_values": [0, 1],
        "q_values": [0, 1, 2],
        "selection_metric": "aic",
        "trend": "n",
        "enforce_stationarity": False,
        "enforce_invertibility": False,
        "max_test_steps": None,
        "out": "Arima/result/arima_ohlc_preds.csv",
        "summary_out": "Arima/result/arima_ohlc_summary.json",
    }

    def pick(key):
        cli_key = key.replace("-", "_")
        val_cli = getattr(args, cli_key)
        if key == "order_by_column" and isinstance(val_cli, str):
            return json.loads(val_cli)
        if key in {"enforce_stationarity", "enforce_invertibility", "order_search"} and isinstance(val_cli, str):
            val = val_cli.strip().lower()
            if val not in {"true", "false"}:
                raise ValueError(f"--{key.replace('_', '-')} must be true or false")
            return val == "true"
        if val_cli is not None:
            return val_cli
        if key in cfg_data and cfg_data[key] is not None:
            return cfg_data[key]
        return defaults[key]

    inp = InputConfig(
        data=pick("data"),
        sep=str(pick("sep")),
        columns=pick("columns"),
    )
    arima_cfg = ArimaConfig(
        split=float(pick("split")),
        order=_parse_order(pick("order")),
        order_by_column=_parse_order_by_column(pick("order_by_column")),
        order_search=bool(pick("order_search")),
        p_values=_parse_int_values(pick("p_values"), "p_values", (0, 1, 2)),
        d_values=_parse_int_values(pick("d_values"), "d_values", (0, 1)),
        q_values=_parse_int_values(pick("q_values"), "q_values", (0, 1, 2)),
        selection_metric=str(pick("selection_metric")).strip().lower(),
        trend=str(pick("trend")),
        enforce_stationarity=bool(pick("enforce_stationarity")),
        enforce_invertibility=bool(pick("enforce_invertibility")),
        refit_each_step=bool(args.refit_each_step),
        max_test_steps=pick("max_test_steps"),
    )
    rt = RuntimeConfig(
        out_csv=str(pick("out")),
        summary_json=pick("summary_out"),
    )
    return inp, arima_cfg, rt


def _candidate_orders(cfg: ArimaConfig, requested_order) -> list[tuple[int, int, int]]:
    candidates = list(product(cfg.p_values, cfg.d_values, cfg.q_values))
    if requested_order not in candidates:
        candidates.append(requested_order)
    return candidates


def fit_model(train_values, order, cfg: ArimaConfig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = ARIMA(
            train_values,
            order=order,
            trend=cfg.trend,
            enforce_stationarity=cfg.enforce_stationarity,
            enforce_invertibility=cfg.enforce_invertibility,
        )
        return model.fit()


def fit_with_fallback(train_values, requested_order, cfg: ArimaConfig):
    attempts = [requested_order]
    for fallback in FALLBACK_ORDERS:
        if fallback != requested_order:
            attempts.append(fallback)

    errors = []
    for order in attempts:
        try:
            return fit_model(train_values, order, cfg), order
        except Exception as exc:  # pragma: no cover - fallback path
            errors.append(f"{order}: {exc}")

    raise RuntimeError("Unable to fit ARIMA model. Attempts: " + " | ".join(errors))


def select_order_by_ic(train_values, requested_order, cfg: ArimaConfig):
    if cfg.selection_metric not in {"aic", "bic"}:
        raise ValueError("selection_metric must be either 'aic' or 'bic'.")

    best_result = None
    best_order = None
    best_score = None
    candidate_orders = _candidate_orders(cfg, requested_order)
    failed = []
    successful = 0

    for order in candidate_orders:
        try:
            result = fit_model(train_values, order, cfg)
        except Exception as exc:  # pragma: no cover - statsmodels fit failures depend on data
            failed.append({"order": list(order), "error": str(exc)})
            continue

        successful += 1
        score = float(getattr(result, cfg.selection_metric))
        if best_score is None or score < best_score:
            best_score = score
            best_result = result
            best_order = order

    if best_result is None:
        result, best_order = fit_with_fallback(train_values, requested_order, cfg)
        best_score = float(getattr(result, cfg.selection_metric))
        return result, best_order, {
            "selection_mode": "fallback_only",
            "selection_metric": cfg.selection_metric,
            "selection_score": best_score,
            "candidate_count": len(candidate_orders),
            "successful_candidates": 0,
            "failed_candidates": len(candidate_orders),
            "failed_details": failed,
        }

    return best_result, best_order, {
        "selection_mode": "grid_search",
        "selection_metric": cfg.selection_metric,
        "selection_score": best_score,
        "candidate_count": len(candidate_orders),
        "successful_candidates": successful,
        "failed_candidates": len(failed),
        "failed_details": failed,
    }


def walk_forward_forecast(series, split, requested_order, cfg: ArimaConfig):
    split_idx = int(len(series) * split)
    if split_idx < 10:
        raise ValueError("Train split too small for ARIMA baseline.")
    if split_idx >= len(series):
        raise ValueError("Split ratio leaves no test samples.")

    train_values = np.asarray(series[:split_idx], dtype=float)
    test_values = np.asarray(series[split_idx:], dtype=float)
    if cfg.max_test_steps is not None:
        test_values = test_values[: int(cfg.max_test_steps)]
    if len(test_values) == 0:
        raise ValueError("No test samples available after applying max_test_steps.")

    search_meta = None
    if cfg.order_search:
        result, selected_order, search_meta = select_order_by_ic(train_values, requested_order, cfg)
        used_order = selected_order
    else:
        result, used_order = fit_with_fallback(train_values, requested_order, cfg)
        selected_order = used_order

    preds = []
    state = result

    for actual in test_values:
        next_pred = float(state.forecast(steps=1)[0])
        preds.append(next_pred)
        if cfg.refit_each_step:
            train_values = np.append(train_values, actual)
            state, used_order = fit_with_fallback(train_values, used_order, cfg)
        else:
            state = state.append([actual], refit=False)

    return {
        "preds": np.asarray(preds, dtype=float),
        "targets": test_values,
        "split_idx": split_idx,
        "train_samples": int(split_idx),
        "test_samples": int(len(test_values)),
        "selected_order": selected_order,
        "used_order": used_order,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "search": search_meta,
    }


def build_summary(out_df: pd.DataFrame, model_meta: Dict[str, Dict], split: float):
    preds = out_df[[f"pred_{c}" for c in BASE_COLS]].values
    targets = out_df[[f"true_{c}" for c in BASE_COLS]].values
    diffs = np.abs(preds - targets)
    mae = diffs.mean(axis=0)
    mae_pips = mae * PIP_FACTOR

    summary = {
        "model": "ARIMA",
        "input_mode": "OHLC_only",
        "split": split,
        "mae": {
            "open": float(mae[0]),
            "high": float(mae[1]),
            "low": float(mae[2]),
            "close": float(mae[3]),
            "avg": float(mae.mean()),
        },
        "mae_pips": {
            "open": float(mae_pips[0]),
            "high": float(mae_pips[1]),
            "low": float(mae_pips[2]),
            "close": float(mae_pips[3]),
            "avg": float(mae_pips.mean()),
        },
        "targets": model_meta,
    }
    return summary


def main():
    inp, cfg, rt = parse_args()

    print("=== CONFIG ===")
    print(asdict(inp))
    print(asdict(cfg))
    print(asdict(rt))
    print("================")

    data_path = resolve_data_path(inp.data)
    df_raw = load_data(data_path, inp.sep)
    ohlc_df = find_ohlc(df_raw, inp.columns)
    if len(ohlc_df) < 50:
        raise ValueError(f"Not enough rows for ARIMA baseline: {len(ohlc_df)}")

    by_col = cfg.order_by_column or {}
    col_results = {}
    for col in BASE_COLS:
        requested_order = by_col.get(col, cfg.order)
        if cfg.order_search:
            print(
                f"[ARIMA] Searching {col} order on train set only with "
                f"{cfg.selection_metric.upper()} over {len(_candidate_orders(cfg, requested_order))} candidates"
            )
        else:
            print(f"[ARIMA] Fitting {col} with requested order={requested_order}")
        col_results[col] = walk_forward_forecast(ohlc_df[col].values, cfg.split, requested_order, cfg)
        print(
            f"[ARIMA] {col}: selected_order={col_results[col]['selected_order']} "
            f"used_order={col_results[col]['used_order']} "
            f"train={col_results[col]['train_samples']} test={col_results[col]['test_samples']}"
        )

    test_sizes = {col_results[col]["test_samples"] for col in BASE_COLS}
    if len(test_sizes) != 1:
        raise RuntimeError(f"Per-column test lengths differ unexpectedly: {test_sizes}")

    out_df = pd.DataFrame(
        {
            "pred_open": col_results["open"]["preds"],
            "pred_high": col_results["high"]["preds"],
            "pred_low": col_results["low"]["preds"],
            "pred_close": col_results["close"]["preds"],
            "true_open": col_results["open"]["targets"],
            "true_high": col_results["high"]["targets"],
            "true_low": col_results["low"]["targets"],
            "true_close": col_results["close"]["targets"],
        }
    )

    out_path = _resolve_runtime_path(rt.out_csv)
    if out_path is None:
        raise ValueError("Output CSV path must not be empty.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path.resolve()}")

    model_meta = {
        col: {
            "requested_order": list((cfg.order_by_column or {}).get(col, cfg.order)),
            "order_search": cfg.order_search,
            "selected_order": list(col_results[col]["selected_order"]),
            "used_order": list(col_results[col]["used_order"]),
            "train_samples": col_results[col]["train_samples"],
            "test_samples": col_results[col]["test_samples"],
            "aic": col_results[col]["aic"],
            "bic": col_results[col]["bic"],
            "search": col_results[col]["search"],
        }
        for col in BASE_COLS
    }
    summary = build_summary(out_df, model_meta, cfg.split)
    print(
        "MAE pips (open/high/low/close/avg):",
        ", ".join(
            f"{summary['mae_pips'][name]:.4f}"
            for name in ["open", "high", "low", "close", "avg"]
        ),
    )

    if rt.summary_json:
        summary_path = _resolve_runtime_path(rt.summary_json)
        if summary_path is None:
            raise ValueError("Summary JSON path must not be empty.")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
