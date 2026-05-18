#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


ROOT = Path(__file__).resolve().parents[1]
PIP_FACTOR = 10_000.0
TARGETS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class ProfileConfig:
    profile: str
    title_label: str
    result_dir: Path
    comparison_metrics: Path


PROFILES = (
    ProfileConfig(
        profile="h4",
        title_label="EUR/USD H4 TVT v02 Last3",
        result_dir=ROOT / "Arima" / "result" / "tvt_v02" / "h4_evaluation_last3",
        comparison_metrics=ROOT
        / "comparison"
        / "tvt_v02"
        / "h4"
        / "comparison_models_h4_tvt_v02_last3_v01_arima_metrics.csv",
    ),
    ProfileConfig(
        profile="d1",
        title_label="EUR/USD D1 TVT v02 Last3",
        result_dir=ROOT / "Arima" / "result" / "tvt_v02" / "d1_evaluation_last3",
        comparison_metrics=ROOT
        / "comparison"
        / "tvt_v02"
        / "d1"
        / "comparison_models_d1_tvt_v02_last3_v01_arima_metrics.csv",
    ),
)


def safe_float(value: float | np.floating | None) -> float:
    if value is None:
        return float("nan")
    try:
        if math.isnan(float(value)):
            return float("nan")
    except TypeError:
        return float("nan")
    return float(value)


def lag1_autocorr(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    x = values[:-1]
    y = values[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ljung_box_pvalue(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag + 1:
        return float("nan")
    result = acorr_ljungbox(values, lags=[lag], return_df=True)
    return float(result["lb_pvalue"].iloc[0])


def jarque_bera_pvalue(values: np.ndarray) -> float:
    if len(values) < 3:
        return float("nan")
    return float(stats.jarque_bera(values).pvalue)


def summarize_residuals(df: pd.DataFrame, *, scope: str, fold: int | str, target: str) -> dict:
    residual = df[f"resid_{target}_pips"].to_numpy(dtype=float)
    return {
        "scope": scope,
        "fold": fold,
        "target": target,
        "n": int(len(residual)),
        "bias_mean_pips": safe_float(np.mean(residual)),
        "median_resid_pips": safe_float(np.median(residual)),
        "std_resid_pips": safe_float(np.std(residual, ddof=1)) if len(residual) > 1 else float("nan"),
        "mae_pips": safe_float(np.mean(np.abs(residual))),
        "rmse_pips": safe_float(np.sqrt(np.mean(residual**2))),
        "min_resid_pips": safe_float(np.min(residual)),
        "max_resid_pips": safe_float(np.max(residual)),
        "lag1_autocorr": safe_float(lag1_autocorr(residual)),
        "ljungbox_p_lag5": safe_float(ljung_box_pvalue(residual, 5)),
        "ljungbox_p_lag10": safe_float(ljung_box_pvalue(residual, 10)),
        "ljungbox_p_lag20": safe_float(ljung_box_pvalue(residual, 20)),
        "jarque_bera_pvalue": safe_float(jarque_bera_pvalue(residual)),
    }


def load_residuals(config: ProfileConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in (19, 20, 21):
        path = config.result_dir / f"fold{fold}_preds.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        expected = {f"pred_{target}" for target in TARGETS} | {f"true_{target}" for target in TARGETS}
        missing = expected.difference(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        out = pd.DataFrame({"profile": config.profile, "fold": fold, "row_in_fold": np.arange(len(df))})
        for target in TARGETS:
            out[f"true_{target}"] = df[f"true_{target}"].astype(float)
            out[f"pred_{target}"] = df[f"pred_{target}"].astype(float)
            out[f"resid_{target}_pips"] = (out[f"true_{target}"] - out[f"pred_{target}"]) * PIP_FACTOR
            out[f"abs_resid_{target}_pips"] = out[f"resid_{target}_pips"].abs()
        frames.append(out)
    combined = pd.concat(frames, ignore_index=True)
    combined["global_index"] = np.arange(len(combined))
    return combined


def build_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target in TARGETS:
        rows.append(summarize_residuals(residuals, scope="combined", fold="all", target=target))
    residuals = residuals.copy()
    residuals["resid_avg_pips"] = residuals[[f"resid_{target}_pips" for target in TARGETS]].mean(axis=1)
    residuals["abs_resid_avg_pips"] = residuals[[f"abs_resid_{target}_pips" for target in TARGETS]].mean(axis=1)
    for fold, fold_df in residuals.groupby("fold", sort=True):
        for target in TARGETS:
            rows.append(summarize_residuals(fold_df, scope="fold", fold=int(fold), target=target))
    summary = pd.DataFrame(rows)
    return summary


def build_validation(config: ProfileConfig, summary: pd.DataFrame) -> pd.DataFrame:
    comparison = pd.read_csv(config.comparison_metrics)
    rows: list[dict] = []
    for _, comp in comparison.iterrows():
        fold = int(comp["fold"])
        fold_summary = summary[(summary["scope"] == "fold") & (summary["fold"] == fold)]
        for target in TARGETS:
            calc = float(fold_summary.loc[fold_summary["target"] == target, "mae_pips"].iloc[0])
            expected = float(comp[f"mae_{target}_pips"])
            rows.append(
                {
                    "profile": config.profile,
                    "fold": fold,
                    "target": target,
                    "computed_mae_pips": calc,
                    "comparison_mae_pips": expected,
                    "abs_diff_pips": abs(calc - expected),
                    "status": "ok" if abs(calc - expected) < 1e-9 else "mismatch",
                }
            )
    return pd.DataFrame(rows)


def format_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def make_html(config: ProfileConfig, residuals: pd.DataFrame, summary: pd.DataFrame, validation: pd.DataFrame) -> str:
    combined = summary[summary["scope"] == "combined"].copy()
    close = combined[combined["target"] == "close"].iloc[0]
    avg_mae = combined["mae_pips"].mean()
    validation_status = "OK" if (validation["status"] == "ok").all() else "MISMATCH"

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Close residual across combined fold 19-21",
            "Close residual histogram",
            "Mean residual bias by OHLC component",
            "Ljung-Box p-value by OHLC component",
        ),
        vertical_spacing=0.09,
    )
    fig.add_trace(
        go.Scatter(
            x=residuals["global_index"],
            y=residuals["resid_close_pips"],
            mode="lines",
            name="Close residual",
            line=dict(color="#2563eb", width=1.3),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_trace(
        go.Histogram(
            x=residuals["resid_close_pips"],
            nbinsx=40,
            name="Close residual histogram",
            marker_color="#1d4ed8",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=combined["target"],
            y=combined["bias_mean_pips"],
            name="Mean residual",
            marker_color="#7c3aed",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_trace(
        go.Bar(
            x=combined["target"],
            y=combined["ljungbox_p_lag10"],
            name="Ljung-Box p lag10",
            marker_color="#dc2626",
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="green", row=4, col=1)
    fig.update_layout(height=1400, showlegend=False, template="plotly_white")
    fig.update_yaxes(title_text="pips", row=1, col=1)
    fig.update_yaxes(title_text="count", row=2, col=1)
    fig.update_yaxes(title_text="pips", row=3, col=1)
    fig.update_yaxes(title_text="p-value", row=4, col=1)
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    combined_display_cols = [
        "target",
        "n",
        "bias_mean_pips",
        "std_resid_pips",
        "mae_pips",
        "rmse_pips",
        "lag1_autocorr",
        "ljungbox_p_lag10",
        "ljungbox_p_lag20",
        "jarque_bera_pvalue",
    ]
    fold_display_cols = [
        "fold",
        "target",
        "n",
        "bias_mean_pips",
        "std_resid_pips",
        "mae_pips",
        "rmse_pips",
        "lag1_autocorr",
        "ljungbox_p_lag10",
    ]

    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 28px; color: #111827; }
      h1, h2 { color: #111827; }
      table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
      th, td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; }
      th:first-child, td:first-child { text-align: left; }
      th { background: #f3f4f6; }
      .note { background: #f9fafb; border-left: 4px solid #2563eb; padding: 10px 12px; }
      .warn { background: #fff7ed; border-left: 4px solid #f97316; padding: 10px 12px; }
    </style>
    """
    combined_html = combined[combined_display_cols].to_html(index=False, float_format=lambda x: f"{x:.6f}")
    fold_html = summary[summary["scope"] == "fold"][fold_display_cols].to_html(
        index=False, float_format=lambda x: f"{x:.6f}"
    )
    validation_html = validation.to_html(index=False, float_format=lambda x: f"{x:.12f}")

    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>ARIMA Residual Diagnostics - {config.title_label}</title>{style}</head>
<body>
<h1>ARIMA Residual Diagnostics - {config.title_label}</h1>
<p class="note"><strong>Scope:</strong> TVT v02 last3, fold 19-21, sumber data hanya dari <code>{config.result_dir.relative_to(ROOT)}</code>. Residual dihitung sebagai <code>actual - predicted</code> dalam pips.</p>
<p class="note"><strong>Validasi sumber:</strong> MAE residual dibandingkan ulang dengan <code>{config.comparison_metrics.relative_to(ROOT)}</code>. Status: <strong>{validation_status}</strong>.</p>
<p><strong>Ringkasan close residual:</strong> n = {int(close['n'])}, bias = {format_float(close['bias_mean_pips'])} pips, std = {format_float(close['std_resid_pips'])} pips, MAE = {format_float(close['mae_pips'])} pips, RMSE = {format_float(close['rmse_pips'])} pips, lag-1 autocorr = {format_float(close['lag1_autocorr'])}, Ljung-Box p lag 10 = {format_float(close['ljungbox_p_lag10'])}, lag 20 = {format_float(close['ljungbox_p_lag20'])}.</p>
<p><strong>Mean MAE empat komponen:</strong> {format_float(avg_mae)} pips.</p>
<p class="warn"><strong>Catatan interpretasi:</strong> p-value Ljung-Box di bawah 0.05 mengindikasikan residual masih memiliki autokorelasi pada lag yang diuji. P-value Jarque-Bera di bawah 0.05 mengindikasikan residual tidak berdistribusi normal; ini umum pada data finansial dan tidak sendirian menentukan kelayakan model.</p>
<h2>Combined Fold 19-21 Summary</h2>
{combined_html}
<h2>Per-Fold Summary</h2>
{fold_html}
<h2>Validation Against Comparison Metrics</h2>
{validation_html}
<h2>Diagnostic Plots</h2>
{plot_html}
</body>
</html>
"""


def write_profile(config: ProfileConfig) -> None:
    residuals = load_residuals(config)
    summary = build_summary(residuals)
    validation = build_validation(config, summary)

    residuals_path = config.result_dir / "arima_residual_diagnostics_tvt_v02_last3_residuals.csv"
    summary_path = config.result_dir / "arima_residual_diagnostics_tvt_v02_last3_summary.csv"
    validation_path = config.result_dir / "arima_residual_diagnostics_tvt_v02_last3_validation.csv"
    html_path = config.result_dir / "arima_residual_diagnostics_tvt_v02_last3.html"

    residuals.to_csv(residuals_path, index=False)
    summary.to_csv(summary_path, index=False)
    validation.to_csv(validation_path, index=False)
    html_path.write_text(make_html(config, residuals, summary, validation), encoding="utf-8")

    if not (validation["status"] == "ok").all():
        raise RuntimeError(f"Validation mismatch for {config.profile}: {validation_path}")

    print(f"[{config.profile}] residuals: {residuals_path.relative_to(ROOT)}")
    print(f"[{config.profile}] summary:   {summary_path.relative_to(ROOT)}")
    print(f"[{config.profile}] html:      {html_path.relative_to(ROOT)}")


def main() -> int:
    for profile in PROFILES:
        write_profile(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
