import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.diagnostic import acorr_ljungbox


PROJECT_ROOT = Path("/home/hduser/jupyter/gust/RisetEU")
FOLDS = [17, 18, 19, 20, 21]


def analyze_nn_histories(base_dir: Path, model_name: str):
    rows = []
    for fold in FOLDS:
        hist = pd.read_csv(base_dir / f"fold{fold:02d}_history.csv")
        val = hist["val_loss"]
        val_min_idx = int(val.idxmin())
        overfit_after_best = bool(
            (hist.loc[val_min_idx:, "loss"].iloc[-1] < hist.loc[val_min_idx, "loss"])
            and (val.iloc[-1] > val.min())
        )
        dloss = hist["loss"].diff().dropna()
        dval = val.diff().dropna()
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "epochs_run": int(len(hist)),
                "loss_start": float(hist["loss"].iloc[0]),
                "loss_end": float(hist["loss"].iloc[-1]),
                "loss_min": float(hist["loss"].min()),
                "loss_min_epoch": int(hist["loss"].idxmin() + 1),
                "val_start": float(val.iloc[0]),
                "val_end": float(val.iloc[-1]),
                "val_min": float(val.min()),
                "val_min_epoch": int(val_min_idx + 1),
                "final_gap": float(val.iloc[-1] - hist["loss"].iloc[-1]),
                "overfit_after_best": overfit_after_best,
                "mean_dloss": float((-dloss).mean()),
                "last5_abs_mean_dloss": float(dloss.tail(5).abs().mean() if len(dloss) >= 5 else dloss.abs().mean()),
                "loss_up_steps": int((dloss > 0).sum()),
                "loss_down_steps": int((dloss < 0).sum()),
                "val_loss_up_steps": int((dval > 0).sum()),
                "val_loss_down_steps": int((dval < 0).sum()),
                "val_min_frac": float((val_min_idx + 1) / len(hist)),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "mean_epochs_run": float(df["epochs_run"].mean()),
        "mean_loss_end": float(df["loss_end"].mean()),
        "mean_val_end": float(df["val_end"].mean()),
        "mean_val_min": float(df["val_min"].mean()),
        "mean_final_gap": float(df["final_gap"].mean()),
        "overfit_fold_count": int(df["overfit_after_best"].sum()),
        "mean_val_min_frac": float(df["val_min_frac"].mean()),
        "mean_last5_abs_mean_dloss": float(df["last5_abs_mean_dloss"].mean()),
    }
    return df, summary


def classify_nn(summary: dict):
    if summary["mean_loss_end"] > 0.005:
        return "indikasi underfitting"
    if summary["overfit_fold_count"] == 0:
        return "konvergen stabil tanpa indikasi overfitting yang berarti"
    if summary["mean_val_min_frac"] >= 0.94:
        return "konvergen dengan overfitting ringan"
    return "konvergen dengan overfitting ringan hingga moderat"


def analyze_arima(arima_dir: Path):
    fold_rows = []
    order_rows = []
    combined_close_resid_pips = []
    combined_mae_avg_pips = []

    for fold in FOLDS:
        preds = pd.read_csv(arima_dir / f"fold{fold:02d}_preds.csv")
        summary = json.loads((arima_dir / f"fold{fold:02d}_summary.json").read_text(encoding="utf-8"))

        resid_close_pips = (preds["true_close"] - preds["pred_close"]) * 10000
        resid_avg_pips = (
            preds[["true_open", "true_high", "true_low", "true_close"]].values
            - preds[["pred_open", "pred_high", "pred_low", "pred_close"]].values
        ).mean(axis=1) * 10000
        mae_avg_pips = np.abs(
            preds[["pred_open", "pred_high", "pred_low", "pred_close"]].values
            - preds[["true_open", "true_high", "true_low", "true_close"]].values
        ).mean(axis=1) * 10000

        lb_lag = min(10, max(1, len(resid_close_pips) // 5))
        lb = acorr_ljungbox(resid_close_pips, lags=[lb_lag], return_df=True)

        fold_rows.append(
            {
                "fold": fold,
                "mae_avg_pips": float(mae_avg_pips.mean()),
                "close_bias_pips": float(resid_close_pips.mean()),
                "close_std_pips": float(resid_close_pips.std(ddof=1)),
                "close_lag1_acf": float(pd.Series(resid_close_pips).autocorr(lag=1)),
                "ljung_box_lag": int(lb_lag),
                "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
            }
        )

        for target in ["open", "high", "low", "close"]:
            t = summary["targets"][target]
            order_rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "selected_order": tuple(t["selected_order"]),
                    "aic": float(t["aic"]),
                    "bic": float(t["bic"]),
                }
            )

        combined_close_resid_pips.extend(resid_close_pips.tolist())
        combined_mae_avg_pips.extend(mae_avg_pips.tolist())

    fold_df = pd.DataFrame(fold_rows)
    order_df = pd.DataFrame(order_rows)

    combined_close_resid_pips = pd.Series(combined_close_resid_pips, dtype=float)
    combined_lb_10 = acorr_ljungbox(combined_close_resid_pips, lags=[10], return_df=True)
    combined_lb_20 = acorr_ljungbox(combined_close_resid_pips, lags=[20], return_df=True)

    summary = {
        "mean_mae_avg_pips": float(fold_df["mae_avg_pips"].mean()),
        "mean_close_bias_pips": float(fold_df["close_bias_pips"].mean()),
        "mean_close_std_pips": float(fold_df["close_std_pips"].mean()),
        "mean_ljung_box_pvalue": float(fold_df["ljung_box_pvalue"].mean()),
        "all_folds_lb_pass": bool((fold_df["ljung_box_pvalue"] > 0.05).all()),
        "combined_close_bias_pips": float(combined_close_resid_pips.mean()),
        "combined_close_std_pips": float(combined_close_resid_pips.std(ddof=1)),
        "combined_close_lag1_acf": float(combined_close_resid_pips.autocorr(lag=1)),
        "combined_ljung_box_pvalue_lag10": float(combined_lb_10["lb_pvalue"].iloc[0]),
        "combined_ljung_box_pvalue_lag20": float(combined_lb_20["lb_pvalue"].iloc[0]),
        "combined_mae_avg_pips": float(np.mean(combined_mae_avg_pips)),
    }
    return fold_df, order_df, summary, combined_close_resid_pips


def classify_arima(summary: dict):
    if summary["all_folds_lb_pass"] and summary["combined_ljung_box_pvalue_lag10"] > 0.05:
        return "forecast residual relatif stabil tanpa bukti kuat autokorelasi serial pada lag 10"
    if summary["combined_ljung_box_pvalue_lag10"] > 0.05 and summary["combined_ljung_box_pvalue_lag20"] <= 0.05:
        return "forecast residual cukup stabil, tetapi masih ada indikasi serial dependence lemah pada horizon gabungan yang lebih panjang"
    return "forecast residual masih menunjukkan indikasi autokorelasi yang perlu dicermati"


def make_arima_html(
    fold_df: pd.DataFrame,
    order_df: pd.DataFrame,
    summary: dict,
    combined_close_resid_pips: pd.Series,
    out_path: Path,
):
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Close Residual (All Last5 Fold Test Candles Combined)",
            "Histogram Close Residual (pips)",
            "Ljung-Box p-value per Fold (Close Residual)",
        ],
        vertical_spacing=0.09,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(combined_close_resid_pips))),
            y=combined_close_resid_pips,
            mode="lines",
            name="Close Residual (pips)",
            line=dict(color="#2563eb", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, row=1, col=1, line_dash="dash", line_color="gray")
    fig.add_trace(
        go.Histogram(
            x=combined_close_resid_pips,
            nbinsx=40,
            name="Residual Histogram",
            marker_color="#1d4ed8",
            opacity=0.85,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=fold_df["fold"],
            y=fold_df["ljung_box_pvalue"],
            name="Ljung-Box p-value",
            marker_color="#dc2626",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0.05, row=3, col=1, line_dash="dash", line_color="green")
    fig.update_yaxes(title_text="Residual (pips)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="p-value", row=3, col=1)
    fig.update_xaxes(title_text="Combined Test Index", row=1, col=1)
    fig.update_xaxes(title_text="Residual (pips)", row=2, col=1)
    fig.update_xaxes(title_text="Fold", row=3, col=1)
    fig.update_layout(template="plotly_white", height=1400, showlegend=False)

    order_count_df = (
        order_df.groupby(["target", "selected_order"]).size().reset_index(name="count").sort_values(["target", "count"], ascending=[True, False])
    )

    style = """
    body { font-family: Arial, sans-serif; margin: 20px; color: #222; }
    table { border-collapse: collapse; margin-bottom: 20px; }
    th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: right; }
    th { background: #f5f5f5; }
    td:first-child, th:first-child { text-align: left; }
    .note { max-width: 1100px; line-height: 1.45; }
    """

    html = [
        "<html><head><meta charset='UTF-8'><title>ARIMA Residual Diagnostics WF72m/1m Last5</title>",
        f"<style>{style}</style></head><body>",
        "<h1>ARIMA Residual Diagnostics WF72m/1m Last5</h1>",
        "<p class='note'><strong>Ruang lingkup:</strong> baseline ARIMA OHLC-only, walk-forward fixed 72 bulan train / 1 bulan test, fold 17-21.</p>",
        "<p class='note'><strong>Interpretasi singkat:</strong> "
        f"mean MAE avg = {summary['mean_mae_avg_pips']:.4f} pips; "
        f"combined close bias = {summary['combined_close_bias_pips']:.4f} pips; "
        f"Ljung-Box p-value gabungan lag 10 = {summary['combined_ljung_box_pvalue_lag10']:.4f}; "
        f"lag 20 = {summary['combined_ljung_box_pvalue_lag20']:.4f}.</p>",
        "<h2>Ringkasan Fold</h2>",
        fold_df.to_html(index=False, float_format="%.4f"),
        "<h2>Frekuensi Orde Terpilih</h2>",
        order_count_df.to_html(index=False),
        "<h2>Visual</h2>",
        fig.to_html(full_html=False, include_plotlyjs="cdn"),
        "</body></html>",
    ]
    out_path.write_text("".join(html), encoding="utf-8")


def write_documentation(md_path: Path, html_path: Path, lstm_summary: dict, bilstm_summary: dict, arima_summary: dict):
    lstm_class = classify_nn(lstm_summary)
    bilstm_class = classify_nn(bilstm_summary)
    arima_class = classify_arima(arima_summary)

    best_title = "Analisis Konvergensi, Overfitting, dan Diagnostik Residual Model"
    alt_title_1 = "Analisis Dinamika Pelatihan dan Diagnostik Error/Residual Model"
    alt_title_2 = "Analisis Generalisasi Model pada Skenario Walk-Forward 72 Bulan/1 Bulan"

    md = f"""# Analisis Konvergensi, Overfitting, dan Diagnostik Residual Model

## Saran Judul

Judul yang paling saya sarankan untuk analisis ini adalah:

**{best_title}**

Alternatif yang masih aman:
- {alt_title_1}
- {alt_title_2}

## Ruang Lingkup Analisis

Analisis ini mencakup tiga hal yang berbeda secara metodologis:

1. **FLF-LSTM dan FLF-BiLSTM** dianalisis melalui kurva `loss` dan `val_loss` untuk melihat konvergensi, indikasi overfitting, dan stabilitas generalisasi.
2. **Gradient loss** pada report repo ini diperlakukan sebagai **proxy perubahan loss per epoch (`dLoss`)**, sehingga fungsinya adalah membaca kecepatan konvergensi dan plateau, bukan gradient parameter model.
3. **ARIMA** tidak memiliki kurva loss epoch seperti neural network, sehingga analisis yang setara dilakukan melalui **diagnostik residual forecast**, AIC/BIC, dan uji autokorelasi residual.

Skenario evaluasi yang dipakai sama untuk semua model, yaitu **walk-forward fixed 72 bulan training, 1 bulan testing, fold 17-21**.

## Hasil FLF-LSTM

- Rata-rata epoch berjalan: **{lstm_summary['mean_epochs_run']:.1f}**
- Rata-rata `loss_end`: **{lstm_summary['mean_loss_end']:.6f}**
- Rata-rata `val_end`: **{lstm_summary['mean_val_end']:.6f}**
- Rata-rata gap akhir `val_loss - loss`: **{lstm_summary['mean_final_gap']:.6f}**
- Fold yang menunjukkan pola overfitting setelah epoch terbaik: **{lstm_summary['overfit_fold_count']} dari 5 fold**
- Rata-rata posisi epoch terbaik validation terhadap total epoch: **{lstm_summary['mean_val_min_frac']*100:.2f}%**

Interpretasi:

**FLF-LSTM {lstm_class}.**

Secara umum training loss dan validation loss sama-sama turun hingga sangat kecil. Tidak ada indikasi underfitting dan tidak ada tanda divergensi training. Overfitting yang muncul bersifat ringan, karena validation loss pada beberapa fold hanya memburuk tipis setelah mencapai titik minimum. Plateau di akhir training juga jelas, ditunjukkan oleh rata-rata perubahan loss 5 epoch terakhir yang sangat kecil.

## Hasil FLF-BiLSTM

- Rata-rata epoch berjalan: **{bilstm_summary['mean_epochs_run']:.1f}**
- Rata-rata `loss_end`: **{bilstm_summary['mean_loss_end']:.6f}**
- Rata-rata `val_end`: **{bilstm_summary['mean_val_end']:.6f}**
- Rata-rata gap akhir `val_loss - loss`: **{bilstm_summary['mean_final_gap']:.6f}**
- Fold yang menunjukkan pola overfitting setelah epoch terbaik: **{bilstm_summary['overfit_fold_count']} dari 5 fold**
- Rata-rata posisi epoch terbaik validation terhadap total epoch: **{bilstm_summary['mean_val_min_frac']*100:.2f}%**

Interpretasi:

**FLF-BiLSTM {bilstm_class}.**

Model ini juga tidak underfit dan berhasil konvergen. Namun, dibanding FLF-LSTM, titik minimum validation loss lebih sering tercapai lebih awal, lalu validation loss bergerak naik ketika training loss masih turun. Itu menunjukkan kecenderungan overfitting yang lebih nyata daripada FLF-LSTM. Dengan kata lain, BiLSTM masih belajar, tetapi generalisasi mulai jenuh lebih cepat.

## Perbandingan LSTM vs BiLSTM

- Kedua model **tidak underfit**
- Kedua model **tidak menunjukkan training yang gagal atau divergen**
- Keduanya berada pada rezim **konvergen**
- Perbedaannya ada pada **stabilitas generalisasi**

Kesimpulan komparatif:

**FLF-LSTM memperlihatkan dinamika pelatihan yang lebih sehat dan generalisasi yang lebih stabil daripada FLF-BiLSTM pada skenario WF72m/1m fold 17-21.**

## Hasil ARIMA

- Mean MAE avg: **{arima_summary['mean_mae_avg_pips']:.4f} pips**
- Mean close bias per fold: **{arima_summary['mean_close_bias_pips']:.4f} pips**
- Combined close bias: **{arima_summary['combined_close_bias_pips']:.4f} pips**
- Combined close residual std: **{arima_summary['combined_close_std_pips']:.4f} pips**
- Combined lag-1 autocorrelation residual close: **{arima_summary['combined_close_lag1_acf']:.4f}**
- Combined Ljung-Box p-value lag 10: **{arima_summary['combined_ljung_box_pvalue_lag10']:.4f}**
- Combined Ljung-Box p-value lag 20: **{arima_summary['combined_ljung_box_pvalue_lag20']:.4f}**

Interpretasi:

**ARIMA {arima_class}.**

Karena ARIMA bukan model berbasis epoch, maka ia tidak dianalisis dengan kurva `loss`/`val_loss`. Diagnostik yang relevan adalah apakah residual forecast masih menyimpan pola serial. Pada hasil ini, p-value Ljung-Box per fold masih berada di atas 0.05, sehingga tidak ada bukti kuat autokorelasi residual close pada masing-masing fold. Namun ketika residual digabung lintas fold, nilai lag 20 berada sangat dekat dengan batas 0.05, sehingga masih ada indikasi lemah bahwa struktur temporal belum sepenuhnya hilang.

## Makna Metodologis

1. **Analisis LSTM/BiLSTM** sebaiknya disebut:
   - analisis konvergensi dan overfitting
   - atau analisis dinamika pelatihan dan generalisasi
2. **Analisis ARIMA** sebaiknya disebut:
   - analisis diagnostik residual
   - atau analisis error/residual forecast
3. Jika ingin semua digabung dalam satu subbagian, istilah paling aman adalah:
   - **analisis konvergensi, overfitting, dan diagnostik residual model**

## Kesimpulan Akhir

- **FLF-LSTM**: konvergen, tidak underfit, dengan overfitting ringan
- **FLF-BiLSTM**: konvergen, tidak underfit, tetapi lebih rentan overfitting
- **ARIMA**: tidak memakai analisis loss curve; evaluasi yang tepat adalah diagnostik residual, AIC/BIC, dan autokorelasi residual

Dokumen ini paling tepat ditempatkan sebagai bahan untuk:
- pembahasan hasil eksperimen
- jawaban sidang/penguji
- atau lampiran analisis teknis model
"""

    md_path.write_text(md, encoding="utf-8")

    html = [
        "<html><head><meta charset='UTF-8'><title>Analisis Konvergensi, Overfitting, dan Diagnostik Residual Model</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.6;color:#222;} h1,h2{color:#163a6b;} code{background:#f5f5f5;padding:1px 4px;border-radius:4px;} ul,ol{line-height:1.6;} strong{color:#111;}</style>",
        "</head><body>",
    ]
    for line in md.splitlines():
        if line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("- "):
            if not html[-1].startswith("<ul>"):
                html.append("<ul>")
            html.append(f"<li>{line[2:]}</li>")
        elif line.startswith("1. "):
            if not html[-1].startswith("<ol>"):
                html.append("<ol>")
            html.append(f"<li>{line[3:]}</li>")
        elif line.startswith("2. ") or line.startswith("3. "):
            html.append(f"<li>{line[3:]}</li>")
        elif line.strip() == "":
            if html[-1] == "<ul>":
                html.append("</ul>")
            elif html[-1] == "<ol>":
                html.append("</ol>")
            else:
                html.append("")
        else:
            html.append(f"<p>{line}</p>")
    if html[-1] == "<ul>":
        html.append("</ul>")
    if html[-1] == "<ol>":
        html.append("</ol>")
    html.append("</body></html>")
    html_path.write_text("\n".join(html), encoding="utf-8")


def main():
    lstm_hist_dir = PROJECT_ROOT / "FLF_LSTM" / "results" / "wf72_test1_lstm_last5"
    bilstm_hist_dir = PROJECT_ROOT / "results" / "rolling_train72_test1_last5"
    arima_dir = PROJECT_ROOT / "Arima" / "result" / "arima_wf72_test1_last5"

    lstm_df, lstm_summary = analyze_nn_histories(lstm_hist_dir, "FLF-LSTM")
    bilstm_df, bilstm_summary = analyze_nn_histories(bilstm_hist_dir, "FLF-BiLSTM")
    arima_fold_df, arima_order_df, arima_summary, combined_close_resid_pips = analyze_arima(arima_dir)

    arima_html_path = PROJECT_ROOT / "Arima" / "result" / "arima_residual_diagnostics_wf72_test1_last5.html"
    make_arima_html(arima_fold_df, arima_order_df, arima_summary, combined_close_resid_pips, arima_html_path)

    md_path = PROJECT_ROOT / "bukuThesis" / "bahan" / "analisis_konvergensi_dan_diagnostik_model_wf72_test1_last5.md"
    html_path = PROJECT_ROOT / "bukuThesis" / "bahan" / "analisis_konvergensi_dan_diagnostik_model_wf72_test1_last5.html"
    write_documentation(md_path, html_path, lstm_summary, bilstm_summary, arima_summary)

    print(f"Wrote {arima_html_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
