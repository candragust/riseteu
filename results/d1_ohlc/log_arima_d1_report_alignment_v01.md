# Log Penyelarasan Report ARIMA 1D

Tanggal kerja: 2026-04-23

## Tujuan

Menyetarakan paket report `ARIMA 1D` dengan pola report `4H`, sehingga `D1` tidak hanya memiliki halaman output utama, tetapi juga:

- halaman diagnostik residual
- halaman detail `MAE vs ATR`
- aset statis untuk kebutuhan Word
- index report `D1` yang menunjuk ke artefak terbaru

## File Baru

- `Arima/result/d1_ohlc/arima_residual_diagnostics_wf72_test1_last5_d1_v01.html`
- `Arima/result/d1_ohlc/mae_atr_wf72_test1_arima_d1_last5_v02.html`
- `Arima/result/d1_ohlc/screenshots/ohlc_dot_wf72_test1_last5_arima_d1_v01.png`
- `Arima/result/d1_ohlc/screenshots/mae_atr_fold21_overlay_arima_d1_v01.png`
- `Arima/result/d1_ohlc/screenshots/mae_atr_fold21_error_avg_atr_arima_d1_v01.png`
- `results/d1_ohlc/index_d1_reports_v03.html`
- `build_arima_d1_reports_v01.py`

## File Yang Diubah

- `generate_model_diagnostics_docs.py`
  - `make_arima_html(...)` diparameterisasi agar bisa dipakai ulang untuk `4H` dan `D1`

## Ringkasan Hasil

- `ARIMA Output Report D1` tetap memakai:
  - `Arima/result/d1_ohlc/arima_wf72_test1_last5_report_v01.html`
- `ARIMA Residual Diagnostics D1` sekarang tersedia:
  - `Arima/result/d1_ohlc/arima_residual_diagnostics_wf72_test1_last5_d1_v01.html`
- `ARIMA Detail MAE vs ATR D1` sekarang tersedia:
  - `Arima/result/d1_ohlc/mae_atr_wf72_test1_arima_d1_last5_v02.html`

## Catatan Analitis

Detail `fold 21` yang dipakai di halaman `MAE vs ATR`:

- sampel test: `22`
- `MAE avg = 36.0492 pips`
- `ATR6 mean = 74.6227 pips`
- `ATR12 mean = 76.3758 pips`
- `MAE avg ≈ 48.31% ATR6`
- `MAE avg ≈ 47.20% ATR12`
- `100%` error avg berada di bawah `ATR6`
- `100%` error avg berada di bawah `ATR12`

## Catatan Integrasi

- Belum ada perubahan ke `master_buku_progres_tesis_v17.docx`
- Jika hasil `ARIMA 1D` ini nanti ingin dimasukkan ke Word, gunakan `v18` atau lebih baru
- Untuk review cepat seluruh report `D1`, gunakan:
  - `results/d1_ohlc/index_d1_reports_v03.html`
