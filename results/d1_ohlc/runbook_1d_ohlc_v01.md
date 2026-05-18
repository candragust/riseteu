# Runbook 1D OHLC v01

Tanggal: 2026-04-22
Lokasi kerja: `/home/hduser/jupyter/gust/RisetEU`

## Tujuan

Dokumen ini menyimpan status eksperimen `EUR/USD 1D OHLC` agar konteks kerja tetap bisa dilanjutkan walaupun riwayat chat terpotong.

## Keputusan Eksperimen

- Data: `results/EURUSD_D1_clean.csv`
- Protokol: `walk-forward 72 bulan train + 1 bulan test`
- Mode: `fixed window`
- Fold utama untuk analisis: `17-21`
- Metrik utama: `MAE avg pips`
- Metrik pendukung: `corr2 avg HLC`, `Directional Accuracy body candle`

## Struktur Output

- Data bersih: `results/EURUSD_D1_clean.csv`
- Manifest setup: `results/d1_ohlc/setup_manifest_v01.json`
- Ringkasan eksekusi 1D: folder `results/d1_ohlc/`
- LSTM 1D last5: `FLF_LSTM/results/d1_ohlc/wf72_test1_last5_v01/`
- BiLSTM 1D last5: `FLF_BILSTM/results/d1_ohlc/wf72_test1_last5_v01/`
- ARIMA 1D last5: `Arima/result/d1_ohlc/arima_wf72_test1_last5_v01/`
- Comparison final 1D: `comparison/d1_ohlc/`

## Config Beku Yang Dipakai

- LSTM final: `FLF_LSTM/configs/d1_ohlc/lstm_flf_config_d1_ohlc_best_v02.json`
- BiLSTM final: `FLF_BILSTM/configs/d1_ohlc/bilstm_flf_config_d1_ohlc_best_v02.json`
- ARIMA final: `Arima/arima_baseline_config_d1_ohlc_v01.json`

## Ringkasan Tuning 1D

### FLF-LSTM

- Config terpilih:
  - `window=4`
  - `units=128`
  - `lr=0.0009`
  - `epochs=60`
  - `lambda=0.9`
  - `sigma=0.1`
  - `batch=64`
  - `dropout=0.0`
  - `l2_reg=0.0`
- Sumber audit:
  - `FLF_LSTM/results/d1_ohlc/tuning_v01/best_progression.csv`

### FLF-BiLSTM

- Config terpilih:
  - `window=6`
  - `units=256`
  - `lr=0.0005`
  - `epochs=50`
  - `lambda=1.0`
  - `sigma=0.05`
  - `batch=32`
  - `dropout=0.0`
  - `l2_reg=0.0`
- Sumber audit:
  - `FLF_BILSTM/results/d1_ohlc/tuning_v01/best_progression.csv`

## Hasil Last 5 Yang Sudah Selesai

### FLF-LSTM

- Metrics CSV: `results/d1_ohlc/lstm_d1_wf72_test1_last5_metrics_v01.csv`
- Mean `MAE avg`: `33.2964 pips`
- Mean `corr2 avg HLC`: `0.6436`
- Mean `Directional Accuracy`: `51.7424%`
- Mean `epochs_ran`: `25.8`
- Mean `best_epoch`: `20.8`

### FLF-BiLSTM

- Metrics CSV: `results/d1_ohlc/bilstm_d1_wf72_test1_last5_metrics_v01.csv`
- Mean `MAE avg`: `28.5463 pips`
- Mean `corr2 avg HLC`: `0.6919`
- Mean `Directional Accuracy`: `46.3636%`
- Mean `epochs_ran`: `16.8`
- Mean `best_epoch`: `11.8`

### Perbandingan Sementara

- Comparison CSV: `results/d1_ohlc/lstm_vs_bilstm_d1_wf72_test1_last5_comparison_v01.csv`
- Summary JSON: `results/d1_ohlc/d1_wf72_test1_last5_summary_v01.json`
- Status:
  - `BiLSTM` unggul pada `MAE` dan `corr2`
  - `LSTM` unggul pada `Directional Accuracy`

## Hasil ARIMA Last 5 Yang Sudah Selesai

- Output folder: `Arima/result/d1_ohlc/arima_wf72_test1_last5_v01/`
- Metrics CSV: `results/d1_ohlc/arima_d1_wf72_test1_last5_metrics_v01.csv`
- Mean `MAE avg`: `40.2983 pips`
- Mean `corr2 avg HLC`: `0.6019`
- Mean `Directional Accuracy`: `43.6364%`
- Dominant selected order pada `last 5`:
  - `open=(1,0,0)`
  - `high=(1,0,1)`
  - `low=(1,1,0)`
  - `close=(1,0,0)`

## Comparison Final 1D

- HTML: `comparison/d1_ohlc/comparison_models_d1_wf72_test1_last5_v01.html`
- Summary JSON: `comparison/d1_ohlc/comparison_models_d1_wf72_test1_last5_v01_summary.json`
- Pairwise CSV: `comparison/d1_ohlc/comparison_models_d1_wf72_test1_last5_v01_pairwise.csv`
- Sinkronisasi summary ke folder kerja: `results/d1_ohlc/d1_wf72_test1_last5_summary_v02.json`

### Ringkasan Final Tiga Model

- `ARIMA`
  - mean `MAE avg = 40.2983 pips`
  - mean `corr2 avg HLC = 0.6019`
  - mean `Directional Accuracy = 43.6364%`
- `FLF-LSTM`
  - mean `MAE avg = 33.2964 pips`
  - mean `corr2 avg HLC = 0.6436`
  - mean `Directional Accuracy = 51.7424%`
- `FLF-BiLSTM`
  - mean `MAE avg = 28.5463 pips`
  - mean `corr2 avg HLC = 0.6919`
  - mean `Directional Accuracy = 46.3636%`

### Kesimpulan Sementara 1D

- Pada `EUR/USD 1D last 5`, `FLF-BiLSTM` adalah model terbaik untuk `MAE avg pips` dan `corr2 avg HLC`.
- `FLF-LSTM` tetap unggul pada `Directional Accuracy`.
- `ARIMA` berada di bawah dua model deep learning pada ketiga metrik utama dalam skenario `1D last 5`.

## Catatan Teknis Penting

- Runner tuning `1D` sudah diperbaiki agar:
  - stage yang lebih buruk tidak menimpa incumbent
  - field `dropout`, `recurrent_dropout`, `l2_reg` tidak lagi `NaN`
  - `Stage 6` LSTM membandingkan juga `epoch=60`
- Untuk rolling evaluation `1D`, `model_out` di config final diset `null` agar runner tidak gagal saat hanya butuh `preds/history`.
- `FLF_BILSTM/rolling_fixed_runner.py` sudah ditambah argumen `--last-n-folds`.

## Checklist Status

- [x] Data `1D` dibersihkan
- [x] Fold `1D` `wf72_test1` dibentuk
- [x] Smoke test `LSTM` dan `BiLSTM`
- [x] Tuning `Stage 1-7`
- [x] Evaluasi `LSTM last 5`
- [x] Evaluasi `BiLSTM last 5`
- [x] Evaluasi `ARIMA last 5`
- [x] Comparison final `ARIMA vs FLF-LSTM vs FLF-BiLSTM`
- [ ] Update narasi tesis untuk hasil `1D`

## Langkah Lanjut Yang Benar

1. Update narasi tesis untuk bab metodologi, hasil, dan progress agar memasukkan temuan `1D`.
2. Putuskan apakah tahap berikutnya adalah `feature engineering` pada model/timeframe terbaik.
3. Jika lanjut ke feature engineering, gunakan baseline `1D` final ini sebagai titik pembanding tetap.
