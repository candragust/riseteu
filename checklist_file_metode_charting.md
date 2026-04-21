# Checklist File Metode untuk Software Charting

Root project: `/home/hduser/jupyter/gust/RisetEU`

Dokumen ini fokus ke **pipeline aktif** yang relevan untuk software charting:
- menu menjalankan kalkulasi
- membaca output prediksi
- menampilkan chart/visual
- menampilkan comparison dan rekap

## 1. Data Bersama dan Preprocessing

- [ ] `EURUSD_H4_25Oct17.csv`
  Data mentah utama EURUSD H4.
- [ ] `results/EURUSD_H4_clean.csv`
  Data bersih yang dipakai baseline aktif ARIMA dan FLF-BiLSTM.
- [ ] `data_prep_eurusd.py`
  Script pembersihan/preprocessing untuk menghasilkan data bersih.
- [ ] `data_prep_report.py`
  Report HTML untuk validasi hasil preprocessing.
- [ ] `results/data_prep_report.html`
  Output report preprocessing yang sudah ada.

## 2. ARIMA

### Core dan Runner

- [ ] `Arima/arima_ohlc_experiment.py`
  Entry point utama untuk run ARIMA OHLC.
- [ ] `Arima/arima_rolling_runner.py`
  Runner walk-forward/fixed-window untuk ARIMA.
- [ ] `Arima/eurusd_ohlc_utils.py`
  Utility loading dan parsing data OHLC untuk ARIMA.
- [ ] `Arima/arima_baseline_config.json`
  Config baseline ARIMA aktif.
- [ ] `Arima/generate_arima_html_report.py`
  Generator HTML report hasil ARIMA.

### Hasil yang Bisa Dibaca Software

- [ ] `Arima/result/arima_wf72_test1_last5/rolling_fixed_summary.csv`
  Rekap fold ARIMA untuk skenario last5.
- [ ] `Arima/result/arima_wf72_test1_last5/fold17_data.csv` s.d. `Arima/result/arima_wf72_test1_last5/fold21_data.csv`
  Data input per fold.
- [ ] `Arima/result/arima_wf72_test1_last5/fold17_preds.csv` s.d. `Arima/result/arima_wf72_test1_last5/fold21_preds.csv`
  Output prediksi per fold untuk chart.
- [ ] `Arima/result/arima_wf72_test1_last5/fold17_summary.json` s.d. `Arima/result/arima_wf72_test1_last5/fold21_summary.json`
  Summary metrik dan orde ARIMA per fold.
- [ ] `Arima/result/arima_residual_diagnostics_wf72_test1_last5.html`
  Visual residual diagnostics ARIMA.
- [ ] `Arima/result/index.html`
  Index report hasil ARIMA.

### Output Tambahan yang Sudah Ada

- [ ] `Arima/result/arima_smoke_preds.csv`
- [ ] `Arima/result/arima_smoke_summary.json`
- [ ] `Arima/result/arima_pathcheck_preds.csv`
- [ ] `Arima/result/arima_pathcheck_summary.json`
- [ ] `Arima/result/arima_wf72_test1_smoke/`

## 3. FLF-LSTM

### Core dan Runner

- [ ] `FLF_LSTM/lstm_flf_experiment.py`
  Entry point utama FLF-LSTM.
- [ ] `FLF_LSTM/rolling_fixed_lstm_runner.py`
  Runner walk-forward/fixed-window untuk FLF-LSTM.
- [ ] `FLF_LSTM/eurusd_lstm_pipeline_runner.py`
  Pipeline tuning bertahap FLF-LSTM.
- [ ] `FLF_LSTM/build_wf72_test1_reports.py`
  Generator report validasi, OHLC, loss, dan gradient proxy.
- [ ] `FLF_LSTM/lstm_flf_config.json`
  Config baseline FLF-LSTM berbasis data bersih.
- [ ] `FLF_LSTM/lstm_flf_config_wf72_test1_latest.json`
  Config tuning/eksperimen terbaru.
- [ ] `FLF_LSTM/lstm_flf_config_wf72_test1_best.json`
  Config terbaik untuk WF72m/1m.
- [ ] `FLF_LSTM/README.md`
  Penjelasan pipeline aktif FLF-LSTM.

### Hasil Evaluasi yang Bisa Dibaca Software

- [ ] `FLF_LSTM/results/wf72_test1_lstm/rolling_fixed_summary.csv`
  Rekap fold FLF-LSTM.
- [ ] `FLF_LSTM/results/wf72_test1_lstm/fold19_data.csv` s.d. `FLF_LSTM/results/wf72_test1_lstm/fold21_data.csv`
  Data input per fold.
- [ ] `FLF_LSTM/results/wf72_test1_lstm/fold19_preds.csv` s.d. `FLF_LSTM/results/wf72_test1_lstm/fold21_preds.csv`
  Output prediksi per fold untuk chart.
- [ ] `FLF_LSTM/results/wf72_test1_lstm/fold19_history.csv` s.d. `FLF_LSTM/results/wf72_test1_lstm/fold21_history.csv`
  History loss per fold.
- [ ] `FLF_LSTM/results/wf72_test1_lstm/ohlc_dot_wf72_test1_lstm_all.html`
  Chart OHLC gabungan semua fold.
- [ ] `FLF_LSTM/results/wf72_test1_lstm/loss_wf72_test1_lstm.html`
  Kurva loss.
- [ ] `FLF_LSTM/results/wf72_test1_lstm/loss_gradient_wf72_test1_lstm.html`
  Gradient proxy / perubahan loss.
- [ ] `FLF_LSTM/results/wf72_test1_lstm/mae_atr_fold19_tail30.html`
- [ ] `FLF_LSTM/results/wf72_test1_lstm/mae_atr_fold20_tail30.html`
- [ ] `FLF_LSTM/results/wf72_test1_lstm/mae_atr_fold21_tail30.html`
- [ ] `FLF_LSTM/results/wf72_test1_lstm/mae_atr_fold21_full.html`

### Hasil Last5 yang Bisa Dipakai untuk Perbandingan

- [ ] `FLF_LSTM/results/wf72_test1_lstm_last5/rolling_fixed_summary.csv`
- [ ] `FLF_LSTM/results/wf72_test1_lstm_last5/fold17_data.csv` s.d. `FLF_LSTM/results/wf72_test1_lstm_last5/fold21_data.csv`
- [ ] `FLF_LSTM/results/wf72_test1_lstm_last5/fold17_preds.csv` s.d. `FLF_LSTM/results/wf72_test1_lstm_last5/fold21_preds.csv`
- [ ] `FLF_LSTM/results/wf72_test1_lstm_last5/fold17_history.csv` s.d. `FLF_LSTM/results/wf72_test1_lstm_last5/fold21_history.csv`
- [ ] `FLF_LSTM/results/wf72_test1_lstm_last5/ohlc_dot_wf72_test1_lstm_last5_all.html`
- [ ] `FLF_LSTM/results/wf72_test1_lstm_last5/loss_wf72_test1_lstm_last5.html`
- [ ] `FLF_LSTM/results/wf72_test1_lstm_last5/loss_gradient_wf72_test1_lstm_last5.html`

### Hasil Tuning / Pipeline

- [ ] `FLF_LSTM/results/lstm_pipeline_wf72_test1/`
  Folder utama hasil tuning bertahap.
- [ ] `FLF_LSTM/results/lstm_pipeline_wf72_test1/stage1_summary.csv` s.d. `FLF_LSTM/results/lstm_pipeline_wf72_test1/stage6_summary.csv`
  Rekap per stage.
- [ ] `FLF_LSTM/results/lstm_pipeline_wf72_test1/best_progression.csv`
  Rekap best config tiap stage.
- [ ] `FLF_LSTM/results/lstm_pipeline_wf72_test1/eurusd_lstm_pipeline_summary.html`
  Summary HTML pipeline tuning.
- [ ] `FLF_LSTM/results/lstm_flf_model.keras`
  Model artifact FLF-LSTM.
- [ ] `FLF_LSTM/results/validation_report_wf72_test1_lstm.html`
- [ ] `FLF_LSTM/results/validation_report_wf72_test1_lstm_last5.html`

## 4. FLF-BiLSTM

### Core dan Runner

- [ ] `bilstm_flf_experiment.py`
  Entry point utama FLF-BiLSTM.
- [ ] `rolling_fixed_runner.py`
  Runner walk-forward/fixed-window utama untuk BiLSTM.
- [ ] `rolling_fixed_runner_days.py`
  Runner alternatif berbasis test-days.
- [ ] `eurusd_pipeline_runner.py`
  Pipeline tuning bertahap FLF-BiLSTM.
- [ ] `pipeline_summary_generator.py`
  Generator summary HTML pipeline lama berbasis hasil hp sweep.
- [ ] `generate_final_config.py`
  Generator `final_config.json` dari hasil hp sweep.
- [ ] `bilstm_flf_config.json`
  Config baseline BiLSTM.
- [ ] `final_config.json`
  Config final aktif BiLSTM untuk pipeline.
- [ ] `temp_config.json`
  Config eksperimen sementara.
- [ ] `temp_wf72_test1_bilstm_last5_config.json`
  Config sementara untuk skenario last5.

### Hasil Evaluasi yang Bisa Dibaca Software

- [ ] `results/rolling_train72_test1/rolling_fixed_summary.csv`
  Rekap fold FLF-BiLSTM.
- [ ] `results/rolling_train72_test1/fold19_data.csv` s.d. `results/rolling_train72_test1/fold21_data.csv`
  Data input per fold.
- [ ] `results/rolling_train72_test1/fold19_preds.csv` s.d. `results/rolling_train72_test1/fold21_preds.csv`
  Output prediksi per fold untuk chart.
- [ ] `results/rolling_train72_test1/fold19_history.csv` s.d. `results/rolling_train72_test1/fold21_history.csv`
  History loss per fold.
- [ ] `results/rolling_train72_test1/fold19_clean_with_atr.csv` s.d. `results/rolling_train72_test1/fold21_clean_with_atr.csv`
  Data fold yang sudah diberi ATR.
- [ ] `results/rolling_train72_test1/wf72_test1_combined_preds.csv`
  Prediksi gabungan beberapa fold.
- [ ] `results/rolling_train72_test1/wf72_test1_tail.csv`
  Data tail untuk analisis ringkas.
- [ ] `results/rolling_train72_test1/rolling_train72_test1_report.csv`
  Report CSV ringkasan.
- [ ] `results/mae_atr_wf72_test1_fold19_tail30.html`
- [ ] `results/mae_atr_wf72_test1_fold20_tail30.html`
- [ ] `results/mae_atr_wf72_test1_fold21_tail30.html`
- [ ] `results/mae_atr_wf72_test1_fold21_full.html`
- [ ] `results/ohlc_dot_wf72_test1_all_folds.html`

### Hasil Last5 yang Bisa Dipakai untuk Perbandingan

- [ ] `results/rolling_train72_test1_last5/rolling_fixed_summary.csv`
- [ ] `results/rolling_train72_test1_last5/fold17_data.csv` s.d. `results/rolling_train72_test1_last5/fold21_data.csv`
- [ ] `results/rolling_train72_test1_last5/fold17_preds.csv` s.d. `results/rolling_train72_test1_last5/fold21_preds.csv`
- [ ] `results/rolling_train72_test1_last5/fold17_history.csv` s.d. `results/rolling_train72_test1_last5/fold21_history.csv`
- [ ] `results/loss_wf72_test1_last5_bilstm.html`
- [ ] `results/loss_gradient_wf72_test1_last5_bilstm.html`
- [ ] `results/validation_report_wf72_test1_last5_bilstm.html`
- [ ] `results/ohlc_dot_wf72_test1_last5_all_folds.html`

### Hasil Tuning / Pipeline

- [ ] `results/eurusd_pipeline/`
  Folder utama hasil tuning bertahap BiLSTM.
- [ ] `results/eurusd_pipeline/stage1_summary.csv` s.d. `results/eurusd_pipeline/stage6_summary.csv`
- [ ] `results/eurusd_pipeline/best_progression.csv`
- [ ] `results/validation_report.html`
- [ ] `results/pipeline_summary.html`

## 5. Comparison, Rekap, dan Visual Bersama

### Script Rekap / Comparison

- [ ] `compare_lstm_bilstm_wf72_test1.py`
  Perbandingan FLF-LSTM vs FLF-BiLSTM.
- [ ] `compare_models_wf72_test1_last5.py`
  Perbandingan ARIMA vs FLF-LSTM vs FLF-BiLSTM.
- [ ] `generate_model_diagnostics_docs.py`
  Rekap diagnostik lintas model.

### Output Comparison / Rekap

- [ ] `results/comparison/README.md`
- [ ] `results/comparison/comparison_lstm_vs_bilstm_wf72_test1.html`
- [ ] `results/comparison/comparison_lstm_vs_bilstm_wf72_test1_last5.html`
- [ ] `results/comparison/comparison_models_wf72_test1_last5.html`
- [ ] `bukuThesis/bahan/analisis_konvergensi_dan_diagnostik_model_wf72_test1_last5.md`
- [ ] `bukuThesis/bahan/analisis_konvergensi_dan_diagnostik_model_wf72_test1_last5.html`

### Utility Visual yang Bisa Dipakai UI

- [ ] `mae_atr_report.py`
  Report MAE vs ATR dalam HTML.
- [ ] `ohlc_dot_report.py`
  Plot OHLC aktual vs prediksi.
- [ ] `plot_close_actual_vs_pred.py`
  Plot close aktual vs prediksi.
- [ ] `plot_ohlc_actual_vs_pred.py`
  Plot OHLC aktual vs prediksi.

## 6. Arsip / Legacy / Eksperimen Lama

File-file ini masih terkait topik model, tetapi **bukan prioritas** untuk software charting aktif:

- [ ] `risetBiLstmPy/FLF_BiLSTM_Forex_configurable.py`
- [ ] `risetBiLstmPy/BiLSTM_FLF_Vjn1.ipynb`
- [ ] `risetBiLstmPy/BiLSTM_FLF_Experiment_Setup.csv`
- [ ] `Bilstm2candle/bilstm_flf_experiment.py`
- [ ] `Bilstm2candle/rollout_two_step.py`
- [ ] `Bilstm2candle/final_config.json`
- [ ] `CodeLstm/FLF-LSTM Different Activations.ipynb`
- [ ] `CodeLstm/OHLC-LSTM-Different_activations.ipynb`
- [ ] `CodeLstm/Multi-LSTM_different_activations.ipynb`
- [ ] `CodeLstm/FLF-SimpleRNN Different-activations.ipynb`
- [ ] `CodeLstm/flf_lstm_paper.txt`
- [ ] `CodeLstm/FLF-LSTM A novel prediction system using Forex Loss Function 10.1016@j.asoc.2020.106780.pdf`

## 7. Catatan Penting untuk Integrasi Software

- FLF-LSTM saat ini bergantung pada data fold hasil skenario rolling, terutama `results/rolling_train72_test1/fold21_data.csv`, karena path itu dipakai di `FLF_LSTM/lstm_flf_config_wf72_test1_latest.json` dan `FLF_LSTM/lstm_flf_config_wf72_test1_best.json`.
- ARIMA tidak punya file `history.csv` seperti LSTM/BiLSTM. Rekap ARIMA ada di `foldXX_summary.json` dan visual residual HTML.
- Untuk software charting, file yang paling penting dibaca UI biasanya:
  - `foldXX_preds.csv`
  - `rolling_fixed_summary.csv`
  - `best_progression.csv`
  - file comparison HTML
  - file visual HTML per metode
- Ada inkonsistensi nama default di `FLF_LSTM/lstm_flf_experiment.py`: default `out` masih bernama `bilstm_flf_predictions.csv`. Aman jika selalu menjalankan script dengan config atau override CLI.
