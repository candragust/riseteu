# Indeks Eksperimen TVT v02

Dokumen ini memetakan file terkait eksperimen `tvt_v02` dari proses split data, tuning, evaluasi, sampai result/report. Path ditulis relatif dari root repo:

`/home/hduser/jupyter/gust/RisetEU`

## Ringkasan Scope

- Timeframe: `H4` dan `D1`.
- Model: `FLF-LSTM`, `FLF-BiLSTM`, dan baseline `ARIMA`.
- Policy fold TVT v02:
  - total fold tersedia per timeframe: 21 fold.
  - tuning: fold `13-18`.
  - evaluasi final: fold `19-21`.
  - tuning dan evaluasi tidak overlap.
- Result utama yang diminta:
  - `FLF_BILSTM/results/tvt_v02`
  - `FLF_LSTM/results/tvt_v02`
  - `Arima/result/tvt_v02`
- Result pendukung:
  - `comparison/tvt_v02`
  - `results/splits/tvt_v02`
  - `bukuThesis/penyusunan_buku_tesis_tvt_v02`

## Alur Pemrosesan

| Tahap | Input | Pemroses | Output utama |
| --- | --- | --- | --- |
| Split TVT v02 | `EURUSD_H4_25Oct17.csv`, `EURUSD_D1_25Oct17.csv` | `split_fold_data_tvt_v02.py` | `results/splits/tvt_v02/h4`, `results/splits/tvt_v02/d1` |
| Tuning FLF-LSTM H4 | split tuning H4 fold 13-18 | `FLF_LSTM/eurusd_lstm_pipeline_runner_tvt_v02.py` | `FLF_LSTM/results/tvt_v02/lstm_pipeline_h4_tuning_last6` |
| Tuning FLF-BiLSTM H4 | split tuning H4 fold 13-18 | `FLF_BILSTM/eurusd_bilstm_pipeline_runner_tvt_v02.py` | `FLF_BILSTM/results/tvt_v02/bilstm_pipeline_h4_tuning_last6` |
| Tuning FLF-LSTM D1 | split tuning D1 fold 13-18 | `FLF_LSTM/eurusd_lstm_d1_pipeline_runner_tvt_v02.py` | `FLF_LSTM/results/tvt_v02/d1_ohlc/lstm_pipeline_d1_tuning_last6` |
| Tuning FLF-BiLSTM D1 | split tuning D1 fold 13-18 | `FLF_BILSTM/eurusd_bilstm_d1_pipeline_runner_tvt_v02.py` | `FLF_BILSTM/results/tvt_v02/d1_ohlc/bilstm_pipeline_d1_tuning_last6` |
| Freeze config + evaluasi final LSTM/BiLSTM | `best_progression.csv` dari tuning | `scripts/finalize_tvt_v02.py` | config `*_best.json`, `*_evaluation_last3` |
| Evaluasi ARIMA | split evaluation fold 19-21 | `Arima/arima_tvt_runner_v02.py`, `Arima/arima_ohlc_experiment.py` | `Arima/result/tvt_v02/h4_evaluation_last3`, `Arima/result/tvt_v02/d1_evaluation_last3` |
| Report validasi model | evaluation result LSTM/BiLSTM | `FLF_LSTM/build_wf72_test1_reports.py` | `validation_report_*.html`, `ohlc_dot_*.html`, `loss_*.html` |
| Report MAE/ATR | evaluation result semua model | `build_tvt_v02_mae_atr_combined_reports.py` | `mae_atr_fold_allfull.html` dan CSV metrik |
| Report ARIMA | ARIMA result | `Arima/generate_arima_html_report.py`, `Arima/generate_tvt_v02_arima_residual_diagnostics.py` | `index.html`, `arima_residual_diagnostics_*.html` |
| Perbandingan model | result LSTM, BiLSTM, ARIMA | `compare_models_h4_tvt_v02_last3_v01.py` | `comparison/tvt_v02/{h4,d1}` |
| Konversi PDF | semua HTML berisi `tvt_v02` | `scripts/convert_tvt_v02_html_to_pdf.py` | `bukuThesis/penyusunan_buku_tesis_tvt_v02/04_lampiran_artefak/pdf_tvt_v02` |

## Data dan Split

| File/folder | Isi |
| --- | --- |
| `EURUSD_H4_25Oct17.csv` | data mentah EUR/USD H4 untuk profile H4 |
| `EURUSD_D1_25Oct17.csv` | data mentah EUR/USD D1 untuk profile D1 |
| `results/EURUSD_H4_clean.csv` | data bersih H4, dipakai juga oleh konfigurasi ARIMA lama |
| `results/EURUSD_D1_clean.csv` | data bersih D1, dipakai juga oleh konfigurasi ARIMA D1 |
| `results/splits/tvt_v02/h4/fold_schedule.csv` | semua fold H4 |
| `results/splits/tvt_v02/h4/fold_schedule_tuning.csv` | H4 tuning fold 13-18 |
| `results/splits/tvt_v02/h4/fold_schedule_evaluation.csv` | H4 final evaluation fold 19-21 |
| `results/splits/tvt_v02/h4/fold_scope_tvt_v02.json` | metadata scope split H4 |
| `results/splits/tvt_v02/h4/foldXX/{train_core,validation,test,combined}.csv` | data split per fold H4 |
| `results/splits/tvt_v02/d1/fold_schedule.csv` | semua fold D1 |
| `results/splits/tvt_v02/d1/fold_schedule_tuning.csv` | D1 tuning fold 13-18 |
| `results/splits/tvt_v02/d1/fold_schedule_evaluation.csv` | D1 final evaluation fold 19-21 |
| `results/splits/tvt_v02/d1/fold_scope_tvt_v02.json` | metadata scope split D1 |
| `results/splits/tvt_v02/d1/foldXX/{train_core,validation,test,combined}.csv` | data split per fold D1 |

Profil split dari kode:

| Timeframe | Train core | Validation | Test | Tuning fold | Evaluation fold |
| --- | ---: | ---: | ---: | --- | --- |
| H4 | 71 bulan | 1 bulan | 1 bulan | 13-18 | 19-21 |
| D1 | 69 bulan | 3 bulan | 1 bulan | 13-18 | 19-21 |

Window evaluasi final:

| Timeframe | Fold | Test start | Test end | Test samples |
| --- | ---: | --- | --- | ---: |
| H4 | 19 | 2025-07-02 00:00:00 | 2025-08-01 20:00:00 | 138 |
| H4 | 20 | 2025-08-04 00:00:00 | 2025-09-01 20:00:00 | 126 |
| H4 | 21 | 2025-09-02 00:00:00 | 2025-10-01 20:00:00 | 132 |
| D1 | 19 | 2025-07-02 00:00:00 | 2025-08-01 00:00:00 | 23 |
| D1 | 20 | 2025-08-04 00:00:00 | 2025-09-01 00:00:00 | 21 |
| D1 | 21 | 2025-09-02 00:00:00 | 2025-10-01 00:00:00 | 22 |

## File Python dan Shell Terkait

### Entrypoint dan kontrol run

| File | Fungsi |
| --- | --- |
| `scripts/run_tvt_v02.sh` | wrapper umum: memilih `h4/d1` dan `lstm/bilstm` |
| `scripts/run_tvt_h4_detached_v02.sh` | menjalankan tuning H4 LSTM/BiLSTM di `screen` |
| `scripts/run_tvt_d1_detached_v02.sh` | menjalankan tuning D1 LSTM/BiLSTM di `screen` |
| `scripts/run_d1_tvt_v02_end_to_end.sh` | workflow D1 penuh: tuning LSTM, finalize LSTM, tuning BiLSTM, finalize BiLSTM, ARIMA, report |
| `scripts/launch_d1_tvt_v02_end_to_end_detached.sh` | menjalankan workflow D1 end-to-end di `screen` |
| `scripts/status_tvt_v02.sh` | status screen/log TVT v02 |
| `scripts/audit_d1_tvt_v02_readiness.py` | audit kesiapan D1 TVT v02 |

### Split dan finalisasi

| File | Fungsi |
| --- | --- |
| `split_fold_data_tvt_v02.py` | membuat `train_core`, `validation`, `test`, `combined`, schedule, dan scope split |
| `scripts/finalize_tvt_v02.py` | audit hasil tuning, memilih config final, menulis `*_best.json`, menjalankan evaluasi fold 19-21 |

### FLF-LSTM

| File | Fungsi |
| --- | --- |
| `FLF_LSTM/lstm_flf_experiment_tvt_v01.py` | training/evaluasi LSTM single run dengan FLF, menerima split train/validation/test |
| `FLF_LSTM/rolling_tvt_lstm_runner_v02.py` | runner evaluasi H4 per fold TVT v02 |
| `FLF_LSTM/rolling_tvt_lstm_d1_runner_v02.py` | runner evaluasi D1 per fold TVT v02 |
| `FLF_LSTM/eurusd_lstm_pipeline_runner_tvt_v02.py` | staged hyperparameter tuning H4 fold 13-18 |
| `FLF_LSTM/eurusd_lstm_d1_pipeline_runner_tvt_v02.py` | staged hyperparameter tuning D1 fold 13-18 |
| `FLF_LSTM/build_wf72_test1_reports.py` | HTML report validasi, OHLC plot, loss, dan loss gradient untuk LSTM/BiLSTM |

### FLF-BiLSTM

| File | Fungsi |
| --- | --- |
| `FLF_BILSTM/bilstm_flf_experiment_tvt_v01.py` | training/evaluasi BiLSTM single run dengan FLF, menerima split train/validation/test |
| `FLF_BILSTM/rolling_tvt_bilstm_runner_v02.py` | runner evaluasi H4 per fold TVT v02 |
| `FLF_BILSTM/rolling_tvt_bilstm_d1_runner_v02.py` | runner evaluasi D1 per fold TVT v02 |
| `FLF_BILSTM/eurusd_bilstm_pipeline_runner_tvt_v02.py` | staged hyperparameter tuning H4 fold 13-18 |
| `FLF_BILSTM/eurusd_bilstm_d1_pipeline_runner_tvt_v02.py` | staged hyperparameter tuning D1 fold 13-18 |

### ARIMA dan report gabungan

| File | Fungsi |
| --- | --- |
| `Arima/arima_ohlc_experiment.py` | eksperimen ARIMA OHLC per fold |
| `Arima/arima_tvt_runner_v02.py` | runner ARIMA TVT v02 fold 19-21 |
| `Arima/generate_arima_html_report.py` | report HTML ARIMA |
| `Arima/generate_tvt_v02_arima_residual_diagnostics.py` | diagnostik residual ARIMA TVT v02 |
| `build_tvt_v02_mae_atr_combined_reports.py` | report MAE/ATR fold 19-21 untuk LSTM, BiLSTM, ARIMA |
| `mae_atr_report.py` | helper visual dan metrik MAE/ATR |
| `compare_models_h4_tvt_v02_last3_v01.py` | perbandingan ARIMA vs LSTM vs BiLSTM; dipakai juga untuk D1 via argumen |
| `scripts/generate_d1_tvt_v02_reports.sh` | report D1 lengkap |
| `scripts/normalize_tvt_html_titles.py` | normalisasi judul HTML TVT |
| `scripts/convert_tvt_v02_html_to_pdf.py` | konversi semua HTML `tvt_v02` ke PDF |
| `scripts/run_tvt_v02_html_pdf_conversion.sh` | wrapper konversi HTML ke PDF |
| `scripts/launch_tvt_v02_html_pdf_detached.sh` | konversi HTML ke PDF di `screen` |

## Konfigurasi Final

| Model | Timeframe | File config final | Parameter final utama |
| --- | --- | --- | --- |
| FLF-LSTM | H4 | `FLF_LSTM/lstm_flf_config_h4_tvt_v02_best.json` | window=10, units=256, activation=relu, lr=0.0009, lambda=0.8, sigma=0.1, batch=96, epochs=60 |
| FLF-BiLSTM | H4 | `FLF_BILSTM/bilstm_flf_config_h4_tvt_v02_best.json` | window=12, units=256, activation=tanh, lr=0.0003, lambda=0.8, sigma=0.15, batch=128, epochs=50 |
| FLF-LSTM | D1 | `FLF_LSTM/configs/d1_ohlc/lstm_flf_config_d1_tvt_v02_best.json` | window=4, units=128, activation=relu, lr=0.0009, lambda=0.8, sigma=0.1, batch=96, epochs=60, l2_reg=1e-05 |
| FLF-BiLSTM | D1 | `FLF_BILSTM/configs/d1_ohlc/bilstm_flf_config_d1_tvt_v02_best.json` | window=4, units=256, activation=relu, lr=0.0007, lambda=1.0, sigma=0.05, batch=32, epochs=50 |
| ARIMA | H4 | `Arima/arima_baseline_config.json` | order search AIC, p=[0,1,2], d=[0,1], q=[0,1,2] |
| ARIMA | D1 | `Arima/arima_baseline_config_d1_ohlc_v01.json` | order search AIC, p=[0,1,2], d=[0,1], q=[0,1,2] |

## Result FLF-LSTM

Root: `FLF_LSTM/results/tvt_v02`

| Folder | Isi penting |
| --- | --- |
| `lstm_pipeline_h4_tuning_last6` | tuning H4 fold 13-18, `stage1_summary.csv` sampai `stage6_summary.csv`, `best_progression.csv`, `audit_finalize_report.*`, `eurusd_lstm_pipeline_tvt_summary.html` |
| `lstm_pipeline_h4_tuning_last6/lstm_tvt_*` | hasil tiap kandidat tuning H4: `fold13-18_preds.csv`, `fold13-18_history.csv`, `rolling_tvt_summary.csv`, `run_manifest.json` |
| `h4_evaluation_last3` | evaluasi final H4 fold 19-21: `fold19-21_preds.csv`, `fold19-21_history.csv`, `rolling_tvt_summary.csv`, `validation_report_h4_tvt_v02_lstm_last3.html`, `ohlc_dot_h4_tvt_v02_lstm_last3_all.html`, `loss_h4_tvt_v02_lstm_last3.html`, `loss_gradient_h4_tvt_v02_lstm_last3.html`, `mae_atr_fold_allfull.html` |
| `d1_ohlc/lstm_pipeline_d1_tuning_last6` | tuning D1 fold 13-18, `stage1_summary.csv` sampai `stage7_summary.csv`, `best_progression.csv`, `audit_finalize_report.*`, `eurusd_lstm_d1_pipeline_tvt_v02_summary.html` |
| `d1_ohlc/lstm_pipeline_d1_tuning_last6/lstm_d1_*` | hasil tiap kandidat tuning D1: `fold13-18_preds.csv`, `fold13-18_history.csv`, `rolling_tvt_summary.csv`, `run_manifest.json` |
| `d1_ohlc/d1_evaluation_last3` | evaluasi final D1 fold 19-21: `fold19-21_preds.csv`, `fold19-21_history.csv`, `rolling_tvt_summary.csv`, `validation_report_d1_tvt_v02_lstm_last3.html`, `ohlc_dot_d1_tvt_v02_lstm_last3_all.html`, `loss_d1_tvt_v02_lstm_last3.html`, `loss_gradient_d1_tvt_v02_lstm_last3.html`, `mae_atr_fold_allfull.html` |

## Result FLF-BiLSTM

Root: `FLF_BILSTM/results/tvt_v02`

| Folder | Isi penting |
| --- | --- |
| `bilstm_pipeline_h4_tuning_last6` | tuning H4 fold 13-18, `stage1_summary.csv` sampai `stage6_summary.csv`, `best_progression.csv`, `audit_finalize_report.*`, `eurusd_bilstm_pipeline_tvt_summary.html` |
| `bilstm_pipeline_h4_tuning_last6/bilstm_tvt_*` | hasil tiap kandidat tuning H4: `fold13-18_preds.csv`, `fold13-18_history.csv`, `rolling_tvt_summary.csv`, `run_manifest.json` |
| `h4_evaluation_last3` | evaluasi final H4 fold 19-21: `fold19-21_preds.csv`, `fold19-21_history.csv`, `rolling_tvt_summary.csv`, `validation_report_h4_tvt_v02_bilstm_last3.html`, `ohlc_dot_h4_tvt_v02_bilstm_last3_all.html`, `loss_h4_tvt_v02_bilstm_last3.html`, `loss_gradient_h4_tvt_v02_bilstm_last3.html`, `mae_atr_fold_allfull.html` |
| `d1_ohlc/bilstm_pipeline_d1_tuning_last6` | tuning D1 fold 13-18, `stage1_summary.csv` sampai `stage7_summary.csv`, `best_progression.csv`, `audit_finalize_report.*`, `eurusd_bilstm_d1_pipeline_tvt_v02_summary.html` |
| `d1_ohlc/bilstm_pipeline_d1_tuning_last6/bilstm_d1_*` | hasil tiap kandidat tuning D1: `fold13-18_preds.csv`, `fold13-18_history.csv`, `rolling_tvt_summary.csv`, `run_manifest.json` |
| `d1_ohlc/d1_evaluation_last3` | evaluasi final D1 fold 19-21: `fold19-21_preds.csv`, `fold19-21_history.csv`, `rolling_tvt_summary.csv`, `validation_report_d1_tvt_v02_bilstm_last3.html`, `ohlc_dot_d1_tvt_v02_bilstm_last3_all.html`, `loss_d1_tvt_v02_bilstm_last3.html`, `loss_gradient_d1_tvt_v02_bilstm_last3.html`, `mae_atr_fold_allfull.html` |

## Result ARIMA

Root: `Arima/result/tvt_v02`

| Folder | Isi penting |
| --- | --- |
| `h4_evaluation_last3` | ARIMA H4 fold 19-21: `fold19-21_data.csv`, `fold19-21_preds.csv`, `fold19-21_summary.json`, `rolling_tvt_summary.csv`, `index.html`, `index_metrics.csv`, `mae_atr_fold_allfull.html`, `arima_residual_diagnostics_tvt_v02_last3.html` |
| `d1_evaluation_last3` | ARIMA D1 fold 19-21: `fold19-21_data.csv`, `fold19-21_preds.csv`, `fold19-21_summary.json`, `rolling_tvt_summary.csv`, `index.html`, `index_metrics.csv`, `mae_atr_fold_allfull.html`, `arima_residual_diagnostics_tvt_v02_last3.html` |

## Result Perbandingan Model

Root: `comparison/tvt_v02`

| Folder | File penting |
| --- | --- |
| `h4` | `comparison_models_h4_tvt_v02_last3_v01.html`, `*_summary.csv/json`, `*_lstm_metrics.csv`, `*_bilstm_metrics.csv`, `*_arima_metrics.csv`, `*_pairwise.csv`, `*_atr_relation.csv`, `*_arima_candle_audit.csv` |
| `d1` | `comparison_models_d1_tvt_v02_last3_v01.html`, `*_summary.csv/json`, `*_lstm_metrics.csv`, `*_bilstm_metrics.csv`, `*_arima_metrics.csv`, `*_pairwise.csv`, `*_atr_relation.csv`, `*_arima_candle_audit.csv` |

Ringkasan metrik evaluasi fold 19-21:

| Timeframe | Model | Mean MAE avg pips | Mean corr2 avg OHLC | Mean DA body % | Best fold | Total samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H4 | FLF-LSTM | 10.5792 | 0.9325 | 49.9979 | 21 | 396 |
| H4 | FLF-BiLSTM | 11.3987 | 0.9160 | 52.2089 | 21 | 396 |
| H4 | ARIMA | 14.1898 | 0.8798 | 50.2494 | 21 | 396 |
| D1 | FLF-BiLSTM | 26.2271 | 0.6929 | 51.6563 | 20 | 66 |
| D1 | FLF-LSTM | 32.3139 | 0.6209 | 51.6563 | 20 | 66 |
| D1 | ARIMA | 37.6320 | 0.4762 | 40.6895 | 20 | 66 |

## File Report HTML yang Cocok untuk Tampilan

### H4

| Model/grup | HTML utama |
| --- | --- |
| Perbandingan | `comparison/tvt_v02/h4/comparison_models_h4_tvt_v02_last3_v01.html` |
| FLF-LSTM | `FLF_LSTM/results/tvt_v02/h4_evaluation_last3/validation_report_h4_tvt_v02_lstm_last3.html` |
| FLF-LSTM | `FLF_LSTM/results/tvt_v02/h4_evaluation_last3/ohlc_dot_h4_tvt_v02_lstm_last3_all.html` |
| FLF-LSTM | `FLF_LSTM/results/tvt_v02/h4_evaluation_last3/loss_h4_tvt_v02_lstm_last3.html` |
| FLF-LSTM | `FLF_LSTM/results/tvt_v02/h4_evaluation_last3/mae_atr_fold_allfull.html` |
| FLF-BiLSTM | `FLF_BILSTM/results/tvt_v02/h4_evaluation_last3/validation_report_h4_tvt_v02_bilstm_last3.html` |
| FLF-BiLSTM | `FLF_BILSTM/results/tvt_v02/h4_evaluation_last3/ohlc_dot_h4_tvt_v02_bilstm_last3_all.html` |
| FLF-BiLSTM | `FLF_BILSTM/results/tvt_v02/h4_evaluation_last3/loss_h4_tvt_v02_bilstm_last3.html` |
| FLF-BiLSTM | `FLF_BILSTM/results/tvt_v02/h4_evaluation_last3/mae_atr_fold_allfull.html` |
| ARIMA | `Arima/result/tvt_v02/h4_evaluation_last3/index.html` |
| ARIMA | `Arima/result/tvt_v02/h4_evaluation_last3/mae_atr_fold_allfull.html` |
| ARIMA | `Arima/result/tvt_v02/h4_evaluation_last3/arima_residual_diagnostics_tvt_v02_last3.html` |

### D1

| Model/grup | HTML utama |
| --- | --- |
| Perbandingan | `comparison/tvt_v02/d1/comparison_models_d1_tvt_v02_last3_v01.html` |
| FLF-LSTM | `FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/validation_report_d1_tvt_v02_lstm_last3.html` |
| FLF-LSTM | `FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/ohlc_dot_d1_tvt_v02_lstm_last3_all.html` |
| FLF-LSTM | `FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/loss_d1_tvt_v02_lstm_last3.html` |
| FLF-LSTM | `FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/mae_atr_fold_allfull.html` |
| FLF-BiLSTM | `FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/validation_report_d1_tvt_v02_bilstm_last3.html` |
| FLF-BiLSTM | `FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/ohlc_dot_d1_tvt_v02_bilstm_last3_all.html` |
| FLF-BiLSTM | `FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/loss_d1_tvt_v02_bilstm_last3.html` |
| FLF-BiLSTM | `FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/mae_atr_fold_allfull.html` |
| ARIMA | `Arima/result/tvt_v02/d1_evaluation_last3/index.html` |
| ARIMA | `Arima/result/tvt_v02/d1_evaluation_last3/mae_atr_fold_allfull.html` |
| ARIMA | `Arima/result/tvt_v02/d1_evaluation_last3/arima_residual_diagnostics_tvt_v02_last3.html` |

## File CSV/JSON yang Cocok untuk Backend Tampilan

Gunakan file berikut sebagai sumber data terstruktur:

| Kebutuhan | File/pattern |
| --- | --- |
| Jadwal split tuning/evaluasi | `results/splits/tvt_v02/{h4,d1}/fold_schedule_tuning.csv`, `fold_schedule_evaluation.csv` |
| Scope split | `results/splits/tvt_v02/{h4,d1}/fold_scope_tvt_v02.json` |
| Prediksi final LSTM/BiLSTM | `FLF_LSTM/results/tvt_v02/**/d1_evaluation_last3/fold*_preds.csv`, `FLF_LSTM/results/tvt_v02/h4_evaluation_last3/fold*_preds.csv`, dan pola sama di `FLF_BILSTM` |
| History training final LSTM/BiLSTM | `FLF_LSTM/results/tvt_v02/**/fold*_history.csv`, `FLF_BILSTM/results/tvt_v02/**/fold*_history.csv` |
| Summary final per model | `FLF_LSTM/results/tvt_v02/**/*evaluation_last3/rolling_tvt_summary.csv`, `FLF_BILSTM/results/tvt_v02/**/*evaluation_last3/rolling_tvt_summary.csv`, `Arima/result/tvt_v02/*_evaluation_last3/rolling_tvt_summary.csv` |
| Manifest run final | `FLF_LSTM/results/tvt_v02/**/*evaluation_last3/run_manifest.json`, `FLF_BILSTM/results/tvt_v02/**/*evaluation_last3/run_manifest.json`, `Arima/result/tvt_v02/*_evaluation_last3/run_manifest.json` |
| Summary tuning | `FLF_LSTM/results/tvt_v02/**/stage*_summary.csv`, `FLF_BILSTM/results/tvt_v02/**/stage*_summary.csv` |
| Progress best tuning | `FLF_LSTM/results/tvt_v02/**/best_progression.csv`, `FLF_BILSTM/results/tvt_v02/**/best_progression.csv` |
| Audit finalisasi | `FLF_LSTM/results/tvt_v02/**/audit_finalize_report.json`, `FLF_BILSTM/results/tvt_v02/**/audit_finalize_report.json` |
| Summary perbandingan | `comparison/tvt_v02/{h4,d1}/*_summary.csv`, `comparison/tvt_v02/{h4,d1}/*_summary.json` |
| Detail metrik model | `comparison/tvt_v02/{h4,d1}/*_lstm_metrics.csv`, `*_bilstm_metrics.csv`, `*_arima_metrics.csv` |
| Pairwise/perbandingan tambahan | `comparison/tvt_v02/{h4,d1}/*_pairwise.csv`, `*_atr_relation.csv`, `*_arima_candle_audit.csv` |
| PDF hasil report | `bukuThesis/penyusunan_buku_tesis_tvt_v02/04_lampiran_artefak/pdf_tvt_v02/pdf_manifest.csv` |

## Artefak Buku/Presentasi

| Folder/file | Isi |
| --- | --- |
| `bukuThesis/penyusunan_buku_tesis_tvt_v02/01_bahan_tvt_v02` | bahan narasi, workflow, audit sumber, desain grid, laporan split |
| `bukuThesis/penyusunan_buku_tesis_tvt_v02/02_draft_bab` | draft implementasi/pembahasan hasil TVT v02 dalam `.md` dan `.docx` |
| `bukuThesis/penyusunan_buku_tesis_tvt_v02/03_tabel_gambar/assets_bagian_i_tvt_v02` | aset PNG turunan dari report HTML/PDF |
| `bukuThesis/penyusunan_buku_tesis_tvt_v02/03_tabel_gambar/assets_bagian_i_tvt_v02/figure_manifest.csv` | mapping figure PNG ke sumber PDF/HTML |
| `bukuThesis/penyusunan_buku_tesis_tvt_v02/04_lampiran_artefak/pdf_tvt_v02` | hasil konversi PDF dari HTML TVT v02 |
| `bukuThesis/metodologi/generate_gambar_metodologi_tvt_v02.py` | generator gambar metodologi TVT v02 |
| `bukuThesis/metodologi/generate_gambar_metodologi_1halaman_tvt_v02.py` | generator gambar metodologi ringkas satu halaman |

## Log Eksekusi

| File/folder | Isi |
| --- | --- |
| `agentactivity.md` | catatan aktivitas run, audit, konversi PDF |
| `logs/detached/tvt_h4_v02_lstm_20260424_175633.log` | log tuning H4 FLF-LSTM |
| `logs/detached/tvt_h4_v02_bilstm_20260424_201124.log` | log tuning H4 FLF-BiLSTM |
| `logs/detached/tvt_d1_v02_e2e_full_20260425_083059.log` | log D1 end-to-end sukses |
| `logs/detached/tvt_v02_html_pdf_20260425_130212.log` | log konversi HTML ke PDF sukses |

## Catatan untuk Software Tampilan

- Untuk dashboard utama, mulai dari `comparison/tvt_v02/{h4,d1}/*_summary.csv` dan HTML comparison.
- Untuk halaman detail model, gunakan `rolling_tvt_summary.csv`, `validation_report_*.html`, `ohlc_dot_*.html`, `loss_*.html`, dan `mae_atr_fold_allfull.html`.
- Untuk halaman tuning, gunakan `stage*_summary.csv`, `best_progression.csv`, dan `audit_finalize_report.json`.
- Untuk audit reproducibility, tampilkan `run_manifest.json`, `selected_schedule.csv`, dan `fold_scope_tvt_v02.json`.
- Jangan menjadikan folder kandidat tuning (`lstm_tvt_*`, `bilstm_tvt_*`, `lstm_d1_*`, `bilstm_d1_*`) sebagai menu utama kecuali aplikasi memang punya tab "Tuning Detail"; jumlah file di sana paling besar.

Jumlah file terkait yang terdeteksi pada root result/split/comparison utama:

| Root | Jumlah file |
| --- | ---: |
| `FLF_LSTM/results/tvt_v02` | 700 |
| `FLF_BILSTM/results/tvt_v02` | 716 |
| `Arima/result/tvt_v02` | 44 |
| `comparison/tvt_v02` | 18 |
| `results/splits/tvt_v02` | 224 |

Total pada lima root tersebut: 1702 file.
