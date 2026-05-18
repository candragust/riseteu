# Runbook D1 TVT v02 Detached

Scope: EUR/USD D1 OHLC, TVT v02, tuning fold 13-18, final evaluation fold 19-21.

## Status Saat Ini

- Readiness: `True`
- Results complete: `True`
- Audit readiness: `results/d1_ohlc/tvt_v02/audit_d1_tvt_v02_readiness.md`
- Blocker lama sudah diperbaiki: `Arima/generate_arima_html_report.py` sekarang kompatibel dengan `rolling_fixed_summary.csv` ARIMA TVT v02 yang sudah berisi kolom metrik `mae_avg_pips`.

## Command Utama

Full detached dari awal stage tuning:

```bash
scripts/launch_d1_tvt_v02_end_to_end_detached.sh full 1 7
```

Resume detached dari stage tertentu, contoh mulai stage 4:

```bash
scripts/launch_d1_tvt_v02_end_to_end_detached.sh full 4 7
```

Rebuild dari artefak training yang sudah ada, termasuk audit, evaluation, ARIMA, report HTML, MAE/ATR, dan comparison:

```bash
scripts/launch_d1_tvt_v02_end_to_end_detached.sh skip 1 7
```

Generate report saja tanpa training/evaluation ulang:

```bash
scripts/generate_d1_tvt_v02_reports.sh
```

Audit kesiapan:

```bash
python scripts/audit_d1_tvt_v02_readiness.py
```

Status detached:

```bash
scripts/status_tvt_v02.sh
screen -ls
```

Catatan aktivitas agent:

```bash
tail -f agentactivity.md
```

Tail log aktif:

```bash
tail -f logs/detached/<nama_session>.log
```

## Output Utama

- FLF-LSTM evaluation: `FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/`
- FLF-BiLSTM evaluation: `FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/`
- ARIMA evaluation: `Arima/result/tvt_v02/d1_evaluation_last3/`
- Comparison: `comparison/tvt_v02/d1/comparison_models_d1_tvt_v02_last3_v01.html`
- MAE/ATR LSTM: `FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/mae_atr_fold_allfull.html`
- MAE/ATR BiLSTM: `FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3/mae_atr_fold_allfull.html`
- MAE/ATR ARIMA: `Arima/result/tvt_v02/d1_evaluation_last3/mae_atr_fold_allfull.html`

## Pipeline End-to-End

1. FLF-LSTM D1 tuning stage 1-7.
2. Audit tuning, freeze config, evaluate FLF-LSTM fold 19-21.
3. FLF-BiLSTM D1 tuning stage 1-7.
4. Audit tuning, freeze config, evaluate FLF-BiLSTM fold 19-21.
5. ARIMA D1 evaluation fold 19-21 memakai split TVT v02 yang sama.
6. Generate validation report, OHLC plot, loss plot, MAE/ATR report, ARIMA index, dan comparison.

## Hasil Ringkas D1 TVT v02 Last3

| Model | MAE AVG | corr2 AVG HLC | Directional Accuracy |
| --- | ---: | ---: | ---: |
| ARIMA | 38.3932 | 0.5368 | 42.42% |
| FLF-LSTM | 38.0826 | 0.5222 | 46.89% |
| FLF-BiLSTM | 26.1765 | 0.5950 | 46.89% |

Kesimpulan operasional: untuk skenario D1 TVT v02 last3, FLF-BiLSTM menjadi model utama berdasarkan MAE AVG dan corr2 AVG HLC. Directional Accuracy seri antara FLF-LSTM dan FLF-BiLSTM.
