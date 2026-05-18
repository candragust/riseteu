# Audit Readiness D1 TVT v02

- Ready to launch detached: `True`
- Results complete: `True`
- Note: readiness checks environment, split files, runners, launchers, and dependencies. Output status below shows whether D1 artifacts have already been produced.
- Project root: `/home/hduser/jupyter/gust/RisetEU`
- Full detached: `scripts/launch_d1_tvt_v02_end_to_end_detached.sh full 1 7`
- Rebuild existing: `scripts/launch_d1_tvt_v02_end_to_end_detached.sh skip 1 7`

## Dependencies

- `screen`: `True`
- TF python: `/home/hduser/miniconda3/envs/test/bin/python`
- TF modules: `{'tensorflow': True, 'statsmodels': True, 'pandas': True, 'numpy': True}`
- Report python: `python`
- Report modules: `{'plotly': True, 'pandas': True, 'numpy': True, 'statsmodels': True}`

## Schedules

- Tuning folds: `[13, 14, 15, 16, 17, 18]`
- Evaluation folds: `[19, 20, 21]`

## Current Output Status

| Group | Exists | File count | Key files |
| --- | ---: | ---: | --- |
| lstm_tuning | True | 35 | audit_finalize_report.md, best_progression.csv |
| bilstm_tuning | True | 33 | audit_finalize_report.md, best_progression.csv |
| lstm_eval | True | 18 | mae_atr_fold_allfull.html, rolling_fixed_summary.csv, rolling_tvt_summary.csv |
| bilstm_eval | True | 18 | mae_atr_fold_allfull.html, rolling_fixed_summary.csv, rolling_tvt_summary.csv |
| arima_eval | True | 18 | index.html, mae_atr_fold_allfull.html, rolling_fixed_summary.csv, rolling_tvt_summary.csv |
| comparison | True | 9 | comparison_models_d1_tvt_v02_last3_v01.html |

## Required Paths

| Path | Exists |
| --- | ---: |
| `/home/hduser/jupyter/gust/RisetEU/results/splits/tvt_v02/d1` | True |
| `/home/hduser/jupyter/gust/RisetEU/results/splits/tvt_v02/d1/fold_schedule_tuning.csv` | True |
| `/home/hduser/jupyter/gust/RisetEU/results/splits/tvt_v02/d1/fold_schedule_evaluation.csv` | True |
| `/home/hduser/jupyter/gust/RisetEU/FLF_LSTM/eurusd_lstm_d1_pipeline_runner_tvt_v02.py` | True |
| `/home/hduser/jupyter/gust/RisetEU/FLF_BILSTM/eurusd_bilstm_d1_pipeline_runner_tvt_v02.py` | True |
| `/home/hduser/jupyter/gust/RisetEU/FLF_LSTM/rolling_tvt_lstm_d1_runner_v02.py` | True |
| `/home/hduser/jupyter/gust/RisetEU/FLF_BILSTM/rolling_tvt_bilstm_d1_runner_v02.py` | True |
| `/home/hduser/jupyter/gust/RisetEU/scripts/finalize_tvt_v02.py` | True |
| `/home/hduser/jupyter/gust/RisetEU/Arima/arima_tvt_runner_v02.py` | True |
| `/home/hduser/jupyter/gust/RisetEU/scripts/run_d1_tvt_v02_end_to_end.sh` | True |
| `/home/hduser/jupyter/gust/RisetEU/scripts/launch_d1_tvt_v02_end_to_end_detached.sh` | True |
| `/home/hduser/jupyter/gust/RisetEU/scripts/generate_d1_tvt_v02_reports.sh` | True |
| `/home/hduser/jupyter/gust/RisetEU/FLF_LSTM/configs/d1_ohlc/lstm_flf_config_d1_tvt_v02_base.json` | True |
| `/home/hduser/jupyter/gust/RisetEU/FLF_BILSTM/configs/d1_ohlc/bilstm_flf_config_d1_tvt_v02_base.json` | True |
| `/home/hduser/jupyter/gust/RisetEU/Arima/arima_baseline_config_d1_ohlc_v01.json` | True |
