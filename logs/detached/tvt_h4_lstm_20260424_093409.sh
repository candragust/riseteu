#!/usr/bin/env bash
set -euo pipefail
cd "/home/hduser/jupyter/gust/RisetEU"
{
  echo "[START] $(date '+%F %T')"
  echo "[MODEL] lstm"
  echo "[RUNNER] /home/hduser/jupyter/gust/RisetEU/FLF_LSTM/eurusd_lstm_pipeline_runner_tvt_v01.py"
  echo "[MAX_STAGE] 6"
  echo "[PYTHON] /home/hduser/miniconda3/envs/test/bin/python"
} >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_h4_lstm_20260424_093409.log"
"/home/hduser/miniconda3/envs/test/bin/python" "/home/hduser/jupyter/gust/RisetEU/FLF_LSTM/eurusd_lstm_pipeline_runner_tvt_v01.py" --skip-training --max-stage "6" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_h4_lstm_20260424_093409.log" 2>&1
status=$?
echo "[END] $(date '+%F %T') exit=$status" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_h4_lstm_20260424_093409.log"
exit "$status"
