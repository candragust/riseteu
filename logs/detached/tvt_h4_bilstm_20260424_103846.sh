#!/usr/bin/env bash
set -euo pipefail
cd "/home/hduser/jupyter/gust/RisetEU"
{
  echo "[START] $(date '+%F %T')"
  echo "[MODEL] bilstm"
  echo "[RUNNER] /home/hduser/jupyter/gust/RisetEU/FLF_BILSTM/eurusd_bilstm_pipeline_runner_tvt_v01.py"
  echo "[MAX_STAGE] 6"
  echo "[PYTHON] /home/hduser/miniconda3/envs/test/bin/python"
} >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_h4_bilstm_20260424_103846.log"
"/home/hduser/miniconda3/envs/test/bin/python" "/home/hduser/jupyter/gust/RisetEU/FLF_BILSTM/eurusd_bilstm_pipeline_runner_tvt_v01.py" --skip-training --max-stage "6" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_h4_bilstm_20260424_103846.log" 2>&1
status=$?
echo "[END] $(date '+%F %T') exit=$status" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_h4_bilstm_20260424_103846.log"
exit "$status"
