#!/usr/bin/env bash
set -euo pipefail
cd "/home/hduser/jupyter/gust/RisetEU"
{
  echo "[START] $(date '+%F %T')"
  echo "[MODEL] bilstm"
  echo "[RUNNER] /home/hduser/jupyter/gust/RisetEU/FLF_BILSTM/eurusd_bilstm_pipeline_runner_tvt_v02.py"
  echo "[MIN_STAGE] 1"
  echo "[MAX_STAGE] 6"
  echo "[MODE] full"
  echo "[PYTHON] /home/hduser/miniconda3/envs/test/bin/python"
} >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_h4_v02_bilstm_20260424_201124.log"
cmd=("/home/hduser/miniconda3/envs/test/bin/python" "/home/hduser/jupyter/gust/RisetEU/FLF_BILSTM/eurusd_bilstm_pipeline_runner_tvt_v02.py" --min-stage "1" --max-stage "6")
if [[ "full" == "skip" ]]; then
  cmd+=(--skip-training)
fi
"${cmd[@]}" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_h4_v02_bilstm_20260424_201124.log" 2>&1
status=$?
echo "[END] $(date '+%F %T') exit=$status" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_h4_v02_bilstm_20260424_201124.log"
exit "$status"
