#!/usr/bin/env bash
set -euo pipefail
cd "/home/hduser/jupyter/gust/RisetEU"
export AGENT_ACTIVITY_FILE="/home/hduser/jupyter/gust/RisetEU/agentactivity.md"
{
  echo "[START] $(date '+%F %T')"
  echo "[EXPERIMENT] tvt_d1_v02_end_to_end"
  echo "[MODE] full"
  echo "[MIN_STAGE] 1"
  echo "[MAX_STAGE] 7"
  echo "[RUNNER] /home/hduser/jupyter/gust/RisetEU/scripts/run_d1_tvt_v02_end_to_end.sh"
  echo "[ACTIVITY] /home/hduser/jupyter/gust/RisetEU/agentactivity.md"
} >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_d1_v02_e2e_full_20260425_083059.log"
printf -- "- %s | [SCREEN START] session=tvt_d1_v02_e2e_full_20260425_083059 log=/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_d1_v02_e2e_full_20260425_083059.log\n" "$(date '+%F %T')" >> "/home/hduser/jupyter/gust/RisetEU/agentactivity.md"
set +e
"/home/hduser/jupyter/gust/RisetEU/scripts/run_d1_tvt_v02_end_to_end.sh" "full" "1" "7" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_d1_v02_e2e_full_20260425_083059.log" 2>&1
status=$?
set -e
echo "[END] $(date '+%F %T') exit=$status" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_d1_v02_e2e_full_20260425_083059.log"
printf -- "- %s | [SCREEN END] session=tvt_d1_v02_e2e_full_20260425_083059 exit=$status log=/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_d1_v02_e2e_full_20260425_083059.log\n" "$(date '+%F %T')" >> "/home/hduser/jupyter/gust/RisetEU/agentactivity.md"
exit "$status"
