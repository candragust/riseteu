#!/usr/bin/env bash
set -euo pipefail
cd "/home/hduser/jupyter/gust/RisetEU"
{
  echo "[START] $(date '+%F %T')"
  echo "[EXPERIMENT] tvt_d1_v02_end_to_end"
  echo "[MODE] full"
  echo "[MIN_STAGE] 1"
  echo "[MAX_STAGE] 7"
  echo "[RUNNER] /home/hduser/jupyter/gust/RisetEU/scripts/run_d1_tvt_v02_end_to_end.sh"
} >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_d1_v02_e2e_full_20260425_000842.log"
set +e
"/home/hduser/jupyter/gust/RisetEU/scripts/run_d1_tvt_v02_end_to_end.sh" "full" "1" "7" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_d1_v02_e2e_full_20260425_000842.log" 2>&1
status=$?
set -e
echo "[END] $(date '+%F %T') exit=$status" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_d1_v02_e2e_full_20260425_000842.log"
exit "$status"
