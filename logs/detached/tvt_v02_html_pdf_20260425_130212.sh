#!/usr/bin/env bash
set -euo pipefail
cd "/home/hduser/jupyter/gust/RisetEU"
export AGENT_ACTIVITY_FILE="/home/hduser/jupyter/gust/RisetEU/agentactivity.md"
export OUTPUT_ROOT="/home/hduser/jupyter/gust/RisetEU/bukuThesis/penyusunan_buku_tesis_tvt_v02/04_lampiran_artefak/pdf_tvt_v02"
{
  echo "[START] $(date '+%F %T')"
  echo "[JOB] tvt_v02_html_pdf_conversion"
  echo "[RUNNER] /home/hduser/jupyter/gust/RisetEU/scripts/run_tvt_v02_html_pdf_conversion.sh"
  echo "[OUTPUT] /home/hduser/jupyter/gust/RisetEU/bukuThesis/penyusunan_buku_tesis_tvt_v02/04_lampiran_artefak/pdf_tvt_v02"
  echo "[ARGS] --overwrite"
  echo "[ACTIVITY] /home/hduser/jupyter/gust/RisetEU/agentactivity.md"
} >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_v02_html_pdf_20260425_130212.log"
printf -- "- %s | [SCREEN START] session=tvt_v02_html_pdf_20260425_130212 log=/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_v02_html_pdf_20260425_130212.log\n" "$(date '+%F %T')" >> "/home/hduser/jupyter/gust/RisetEU/agentactivity.md"
set +e
"/home/hduser/jupyter/gust/RisetEU/scripts/run_tvt_v02_html_pdf_conversion.sh" --overwrite >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_v02_html_pdf_20260425_130212.log" 2>&1
status=$?
set -e
echo "[END] $(date '+%F %T') exit=$status" >> "/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_v02_html_pdf_20260425_130212.log"
printf -- "- %s | [SCREEN END] session=tvt_v02_html_pdf_20260425_130212 exit=$status log=/home/hduser/jupyter/gust/RisetEU/logs/detached/tvt_v02_html_pdf_20260425_130212.log\n" "$(date '+%F %T')" >> "/home/hduser/jupyter/gust/RisetEU/agentactivity.md"
exit "$status"
