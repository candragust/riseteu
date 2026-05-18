#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hduser/jupyter/gust/RisetEU"
LOG_DIR="$PROJECT_ROOT/logs/detached"

echo "Active screen sessions:"
screen -ls || true
echo
echo "Recent detached logs:"
if [[ -d "$LOG_DIR" ]]; then
  ls -lt "$LOG_DIR" | sed -n '1,20p'
else
  echo "(no log directory yet)"
fi
