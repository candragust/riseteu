#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hduser/jupyter/gust/RisetEU"
LOG_DIR="$PROJECT_ROOT/logs/detached"
AGENT_ACTIVITY_FILE="${AGENT_ACTIVITY_FILE:-$PROJECT_ROOT/agentactivity.md}"

echo "Active TVT v02 screen sessions:"
screen -ls 2>/dev/null | grep 'tvt_.*_v02_' || echo "(none)"
echo
echo "Recent TVT v02 logs:"
if [[ -d "$LOG_DIR" ]]; then
  ls -lt "$LOG_DIR" | grep 'tvt_.*_v02_' | sed -n '1,20p' || echo "(no v02 logs yet)"
else
  echo "(no log directory yet)"
fi
echo
echo "Recent agent activity:"
if [[ -f "$AGENT_ACTIVITY_FILE" ]]; then
  tail -20 "$AGENT_ACTIVITY_FILE"
else
  echo "(no activity note yet: $AGENT_ACTIVITY_FILE)"
fi
