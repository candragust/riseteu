#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hduser/jupyter/gust/RisetEU"

usage() {
  cat <<'EOF'
Usage:
  run_tvt_v02.sh <timeframe> <model> [max_stage] [min_stage] [mode]

Arguments:
  timeframe  h4 | d1
  model      lstm | bilstm
  max_stage  Optional. H4 default 6, D1 default 7.
  min_stage  Optional. Default 1.
  mode       Optional. skip | full. Default skip.

Examples:
  run_tvt_v02.sh h4 lstm
  run_tvt_v02.sh h4 bilstm 6 1 full
  run_tvt_v02.sh d1 lstm
  run_tvt_v02.sh d1 bilstm 7 1 full
EOF
}

TIMEFRAME="${1:-}"
MODEL="${2:-}"
MAX_STAGE="${3:-}"
MIN_STAGE="${4:-1}"
MODE="${5:-skip}"

if [[ -z "$TIMEFRAME" || -z "$MODEL" ]]; then
  usage
  exit 1
fi

case "$TIMEFRAME" in
  h4)
    LAUNCHER="$PROJECT_ROOT/scripts/run_tvt_h4_detached_v02.sh"
    DEFAULT_MAX_STAGE="6"
    ;;
  d1)
    LAUNCHER="$PROJECT_ROOT/scripts/run_tvt_d1_detached_v02.sh"
    DEFAULT_MAX_STAGE="7"
    ;;
  *)
    echo "Unknown timeframe: $TIMEFRAME" >&2
    usage
    exit 1
    ;;
esac

if [[ -z "$MAX_STAGE" ]]; then
  MAX_STAGE="$DEFAULT_MAX_STAGE"
fi

exec "$LAUNCHER" "$MODEL" "$MAX_STAGE" "$MIN_STAGE" "$MODE"
