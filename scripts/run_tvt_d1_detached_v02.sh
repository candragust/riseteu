#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hduser/jupyter/gust/RisetEU"
PYTHON_BIN="/home/hduser/miniconda3/envs/test/bin/python"
LOG_DIR="$PROJECT_ROOT/logs/detached"

usage() {
  cat <<'EOF'
Usage:
  run_tvt_d1_detached_v02.sh <model> [max_stage] [min_stage] [mode]

Arguments:
  model      lstm | bilstm
  max_stage  Optional. Default 7.
  min_stage  Optional. Default 1.
  mode       Optional. skip | full. Default skip.

Behavior:
  - Runs EUR/USD 1D TVT v02 tuning in a detached screen session.
  - `skip` mode reuses completed job outputs with `--skip-training`.
  - `full` mode retrains the requested stage range.

Examples:
  run_tvt_d1_detached_v02.sh lstm
  run_tvt_d1_detached_v02.sh lstm 7
  run_tvt_d1_detached_v02.sh bilstm 7
  run_tvt_d1_detached_v02.sh bilstm 7 4 full
EOF
}

MODEL="${1:-}"
MAX_STAGE="${2:-7}"
MIN_STAGE="${3:-1}"
MODE="${4:-skip}"

if [[ -z "$MODEL" ]]; then
  usage
  exit 1
fi

case "$MODEL" in
  lstm)
    RUNNER="$PROJECT_ROOT/FLF_LSTM/eurusd_lstm_d1_pipeline_runner_tvt_v02.py"
    BASE_CONFIG="$PROJECT_ROOT/FLF_LSTM/configs/d1_ohlc/lstm_flf_config_d1_tvt_v02_base.json"
    OUT_DIR="$PROJECT_ROOT/FLF_LSTM/results/tvt_v02/d1_ohlc/lstm_pipeline_d1_tuning_last6"
    ;;
  bilstm)
    RUNNER="$PROJECT_ROOT/FLF_BILSTM/eurusd_bilstm_d1_pipeline_runner_tvt_v02.py"
    BASE_CONFIG="$PROJECT_ROOT/FLF_BILSTM/configs/d1_ohlc/bilstm_flf_config_d1_tvt_v02_base.json"
    OUT_DIR="$PROJECT_ROOT/FLF_BILSTM/results/tvt_v02/d1_ohlc/bilstm_pipeline_d1_tuning_last6"
    ;;
  *)
    echo "Unknown model: $MODEL" >&2
    usage
    exit 1
    ;;
esac

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required but not found." >&2
  exit 1
fi

for path in "$RUNNER" "$BASE_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done

if [[ "$MODE" != "skip" && "$MODE" != "full" ]]; then
  echo "Unknown mode: $MODE" >&2
  usage
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="tvt_d1_v02_${MODEL}_${STAMP}"
LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"
WRAPPER="$LOG_DIR/${SESSION_NAME}.sh"

cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
{
  echo "[START] \$(date '+%F %T')"
  echo "[EXPERIMENT] tvt_d1_v02_tuning"
  echo "[MODEL] $MODEL"
  echo "[RUNNER] $RUNNER"
  echo "[BASE_CONFIG] $BASE_CONFIG"
  echo "[OUT_DIR] $OUT_DIR"
  echo "[MIN_STAGE] $MIN_STAGE"
  echo "[MAX_STAGE] $MAX_STAGE"
  echo "[MODE] $MODE"
  echo "[PYTHON] $PYTHON_BIN"
} >> "$LOG_FILE"
cmd=("$PYTHON_BIN" "$RUNNER" --base-config "$BASE_CONFIG" --out-dir "$OUT_DIR" --min-stage "$MIN_STAGE" --max-stage "$MAX_STAGE")
if [[ "$MODE" == "skip" ]]; then
  cmd+=(--skip-training)
fi
"\${cmd[@]}" >> "$LOG_FILE" 2>&1
status=\$?
echo "[END] \$(date '+%F %T') exit=\$status" >> "$LOG_FILE"
exit "\$status"
EOF

chmod +x "$WRAPPER"
screen -dmS "$SESSION_NAME" bash "$WRAPPER"

echo "Detached run started."
echo "Session : $SESSION_NAME"
echo "Log     : $LOG_FILE"
echo "Wrapper : $WRAPPER"
echo "Out dir : $OUT_DIR"
echo "Attach  : screen -r $SESSION_NAME"
echo "List    : screen -ls"
echo "Tail    : tail -f $LOG_FILE"
