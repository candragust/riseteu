#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hduser/jupyter/gust/RisetEU"
PYTHON_BIN="/home/hduser/miniconda3/envs/test/bin/python"
RUNNER="$PROJECT_ROOT/FLF_LSTM/eurusd_lstm_d1_pipeline_runner_v01.py"
BASE_CONFIG="$PROJECT_ROOT/FLF_LSTM/configs/d1_ohlc/lstm_flf_config_d1_ohlc_base_v01.json"
OUT_DIR="$PROJECT_ROOT/FLF_LSTM/results/d1_ohlc/tuning_v02"
LOG_DIR="$PROJECT_ROOT/logs/detached"

usage() {
  cat <<'EOF'
Usage:
  run_d1_lstm_tuning_detached.sh [min_stage] [max_stage]

Arguments:
  min_stage  Optional. Default 1.
  max_stage  Optional. Default 7.

Behavior:
  - Runs EUR/USD 1D FLF-LSTM tuning in a detached screen session.
  - Uses a fresh output root: FLF_LSTM/results/d1_ohlc/tuning_v02
  - Preserves existing tuning_v01 artifacts.

Examples:
  run_d1_lstm_tuning_detached.sh
  run_d1_lstm_tuning_detached.sh 1 7
  run_d1_lstm_tuning_detached.sh 4 7
EOF
}

MIN_STAGE="${1:-1}"
MAX_STAGE="${2:-7}"

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

mkdir -p "$LOG_DIR" "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="d1_lstm_tuning_${STAMP}"
LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"
WRAPPER="$LOG_DIR/${SESSION_NAME}.sh"

cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
{
  echo "[START] \$(date '+%F %T')"
  echo "[EXPERIMENT] d1_lstm_tuning"
  echo "[RUNNER] $RUNNER"
  echo "[BASE_CONFIG] $BASE_CONFIG"
  echo "[OUT_DIR] $OUT_DIR"
  echo "[MIN_STAGE] $MIN_STAGE"
  echo "[MAX_STAGE] $MAX_STAGE"
  echo "[PYTHON] $PYTHON_BIN"
} >> "$LOG_FILE"
"$PYTHON_BIN" "$RUNNER" \\
  --base-config "$BASE_CONFIG" \\
  --out-dir "$OUT_DIR" \\
  --min-stage "$MIN_STAGE" \\
  --max-stage "$MAX_STAGE" >> "$LOG_FILE" 2>&1
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
