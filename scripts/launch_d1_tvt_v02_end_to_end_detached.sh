#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hduser/jupyter/gust/RisetEU}"
LOG_DIR="$PROJECT_ROOT/logs/detached"
AGENT_ACTIVITY_FILE="${AGENT_ACTIVITY_FILE:-$PROJECT_ROOT/agentactivity.md}"
MODE="${1:-skip}"
MIN_STAGE="${2:-1}"
MAX_STAGE="${3:-7}"

usage() {
  cat <<'EOF'
Usage:
  launch_d1_tvt_v02_end_to_end_detached.sh [mode] [min_stage] [max_stage]

Arguments:
  mode       skip | full. Default: skip.
  min_stage  Tuning stage start. Default: 1.
  max_stage  Tuning stage end. Default: 7.

Examples:
  launch_d1_tvt_v02_end_to_end_detached.sh skip
  launch_d1_tvt_v02_end_to_end_detached.sh full 1 7
  launch_d1_tvt_v02_end_to_end_detached.sh full 4 7
EOF
}

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" != "skip" && "$MODE" != "full" ]]; then
  echo "Unknown mode: $MODE" >&2
  usage
  exit 1
fi

if ! [[ "$MIN_STAGE" =~ ^[0-9]+$ && "$MAX_STAGE" =~ ^[0-9]+$ ]]; then
  echo "min_stage and max_stage must be integers." >&2
  usage
  exit 1
fi

if (( MAX_STAGE < 7 )); then
  echo "End-to-end finalization requires max_stage=7 because audit/freeze/evaluation depend on all D1 tuning stages." >&2
  echo "Use max_stage=7, or use scripts/run_tvt_d1_detached_v02.sh for model-only partial tuning." >&2
  exit 1
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required but not found." >&2
  exit 1
fi

RUNNER="$PROJECT_ROOT/scripts/run_d1_tvt_v02_end_to_end.sh"
if [[ ! -x "$RUNNER" ]]; then
  echo "Runner is not executable: $RUNNER" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$AGENT_ACTIVITY_FILE")"

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="tvt_d1_v02_e2e_${MODE}_${STAMP}"
LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"
WRAPPER="$LOG_DIR/${SESSION_NAME}.sh"

printf -- "- %s | [LAUNCH] D1 TVT v02 detached session=%s mode=%s stages=%s-%s log=%s\n" \
  "$(date '+%F %T')" "$SESSION_NAME" "$MODE" "$MIN_STAGE" "$MAX_STAGE" "$LOG_FILE" >> "$AGENT_ACTIVITY_FILE"

cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
export AGENT_ACTIVITY_FILE="$AGENT_ACTIVITY_FILE"
{
  echo "[START] \$(date '+%F %T')"
  echo "[EXPERIMENT] tvt_d1_v02_end_to_end"
  echo "[MODE] $MODE"
  echo "[MIN_STAGE] $MIN_STAGE"
  echo "[MAX_STAGE] $MAX_STAGE"
  echo "[RUNNER] $RUNNER"
  echo "[ACTIVITY] $AGENT_ACTIVITY_FILE"
} >> "$LOG_FILE"
printf -- "- %s | [SCREEN START] session=$SESSION_NAME log=$LOG_FILE\n" "\$(date '+%F %T')" >> "$AGENT_ACTIVITY_FILE"
set +e
"$RUNNER" "$MODE" "$MIN_STAGE" "$MAX_STAGE" >> "$LOG_FILE" 2>&1
status=\$?
set -e
echo "[END] \$(date '+%F %T') exit=\$status" >> "$LOG_FILE"
printf -- "- %s | [SCREEN END] session=$SESSION_NAME exit=\$status log=$LOG_FILE\n" "\$(date '+%F %T')" >> "$AGENT_ACTIVITY_FILE"
exit "\$status"
EOF

chmod +x "$WRAPPER"
screen -dmS "$SESSION_NAME" bash "$WRAPPER"

echo "Detached D1 TVT v02 end-to-end run started."
echo "Session : $SESSION_NAME"
echo "Log     : $LOG_FILE"
echo "Wrapper : $WRAPPER"
echo "Activity: $AGENT_ACTIVITY_FILE"
echo "Attach  : screen -r $SESSION_NAME"
echo "List    : screen -ls"
echo "Tail    : tail -f $LOG_FILE"
echo "Note    : tail -f $AGENT_ACTIVITY_FILE"
