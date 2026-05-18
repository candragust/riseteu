#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hduser/jupyter/gust/RisetEU}"
LOG_DIR="$PROJECT_ROOT/logs/detached"
AGENT_ACTIVITY_FILE="${AGENT_ACTIVITY_FILE:-$PROJECT_ROOT/agentactivity.md}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/bukuThesis/penyusunan_buku_tesis_tvt_v02/04_lampiran_artefak/pdf_tvt_v02}"

usage() {
  cat <<'EOF'
Usage:
  launch_tvt_v02_html_pdf_detached.sh [converter args...]

Examples:
  launch_tvt_v02_html_pdf_detached.sh --overwrite
  launch_tvt_v02_html_pdf_detached.sh --include 'comparison/tvt_v02' --overwrite

Monitoring:
  screen -ls
  tail -f logs/detached/<session>.log
  tail -f agentactivity.md
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required but not found." >&2
  exit 1
fi

RUNNER="$PROJECT_ROOT/scripts/run_tvt_v02_html_pdf_conversion.sh"
if [[ ! -x "$RUNNER" ]]; then
  echo "Runner is not executable: $RUNNER" >&2
  echo "Run: chmod +x $RUNNER" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$AGENT_ACTIVITY_FILE")"
mkdir -p "$OUTPUT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="tvt_v02_html_pdf_${STAMP}"
LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"
WRAPPER="$LOG_DIR/${SESSION_NAME}.sh"
RUN_ARGS_Q=()
for arg in "$@"; do
  RUN_ARGS_Q+=("$(printf '%q' "$arg")")
done
RUN_ARGS_TEXT="${RUN_ARGS_Q[*]}"

printf -- "- %s | [LAUNCH] TVT v02 HTML PDF detached session=%s log=%s output=%s args=%s\n" \
  "$(date '+%F %T')" "$SESSION_NAME" "$LOG_FILE" "$OUTPUT_ROOT" "$RUN_ARGS_TEXT" >> "$AGENT_ACTIVITY_FILE"

cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
export AGENT_ACTIVITY_FILE="$AGENT_ACTIVITY_FILE"
export OUTPUT_ROOT="$OUTPUT_ROOT"
{
  echo "[START] \$(date '+%F %T')"
  echo "[JOB] tvt_v02_html_pdf_conversion"
  echo "[RUNNER] $RUNNER"
  echo "[OUTPUT] $OUTPUT_ROOT"
  echo "[ARGS] $RUN_ARGS_TEXT"
  echo "[ACTIVITY] $AGENT_ACTIVITY_FILE"
} >> "$LOG_FILE"
printf -- "- %s | [SCREEN START] session=$SESSION_NAME log=$LOG_FILE\n" "\$(date '+%F %T')" >> "$AGENT_ACTIVITY_FILE"
set +e
"$RUNNER" $RUN_ARGS_TEXT >> "$LOG_FILE" 2>&1
status=\$?
set -e
echo "[END] \$(date '+%F %T') exit=\$status" >> "$LOG_FILE"
printf -- "- %s | [SCREEN END] session=$SESSION_NAME exit=\$status log=$LOG_FILE\n" "\$(date '+%F %T')" >> "$AGENT_ACTIVITY_FILE"
exit "\$status"
EOF

chmod +x "$WRAPPER"
screen -dmS "$SESSION_NAME" bash "$WRAPPER"

echo "Detached TVT v02 HTML to PDF conversion started."
echo "Session : $SESSION_NAME"
echo "Log     : $LOG_FILE"
echo "Wrapper : $WRAPPER"
echo "Output  : $OUTPUT_ROOT"
echo "Activity: $AGENT_ACTIVITY_FILE"
echo "Attach  : screen -r $SESSION_NAME"
echo "List    : screen -ls"
echo "Tail    : tail -f $LOG_FILE"
echo "Note    : tail -f $AGENT_ACTIVITY_FILE"
