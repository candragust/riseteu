#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hduser/jupyter/gust/RisetEU}"
TF_PYTHON="${TF_PYTHON:-/home/hduser/miniconda3/envs/test/bin/python}"
REPORT_PYTHON="${REPORT_PYTHON:-python}"
AGENT_ACTIVITY_FILE="${AGENT_ACTIVITY_FILE:-$PROJECT_ROOT/agentactivity.md}"

MODE="${1:-skip}"
MIN_STAGE="${2:-1}"
MAX_STAGE="${3:-7}"

usage() {
  cat <<'EOF'
Usage:
  run_d1_tvt_v02_end_to_end.sh [mode] [min_stage] [max_stage]

Arguments:
  mode       skip | full. Default: skip.
  min_stage  Tuning stage start. Default: 1.
  max_stage  Tuning stage end. Default: 7.

Behavior:
  1. Run/refresh FLF-LSTM D1 TVT v02 tuning.
  2. Audit, freeze config, and evaluate FLF-LSTM on fold 19-21.
  3. Run/refresh FLF-BiLSTM D1 TVT v02 tuning.
  4. Audit, freeze config, and evaluate FLF-BiLSTM on fold 19-21.
  5. Run ARIMA on exact TVT v02 D1 fold 19-21 combined split files.
  6. Generate validation, MAE/ATR, ARIMA, and comparison reports.

Use mode=full for real training. Use mode=skip to rebuild from existing stage artifacts.
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

cd "$PROJECT_ROOT"

log_activity() {
  local message="$1"
  mkdir -p "$(dirname "$AGENT_ACTIVITY_FILE")"
  printf -- "- %s | %s\n" "$(date '+%F %T')" "$message" >> "$AGENT_ACTIVITY_FILE" || true
}

announce_step() {
  local message="$1"
  echo "$message"
  log_activity "$message"
}

log_activity "[START] D1 TVT v02 end-to-end mode=$MODE stages=$MIN_STAGE-$MAX_STAGE"
trap 'status=$?; if (( status != 0 )); then log_activity "[FAILED] D1 TVT v02 end-to-end exit=$status"; fi' EXIT

for path in \
  "$TF_PYTHON" \
  "$PROJECT_ROOT/FLF_LSTM/eurusd_lstm_d1_pipeline_runner_tvt_v02.py" \
  "$PROJECT_ROOT/FLF_BILSTM/eurusd_bilstm_d1_pipeline_runner_tvt_v02.py" \
  "$PROJECT_ROOT/scripts/finalize_tvt_v02.py" \
  "$PROJECT_ROOT/Arima/arima_tvt_runner_v02.py" \
  "$PROJECT_ROOT/scripts/generate_d1_tvt_v02_reports.sh"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 1
  fi
done

skip_arg=()
if [[ "$MODE" == "skip" ]]; then
  skip_arg=(--skip-training)
fi

announce_step "[1/6] FLF-LSTM D1 TVT v02 tuning mode=$MODE stages=$MIN_STAGE-$MAX_STAGE"
"$TF_PYTHON" "$PROJECT_ROOT/FLF_LSTM/eurusd_lstm_d1_pipeline_runner_tvt_v02.py" \
  --min-stage "$MIN_STAGE" \
  --max-stage "$MAX_STAGE" \
  "${skip_arg[@]}"
log_activity "[1/6 DONE] FLF-LSTM D1 TVT v02 tuning"

announce_step "[2/6] FLF-LSTM audit/freeze/evaluation"
"$TF_PYTHON" "$PROJECT_ROOT/scripts/finalize_tvt_v02.py" d1 lstm \
  --python-bin "$TF_PYTHON"
log_activity "[2/6 DONE] FLF-LSTM audit/freeze/evaluation"

announce_step "[3/6] FLF-BiLSTM D1 TVT v02 tuning mode=$MODE stages=$MIN_STAGE-$MAX_STAGE"
"$TF_PYTHON" "$PROJECT_ROOT/FLF_BILSTM/eurusd_bilstm_d1_pipeline_runner_tvt_v02.py" \
  --min-stage "$MIN_STAGE" \
  --max-stage "$MAX_STAGE" \
  "${skip_arg[@]}"
log_activity "[3/6 DONE] FLF-BiLSTM D1 TVT v02 tuning"

announce_step "[4/6] FLF-BiLSTM audit/freeze/evaluation"
"$TF_PYTHON" "$PROJECT_ROOT/scripts/finalize_tvt_v02.py" d1 bilstm \
  --python-bin "$TF_PYTHON"
log_activity "[4/6 DONE] FLF-BiLSTM audit/freeze/evaluation"

announce_step "[5/6] ARIMA D1 TVT v02 evaluation"
"$TF_PYTHON" "$PROJECT_ROOT/Arima/arima_tvt_runner_v02.py" \
  --schedule "$PROJECT_ROOT/results/splits/tvt_v02/d1/fold_schedule_evaluation.csv" \
  --fold-root "$PROJECT_ROOT/results/splits/tvt_v02/d1" \
  --base-config "$PROJECT_ROOT/Arima/arima_baseline_config_d1_ohlc_v01.json" \
  --out-dir "$PROJECT_ROOT/Arima/result/tvt_v02/d1_evaluation_last3" \
  --python-bin "$TF_PYTHON"
log_activity "[5/6 DONE] ARIMA D1 TVT v02 evaluation"

announce_step "[6/6] D1 TVT v02 reports"
REPORT_PYTHON="$REPORT_PYTHON" "$PROJECT_ROOT/scripts/generate_d1_tvt_v02_reports.sh"
log_activity "[6/6 DONE] D1 TVT v02 reports"

echo "[DONE] D1 TVT v02 end-to-end workflow completed."
log_activity "[DONE] D1 TVT v02 end-to-end workflow completed"
