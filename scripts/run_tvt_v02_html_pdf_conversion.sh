#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hduser/jupyter/gust/RisetEU}"
REPORT_PYTHON="${REPORT_PYTHON:-python3}"
AGENT_ACTIVITY_FILE="${AGENT_ACTIVITY_FILE:-$PROJECT_ROOT/agentactivity.md}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/bukuThesis/penyusunan_buku_tesis_tvt_v02/04_lampiran_artefak/pdf_tvt_v02}"

usage() {
  cat <<'EOF'
Usage:
  run_tvt_v02_html_pdf_conversion.sh [converter args...]

Examples:
  run_tvt_v02_html_pdf_conversion.sh --dry-run
  run_tvt_v02_html_pdf_conversion.sh --overwrite
  run_tvt_v02_html_pdf_conversion.sh --include 'comparison/tvt_v02' --overwrite

Defaults:
  Converts every .html path containing tvt_v02.
  Writes PDFs under:
    bukuThesis/penyusunan_buku_tesis_tvt_v02/04_lampiran_artefak/pdf_tvt_v02

Print safety:
  Default paper is A3 landscape with injected print CSS/JS to reduce chart
  clipping in wide Plotly reports.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$PROJECT_ROOT"

log_activity() {
  local message="$1"
  mkdir -p "$(dirname "$AGENT_ACTIVITY_FILE")"
  printf -- "- %s | %s\n" "$(date '+%F %T')" "$message" >> "$AGENT_ACTIVITY_FILE" || true
}

CONVERTER="$PROJECT_ROOT/scripts/convert_tvt_v02_html_to_pdf.py"
if [[ ! -f "$CONVERTER" ]]; then
  echo "Converter not found: $CONVERTER" >&2
  exit 1
fi

if ! command -v google-chrome >/dev/null 2>&1 && [[ -z "${CHROME_BIN:-}" ]]; then
  echo "google-chrome is required, or set CHROME_BIN." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
log_activity "[START] TVT v02 HTML PDF conversion output=$OUTPUT_ROOT args=$*"

set +e
"$REPORT_PYTHON" "$CONVERTER" \
  --root "$PROJECT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --activity-file "$AGENT_ACTIVITY_FILE" \
  "$@"
status=$?
set -e

if (( status == 0 )); then
  log_activity "[DONE] TVT v02 HTML PDF conversion output=$OUTPUT_ROOT"
else
  log_activity "[FAILED] TVT v02 HTML PDF conversion exit=$status output=$OUTPUT_ROOT"
fi

exit "$status"

