#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hduser/jupyter/gust/RisetEU}"
REPORT_PYTHON="${REPORT_PYTHON:-python}"

LSTM_DIR="$PROJECT_ROOT/FLF_LSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3"
BILSTM_DIR="$PROJECT_ROOT/FLF_BILSTM/results/tvt_v02/d1_ohlc/d1_evaluation_last3"
ARIMA_DIR="$PROJECT_ROOT/Arima/result/tvt_v02/d1_evaluation_last3"
COMPARISON_OUT="$PROJECT_ROOT/comparison/tvt_v02/d1/comparison_models_d1_tvt_v02_last3_v01.html"

LSTM_CFG="$PROJECT_ROOT/FLF_LSTM/configs/d1_ohlc/lstm_flf_config_d1_tvt_v02_best.json"
BILSTM_CFG="$PROJECT_ROOT/FLF_BILSTM/configs/d1_ohlc/bilstm_flf_config_d1_tvt_v02_best.json"
ARIMA_CFG="$PROJECT_ROOT/Arima/arima_baseline_config_d1_ohlc_v01.json"

cd "$PROJECT_ROOT"

for path in \
  "$LSTM_DIR/rolling_tvt_summary.csv" \
  "$BILSTM_DIR/rolling_tvt_summary.csv" \
  "$ARIMA_DIR/rolling_fixed_summary.csv" \
  "$LSTM_CFG" \
  "$BILSTM_CFG" \
  "$ARIMA_CFG"; do
  if [[ ! -e "$path" ]]; then
    echo "Required artifact not found: $path" >&2
    exit 1
  fi
done

mkdir -p "$PROJECT_ROOT/comparison/tvt_v02/d1"

"$REPORT_PYTHON" "$PROJECT_ROOT/FLF_LSTM/build_wf72_test1_reports.py" \
  --rolling-dir "$LSTM_DIR" \
  --out-validation "$LSTM_DIR/validation_report_d1_tvt_v02_lstm_last3.html" \
  --out-ohlc "$LSTM_DIR/ohlc_dot_d1_tvt_v02_lstm_last3_all.html" \
  --out-loss "$LSTM_DIR/loss_d1_tvt_v02_lstm_last3.html" \
  --out-gradient "$LSTM_DIR/loss_gradient_d1_tvt_v02_lstm_last3.html" \
  --title-prefix "EUR/USD D1 FLF-LSTM Last3" \
  --config-path "$LSTM_CFG" \
  --method-label "FLF-LSTM"

"$REPORT_PYTHON" "$PROJECT_ROOT/FLF_LSTM/build_wf72_test1_reports.py" \
  --rolling-dir "$BILSTM_DIR" \
  --out-validation "$BILSTM_DIR/validation_report_d1_tvt_v02_bilstm_last3.html" \
  --out-ohlc "$BILSTM_DIR/ohlc_dot_d1_tvt_v02_bilstm_last3_all.html" \
  --out-loss "$BILSTM_DIR/loss_d1_tvt_v02_bilstm_last3.html" \
  --out-gradient "$BILSTM_DIR/loss_gradient_d1_tvt_v02_bilstm_last3.html" \
  --title-prefix "EUR/USD D1 FLF-BiLSTM Last3" \
  --config-path "$BILSTM_CFG" \
  --method-label "FLF-BiLSTM"

"$REPORT_PYTHON" "$PROJECT_ROOT/Arima/generate_arima_html_report.py" \
  --result-dir "$ARIMA_DIR" \
  --out "$ARIMA_DIR/index.html"

"$REPORT_PYTHON" "$PROJECT_ROOT/build_tvt_v02_mae_atr_combined_reports.py" \
  --timeframe d1

"$REPORT_PYTHON" "$PROJECT_ROOT/compare_models_h4_tvt_v02_last3_v01.py" \
  --arima-dir "$ARIMA_DIR" \
  --lstm-dir "$LSTM_DIR" \
  --bilstm-dir "$BILSTM_DIR" \
  --lstm-config "$LSTM_CFG" \
  --bilstm-config "$BILSTM_CFG" \
  --arima-config "$ARIMA_CFG" \
  --out-html "$COMPARISON_OUT" \
  --profile-label "D1 TVT v02" \
  --timeframe-label "D1" \
  --tuning-folds-label "13-18" \
  --evaluation-folds-label "19-21"

echo "D1 TVT v02 reports generated."
echo "LSTM      : $LSTM_DIR/validation_report_d1_tvt_v02_lstm_last3.html"
echo "BiLSTM    : $BILSTM_DIR/validation_report_d1_tvt_v02_bilstm_last3.html"
echo "ARIMA     : $ARIMA_DIR/index.html"
echo "Comparison: $COMPARISON_OUT"
