#!/bin/bash
# Launch MLflow UI for AIVC v1.1 sweep visualisation.
# Run this in a terminal, then open http://localhost:5000 in browser.
#
# The UI shows:
#   - All 36 sweep configurations in a table
#   - Parallel coordinates plot: lfc_beta x K x lambda_l1 -> test_r
#   - W density trajectory per run (time-series)
#   - JAK-STAT recovery vs Pearson r scatter
#   - Model checkpoint artifacts per run
#
# Usage: ./scripts/launch_mlflow_ui.sh [port]

PORT=${1:-5000}
echo "Starting MLflow UI at http://localhost:${PORT}"
echo "Tracking store: ./mlruns (local file store)"
echo "Press Ctrl+C to stop."
mlflow ui --port "${PORT}" --backend-store-uri ./mlruns
