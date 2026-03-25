#!/bin/bash
# Start the AIVC inference API.
#
# Usage:
#   ./scripts/start_api.sh              # development (auto-reload)
#   ./scripts/start_api.sh --prod       # production (4 workers)
#   ./scripts/start_api.sh --port 9000  # custom port
#
# Environment variables:
#   AIVC_CHECKPOINT      Path to model checkpoint (default: auto-detect)
#   MLFLOW_TRACKING_URI  MLflow server (default: http://localhost:5000)
#
# Once running:
#   Health:   curl http://localhost:8000/health
#   Docs:     http://localhost:8000/docs  (Swagger UI)
#   Redoc:    http://localhost:8000/redoc

PORT=8000
MODE="dev"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prod) MODE="prod"; shift ;;
        --port) PORT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "Starting AIVC API on port ${PORT} (mode: ${MODE})"

if [ "$MODE" = "prod" ]; then
    uvicorn api.server:app --host 0.0.0.0 --port "${PORT}" --workers 4
else
    uvicorn api.server:app --host 0.0.0.0 --port "${PORT}" --reload
fi
