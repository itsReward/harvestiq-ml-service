#!/bin/bash
# Start the ML service with proper configuration

set -e

echo "🌾 Starting HarvestIQ ML Model Service..."

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Load environment variables (safely handling comments)
if [ -f .env ]; then
    # Export only lines that are not comments and contain =
    export $(grep -v '^#' .env | grep '=' | xargs)
fi

# Set default values
export MODEL_STORAGE_PATH=${MODEL_STORAGE_PATH:-"./storage/models"}
export API_PORT=${API_PORT:-8001}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Create required directories
mkdir -p "${MODEL_STORAGE_PATH}/versions"
mkdir -p "${MODEL_STORAGE_PATH}/active"
mkdir -p "${MODEL_STORAGE_PATH}/archive"
mkdir -p "./data/training"
mkdir -p "./logs"

echo "🚀 Starting ML service on port ${API_PORT}..."
echo "📁 Project root: ${PROJECT_ROOT}"
echo "🐍 PYTHONPATH: ${PYTHONPATH}"

uvicorn main:app \
    --host 0.0.0.0 \
    --port "${API_PORT}" \
    --log-level "${LOG_LEVEL,,}" \
    --reload