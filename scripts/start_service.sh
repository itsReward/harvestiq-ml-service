#!/bin/bash
# Start the ML service with proper configuration

set -e

echo "🌾 Starting HarvestIQ ML Model Service..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Set default values
export MODEL_STORAGE_PATH=${MODEL_STORAGE_PATH:-"./storage/models"}
export API_PORT=${API_PORT:-8001}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# Create required directories
mkdir -p "${MODEL_STORAGE_PATH}/versions"
mkdir -p "${MODEL_STORAGE_PATH}/active"
mkdir -p "${MODEL_STORAGE_PATH}/archive"
mkdir -p "./data/training"
mkdir -p "./logs"

echo "🚀 Starting ML service on port ${API_PORT}..."
uvicorn main:app \
    --host 0.0.0.0 \
    --port "${API_PORT}" \
    --log-level "${LOG_LEVEL,,}" \
    --reload
