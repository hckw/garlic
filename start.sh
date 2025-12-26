#!/bin/bash
set -e

# Railway sets PORT env var, default to 8000 if not set
PORT=${PORT:-8000}

echo "Starting uvicorn on port $PORT"

# Start the FastAPI app
exec uvicorn backend.main:app --host 0.0.0.0 --port "$PORT"

