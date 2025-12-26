#!/usr/bin/env python3
"""Startup script for Railway deployment."""
import os
import sys

# Get PORT from environment, default to 8000
port = int(os.getenv("PORT", "8000"))

# Start uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

