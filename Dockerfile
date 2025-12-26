FROM python:3.10-slim

# Install system dependencies required by OpenCV (via inference-sdk)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-backend.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy application code
COPY backend/ ./backend/

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Run the application
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}

