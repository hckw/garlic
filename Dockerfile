FROM python:3.10-slim

# Install system dependencies required by OpenCV (via inference-sdk)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-backend.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy application code
COPY backend/ ./backend/

# Copy startup script
COPY start.sh ./
RUN chmod +x start.sh

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application using the startup script
CMD ["./start.sh"]

