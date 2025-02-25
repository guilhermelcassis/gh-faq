# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip/*

# Copy the .env file
COPY .env ./

# Copy the rest of the application
COPY . .

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Command to run the application
CMD ["python", "src/main.py"]