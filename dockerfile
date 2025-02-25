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

# Install Python dependencies with specific versions to avoid conflicts
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --upgrade pinecone-client==3.0.0 \
    && pip install tensorflow==2.12.0 \
    && pip install numpy==1.23.5 scikit-learn==1.2.2 \
    && rm -rf ~/.cache/pip/*

# Copy the .env file
COPY .env ./

# Copy the rest of the application
COPY . .

# Make sure the data directory exists and is copied correctly
RUN mkdir -p /app/data
COPY data/FAQ_structured.json /app/data/
COPY data/question_patterns.json /app/data/
COPY data/category_weights.json /app/data/
COPY data/category_patterns.json /app/data/
COPY data/required_keys.json /app/data/
COPY data/synonyms.json /app/data/

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Add this to ensure environment variables are properly set
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app