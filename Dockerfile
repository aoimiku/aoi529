FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# Copy requirements and install dependencies
COPY requirements.txt .
# Install CPU-only torch (remove torchvision to save space)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model to cache it in the image
COPY download_model.py .
RUN python download_model.py

# Copy application code
COPY . .

# Expose the port
EXPOSE 8000

# Start the application using Gunicorn (timeout increased to 120s)
CMD sh -c "gunicorn src.server:app --bind 0.0.0.0:${PORT:-8000} -k uvicorn.workers.UvicornWorker --timeout 120"
