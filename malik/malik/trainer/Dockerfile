FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire trainer code (train.py, api.py, etc.)
COPY . .

# Expose the trainer API port
EXPOSE 9000

# Environment variable (optional, helps if backend connects here)
ENV MLFLOW_TRACKING_URI="http://0.0.0.0:5000"

# Run FastAPI for training (MLflow endpoint)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9000"]
