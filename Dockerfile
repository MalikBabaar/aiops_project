FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for pandas, numpy, scikit-learn, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files including model and .env
COPY . .

# Expose API port
EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
