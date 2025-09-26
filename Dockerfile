FROM python:3.13-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files including model and .env
COPY . .

# Expose API port from .env (default 5000)
EXPOSE 5000

# Use python-dotenv to load .env before starting uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
