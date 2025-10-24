FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

ENV PYTHONPATH="/app:${PYTHONPATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 9000

ENV MLFLOW_TRACKING_URI="http://mlflow:5001"

CMD ["uvicorn", "malik.malik.trainer.api:app", "--host", "0.0.0.0", "--port", "9000"]
