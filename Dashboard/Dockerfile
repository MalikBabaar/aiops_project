FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Dashboard folder into container
COPY Dashboard ./Dashboard

# Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "Dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
