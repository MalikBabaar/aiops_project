FROM python:3.11-slim

# Set working directory to repo root
WORKDIR /app

# Install basic dependencies for pandas/numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy the full repo (one level above Dashboard folder)
COPY ../ .  

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Environment variable so Streamlit can talk to backend container
ENV API_URL="http://backend:5000"

# Run Streamlit app from Dashboard folder
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
