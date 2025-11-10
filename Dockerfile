FROM python:3.11-slim

# Install system dependencies (FFmpeg for audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model for NLP processing
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p output cache

# Clean up unnecessary files
RUN rm -rf datasets/ .git .DS_Store __pycache__ *.pyc

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=info"]
