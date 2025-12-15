# YouTube Transcriber with GPU Support
# Based on NVIDIA CUDA image with cuDNN

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="Kyle Fox"
LABEL description="Local YouTube transcription with faster-whisper and web UI"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create non-root user for security
RUN useradd -m -u 1000 transcriber
RUN mkdir -p /app/output /app/models && chown -R transcriber:transcriber /app
RUN mkdir -p /app/output /app/models /home/transcriber/.cache && chown -R transcriber:transcriber /app /home/transcriber/.cache

# Switch to non-root user
USER transcriber

# Set up Python environment
ENV PATH="/home/transcriber/.local/bin:${PATH}"

# Install Python dependencies
COPY --chown=transcriber:transcriber requirements.txt .
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=transcriber:transcriber . .

# Pre-download the Whisper model (optional - comment out to download on first run)
# This makes the container larger but startup faster
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu')"

# Expose web UI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables for the app
ENV MODEL_SIZE=large-v3
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16
ENV OUTPUT_DIR=/app/output

# Run the web server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
