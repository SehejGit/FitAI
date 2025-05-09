FROM python:3.9-slim

WORKDIR /app

# Install system dependencies - updated list for Debian bookworm
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbbmalloc2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Update pip first
RUN pip install --upgrade pip

# Install Python packages with increased timeouts
RUN pip install --no-cache-dir --timeout 300 --default-timeout=300 -r requirements.txt

# Copy application code
COPY . .

# Set environment variables to help MediaPipe
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV MEDIAPIPE_GPU_MODE=0
ENV VECLIB_MAXIMUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV CLOUD_RUN=true

# Run the application with extra timeout
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]