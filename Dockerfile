FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt
    
# Add user local binaries to PATH
ENV PATH=/root/.local/bin:$PATH

# Copy all files (except those in .dockerignore)
COPY . .

# Use CMD instead of ENTRYPOINT for better flexibility
CMD ["python", "train.py"]