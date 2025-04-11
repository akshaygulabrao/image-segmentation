# ---- Base Stage ----
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- Builder Stage (for dependencies) ----
FROM base as builder

# Install system dependencies (adjust based on your needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# ---- Runtime Stage ----
FROM base as runtime

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure scripts in `.local` are usable
ENV PATH=/root/.local/bin:$PATH

# Copy training code
COPY train.py .
RUN wandb login --relogin e23d90d0b590c3fa55eb5ece24d6d5db0fcb33f4

# Default command (override with `docker run` or `docker compose`)
CMD ["python", "train.py"]