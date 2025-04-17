FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt

ENV PATH=/root/.local/bin:$PATH
COPY train.py .
CMD ["tail", "-f", "/dev/null"]