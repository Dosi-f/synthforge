FROM rocm/rocm:6.2-runtime

LABEL maintainer="Dosi-f@users.noreply.github.com"
LABEL description="SynthForge - Synthetic data generation toolkit"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e ".[gpu]"

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

RUN mkdir -p /app/outputs /app/.cache

ENV SYNTHFORGE_OUTPUT_DIR=/app/outputs
ENV SYNTHFORGE_CACHE_DIR=/app/.cache

ENTRYPOINT ["synthforge"]
CMD ["--help"]
