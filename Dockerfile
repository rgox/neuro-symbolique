# Multi-stage build for neuro-symbolic AI platform
# Stage 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application
COPY nesy/ ./nesy/
COPY examples/ ./examples/

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV NESY_ENV=production
ENV NESY_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import nesy; print('OK')" || exit 1

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "-m", "uvicorn", "nesy.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
