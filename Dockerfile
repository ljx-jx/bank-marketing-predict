FROM python:3.11-slim

LABEL app="bank-marketing-predict"

WORKDIR /app

# Install system deps for healthcheck and entrypoint
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (with configurable PyPI mirror)
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --timeout 120 -i "${PIP_INDEX_URL}" pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -i "${PIP_INDEX_URL}" -r requirements.txt

# Copy application code
COPY docker-entrypoint.sh .
COPY src/ ./src/
COPY models/ ./models/

RUN chmod +x docker-entrypoint.sh

EXPOSE 8501 8502

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8502/healthz || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
