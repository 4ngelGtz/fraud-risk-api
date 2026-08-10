# Fraud Risk API — two-stage image for the frozen xgb-transformed-v1 service.
# Build assumes artifacts/xgb-transformed-v1/ exists locally (not trained here).

# ---------------------------------------------------------------------------
# builder: install the project and runtime Python dependencies into a venv
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src

# Runtime deps only (omit optional extras such as pytest / ruff / jupyter).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# runtime: serve predictions from the frozen artifact (non-root)
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# libgomp1: OpenMP runtime required by the CPU XGBoost wheel on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUD_MODEL_DIR=/app/artifacts/xgb-transformed-v1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --uid 1000 --user-group appuser

COPY --from=builder /opt/venv /opt/venv
COPY artifacts/xgb-transformed-v1 ./artifacts/xgb-transformed-v1

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "fraud_risk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
