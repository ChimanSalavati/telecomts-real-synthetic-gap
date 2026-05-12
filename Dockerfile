# TelecomAudit — origin-aware benchmark audit service.
#
# Builds a minimal image that serves the audit API (POST /audit/origin,
# GET /audit/health) and ships the telecomts-audit CLI used as a CI gate.
#
#   docker build -t telecomts-audit .
#   docker run --rm -p 8765:8765 telecomts-audit
#   curl localhost:8765/audit/health
#
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package + API extra first (cached unless packaging metadata moves).
COPY pyproject.toml README.md LICENSE ./
COPY telecomts_gap ./telecomts_gap
RUN pip install --upgrade pip && pip install ".[api]"

# Copy the operator checklist so `telecomts-audit --print-checklist` works in-image.
COPY CHECKLIST.md ./

EXPOSE 8765

# Default: serve the audit API. Override the entrypoint to use the CLI instead,
# e.g.  docker run --rm telecomts-audit telecomts-audit --print-checklist
CMD ["uvicorn", "telecomts_gap.api:app", "--host", "0.0.0.0", "--port", "8765"]
