# Two images from one file:
#   docker build --target web .          # the Flask app (default)
#   docker build --target forecaster .   # scripts/run_forecasts.py

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# --- Shared build setup ---
FROM ${PYTHON_IMAGE} AS builder-base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- Build stages ---
FROM builder-base AS builder-web

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM builder-base AS builder-forecaster

COPY requirements.txt requirements-forecaster.txt ./
RUN pip install -r requirements-forecaster.txt

# --- Shared runtime setup ---
FROM ${PYTHON_IMAGE} AS runtime-base

RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

ENV PATH="/opt/venv/bin:$PATH" \
    FLASK_APP=wsgi.py \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- forecaster ---
FROM runtime-base AS forecaster

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-forecaster /opt/venv /opt/venv
COPY --chown=appuser:appuser . .

USER appuser

CMD ["python", "-m", "scripts.run_forecasts"]

# --- web ---
FROM runtime-base AS web

COPY --from=builder-web /opt/venv /opt/venv
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "wsgi:app"]
