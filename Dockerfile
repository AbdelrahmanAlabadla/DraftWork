FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app \
    HOST=0.0.0.0 \
    PORT=8000 \
    FORWARDED_ALLOW_IPS=127.0.0.1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Chromium requires libatomic.so.1 during Playwright's browser installation.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Install the version-matched headless Chromium runtime and its Linux font /
# shared-library dependencies for deterministic server-side PDF exports.
RUN python -m playwright install --with-deps chromium

# Keep these ARGs after the dependency layer so changing the source revision
# does not invalidate cached pip installs.
ARG GIT_COMMIT=unknown
ARG SOURCE_FINGERPRINT=unknown
ENV APP_GIT_COMMIT=$GIT_COMMIT \
    APP_SOURCE_FINGERPRINT=$SOURCE_FINGERPRINT

COPY . .

RUN addgroup --system --gid 10001 app \
    && adduser --system --uid 10001 --ingroup app --home /home/app app \
    && mkdir -p /app/data /app/tmp /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
