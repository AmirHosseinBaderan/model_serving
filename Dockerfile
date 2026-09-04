FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements-runtime.txt .

RUN pip install \
    --no-cache-dir \
    --trusted-host web.registery.cloudito.home \
    --index-url http://web.registery.cloudito.home/repository/pypi-group/simple/ \
    -r requirements-runtime.txt

RUN useradd \
    --create-home \
    --shell /usr/sbin/nologin \
    appuser

COPY api ./api
COPY model ./model
COPY serving ./serving

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]