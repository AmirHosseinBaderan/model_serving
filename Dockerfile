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

COPY api ./api
COPY model ./model
COPY serving ./serving

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]