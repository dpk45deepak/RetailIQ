# FastAPI service. Build: docker build -t retailiq-api .   Run: docker run -p 8000:8000 retailiq-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RETAILIQ_ROOT=/app
WORKDIR /app

COPY requirements-api.txt .
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-api.txt -c requirements-lock.txt

COPY src ./src
COPY api ./api
COPY models ./models
COPY data/processed ./data/processed

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
