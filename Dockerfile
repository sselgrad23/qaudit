# --- API service image ------------------------------------------------------
# Slim single-stage build for the FastAPI service. Ships the core deps only
# (BM25 retriever + heuristic judge), so the image stays small and torch-free; the
# dense retriever and the local LLM judge are opt-in on a GPU host. This is exactly
# the CI/CPU configuration, which keeps the container reproducible and cheap.
FROM python:3.11-slim

WORKDIR /app

# Install runtime deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself.
COPY pyproject.toml ./
COPY src ./src
COPY README.md ./
RUN pip install --no-cache-dir --no-deps -e .

ENV QAUDIT_DATA_DIR=/app/data \
    QAUDIT_MODEL_DIR=/app/models \
    QAUDIT_REPORT_DIR=/app/reports \
    QAUDIT_RETRIEVER=bm25 \
    QAUDIT_JUDGE_BACKEND=heuristic \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "qaudit.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
