FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/models/huggingface \
    DOCLING_ARTIFACTS_PATH=/workspace/models/docling

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl git libgl1 libglib2.0-0 tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml ./
COPY rag_app ./rag_app
RUN pip install --upgrade pip && pip install ".[notebook]"

COPY . .
RUN useradd --create-home --uid 10001 raguser \
    && mkdir -p /workspace/data /workspace/artifacts /workspace/models \
    && chown -R raguser:raguser /workspace

USER raguser
EXPOSE 8501 8888
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "rag_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
