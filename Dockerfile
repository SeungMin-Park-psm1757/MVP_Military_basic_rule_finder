FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    APP_CHROMA_PATH=/app/data/chroma \
    APP_RUNTIME_DIR=/app/data/runtime \
    APP_ALLOW_DEBUG_TAB=false \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data/chroma /app/data/runtime /app/.cache/huggingface \
    && python scripts/build_sample_corpus.py \
    && python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl

EXPOSE 8501

CMD ["python", "scripts/start_render_webapp.py"]
