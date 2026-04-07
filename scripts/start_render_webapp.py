from __future__ import annotations

import os
import sys
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()


def configure_env() -> None:
    os.environ.setdefault("APP_CHROMA_PATH", "/app/data/chroma")
    os.environ.setdefault("APP_RUNTIME_DIR", "/app/data/runtime")
    os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")


def has_bootstrapped_chroma() -> bool:
    chroma_path = Path(os.environ["APP_CHROMA_PATH"])
    if not chroma_path.exists():
        return False
    return any(chroma_path.iterdir())


def ensure_demo_corpus() -> None:
    sample_path = Path("data/sample/processed/sample_documents.jsonl")
    if has_bootstrapped_chroma():
        print("Chroma bootstrap files already exist.", flush=True)
        return

    from dotenv import load_dotenv
    from build_sample_corpus import main as build_sample_corpus
    from army_reg_rag.config import load_settings
    from army_reg_rag.retrieval.chroma_store import ChromaStore
    from army_reg_rag.services.ingest_service import ingest_jsonl

    if not sample_path.exists():
        build_sample_corpus()

    load_dotenv()
    settings = load_settings()
    store = ChromaStore(settings)
    current_count = store.count()
    if current_count > 0:
        print(f"Chroma collection already contains {current_count} chunks.", flush=True)
        return

    if not settings.demo_input_path.exists():
        print(f"Demo input missing at {settings.demo_input_path}; skipping bootstrap.", file=sys.stderr, flush=True)
        return

    count = ingest_jsonl(str(settings.demo_input_path), store)
    print(f"Bootstrapped {count} demo chunks into '{settings.app.collection_name}'.", flush=True)


def main() -> None:
    configure_env()
    try:
        ensure_demo_corpus()
    except Exception as exc:
        print(f"Render bootstrap warning: {exc}", file=sys.stderr, flush=True)

    port = os.getenv("PORT", "8501")
    os.execvpe(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            port,
            "--server.headless",
            "true",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
            "--server.enableWebsocketCompression",
            "false",
        ],
        os.environ,
    )


if __name__ == "__main__":
    main()
