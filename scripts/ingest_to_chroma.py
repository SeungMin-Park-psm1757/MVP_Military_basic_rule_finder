from __future__ import annotations

import argparse

from dotenv import load_dotenv

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

from army_reg_rag.config import load_settings
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.services.ingest_service import ingest_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONL 코퍼스를 Chroma에 적재합니다.")
    parser.add_argument("--input", required=True, help="입력 JSONL 경로")
    parser.add_argument("--config", default="config/settings.yaml", help="설정 파일 경로")
    args = parser.parse_args()

    load_dotenv()
    settings = load_settings(args.config)
    store = ChromaStore(settings)
    count = ingest_jsonl(args.input, store)
    print(f"ingested {count} chunks into collection '{settings.app.collection_name}'")
    print(f"current collection count: {store.count()}")


if __name__ == "__main__":
    main()
