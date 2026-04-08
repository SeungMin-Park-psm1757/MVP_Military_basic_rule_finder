from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

from army_reg_rag.config import load_settings
from army_reg_rag.corpus.public_law_pipeline import build_public_corpus
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.services.ingest_service import ingest_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="실제 공개 법령 원문을 수집·정규화·Chroma 적재까지 한 번에 수행합니다.")
    parser.add_argument("--config", default="config/settings.yaml", help="설정 파일 경로")
    parser.add_argument("--manifest", default="data_manifests/public_law_sources.csv", help="공개 법령 manifest 경로")
    parser.add_argument("--raw-dir", default="", help="원문 저장 디렉터리 override")
    parser.add_argument("--output", default="", help="정규화 JSONL 경로 override")
    parser.add_argument("--report-path", default="", help="빌드 리포트 경로 override")
    parser.add_argument("--skip-ingest", action="store_true", help="정규화까지만 수행하고 Chroma 적재는 건너뜁니다.")
    args = parser.parse_args()

    load_dotenv()
    settings = load_settings(args.config)
    report = build_public_corpus(
        settings,
        manifest_path=args.manifest,
        raw_dir=args.raw_dir or None,
        output_path=args.output or None,
        report_path=args.report_path or None,
    )

    if not args.skip_ingest:
        store = ChromaStore(settings)
        store.reset()
        ingested = ingest_jsonl(report["output_path"], store)
        report["ingested_count"] = ingested
        report["collection_count"] = store.count()

    target_report_path = args.report_path or report["report_path"]
    with open(target_report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
