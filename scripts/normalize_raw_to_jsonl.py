from __future__ import annotations

import argparse

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

from army_reg_rag.corpus.public_law_pipeline import normalize_public_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="data/raw의 실제 원문을 data/processed/law_corpus.jsonl로 정규화합니다.")
    parser.add_argument("--raw-dir", default="data/raw/public_law", help="원문 디렉터리")
    parser.add_argument("--output", default="data/processed/law_corpus.jsonl", help="출력 JSONL 경로")
    parser.add_argument("--report-path", default="data/runtime/public_normalize_report.json", help="정규화 리포트 경로")
    args = parser.parse_args()

    report = normalize_public_sources(
        raw_dir=args.raw_dir,
        output_path=args.output,
        report_path=args.report_path,
    )
    print(f"normalized {report['record_count']} records to {args.output}")


if __name__ == "__main__":
    main()
