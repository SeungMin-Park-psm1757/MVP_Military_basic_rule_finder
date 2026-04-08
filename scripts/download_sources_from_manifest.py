from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

from army_reg_rag.corpus.public_law_pipeline import download_public_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="공개 법령 원문을 manifest 기준으로 data/raw에 수집합니다.")
    parser.add_argument("--manifest", required=True, help="CSV manifest 경로")
    parser.add_argument("--output-dir", default="data/raw/public_law", help="원문 저장 디렉터리")
    parser.add_argument("--report-path", default="data/runtime/public_download_report.json", help="다운로드 점검 리포트 경로")
    args = parser.parse_args()

    report = download_public_sources(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        report_path=args.report_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
