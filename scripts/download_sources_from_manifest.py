from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlparse

import requests

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

from army_reg_rag.utils.io import ensure_dir


def sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def guess_extension(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if suffix:
        return suffix
    if "pdf" in content_type:
        return ".pdf"
    if "json" in content_type:
        return ".json"
    if "xml" in content_type:
        return ".xml"
    return ".html"


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 매니페스트에 있는 공개 자료를 다운로드합니다.")
    parser.add_argument("--manifest", required=True, help="CSV 매니페스트 경로")
    parser.add_argument("--output-dir", default="data/raw/downloads", help="다운로드 저장 디렉터리")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    manifest_path = Path(args.manifest)
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8-sig")))
    downloaded = 0

    for row in rows:
        url = (row.get("url") or "").strip()
        if not url:
            continue
        source_id = sanitize_filename(row.get("source_id") or row.get("id") or f"row_{downloaded+1}")
        try:
            response = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "army-reg-rag-mvp/0.1 (+https://www.law.go.kr)"},
            )
            response.raise_for_status()
            ext = guess_extension(url, response.headers.get("content-type", ""))
            file_path = output_dir / f"{source_id}{ext}"
            file_path.write_bytes(response.content)
            meta_path = output_dir / f"{source_id}.meta.json"
            meta_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"downloaded: {file_path}")
            downloaded += 1
        except Exception as exc:
            print(f"failed: {url} -> {exc}")

    print(f"done. downloaded={downloaded}")


if __name__ == "__main__":
    main()
