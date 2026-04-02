from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

from army_reg_rag.utils.io import write_jsonl


def _classify_source_type(filename: str) -> str:
    name = filename.lower()
    if "old_and_new" in name or "compare" in name or "신구" in name:
        return "old_new_comparison"
    if "reason" in name or "이유" in name:
        return "revision_reason"
    if "history" in name or "연혁" in name:
        return "history_note"
    return "law_text"


def _read_text_best_effort(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def _load_sidecar_metadata(path: Path) -> dict:
    sidecar = path.parent / f"{path.stem}.meta.json"
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _record_from_text(path: Path, text: str, idx: int, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    title = path.stem
    source_type = metadata.get("source_type") or _classify_source_type(title)
    law_name = metadata.get("law_name") or title.split("__")[0].replace("_", " ").replace("-", " ")
    scope = metadata.get("scope") or title
    return {
        "id": f"normalized-{idx:04d}",
        "law_name": law_name,
        "law_level": "",
        "source_type": source_type,
        "version_label": metadata.get("scope", "raw-normalized"),
        "promulgation_date": "",
        "effective_date": "",
        "article_no": "",
        "article_title": scope,
        "revision_kind": "",
        "text": " ".join(text.split())[:4000],
        "source_url": metadata.get("url", ""),
        "scope": scope,
        "notes": metadata.get("notes", ""),
    }


def main() -> None:
    raw_dir = Path("data/raw")
    output_path = Path("data/processed/law_corpus.jsonl")
    records = []
    idx = 1

    for path in sorted(raw_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.name.endswith(".meta.json"):
            continue
        metadata = _load_sidecar_metadata(path)

        if path.suffix.lower() in {".html", ".htm", ".do"}:
            html = _read_text_best_effort(path)
            text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
        elif path.suffix.lower() in {".json"}:
            text = json.dumps(json.loads(_read_text_best_effort(path)), ensure_ascii=False)
        elif path.suffix.lower() in {".xml"}:
            text = _read_text_best_effort(path)
        else:
            continue

        if not text.strip():
            continue
        records.append(_record_from_text(path, text, idx, metadata=metadata))
        idx += 1

    write_jsonl(output_path, records)
    print(f"normalized {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
