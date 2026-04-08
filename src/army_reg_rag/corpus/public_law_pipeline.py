from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from army_reg_rag.config import Settings
from army_reg_rag.utils.io import ensure_dir, read_jsonl, write_jsonl

SUPPORTED_CORPUS_SOURCE_TYPES = {
    "law_text",
    "revision_reason",
    "old_new_comparison",
    "history_note",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/133.0.0.0 Safari/537.36"
)

ERROR_MARKERS = (
    "국가법령정보센터 | 오류페이지",
    "현재 사용자가 많아 요청하신 페이지를 정상적으로 제공할 수 없습니다.",
)

HEADER_RE = re.compile(
    r"(?P<law_name>[가-힣A-Za-z0-9·ㆍ\s]+?)\s*"
    r"(?:\(\s*약칭:\s*(?P<alias>[^)]+)\))?\s*"
    r"\[시행\s*(?P<effective>\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.)\]\s*"
    r"\[(?P<level>법률|대통령령|총리령|부령|국방부령)\s*제(?P<number>[0-9]+)호,\s*"
    r"(?P<promulgation>\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.),\s*(?P<revision_kind>[^\]]+)\]"
)
ARTICLE_RE = re.compile(r"(?=(제\d+조(?:의\d+)?\s*\([^)]+\)))")
SUPPLEMENTARY_RE = re.compile(r"(?=(부칙(?:\s*<[^>]+>)?))")


@dataclass(slots=True)
class ManifestRow:
    source_id: str
    law_name: str
    scope: str
    source_type: str
    url: str
    notes: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "ManifestRow":
        return cls(
            source_id=(row.get("source_id") or "").strip(),
            law_name=(row.get("law_name") or "").strip(),
            scope=(row.get("scope") or "").strip(),
            source_type=(row.get("source_type") or "").strip(),
            url=(row.get("url") or "").strip(),
            notes=(row.get("notes") or "").strip(),
        )

    @property
    def is_corpus_source(self) -> bool:
        return self.source_type in SUPPORTED_CORPUS_SOURCE_TYPES


def load_manifest_rows(manifest_path: str | Path) -> list[ManifestRow]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [ManifestRow.from_dict(row) for row in csv.DictReader(handle)]


def get_corpus_input_path(settings: Settings) -> Path:
    processed = settings.processed_dir / "law_corpus.jsonl"
    if processed.exists() and read_jsonl(processed):
        return processed
    return settings.demo_input_path


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    )
    return session


def _is_error_payload(response: requests.Response) -> bool:
    if response.status_code >= 500:
        return True
    if "pdf" in (response.headers.get("content-type") or "").lower():
        return False
    body = response.text or ""
    return any(marker in body for marker in ERROR_MARKERS)


def _request_with_retry(
    session: requests.Session,
    url: str,
    *,
    referer: str | None = None,
    attempts: int = 4,
    timeout: int = 60,
) -> requests.Response:
    headers = {"Referer": referer} if referer else None
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            if _is_error_payload(response):
                raise RuntimeError(f"error payload received from {url}")
            return response
        except Exception as exc:  # pragma: no cover - covered in integration runs
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(1.2 * attempt)
    raise RuntimeError(str(last_error) if last_error else f"request failed: {url}")


def _canonical_text(value: str) -> str:
    return " ".join(value.split())


def _normalize_display_date(value: str) -> str:
    cleaned = value.replace(" ", "")
    if "." in cleaned:
        parts = [part for part in cleaned.split(".") if part]
        if len(parts) >= 3:
            return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    if len(cleaned) == 8 and cleaned.isdigit():
        return f"{cleaned[0:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
    return value.strip()


def _extract_effective_date(html: str) -> str:
    match = re.search(r"var\s+efYd\s*=\s*'(\d{8})';", html)
    if match:
        return match.group(1)

    title_match = HEADER_RE.search(_canonical_text(BeautifulSoup(html, "lxml").get_text(" ", strip=True)))
    if title_match:
        return _normalize_display_date(title_match.group("effective")).replace("-", "")
    raise ValueError("unable to derive effective date from landing page")


def _extract_lsi_seq(html: str) -> str:
    match = re.search(r'<input[^>]+id="lsiSeq"[^>]+value="(\d+)"', html)
    if match:
        return match.group(1)
    match = re.search(r"lsPopViewAll2\('(\d+)'", html)
    if match:
        return match.group(1)
    raise ValueError("unable to derive lsiSeq from landing page")


def _build_pdf_download_url(row: ManifestRow, landing_html: str) -> str:
    lsi_seq = _extract_lsi_seq(landing_html)
    effective_date = _extract_effective_date(landing_html)
    params = {
        "ancYnChk": "" if row.source_type == "history_note" else "0",
        "bylChaChk": "N",
        "efGubun": "Y",
        "efYd": effective_date,
        "joAllCheck": "Y",
        "joEfOutPutYn": "on",
        "lsiSeq": lsi_seq,
        "mokChaChk": "N",
    }
    return f"https://www.law.go.kr/LSW/lsPdfPrint.do?{urlencode(params)}"


def _download_single_source(
    row: ManifestRow,
    *,
    output_dir: Path,
    session: requests.Session,
) -> dict[str, Any]:
    source_dir = ensure_dir(output_dir / row.source_id)
    landing_response = _request_with_retry(session, row.url)
    landing_html = landing_response.text

    page_path = source_dir / "page.html"
    page_path.write_text(landing_html, encoding="utf-8")

    page_text = _canonical_text(BeautifulSoup(landing_html, "lxml").get_text(" ", strip=True))
    title = BeautifulSoup(landing_html, "lxml").title
    title_text = title.get_text(" ", strip=True) if title else ""
    if row.law_name not in f"{title_text} {page_text}":
        raise RuntimeError(f"expected law name not found in landing page: {row.law_name}")

    metadata = {
        "source_id": row.source_id,
        "law_name": row.law_name,
        "scope": row.scope,
        "source_type": row.source_type,
        "source_url": row.url,
        "notes": row.notes,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": ["page.html"],
    }

    if row.source_type in {"law_text", "history_note"}:
        pdf_url = _build_pdf_download_url(row, landing_html)
        pdf_response = _request_with_retry(session, pdf_url, referer=row.url)
        if not pdf_response.content.startswith(b"%PDF"):
            raise RuntimeError(f"non-PDF payload received from {pdf_url}")
        pdf_path = source_dir / "body.pdf"
        pdf_path.write_bytes(pdf_response.content)
        metadata["pdf_url"] = pdf_url
        metadata["artifacts"].append("body.pdf")

    metadata_path = source_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "source_id": row.source_id,
        "law_name": row.law_name,
        "scope": row.scope,
        "source_type": row.source_type,
        "status": "downloaded",
        "artifacts": metadata["artifacts"],
        "source_dir": str(source_dir),
    }


def download_public_sources(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    output_root = ensure_dir(output_dir)
    rows = load_manifest_rows(manifest_path)
    session = _make_session()
    entries: list[dict[str, Any]] = []

    for row in rows:
        if not row.is_corpus_source:
            entries.append(
                {
                    "source_id": row.source_id,
                    "law_name": row.law_name,
                    "scope": row.scope,
                    "source_type": row.source_type,
                    "status": "skipped_non_corpus",
                }
            )
            continue

        try:
            entries.append(_download_single_source(row, output_dir=output_root, session=session))
        except Exception as exc:
            entries.append(
                {
                    "source_id": row.source_id,
                    "law_name": row.law_name,
                    "scope": row.scope,
                    "source_type": row.source_type,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    report = {
        "manifest_path": str(Path(manifest_path)),
        "output_dir": str(output_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "downloaded_count": sum(1 for entry in entries if entry["status"] == "downloaded"),
        "failed_count": sum(1 for entry in entries if entry["status"] == "failed"),
        "sources": entries,
    }
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _law_level_for(law_name: str) -> str:
    if law_name.endswith("시행령"):
        return "시행령"
    if law_name.endswith("시행규칙"):
        return "시행규칙"
    if law_name == "군인복무규율":
        return "대통령령"
    return "법률"


def _clean_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"법제처\s+\d+\s+국가법령정보센\s*터", " ", cleaned)
    cleaned = re.sub(r"\n\s+\n", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return _clean_text("\n".join(page.extract_text() or "" for page in reader.pages))


def _html_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    return _clean_text(soup.get_text("\n", strip=True))


def _chunk_large_text(text: str, *, max_chars: int = 2400, overlap: int = 220) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + int(max_chars * 0.6), end),
                text.rfind(". ", start + int(max_chars * 0.6), end),
                text.rfind("다. ", start + int(max_chars * 0.6), end),
            )
            if boundary > start:
                end = boundary + (0 if text[boundary] == "\n" else 1)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _parse_header(block: str, *, fallback_law_name: str) -> dict[str, str]:
    match = HEADER_RE.search(_canonical_text(block))
    if not match:
        return {
            "law_name": fallback_law_name,
            "effective_date": "",
            "promulgation_date": "",
            "revision_kind": "",
            "version_label": "",
        }
    return {
        "law_name": match.group("law_name").strip(),
        "effective_date": _normalize_display_date(match.group("effective")),
        "promulgation_date": _normalize_display_date(match.group("promulgation")),
        "revision_kind": match.group("revision_kind").strip(),
        "version_label": f"[시행 {match.group('effective').strip()}] {match.group('revision_kind').strip()}",
    }


def _split_revision_sections(text: str) -> list[str]:
    normalized = _canonical_text(text)
    matches = list(HEADER_RE.finditer(normalized))
    if not matches:
        return [normalized]

    sections: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        section = normalized[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _split_article_segments(text: str) -> list[tuple[str, str]]:
    working = text
    supplementary_match = SUPPLEMENTARY_RE.search(working)
    supplementary = ""
    if supplementary_match:
        supplementary = working[supplementary_match.start():].strip()
        working = working[:supplementary_match.start()].strip()

    matches = list(ARTICLE_RE.finditer(working))
    segments: list[tuple[str, str]] = []
    if matches:
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(working)
            segment = working[start:end].strip()
            header_match = re.match(r"(제\d+조(?:의\d+)?)(?:\s*)(\([^)]+\))", segment)
            if not header_match:
                continue
            article_no = header_match.group(1)
            article_title = header_match.group(2).strip("()")
            segments.append((article_no, article_title + "\n" + segment))
    elif working:
        segments.append(("", working))

    if supplementary:
        segments.append(("부칙", supplementary))
    return segments


def _record_id(*parts: str) -> str:
    joined = "::".join(part for part in parts if part)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"public-{digest}"


def _build_records_for_law_text(metadata: dict[str, Any], pdf_path: Path) -> list[dict[str, Any]]:
    text = _pdf_text(pdf_path)
    header = _parse_header(text, fallback_law_name=str(metadata.get("law_name", "")))
    records: list[dict[str, Any]] = []

    for index, (article_no, payload) in enumerate(_split_article_segments(text), start=1):
        first_line, body = (payload.split("\n", 1) + [""])[:2]
        article_title = "부칙" if article_no == "부칙" else first_line
        records.append(
            {
                "id": _record_id(str(metadata.get("source_id")), article_no or f"segment-{index}"),
                "law_name": header["law_name"] or str(metadata.get("law_name", "")),
                "law_level": _law_level_for(str(metadata.get("law_name", ""))),
                "source_type": str(metadata.get("source_type", "")),
                "version_label": header["version_label"] or str(metadata.get("scope", "")),
                "promulgation_date": header["promulgation_date"],
                "effective_date": header["effective_date"],
                "article_no": article_no,
                "article_title": article_title,
                "revision_kind": header["revision_kind"],
                "text": _clean_text(payload if article_no == "부칙" else body),
                "source_url": str(metadata.get("source_url", "")),
                "scope": str(metadata.get("scope", "")),
                "source_id": str(metadata.get("source_id", "")),
            }
        )
    return records


def _build_records_for_revision_like(metadata: dict[str, Any], html_path: Path) -> list[dict[str, Any]]:
    text = _html_text(html_path)
    sections = _split_revision_sections(text)
    article_title = "제정·개정이유" if metadata.get("source_type") == "revision_reason" else "제정·개정문"
    records: list[dict[str, Any]] = []

    for section_index, section in enumerate(sections, start=1):
        header = _parse_header(section, fallback_law_name=str(metadata.get("law_name", "")))
        for chunk_index, chunk in enumerate(_chunk_large_text(section), start=1):
            chunk_title = article_title if chunk_index == 1 else f"{article_title} ({chunk_index})"
            records.append(
                {
                    "id": _record_id(str(metadata.get("source_id")), str(section_index), str(chunk_index)),
                    "law_name": header["law_name"] or str(metadata.get("law_name", "")),
                    "law_level": _law_level_for(str(metadata.get("law_name", ""))),
                    "source_type": str(metadata.get("source_type", "")),
                    "version_label": header["version_label"] or str(metadata.get("scope", "")),
                    "promulgation_date": header["promulgation_date"],
                    "effective_date": header["effective_date"],
                    "article_no": "",
                    "article_title": chunk_title,
                    "revision_kind": header["revision_kind"],
                    "text": chunk,
                    "source_url": str(metadata.get("source_url", "")),
                    "scope": str(metadata.get("scope", "")),
                    "source_id": str(metadata.get("source_id", "")),
                    "section_index": section_index,
                }
            )
    return records


def normalize_public_sources(
    raw_dir: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    raw_root = Path(raw_dir)
    records: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []

    for metadata_path in sorted(raw_root.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_type = str(metadata.get("source_type", ""))
        source_dir = metadata_path.parent
        page_path = source_dir / "page.html"
        pdf_path = source_dir / "body.pdf"

        if source_type in {"law_text", "history_note"}:
            built_records = _build_records_for_law_text(metadata, pdf_path)
        else:
            built_records = _build_records_for_revision_like(metadata, page_path)

        records.extend(built_records)
        source_reports.append(
            {
                "source_id": str(metadata.get("source_id", "")),
                "law_name": str(metadata.get("law_name", "")),
                "scope": str(metadata.get("scope", "")),
                "source_type": source_type,
                "record_count": len(built_records),
            }
        )

    write_jsonl(output_path, records)
    report = {
        "raw_dir": str(raw_root),
        "output_path": str(output_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "sources": source_reports,
    }
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_public_corpus(
    settings: Settings,
    *,
    manifest_path: str | Path = "data_manifests/public_law_sources.csv",
    raw_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    raw_target = Path(raw_dir) if raw_dir else settings.raw_dir / "public_law"
    output_target = Path(output_path) if output_path else settings.processed_dir / "law_corpus.jsonl"
    report_target = Path(report_path) if report_path else settings.runtime_dir / "public_corpus_report.json"

    download_report = download_public_sources(
        manifest_path=manifest_path,
        output_dir=raw_target,
        report_path=report_target,
    )
    if download_report["failed_count"]:
        raise RuntimeError("one or more public sources failed to download")

    normalize_report = normalize_public_sources(
        raw_dir=raw_target,
        output_path=output_target,
        report_path=report_target,
    )

    combined = {
        "manifest_path": str(manifest_path),
        "raw_dir": str(raw_target),
        "output_path": str(output_target),
        "report_path": str(report_target),
        "downloaded_count": download_report["downloaded_count"],
        "failed_count": download_report["failed_count"],
        "record_count": normalize_report["record_count"],
        "sources": [
            {
                **source,
                **next(
                    (
                        item
                        for item in normalize_report["sources"]
                        if item["source_id"] == source.get("source_id")
                    ),
                    {},
                ),
            }
            for source in download_report["sources"]
        ],
    }
    report_target.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    return combined
