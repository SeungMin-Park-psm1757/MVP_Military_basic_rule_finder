from __future__ import annotations

from pathlib import Path

from army_reg_rag.config import AppConfig, DataConfig, Settings
from army_reg_rag.corpus.public_law_pipeline import (
    _build_pdf_download_url,
    _parse_header,
    _split_article_segments,
    _split_revision_sections,
    get_corpus_input_path,
)
from army_reg_rag.utils.io import write_jsonl


def test_get_corpus_input_path_prefers_full_processed_corpus(tmp_path):
    settings = Settings(
        app=AppConfig(chroma_path=str(tmp_path / "chroma")),
        data=DataConfig(
            demo_input_path=str(tmp_path / "data" / "sample.jsonl"),
            raw_dir=str(tmp_path / "data" / "raw"),
            processed_dir=str(tmp_path / "data" / "processed"),
            runtime_dir=str(tmp_path / "data" / "runtime"),
        ),
    )
    settings.ensure_runtime_dirs()

    demo_path = settings.demo_input_path
    processed_path = settings.processed_dir / "law_corpus.jsonl"
    write_jsonl(demo_path, [{"id": "demo-1"}])
    write_jsonl(processed_path, [{"id": "real-1"}])

    assert get_corpus_input_path(settings) == processed_path


def test_build_pdf_download_url_uses_landing_page_state():
    landing_html = """
    <html>
      <head><title>법령 > 본문 > 군인의 지위 및 복무에 관한 기본법 | 국가법령정보센터</title></head>
      <body>
        <input id="lsiSeq" value="268083" />
        <script>
          var efYd = '20260108';
        </script>
      </body>
    </html>
    """
    row = type("Row", (), {"source_type": "law_text"})()

    url = _build_pdf_download_url(row, landing_html)

    assert "lsiSeq=268083" in url
    assert "efYd=20260108" in url
    assert "joAllCheck=Y" in url


def test_split_article_segments_extracts_articles_and_supplementary():
    text = """
    제1조(목적) 이 법은 목적을 정한다.
    제2조(정의) 이 법에서 사용하는 용어를 정한다.
    부칙 이 법은 공포한 날부터 시행한다.
    """

    segments = _split_article_segments(text)

    assert segments[0][0] == "제1조"
    assert "목적" in segments[0][1]
    assert segments[1][0] == "제2조"
    assert segments[-1][0] == "부칙"


def test_split_revision_sections_breaks_on_each_version_header():
    text = (
        "군인의 지위 및 복무에 관한 기본법 [시행 2026. 1. 8.] [법률 제20641호, 2025. 1. 7., 일부개정] "
        "【제정·개정이유】 첫 번째 개정 이유. "
        "군인의 지위 및 복무에 관한 기본법 [시행 2025. 6. 4.] [법률 제20539호, 2024. 12. 3., 일부개정] "
        "【제정·개정이유】 두 번째 개정 이유."
    )

    sections = _split_revision_sections(text)

    assert len(sections) == 2
    assert "첫 번째 개정 이유" in sections[0]
    assert "두 번째 개정 이유" in sections[1]


def test_parse_header_reads_dates_and_revision_kind():
    header = _parse_header(
        "군인복무규율 [시행 2015. 7. 13.] [대통령령 제26394호, 2015. 7. 13., 일부개정]",
        fallback_law_name="군인복무규율",
    )

    assert header["law_name"] == "군인복무규율"
    assert header["effective_date"] == "2015-07-13"
    assert header["promulgation_date"] == "2015-07-13"
    assert header["revision_kind"] == "일부개정"
