from __future__ import annotations

from pathlib import Path

from army_reg_rag.config import AppConfig, DataConfig, Settings
from army_reg_rag.corpus.public_law_pipeline import (
    _collapse_repeated_phrase,
    _build_records_for_revision_like,
    _build_revision_display_text,
    _build_pdf_download_url,
    _build_revision_reason_download_url,
    _clean_text,
    _parse_header,
    _split_revision_reason_chunks,
    _strip_article_heading,
    _strip_revision_heading,
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


def test_build_pdf_download_url_uses_pretty_law_url_iframe_state():
    landing_html = """
    <html>
      <body>
        <iframe src="/LSW//lsInfoP.do?lsiSeq=283197&amp;chrClsCd=010202&amp;urlMode=lsInfoP&amp;efYd=20260203&amp;ancYnChk=0"></iframe>
      </body>
    </html>
    """
    row = type("Row", (), {"source_type": "law_text"})()

    url = _build_pdf_download_url(row, landing_html)

    assert "lsiSeq=283197" in url
    assert "efYd=20260203" in url


def test_build_revision_reason_download_url_uses_txt_export_endpoint():
    row = type(
        "Row",
        (),
        {
            "url": "https://www.law.go.kr/LSW/lsRvsRsnListP.do?lsId=012591&chrClsCd=010102&lsRvsGubun=all",
        },
    )()

    url = _build_revision_reason_download_url(row)

    assert "lsRvsRsnDocInfoR.do" in url
    assert "lsId=012591" in url
    assert "saveExt=txt" in url


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


def test_collapse_repeated_phrase_dedupes_law_name():
    assert _collapse_repeated_phrase("군인복무규율 군인복무규율") == "군인복무규율"


def test_clean_text_joins_wrapped_words_and_drops_standalone_headers():
    raw = (
        "제45조(신고자 보호) ① 누구든지 신고등을 이유\n"
        "로 신고자에게 불이\n"
        "익조치를 하여서는 아니 된다.\n"
        "군인의 지위 및 복무에 관한 기본법\n"
        "제6장 보칙"
    )

    cleaned = _clean_text(raw, law_name="군인의 지위 및 복무에 관한 기본법")

    assert "이유로" in cleaned
    assert "불이익조치" in cleaned
    assert "군인의 지위 및 복무에 관한 기본법" not in cleaned
    assert "제6장 보칙" not in cleaned


def test_clean_text_repairs_common_pdf_spacing_artifacts():
    cleaned = _clean_text(
        "제44조 신고등\n"
        "”이라 한다. 그가신고자임을 알 수있다.\n"
        "제12조 청원휴 가는 왕복소일수 관련 사항를 정한다."
    )

    assert "신고등”이라 한다" in cleaned
    assert "그가 신고자임" in cleaned
    assert "수 있다" in cleaned
    assert "청원휴가" in cleaned
    assert "왕복소요일수" in cleaned
    assert "사항을" in cleaned


def test_strip_article_heading_and_revision_heading_produce_clean_summaries():
    article_text = "제45조(신고자 보호) 누구든지 신고를 이유로 불이익조치를 하여서는 아니 된다."
    revision_text = (
        "국가법령정보센터|전체 제정·개정이유자바스크립트를 지원하지 않아 일부 기능을 사용할 수 없습니다. "
        "군인의 지위 및 복무에 관한 기본법 [시행 2026. 1. 8.] [법률 제123호, 2025. 1. 7., 일부개정] "
        "【제정·개정이유】 [일부개정] 개정이유 신고자 보호 강화를 위해 불이익조치를 금지한다. "
        "주요내용가. 신고자 보호 조문을 정비함."
    )

    stripped_article = _strip_article_heading(article_text, "제45조", "신고자 보호")
    stripped_revision = _strip_revision_heading(revision_text, "군인의 지위 및 복무에 관한 기본법")
    revision_display = _build_revision_display_text(
        "◇ 개정이유 신고자 보호를 강화하려는 것임.\n◇ 주요내용\n가. 신고자 보호 조문을 정비함.\n나. 불이익조치를 금지함."
    )
    revision_chunks = _split_revision_reason_chunks(
        "◇ 개정이유 신고자 보호를 강화하려는 것임.\n◇ 주요내용\n가. 신고자 보호 조문을 정비함.\n나. 불이익조치를 금지함."
    )

    assert stripped_article.startswith("누구든지 신고를 이유로")
    assert stripped_revision.startswith("◇ 개정이유")
    assert "국가법령정보센터" not in stripped_revision
    assert "\n가. 신고자 보호 조문을 정비함." in stripped_revision
    assert revision_display.startswith("개정이유:")
    assert "주요내용:" in revision_display
    assert revision_chunks == [
        ("개정이유", "신고자 보호를 강화하려는 것임."),
        ("주요내용 가", "가. 신고자 보호 조문을 정비함."),
        ("주요내용 나", "나. 불이익조치를 금지함."),
    ]


def test_build_records_for_revision_like_prefers_plain_text_payload(tmp_path):
    source_dir = tmp_path / "revision"
    source_dir.mkdir()
    html_path = source_dir / "page.html"
    text_path = source_dir / "body.txt"
    html_path.write_text("<html><body>목록 프레임</body></html>", encoding="utf-8")
    text_path.write_text(
        "군인의 지위 및 복무에 관한 기본법 시행령 [시행 2024. 8. 7.] [대통령령 제34971호, 2024. 8. 6., 일부개정]\n"
        "◇ 개정이유\n"
        "자녀돌봄휴가와 배우자 출산휴가 기준을 보완하려는 것임.\n"
        "◇ 주요내용\n"
        "가. 자녀돌봄휴가 사용 사유를 명확히 함.\n",
        encoding="utf-8",
    )
    metadata = {
        "source_id": "childcare-reason",
        "law_name": "군인의 지위 및 복무에 관한 기본법 시행령",
        "scope": "시행령 개정이유",
        "source_type": "revision_reason",
        "source_url": "https://example.com",
    }

    records = _build_records_for_revision_like(metadata, html_path, text_path)
    combined = " ".join(record["text"] for record in records)

    assert records
    assert "자녀돌봄휴가" in combined
