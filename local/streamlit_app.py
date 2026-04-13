from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SRC_PATH_STR = str(SRC_PATH)
if SRC_PATH_STR not in sys.path:
    sys.path.insert(0, SRC_PATH_STR)

import streamlit as st
from dotenv import load_dotenv

from army_reg_rag.config import load_settings
from army_reg_rag.corpus import get_corpus_input_path
from army_reg_rag.domain.models import SearchHit
from army_reg_rag.export_docx import ConversationTurn, build_conversation_docx, build_conversation_turns
from army_reg_rag.llm.lm_studio_client import DEFAULT_LM_STUDIO_BASE_URL, LMStudioAnswerClient
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.services.answer_service import AnswerService
from army_reg_rag.services.ingest_service import ingest_jsonl
from army_reg_rag.utils.io import read_jsonl

load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / "local" / ".env", override=True)

SOURCE_TYPE_ORDER = [
    "law_text",
    "revision_reason",
    "old_new_comparison",
    "history_note",
]

SOURCE_TYPE_LABELS = {
    "law_text": "현행 조문",
    "revision_reason": "개정 이유",
    "old_new_comparison": "신구 비교",
    "history_note": "연혁 자료",
}

BACKEND_LABELS = {
    "lm_studio": "LM Studio 로컬 생성",
    "retrieval_fallback": "근거 요약 모드",
    "retrieval_only": "근거 요약 모드",
    "quota_blocked": "질문 차단",
}

CHAT_EXAMPLES = [
    "군인의 지위 및 복무에 관한 기본법에서 휴가 관련 현행 규정을 찾아줘.",
    "육아시간 관련 규정이 왜 바뀌었는지 개정 이유 중심으로 설명해줘.",
    "휴직이나 휴가 관련 사안을 실무적으로 볼 때 어떤 순서로 확인해야 하는지 알려줘.",
]


@st.cache_resource
def get_settings():
    return load_settings()


@st.cache_resource
def get_store():
    return ChromaStore(get_settings())


@st.cache_resource
def get_answer_service(base_url: str):
    settings = get_settings()
    return AnswerService(
        settings,
        store=get_store(),
        client=LMStudioAnswerClient(
            settings,
            base_url=base_url,
            enforce_limits=False,
        ),
    )


@st.cache_data(ttl=3, show_spinner=False)
def probe_lm_studio(base_url: str) -> dict[str, Any]:
    client = LMStudioAnswerClient(
        get_settings(),
        base_url=base_url,
        enforce_limits=False,
    )
    return client.describe_connection()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --canvas: #f6f2e8;
            --surface: rgba(255, 250, 242, 0.82);
            --ink: #1a2530;
            --muted: #5f6c79;
            --line: rgba(26, 37, 48, 0.10);
            --accent: #1f5f4a;
            --warn: #8d5422;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(31, 95, 74, 0.10), transparent 26%),
                radial-gradient(circle at top right, rgba(141, 84, 34, 0.08), transparent 28%),
                linear-gradient(180deg, #fbf8f1 0%, #f2ede2 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5efe4 0%, #ece4d5 100%);
            border-right: 1px solid var(--line);
        }

        .local-hero {
            border: 1px solid var(--line);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255, 250, 242, 0.96), rgba(247, 240, 226, 0.92));
            padding: 1.5rem 1.55rem 1.35rem 1.55rem;
            box-shadow: 0 22px 60px rgba(26, 37, 48, 0.06);
            margin-bottom: 1rem;
        }

        .local-kicker {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .local-title {
            color: var(--ink);
            font-size: 2rem;
            line-height: 1.18;
            font-weight: 800;
            margin-top: 0.35rem;
        }

        .local-copy {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.7;
            margin-top: 0.55rem;
            max-width: 44rem;
        }

        .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.95rem;
        }

        .status-pill {
            border: 1px solid var(--line);
            border-radius: 999px;
            background: var(--surface);
            padding: 0.45rem 0.8rem;
            color: var(--ink);
            font-size: 0.82rem;
        }

        .disclaimer-card {
            border-left: 4px solid var(--warn);
            border-radius: 16px;
            background: rgba(255, 250, 242, 0.88);
            padding: 0.95rem 1rem;
            margin: 0.9rem 0 1.1rem 0;
            color: var(--ink);
        }

        .empty-state {
            border: 1px dashed rgba(26, 37, 48, 0.18);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.48);
            padding: 1.15rem 1.2rem;
            color: var(--muted);
        }

        .section-title {
            color: var(--ink);
            font-weight: 800;
            font-size: 1rem;
            margin: 0.15rem 0 0.7rem 0;
        }

        .hint-text {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.6;
        }

        div[data-testid="stChatMessageContent"] .stMarkdown h2 {
            color: var(--ink);
            font-size: 1.32rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 1.2rem 0 0.55rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid rgba(26, 37, 48, 0.10);
        }

        div[data-testid="stChatMessageContent"] .stMarkdown h3 {
            color: var(--accent);
            font-size: 1.02rem;
            font-weight: 800;
            margin: 0.9rem 0 0.35rem 0;
        }

        div[data-testid="stChatMessageContent"] .stMarkdown h4 {
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 700;
            margin: 0.8rem 0 0.25rem 0;
            padding-left: 0.65rem;
            border-left: 3px solid rgba(31, 95, 74, 0.25);
        }

        div[data-testid="stChatMessageContent"] .stMarkdown p,
        div[data-testid="stChatMessageContent"] .stMarkdown li {
            line-height: 1.72;
        }

        div[data-testid="stChatMessageContent"] .stMarkdown ul {
            margin-top: 0.2rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ordered_source_types(source_types: list[str]) -> list[str]:
    known = [value for value in SOURCE_TYPE_ORDER if value in source_types]
    extra = sorted(value for value in source_types if value not in SOURCE_TYPE_ORDER)
    return known + extra


def format_source_type(source_type: str) -> str:
    return SOURCE_TYPE_LABELS.get(source_type, source_type or "미분류")


def load_corpus_rows() -> list[dict]:
    settings = get_settings()
    rows_by_id: dict[str, dict] = {}
    corpus_path = get_corpus_input_path(settings)
    for row in read_jsonl(corpus_path):
        row_id = str(row.get("id") or f"{row.get('law_name', '')}:{row.get('article_title', '')}:{len(rows_by_id)}")
        rows_by_id[row_id] = row
    return list(rows_by_id.values())


def load_law_options(rows: list[dict]) -> list[str]:
    options = {"전체"}
    for row in rows:
        law_name = str(row.get("law_name", "")).strip()
        if law_name:
            options.add(law_name)
    return sorted(options, key=lambda value: (value != "전체", value))


def load_source_type_options(rows: list[dict]) -> list[str]:
    available = {str(row["source_type"]) for row in rows if row.get("source_type")}
    return ordered_source_types(list(available))


def ensure_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("clear_question_input", False)
    st.session_state.setdefault("local_base_url", DEFAULT_LM_STUDIO_BASE_URL)
    if st.session_state.clear_question_input:
        st.session_state.question_input = ""
        st.session_state.clear_question_input = False


def validate_question(question: str) -> str | None:
    if not question.strip():
        return "질문을 입력해 주세요."
    return None


def format_filter_summary(law_name: str, source_types: list[str], all_source_types: list[str]) -> str:
    parts = []
    if law_name and law_name != "전체":
        parts.append(f"법령: {law_name}")
    if source_types and len(source_types) != len(all_source_types):
        parts.append("자료 유형: " + ", ".join(format_source_type(source_type) for source_type in source_types))
    return " / ".join(parts)


def render_intro(connection_state: dict[str, Any], document_count: int) -> None:
    model_label = connection_state.get("resolved_model") or "LM Studio 자동 추적"
    status_label = "연결됨" if connection_state.get("available") else "근거 요약 모드"
    st.markdown(
        f"""
        <div class="local-hero">
            <div class="local-kicker">Local Legal RAG</div>
            <div class="local-title">로컬 PC용 군 복무 법규 RAG 웹앱</div>
            <div class="local-copy">
                LM Studio 기반 로컬 모델을 사용해 현행 조문과 개정 자료를 함께 보여주는 버전입니다.
                로컬 실행에서는 질문 횟수와 한도 제한을 두지 않습니다.
            </div>
            <div class="status-strip">
                <div class="status-pill">모델: {model_label}</div>
                <div class="status-pill">상태: {status_label}</div>
                <div class="status-pill">문서 수: {document_count}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-card">
            실무 참고용 안내이며 법률자문이 아닙니다. 실제 인사, 징계, 복무 처리에는 최신 원문과 소속 부대 지침을 함께 확인해 주세요.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_buttons() -> str | None:
    st.markdown("<div class='section-title'>예시 질문</div>", unsafe_allow_html=True)
    columns = st.columns(len(CHAT_EXAMPLES))
    for idx, question in enumerate(CHAT_EXAMPLES):
        label = ["현행 규정", "개정 이유", "실무 참고"][idx]
        if columns[idx].button(label, key=f"local_example_{idx}", use_container_width=True):
            return question
    return None


def render_question_box() -> tuple[bool, str]:
    question = st.text_area(
        "질문 입력",
        key="question_input",
        height=120,
        label_visibility="collapsed",
        placeholder=(
            "예: 군인의 지위 및 복무에 관한 기본법에서 휴가 규정이 시행령과 어떻게 이어지는지 보여줘.\n"
            "답변은 근거 중심으로 정리해 줘."
        ),
    )
    footer_left, footer_right = st.columns([5, 1.2], vertical_alignment="bottom")
    with footer_left:
        st.markdown(
            "<div class='hint-text'>최신 질문과 답변이 입력창 바로 아래에 보이도록 정렬됩니다.</div>",
            unsafe_allow_html=True,
        )
    with footer_right:
        submitted = st.button("질문하기", type="primary", use_container_width=True)
    return submitted, question


def render_connection_notice(connection_state: dict[str, Any]) -> None:
    if connection_state.get("available"):
        st.info(
            f"LM Studio 자동 추적 연결됨: {connection_state.get('base_url')} / {connection_state.get('resolved_model') or '단일 로드 모델'}"
        )
    else:
        st.warning(connection_state.get("message") or "LM Studio가 준비되지 않아 근거 요약 모드로 전환됩니다.")


def render_evidence_card(hit: SearchHit, preview_chars: int) -> None:
    chunk = hit.chunk
    article_ref = " ".join(part for part in [chunk.article_no, chunk.article_title] if part).strip()
    meta = [format_source_type(chunk.source_type)]
    if article_ref:
        meta.append(article_ref)
    if chunk.effective_date:
        meta.append(f"시행일 {chunk.effective_date}")

    text = str(chunk.extra.get("display_text") or chunk.extra.get("summary_text") or chunk.text).strip()
    if len(text) > preview_chars:
        text = text[:preview_chars].rstrip() + "..."

    with st.container(border=True):
        st.markdown(f"**{chunk.law_name}**")
        st.caption(" / ".join(meta))
        st.write(text)
        if chunk.source_url:
            st.markdown(f"[원문 링크]({chunk.source_url})")


def render_grouped_evidence(hits: list[SearchHit], preview_chars: int) -> None:
    if not hits:
        st.info("표시할 근거가 없습니다.")
        return

    grouped: defaultdict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.chunk.source_type or "unknown"].append(hit)

    for source_type in ordered_source_types(list(grouped.keys())):
        st.markdown(f"<div class='section-title'>{format_source_type(source_type)}</div>", unsafe_allow_html=True)
        for hit in grouped[source_type]:
            render_evidence_card(hit, preview_chars)


def flatten_turns(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for turn in turns:
        history.append(turn.question)
        if turn.answer is not None:
            history.append(turn.answer)
    return history


def export_filename(turn_index: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"local_military_rule_chat_turn_{turn_index}_{timestamp}.docx"


def render_chat_history(preview_chars: int, all_source_types: list[str]) -> None:
    history = st.session_state.chat_history
    turns = build_conversation_turns(history)
    if not turns:
        st.markdown(
            """
            <div class="empty-state">
                아직 답변 이력이 없습니다. 질문을 입력하면 최신 대화가 입력창 바로 아래에 나타납니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for turn_index in range(len(turns), 0, -1):
        turn = turns[turn_index - 1]
        with st.chat_message("user"):
            st.markdown(turn.question.get("content", ""))
            answer = turn.answer or {}
            summary = format_filter_summary(
                answer.get("law_name", turn.question.get("law_name", "전체")),
                answer.get("source_types", turn.question.get("source_types", all_source_types)),
                all_source_types,
            )
            if summary:
                st.caption(summary)

        if turn.answer is None:
            continue

        with st.chat_message("assistant"):
            top_left, top_right = st.columns([5.5, 1.4], vertical_alignment="center")
            with top_left:
                st.caption("실무 참고용, 법률자문 아님")
            with top_right:
                docx_bytes = build_conversation_docx(flatten_turns(turns[:turn_index]))
                st.download_button(
                    "DOCX 내보내기",
                    data=docx_bytes,
                    file_name=export_filename(turn_index),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"export_turn_{turn_index}",
                    use_container_width=True,
                )

            if turn.answer.get("answer_notice"):
                st.info(turn.answer["answer_notice"])
            st.markdown(turn.answer.get("answer_markdown", ""))

            evidence = turn.answer.get("evidence") or []
            if evidence:
                with st.expander("근거 카드 보기", expanded=True):
                    backend = turn.answer.get("answer_backend")
                    if backend:
                        st.caption(BACKEND_LABELS.get(backend, backend))
                    render_grouped_evidence(evidence, preview_chars)

            with st.expander("디버그 정보", expanded=False):
                st.json(
                    {
                        "intent": turn.answer.get("intent"),
                        "route_rationale": turn.answer.get("route_rationale"),
                        "answer_backend": turn.answer.get("answer_backend"),
                        "lm_studio_usage": turn.answer.get("model_usage", {}),
                    }
                )


def store_answer(result, law_name: str, source_types: list[str]) -> None:
    st.session_state.chat_history.extend(
        [
            {
                "role": "user",
                "content": result.question.strip(),
                "law_name": law_name,
                "source_types": list(source_types),
            },
            {
                "role": "assistant",
                "answer_markdown": result.answer_markdown,
                "intent": result.intent,
                "route_rationale": result.route_rationale,
                "evidence": result.evidence,
                "answer_backend": result.answer_backend,
                "answer_notice": result.answer_notice,
                "model_usage": result.quota_snapshot,
                "law_name": law_name,
                "source_types": list(source_types),
            },
        ]
    )


def handle_question(
    question: str,
    *,
    service: AnswerService,
    law_name: str,
    source_types: list[str],
) -> None:
    error = validate_question(question)
    if error:
        st.error(error)
        return

    with st.spinner("근거 문서를 찾고 로컬 모델로 답변을 구성하고 있습니다..."):
        result = service.answer(
            question=question,
            law_name=law_name,
            source_types=source_types or None,
        )

    store_answer(result, law_name, source_types)
    st.session_state.clear_question_input = True
    st.rerun()


def render_bootstrap_panel(store: ChromaStore, rows: list[dict]) -> None:
    settings = get_settings()
    corpus_path = get_corpus_input_path(settings)
    using_full_corpus = corpus_path == settings.processed_dir / "law_corpus.jsonl"
    if using_full_corpus:
        st.warning("아직 컬렉션이 비어 있습니다. 실제 원문 코퍼스를 먼저 적재해야 질문에 답할 수 있습니다.")
        button_label = "실제 원문 코퍼스 자동 적재"
        spinner_text = "실제 원문 코퍼스를 적재하고 있습니다..."
        success_text = "실제 원문 코퍼스를 적재했습니다. 화면을 다시 불러옵니다."
        missing_text = "실제 원문 코퍼스가 없습니다. 먼저 build_public_corpus.py를 실행해 주세요."
        command_text = "python scripts/build_public_corpus.py"
        caption_label = "실제 원문"
    else:
        st.warning("아직 컬렉션이 비어 있습니다. 샘플 코퍼스를 먼저 적재해야 질문에 답할 수 있습니다.")
        button_label = "샘플 코퍼스 자동 적재"
        spinner_text = "샘플 코퍼스를 적재하고 있습니다..."
        success_text = "샘플 코퍼스를 적재했습니다. 화면을 다시 불러옵니다."
        missing_text = "샘플 입력 파일이 없습니다. 먼저 build_sample_corpus.py를 실행해 주세요."
        command_text = (
            "python scripts/build_sample_corpus.py\n"
            "python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl"
        )
        caption_label = "샘플"

    left_col, right_col = st.columns([1.1, 1.3])
    with left_col:
        if st.button(button_label, use_container_width=True):
            if corpus_path.exists():
                with st.spinner(spinner_text):
                    ingest_jsonl(str(corpus_path), store)
                st.success(success_text)
                st.rerun()
            else:
                st.error(missing_text)
    with right_col:
        st.code(command_text, language="bash")
    st.caption(f"현재 준비된 {caption_label} 문서 수: {len(rows)}")


def ensure_preferred_corpus_loaded(store: ChromaStore) -> bool:
    settings = get_settings()
    corpus_path = get_corpus_input_path(settings)
    if store.count() > 0 or not corpus_path.exists():
        return False
    try:
        ingest_jsonl(str(corpus_path), store)
    except Exception:
        return False
    return store.count() > 0


def render_sidebar(law_options: list[str], source_type_options: list[str]) -> tuple[str, str, list[str], dict[str, Any]]:
    with st.sidebar:
        st.header("로컬 설정")

        default_base_url = st.session_state.get("local_base_url", DEFAULT_LM_STUDIO_BASE_URL)
        base_url = st.text_input("LM Studio Base URL", value=default_base_url)
        st.session_state.local_base_url = base_url.strip() or DEFAULT_LM_STUDIO_BASE_URL

        connection_state = probe_lm_studio(st.session_state.local_base_url)
        if connection_state.get("available"):
            st.success(connection_state["message"])
        else:
            st.warning(connection_state["message"])
        st.caption(f"연결 주소: {connection_state.get('base_url')}")
        loaded_models = connection_state.get("loaded_models") or []
        if loaded_models:
            st.caption("현재 로드된 LLM: " + ", ".join(loaded_models))
        else:
            st.caption("LM Studio에서 LLM 하나만 로드하면 앱이 그 모델을 자동으로 따라갑니다.")
        st.caption("로컬 버전에서는 질문 횟수와 한도를 제한하지 않습니다.")

        st.divider()
        st.header("검색 필터")
        selected_law = st.selectbox("법령", law_options, index=0)
        selected_source_types = st.multiselect(
            "자료 유형",
            source_type_options,
            default=source_type_options,
            format_func=format_source_type,
        )
        st.caption("자료 유형을 좁히면 현행 조문과 개정 자료를 더 분명하게 구분해서 볼 수 있습니다.")

        st.divider()
        if st.button("대화 기록 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.clear_question_input = True
            st.rerun()

    return st.session_state.local_base_url, selected_law, selected_source_types, connection_state


def main() -> None:
    st.set_page_config(
        page_title="로컬 군 복무 법규 RAG",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    ensure_session_state()

    settings = get_settings()
    store = get_store()
    rows = load_corpus_rows()
    law_options = load_law_options(rows)
    source_type_options = load_source_type_options(rows)

    base_url, selected_law, selected_source_types, connection_state = render_sidebar(
        law_options,
        source_type_options,
    )
    service = get_answer_service(base_url)

    if store.count() == 0:
        ensure_preferred_corpus_loaded(store)

    render_intro(connection_state, max(store.count(), len(rows)))
    render_connection_notice(connection_state)

    submitted, typed_question = render_question_box()
    example_question = render_example_buttons()

    if store.count() == 0:
        render_bootstrap_panel(store, rows)
        return

    if example_question:
        handle_question(
            example_question,
            service=service,
            law_name=selected_law,
            source_types=selected_source_types,
        )
    elif submitted:
        handle_question(
            typed_question,
            service=service,
            law_name=selected_law,
            source_types=selected_source_types,
        )

    render_chat_history(settings.ui.evidence_preview_chars, source_type_options)


if __name__ == "__main__":
    main()
