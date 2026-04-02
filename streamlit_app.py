from __future__ import annotations

from collections import defaultdict
import html
import json

from scripts._bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from army_reg_rag.config import load_settings
from army_reg_rag.domain.models import SearchHit
from army_reg_rag.llm.gemini_client import QUOTA_BLOCK_MESSAGE
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.services.answer_service import AnswerService
from army_reg_rag.services.ingest_service import ingest_jsonl
from army_reg_rag.utils.io import read_jsonl

load_dotenv()

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
    "history_note": "연혁 메모",
}

FILTER_GROUP_ORDER = ["법령", "개정이유", "신구 비교", "기타"]

FILTER_GROUP_SOURCE_TYPES = {
    "법령": ["law_text"],
    "개정이유": ["revision_reason"],
    "신구 비교": ["old_new_comparison"],
}

BACKEND_LABELS = {
    "chroma": "Chroma PersistentClient",
    "json_fallback": "JSON fallback store",
    "gemini": "Gemini 2.0 Flash",
    "retrieval_fallback": "근거요약모드",
    "retrieval_only": "근거요약모드",
    "quota_blocked": "제한",
}

CHAT_EXAMPLES = [
    ("현행 규정", "군인의 지위 및 복무에 관한 기본법에서 휴가 관련 현행 규정을 찾아줘."),
    ("개정 배경", "왜 육아시간 관련 규정이 바뀌었는지 개정 이유 중심으로 설명해줘."),
    ("실무 참고", "휴가와 돌봄 관련 사안을 실무상 어떤 순서로 확인해야 하는지 알려줘."),
]

@st.cache_resource
def get_settings():
    return load_settings()


@st.cache_resource
def get_store():
    settings = get_settings()
    return ChromaStore(settings)


@st.cache_resource
def get_answer_service():
    settings = get_settings()
    return AnswerService(settings, store=get_store())


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;800&display=swap');

        :root {
            --paper: #f3efe7;
            --paper-soft: #fbf9f4;
            --ink: #17212b;
            --ink-soft: #51606f;
            --line: rgba(23, 33, 43, 0.10);
            --accent: #8a6a43;
            --accent-soft: rgba(138, 106, 67, 0.10);
            --slate: #304457;
            --warn: #8b4e2d;
        }

        html, body, [class*="css"] {
            font-family: "Noto Sans KR", sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(138, 106, 67, 0.08), transparent 24%),
                linear-gradient(180deg, #f7f4ee 0%, #f1ede4 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(247, 244, 238, 0.82);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f1e8 0%, #efe8dc 100%);
            border-right: 1px solid var(--line);
        }

        .main-shell {
            padding-top: 1.6rem;
        }

        .masthead {
            margin-bottom: 1.2rem;
            text-align: center;
        }

        .masthead-title {
            font-family: "Noto Sans KR", sans-serif;
            font-size: 2.2rem;
            line-height: 1.24;
            margin-bottom: 0.42rem;
            color: var(--ink);
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .masthead-copy {
            color: var(--ink-soft);
            line-height: 1.65;
            font-size: 0.95rem;
            max-width: 36rem;
            margin: 0 auto;
        }

        div[data-testid="stTextArea"] {
            margin-bottom: 0.35rem;
        }

        div[data-testid="stTextArea"] [data-baseweb="textarea"] {
            background: #ffffff !important;
            border: 1px solid var(--line);
            border-radius: 26px;
            box-shadow: 0 14px 32px rgba(23, 33, 43, 0.05);
            padding: 0.35rem;
        }

        div[data-testid="stTextArea"] textarea {
            background: #ffffff !important;
            min-height: 7rem;
        }

        div[data-testid="stTextArea"] textarea::placeholder {
            color: rgba(81, 96, 111, 0.88);
        }

        div[data-testid="InputInstructions"] {
            display: none;
        }

        .question-footer-copy {
            color: var(--ink-soft);
            font-size: 0.84rem;
            margin-top: -0.1rem;
        }

        div[data-testid="stButton"] > button {
            min-height: 2.9rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.84);
            font-weight: 600;
            color: var(--ink);
        }

        div[data-testid="stButton"] > button:hover {
            border-color: rgba(23, 33, 43, 0.24);
            background: rgba(255, 255, 255, 0.98);
            color: var(--ink);
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            background: var(--ink);
            border-color: var(--ink);
            color: #ffffff;
        }

        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #23303d;
            border-color: #23303d;
            color: #ffffff;
        }

        .example-header {
            margin: 0.15rem 0 0.6rem 0;
        }

        .example-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .example-title {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 700;
            margin-top: 0.05rem;
        }

        .example-copy {
            color: var(--ink-soft);
            font-size: 0.9rem;
            margin-top: 0.12rem;
        }

        .counter-text {
            color: var(--ink-soft);
            font-size: 0.84rem;
            padding-top: 0.45rem;
        }

        .empty-state {
            border: 1px solid var(--line);
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.68);
            padding: 1.05rem 1.15rem;
            margin-top: 1rem;
        }

        .empty-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .empty-copy {
            color: var(--ink-soft);
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .sidebar-copy {
            color: var(--ink-soft);
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .streamlit-expanderHeader {
            font-weight: 700;
        }

        .evidence-block-title {
            color: var(--ink);
            font-weight: 700;
            margin: 1rem 0 0.15rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_corpus_rows() -> list[dict]:
    settings = get_settings()
    rows_by_id: dict[str, dict] = {}
    for path in [settings.demo_input_path, settings.processed_dir / "law_corpus.jsonl"]:
        for row in read_jsonl(path):
            row_id = str(row.get("id") or f"{row.get('law_name', '')}:{row.get('article_title', '')}:{len(rows_by_id)}")
            rows_by_id[row_id] = row
    return list(rows_by_id.values())


def ordered_source_types(source_types: list[str]) -> list[str]:
    known = [value for value in SOURCE_TYPE_ORDER if value in source_types]
    extra = sorted(value for value in source_types if value not in SOURCE_TYPE_ORDER)
    return known + extra


def format_source_type(source_type: str) -> str:
    return SOURCE_TYPE_LABELS.get(source_type, source_type or "미분류")


def load_law_options(rows: list[dict]) -> list[str]:
    options = {"전체"}
    for row in rows:
        if row.get("law_name"):
            options.add(str(row["law_name"]))
    return sorted(options, key=lambda value: (value != "전체", value))


def load_source_type_options(rows: list[dict]) -> list[str]:
    available = {str(row["source_type"]) for row in rows if row.get("source_type")}
    return ordered_source_types(list(available))


def _other_source_types(source_types: list[str]) -> list[str]:
    primary = {value for values in FILTER_GROUP_SOURCE_TYPES.values() for value in values}
    return [value for value in ordered_source_types(source_types) if value not in primary]


def load_source_type_filter_groups(rows: list[dict]) -> list[str]:
    available_source_types = load_source_type_options(rows)
    options: list[str] = []
    for label in FILTER_GROUP_ORDER:
        if label == "기타":
            if _other_source_types(available_source_types):
                options.append(label)
            continue
        if any(source_type in available_source_types for source_type in FILTER_GROUP_SOURCE_TYPES[label]):
            options.append(label)
    return options


def expand_filter_groups(filter_groups: list[str], available_source_types: list[str]) -> list[str]:
    if not filter_groups:
        return available_source_types

    selected: list[str] = []
    other_source_types = _other_source_types(available_source_types)
    for label in filter_groups:
        if label == "기타":
            selected.extend(other_source_types)
        else:
            selected.extend(FILTER_GROUP_SOURCE_TYPES.get(label, []))

    expanded = [value for value in ordered_source_types(list(dict.fromkeys(selected))) if value in available_source_types]
    return expanded or available_source_types


def summarize_filter_groups(source_types: list[str], all_source_types: list[str]) -> list[str]:
    selected = set(source_types)
    summary: list[str] = []
    for label in FILTER_GROUP_ORDER:
        if label == "기타":
            if any(value in selected for value in _other_source_types(all_source_types)):
                summary.append(label)
            continue
        if any(value in selected for value in FILTER_GROUP_SOURCE_TYPES[label]):
            summary.append(label)
    return summary


def format_filter_summary(law_name: str, source_types: list[str], all_source_types: list[str]) -> str:
    parts = []
    if law_name != "전체":
        parts.append(f"법령: {law_name}")
    if source_types and len(source_types) != len(all_source_types):
        filter_groups = summarize_filter_groups(source_types, all_source_types)
        if filter_groups:
            parts.append("자료 유형: " + ", ".join(filter_groups))
    return " / ".join(parts)


def ensure_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("clear_question_input", False)
    if st.session_state.clear_question_input:
        st.session_state.question_input = ""
        st.session_state.clear_question_input = False


def global_quota_snapshot(service: AnswerService) -> dict:
    return service.client.usage_tracker.snapshot()


def remaining_global_questions(service: AnswerService) -> int:
    snapshot = global_quota_snapshot(service)
    return int(snapshot.get("remaining_requests", 0))


def validate_question(question: str, service: AnswerService, max_chars: int) -> str | None:
    stripped = question.strip()
    if not stripped:
        return "질문을 입력해 주세요."
    if len(stripped) > max_chars:
        return f"질문은 최대 {max_chars}자까지 입력할 수 있습니다."
    snapshot = global_quota_snapshot(service)
    if not snapshot.get("can_generate", False):
        return QUOTA_BLOCK_MESSAGE
    return None


def render_intro() -> None:
    st.markdown(
        """
        <div class="masthead">
            <div class="masthead-title">군인의 지위 및 복무에 관한 기본법 챗봇</div>
            <div class="masthead-copy">현행 조문, 개정 이유, 신구 비교, 연혁 메모를 근거로 답합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_buttons() -> str | None:
    st.markdown(
        """
        <div class="example-header">
            <div class="example-title">질문 예시</div>
            <div class="example-copy">질문 예시입니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(CHAT_EXAMPLES))
    for idx, (label, question) in enumerate(CHAT_EXAMPLES):
        if cols[idx].button(label, key=f"chat_example_{idx}", use_container_width=True):
            return question
    return None


def render_evidence_card(hit: SearchHit, preview_chars: int) -> None:
    chunk = hit.chunk
    st.markdown(f"**{chunk.law_name}**")
    st.caption(
        " · ".join(
            value
            for value in [
                format_source_type(chunk.source_type),
                chunk.article_no,
                chunk.article_title,
                f"시행일 {chunk.effective_date}" if chunk.effective_date else "",
                chunk.revision_kind,
            ]
            if value
        )
    )

    text = chunk.text.strip()
    if len(text) > preview_chars:
        text = text[:preview_chars] + "..."
    st.write(text)

    if chunk.source_url:
        st.markdown(f"[원문 링크]({chunk.source_url})")


def render_grouped_evidence(hits: list[SearchHit], preview_chars: int) -> None:
    if not hits:
        st.write("표시할 근거가 없습니다.")
        return

    grouped: defaultdict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.chunk.source_type or "unknown"].append(hit)

    for source_type in ordered_source_types(list(grouped.keys())):
        label = format_source_type(source_type)
        with st.expander(f"{label} {len(grouped[source_type])}건", expanded=source_type == "law_text"):
            for hit in grouped[source_type]:
                with st.container(border=True):
                    render_evidence_card(hit, preview_chars)


def build_evidence_summary_line(hit: SearchHit, index: int) -> str:
    chunk = hit.chunk
    preview = " ".join(chunk.text.strip().replace("데모용 요약:", "").split())
    if len(preview) > 170:
        preview = preview[:167].rstrip() + "..."
    article_ref = " ".join(part for part in [chunk.article_no, chunk.article_title] if part).strip()
    meta = [chunk.law_name, format_source_type(chunk.source_type)]
    if article_ref:
        meta.append(article_ref)
    if chunk.effective_date:
        meta.append(f"시행일 {chunk.effective_date}")
    return f"[{index}] {' / '.join(meta)} - {preview}"


def render_copyable_evidence_summary(hits: list[SearchHit]) -> None:
    summary_lines = [build_evidence_summary_line(hit, idx) for idx, hit in enumerate(hits[:5], start=1)]
    row_html = []
    for index, line in enumerate(summary_lines):
        safe_line = html.escape(line)
        copy_payload = json.dumps(line, ensure_ascii=False)
        row_html.append(
            f"""
            <div class="evidence-row">
                <div class="evidence-text">{safe_line}</div>
                <button class="copy-btn" onclick='copyEvidence({copy_payload}, this)' title="복사">📋</button>
            </div>
            """
        )

    block = f"""
    <style>
      body {{
        margin: 0;
        font-family: "Noto Sans KR", sans-serif;
        color: #17212b;
        background: transparent;
      }}
      .evidence-list {{
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
      }}
      .evidence-row {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.7rem;
        border: 1px solid rgba(23, 33, 43, 0.10);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.78);
        padding: 0.65rem 0.75rem;
      }}
      .evidence-text {{
        flex: 1;
        font-size: 0.88rem;
        line-height: 1.5;
        white-space: normal;
        word-break: keep-all;
      }}
      .copy-btn {{
        border: 1px solid rgba(23, 33, 43, 0.12);
        background: #ffffff;
        border-radius: 999px;
        width: 2rem;
        height: 2rem;
        cursor: pointer;
        flex-shrink: 0;
      }}
    </style>
    <div class="evidence-list">
      {''.join(row_html)}
    </div>
    <script>
      async function copyEvidence(text, button) {{
        const original = button.textContent;
        try {{
          if (navigator.clipboard && window.isSecureContext) {{
            await navigator.clipboard.writeText(text);
          }} else {{
            const area = document.createElement('textarea');
            area.value = text;
            document.body.appendChild(area);
            area.select();
            document.execCommand('copy');
            document.body.removeChild(area);
          }}
          button.textContent = '✓';
        }} catch (err) {{
          button.textContent = '!';
        }}
        setTimeout(() => {{
          button.textContent = original;
        }}, 1200);
      }}
    </script>
    """
    components.html(block, height=72 + (len(summary_lines) * 74), scrolling=False)


def render_live_counter_bridge(max_chars: int) -> None:
    components.html(
        f"""
        <script>
          const doc = window.parent.document;
          const attach = () => {{
            const textarea = doc.querySelector('textarea[aria-label="질문 입력"]');
            const counter = doc.getElementById('live-question-counter');
            if (!textarea || !counter) {{
              window.setTimeout(attach, 120);
              return;
            }}
            const update = () => {{
              counter.textContent = `${{textarea.value.length}}/{max_chars}`;
            }};
            if (!textarea.dataset.counterBound) {{
              textarea.dataset.counterBound = '1';
              textarea.addEventListener('input', update);
              textarea.addEventListener('keyup', update);
              textarea.addEventListener('change', update);
            }}
            update();
          }};
          attach();
        </script>
        """,
        height=0,
    )


def render_chat_history(preview_chars: int, all_source_types: list[str], allow_debug: bool) -> None:
    history = st.session_state.chat_history
    if not history:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-title">답변은 이 아래 대화 영역에 표시됩니다.</div>
                <div class="empty-copy">질문을 입력하면 관련 공개 자료를 찾고, 현행 조문과 개정 근거를 구분해 정리합니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
                summary = format_filter_summary(
                    message.get("law_name", "전체"),
                    message.get("source_types", all_source_types),
                    all_source_types,
                )
                if summary:
                    st.caption(summary)
            continue

        with st.chat_message("assistant"):
            st.caption("실무 참고용, 법률자문 아님")
            st.markdown(message["answer_markdown"])
            if message.get("evidence"):
                st.markdown("<div class='evidence-block-title'>근거(하단 원문 링크 참고)</div>", unsafe_allow_html=True)
                if message.get("answer_backend"):
                    st.caption(BACKEND_LABELS.get(message["answer_backend"], message["answer_backend"]))
                render_copyable_evidence_summary(message["evidence"])
                with st.expander("원문 링크", expanded=False):
                    render_grouped_evidence(message["evidence"], preview_chars)
            if allow_debug:
                with st.expander("검색 메모 보기", expanded=False):
                    st.json(
                        {
                            "intent": message["intent"],
                            "route_rationale": message["route_rationale"],
                            "answer_backend": message.get("answer_backend"),
                            "quota_snapshot": message.get("quota_snapshot", {}),
                        }
                    )


def render_quota_panel(service: AnswerService) -> None:
    usage = service.client.usage_tracker.snapshot()
    budget_ratio = float(usage.get("budget_ratio", 0.0))
    request_count = int(usage.get("request_count", 0))
    request_soft_limit = int(usage.get("request_soft_limit", 0))
    remaining_requests = int(usage.get("remaining_requests", 0))
    max_chars = get_settings().app.max_question_chars

    st.header("사용 현황")
    st.metric("남은 질문 수", remaining_requests)
    st.metric("질문 최대 글자 수", max_chars)
    st.metric("생성 응답 상태", "제한" if usage.get("hard_blocked") or not usage.get("can_generate") else "정상")

    if usage.get("hard_blocked"):
        st.error(QUOTA_BLOCK_MESSAGE)
    elif not usage.get("can_generate"):
        st.warning(QUOTA_BLOCK_MESSAGE)
    elif budget_ratio >= 0.75:
        st.info("Gemini 사용량이 누적되고 있습니다. 오늘은 보수적으로 운영합니다.")

    st.caption(f"오늘 허용 질문 {request_count}/{request_soft_limit or '-'}")


def render_sidebar(law_options: list[str], source_type_options: list[str], source_type_filter_groups: list[str], service: AnswerService) -> tuple[str, list[str]]:
    settings = get_settings()
    with st.sidebar:
        render_quota_panel(service)
        st.divider()

        st.header("검색 필터")
        default_law = settings.app.default_law_filter
        default_index = law_options.index(default_law) if default_law in law_options else 0
        selected_law = st.selectbox("법령 필터", law_options, index=default_index)
        selected_filter_groups = st.multiselect(
            "자료 유형 필터",
            source_type_filter_groups,
            default=source_type_filter_groups,
        )
        selected_source_types = expand_filter_groups(selected_filter_groups, source_type_options)
        st.markdown("<div class='sidebar-copy'>필요한 범위만 좁혀서 조회할 수 있습니다.</div>", unsafe_allow_html=True)
        st.divider()

        st.header("세션")
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.clear_question_input = True
            st.rerun()
        st.markdown("<div class='sidebar-copy'>현재 화면의 대화만 초기화합니다.</div>", unsafe_allow_html=True)
    return selected_law, list(selected_source_types)


def _store_answer(result, law_name: str, source_types: list[str], *, replace_history: bool = False) -> None:
    messages = [
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
            "quota_snapshot": result.quota_snapshot,
            "law_name": law_name,
            "source_types": list(source_types),
        },
    ]
    if replace_history:
        st.session_state.chat_history = messages
    else:
        st.session_state.chat_history.extend(messages)


def handle_chat_submission(
    question: str,
    *,
    service: AnswerService,
    law_name: str,
    source_types: list[str],
    settings,
) -> None:
    error = validate_question(question, service, settings.app.max_question_chars)
    if error:
        if error == QUOTA_BLOCK_MESSAGE:
            st.session_state.chat_history.extend(
                [
                    {
                        "role": "user",
                        "content": question.strip(),
                        "law_name": law_name,
                        "source_types": list(source_types),
                    },
                    {
                        "role": "assistant",
                        "answer_markdown": error,
                        "intent": "quota_blocked",
                        "route_rationale": "",
                        "evidence": [],
                        "answer_backend": "quota_blocked",
                        "answer_notice": "",
                        "quota_snapshot": global_quota_snapshot(service),
                        "law_name": law_name,
                        "source_types": list(source_types),
                    },
                ]
            )
            st.session_state.clear_question_input = True
            st.rerun()
        st.error(error)
        return

    with st.spinner("근거 문서를 찾고 답변을 정리하고 있습니다..."):
        result = service.answer(question=question, law_name=law_name, source_types=source_types)

    _store_answer(result, law_name, source_types)
    st.session_state.clear_question_input = True
    st.rerun()


def handle_example_click(
    question: str,
    *,
    service: AnswerService,
    law_name: str,
    source_types: list[str],
) -> None:
    result = service.answer(
        question=question,
        law_name=law_name,
        source_types=source_types,
        allow_generation=False,
    )
    _store_answer(result, law_name, source_types, replace_history=True)
    st.session_state.clear_question_input = True
    st.rerun()


def render_bootstrap_panel(store: ChromaStore, rows: list[dict]) -> None:
    settings = get_settings()
    st.warning("아직 컬렉션이 비어 있습니다. 아래 버튼으로 데모 코퍼스를 자동 적재하거나 README 절차를 먼저 실행해 주세요.")
    col1, col2 = st.columns([1, 1.1])
    with col1:
        if st.button("데모 코퍼스 자동 적재", use_container_width=True):
            if settings.demo_input_path.exists():
                with st.spinner("데모 코퍼스를 적재하고 있습니다..."):
                    ingest_jsonl(str(settings.demo_input_path), store)
                st.success("데모 코퍼스를 적재했습니다. 화면을 다시 불러옵니다.")
                st.rerun()
            else:
                st.error("데모 입력 파일이 없습니다. 먼저 build_sample_corpus.py를 실행해 주세요.")
    with col2:
        st.code(
            "python scripts/build_sample_corpus.py\n"
            "python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl",
            language="bash",
        )
    st.caption(f"로컬 샘플 코퍼스 {len(rows)}건이 준비되어 있습니다.")


def render_question_box(max_chars: int) -> tuple[bool, str]:
    question = st.text_area(
        "질문 입력",
        key="question_input",
        placeholder="이곳이 채팅창입니다. 궁금한 내용을 입력해 주세요.\n예: 군인복무규율에서 군인의 지위 및 복무에 관한 기본법으로 이어지는 변화를 설명해줘.",
        height=110,
        label_visibility="collapsed",
    )
    footer_col_1, footer_col_2 = st.columns([5.8, 1.2], vertical_alignment="bottom")
    with footer_col_1:
        st.markdown(
            f"<div id='live-question-counter' class='question-footer-copy'>{len(question)}/{max_chars}</div>",
            unsafe_allow_html=True,
        )
        render_live_counter_bridge(max_chars)
    with footer_col_2:
        submitted = st.button("전송", key="send_question", type="primary", use_container_width=True)
    return submitted, question


def main():
    st.set_page_config(page_title="군인 기본법 법규 챗봇", page_icon="§", layout="wide")
    inject_styles()
    settings = get_settings()
    ensure_session_state()

    store = get_store()
    service = get_answer_service()
    rows = load_corpus_rows()

    law_options = load_law_options(rows)
    source_type_options = load_source_type_options(rows)
    source_type_filter_groups = load_source_type_filter_groups(rows)
    selected_law, selected_source_types = render_sidebar(law_options, source_type_options, source_type_filter_groups, service)

    _, center_col, _ = st.columns([1.1, 6.2, 1.1])
    with center_col:
        st.markdown("<div class='main-shell'>", unsafe_allow_html=True)
        render_intro()

        if store.count() == 0:
            render_bootstrap_panel(store, rows)
            st.markdown("</div>", unsafe_allow_html=True)
            return

        form_submitted, typed_question = render_question_box(settings.app.max_question_chars)
        example_question = render_example_buttons()

        if example_question:
            handle_example_click(
                example_question,
                service=service,
                law_name=selected_law,
                source_types=selected_source_types,
            )
        elif form_submitted:
            handle_chat_submission(
                typed_question,
                service=service,
                law_name=selected_law,
                source_types=selected_source_types,
                settings=settings,
            )

        render_chat_history(
            preview_chars=settings.ui.evidence_preview_chars,
            all_source_types=source_type_options,
            allow_debug=settings.app.allow_debug_tab,
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
