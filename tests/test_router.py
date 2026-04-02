from army_reg_rag.retrieval.router import decide_route


def test_explain_route():
    result = decide_route("왜 규정이 바뀌었어?")
    assert result.intent == "explain_change"
    assert "history_note" in result.preferred_source_types


def test_practical_route():
    result = decide_route("그럼 실무상 어떻게 처리해야 해?")
    assert result.intent == "practical"


def test_hybrid_route():
    result = decide_route("왜 바뀌었고, 나는 어떻게 해야 해?")
    assert result.intent == "hybrid"
    assert result.preferred_source_types[0] == "revision_reason"


def test_search_route():
    result = decide_route("휴가 관련 현행 규정을 찾아줘.")
    assert result.intent == "search"
    assert result.preferred_source_types[0] == "law_text"
