from army_reg_rag.retrieval.router import decide_route


def test_explain_route():
    result = decide_route("왜 개정되었어?")
    assert result.intent == "explain_change"
    assert "history_note" in result.preferred_source_types


def test_practical_route():
    result = decide_route("그럼 실무적으로 어떻게 처리해야 해?")
    assert result.intent == "practical"


def test_hybrid_route():
    result = decide_route("과거에는 어땠고 지금은 실무적으로 어떻게 봐야 해?")
    assert result.intent == "hybrid"
    assert result.preferred_source_types[0] == "history_note"


def test_search_route():
    result = decide_route("휴가 관련 현행 규정을 찾아줘")
    assert result.intent == "search"
    assert result.preferred_source_types[0] == "law_text"


def test_history_route_prioritizes_history_sources():
    result = decide_route("과거 군인복무규율에서 휴가가 어떻게 이어졌는지 알려줘")
    assert result.intent == "explain_change"
    assert result.preferred_source_types[0] == "history_note"
