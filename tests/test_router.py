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


def test_history_route_handles_timeline_wording():
    result = decide_route("징계 관련 기본법 변천사를 알려줘")
    assert result.intent == "explain_change"


def test_explain_route_handles_change_wording():
    result = decide_route("군에서 육아 관련 휴가나 휴직의 변화도 설명해줘")
    assert result.intent == "explain_change"


def test_transition_route_handles_dallajyeot_wording():
    result = decide_route("군인복무규율에서 기본법 체계로 넘어오면서 징계 기준이 어떻게 달라졌는지 설명해줘")
    assert result.intent == "explain_change"
    assert result.preferred_source_types[0] == "history_note"


def test_transition_route_handles_repeal_reorganization_and_current_scope_question():
    result = decide_route(
        "군인복무규율이 2016년 폐지된 뒤, 복무·군기·권리구제 사항은 기본법 체계로 어떻게 재편되었고, "
        "징계의 종류·절차는 현재 어떤 법령이 담당하는지 구분해서 설명해줘."
    )

    assert result.intent == "hybrid"
    assert result.preferred_source_types == ["history_note", "revision_reason", "old_new_comparison", "law_text"]


def test_transition_route_handles_revision_reason_plus_current_articles():
    result = decide_route(
        "군인복무규율이 폐지되고 기본법 체계가 도입되면서 어떻게 재편되었는지 개정 이유와 현행 조문을 함께 근거로 설명해줘."
    )

    assert result.intent == "hybrid"
    assert "law_text" in result.preferred_source_types
