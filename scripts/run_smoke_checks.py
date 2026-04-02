from __future__ import annotations

from dotenv import load_dotenv

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

from army_reg_rag.config import load_settings
from army_reg_rag.retrieval.router import decide_route
from army_reg_rag.utils.quota import LocalDailyQuota


def main() -> None:
    load_dotenv()
    settings = load_settings()
    quota = LocalDailyQuota(settings)

    assert settings.app.daily_limit == 20
    assert decide_route("왜 개정됐어?").intent == "explain_change"
    assert decide_route("실무상 어떻게 처리해?").intent == "what_should_i_do"
    assert quota.remaining() >= 0

    print("smoke checks passed")
    print(f"collection path: {settings.chroma_path}")
    print(f"remaining quota: {quota.remaining()}")


if __name__ == "__main__":
    main()
