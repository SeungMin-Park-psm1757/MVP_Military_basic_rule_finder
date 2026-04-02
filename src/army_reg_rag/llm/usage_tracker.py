from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from math import ceil
from zoneinfo import ZoneInfo

from army_reg_rag.config import Settings


@dataclass(slots=True)
class GeminiUsageState:
    date: str
    request_count: int = 0
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    hard_blocked: bool = False
    block_reason: str = ""


class GeminiUsageTracker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state_path = settings.runtime_dir / "gemini_usage.json"
        self.tz = ZoneInfo(settings.app.timezone)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _today(self) -> str:
        return datetime.now(self.tz).date().isoformat()

    def _default_state(self) -> GeminiUsageState:
        return GeminiUsageState(date=self._today())

    def load(self) -> GeminiUsageState:
        today = self._today()
        if not self.state_path.exists():
            state = self._default_state()
            self.save(state)
            return state

        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = GeminiUsageState(
            date=raw.get("date", today),
            request_count=int(raw.get("request_count", 0)),
            prompt_tokens=int(raw.get("prompt_tokens", 0)),
            candidate_tokens=int(raw.get("candidate_tokens", 0)),
            total_tokens=int(raw.get("total_tokens", 0)),
            hard_blocked=bool(raw.get("hard_blocked", False)),
            block_reason=str(raw.get("block_reason", "")),
        )
        if state.date != today:
            state = self._default_state()
            self.save(state)
        return state

    def save(self, state: GeminiUsageState) -> None:
        self.state_path.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _soft_limit(self, limit: int) -> int:
        if limit <= 0:
            return 0
        return max(1, ceil(limit * self.settings.llm.budget_cutoff_ratio))

    def snapshot(self) -> dict[str, int | float | bool | str]:
        state = self.load()
        request_limit = int(self.settings.llm.daily_request_budget)
        token_limit = int(self.settings.llm.daily_token_budget)
        request_soft_limit = self._soft_limit(request_limit)
        token_soft_limit = self._soft_limit(token_limit)

        request_ratio = 0.0 if request_limit <= 0 else state.request_count / request_limit
        token_ratio = 0.0 if token_limit <= 0 else state.total_tokens / token_limit
        budget_ratio = request_ratio

        within_request_budget = request_soft_limit == 0 or state.request_count < request_soft_limit
        can_generate = not state.hard_blocked and within_request_budget
        remaining_requests = max(request_soft_limit - state.request_count, 0) if request_soft_limit else 0
        if not can_generate:
            remaining_requests = 0

        return {
            "date": state.date,
            "request_count": state.request_count,
            "prompt_tokens": state.prompt_tokens,
            "candidate_tokens": state.candidate_tokens,
            "total_tokens": state.total_tokens,
            "request_limit": request_limit,
            "token_limit": token_limit,
            "request_soft_limit": request_soft_limit,
            "token_soft_limit": token_soft_limit,
            "budget_cutoff_ratio": float(self.settings.llm.budget_cutoff_ratio),
            "request_ratio": request_ratio,
            "token_ratio": token_ratio,
            "budget_ratio": budget_ratio,
            "remaining_requests": remaining_requests,
            "hard_blocked": state.hard_blocked,
            "block_reason": state.block_reason,
            "can_generate": can_generate,
        }

    def record_success(self, *, prompt_tokens: int, candidate_tokens: int, total_tokens: int) -> dict[str, int | float | bool | str]:
        state = self.load()
        state.request_count += 1
        state.prompt_tokens += max(prompt_tokens, 0)
        state.candidate_tokens += max(candidate_tokens, 0)
        state.total_tokens += max(total_tokens, max(prompt_tokens + candidate_tokens, 0))
        self.save(state)
        return self.snapshot()

    def block_for_today(self, reason: str) -> dict[str, int | float | bool | str]:
        state = self.load()
        state.hard_blocked = True
        state.block_reason = reason
        self.save(state)
        return self.snapshot()
