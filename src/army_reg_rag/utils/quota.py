from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

from army_reg_rag.config import Settings


@dataclass(slots=True)
class QuotaState:
    date: str
    used_count: int


class LocalDailyQuota:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state_path = settings.runtime_dir / "quota_state.json"
        self.tz = ZoneInfo(settings.app.timezone)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _today(self) -> str:
        return datetime.now(self.tz).date().isoformat()

    def load(self) -> QuotaState:
        today = self._today()
        if not self.state_path.exists():
            state = QuotaState(date=today, used_count=0)
            self.save(state)
            return state
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = QuotaState(date=raw.get("date", today), used_count=int(raw.get("used_count", 0)))
        if state.date != today:
            state = QuotaState(date=today, used_count=0)
            self.save(state)
        return state

    def save(self, state: QuotaState) -> None:
        self.state_path.write_text(
            json.dumps({"date": state.date, "used_count": state.used_count}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def remaining(self) -> int:
        state = self.load()
        return max(self.settings.app.daily_limit - state.used_count, 0)

    def can_consume(self) -> bool:
        return self.remaining() > 0

    def consume(self, amount: int = 1) -> QuotaState:
        state = self.load()
        state.used_count += amount
        if state.used_count > self.settings.app.daily_limit:
            state.used_count = self.settings.app.daily_limit
        self.save(state)
        return state
