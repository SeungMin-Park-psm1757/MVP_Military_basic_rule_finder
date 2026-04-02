from __future__ import annotations

import os
from typing import Any


def get_runtime_value(name: str, default: Any = "") -> Any:
    value = os.getenv(name)
    if value not in (None, ""):
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            secret_value = st.secrets[name]
            if secret_value not in (None, ""):
                return secret_value

        env_group = st.secrets.get("env")
        if isinstance(env_group, dict) and name in env_group:
            secret_value = env_group[name]
            if secret_value not in (None, ""):
                return secret_value
    except Exception:
        pass

    return default


def get_runtime_bool(name: str, default: bool = False) -> bool:
    value = get_runtime_value(name, None)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
