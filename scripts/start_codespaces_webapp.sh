#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p /tmp/military-basic-rule-finder

if pgrep -f "streamlit run streamlit_app.py --server.port 8501" >/dev/null; then
  exit 0
fi

nohup python -m streamlit run streamlit_app.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  >/tmp/military-basic-rule-finder/streamlit.log 2>&1 &
