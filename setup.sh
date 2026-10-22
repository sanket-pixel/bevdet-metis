#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-../deeplsd-metis/.venv/bin/python}
if [ ! -x "$PYTHON" ]; then
    echo 'Set PYTHON to your existing Python environment, or create .venv with uv venv --python 3.11 .venv' >&2
    exit 2
fi
uv pip install --python "$PYTHON" --index-url https://download.pytorch.org/whl/cpu 'torch>=2.0'
uv pip install --python "$PYTHON" -r requirements.txt
