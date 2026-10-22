#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-../deeplsd-metis/.venv/bin/python}
if [ ! -x "$PYTHON" ]; then
    echo 'Set PYTHON to your existing Python environment, or create .venv with uv venv --python 3.11 .venv' >&2
    exit 2
fi
uv pip install --python "$PYTHON" -r requirements.txt
upstream=third_party/advanced_fastbev
revision=04c2b3610f7c33ebc323723fe90998121e22cfac
if [ ! -d "$upstream/.git" ]; then
    mkdir -p "$upstream"
    git -C "$upstream" init -q
    git -C "$upstream" remote add origin https://github.com/ymlab/advanced-fastbev.git
    git -C "$upstream" fetch --depth 1 origin "$revision"
    git -C "$upstream" checkout --detach FETCH_HEAD
fi
if [ "$(git -C "$upstream" rev-parse HEAD)" != "$revision" ]; then
    echo "Expected upstream revision $revision in $upstream" >&2
    exit 2
fi
