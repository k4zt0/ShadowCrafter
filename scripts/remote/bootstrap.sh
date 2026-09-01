#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${SHADOWCRAFTER_REMOTE_DIR:-/root/ShadowCrafter}"
cd "$PROJECT_DIR"

if ! python3.12 -m venv --help >/dev/null 2>&1; then
  apt-get update
  apt-get install -y python3.12-venv git-lfs rsync
fi

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e '.[data,serve]'
.venv/bin/python -m pip install -r requirements/train-hf.lock.txt
.venv/bin/python -m pip install -e . --no-deps
mkdir -p artifacts/environment
.venv/bin/python -m pip freeze > artifacts/environment/pip-freeze.txt

echo "Remote environment ready at $PROJECT_DIR/.venv"
