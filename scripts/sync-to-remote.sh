#!/usr/bin/env bash
set -euo pipefail

SSH_KEY_PATH="${SHADOWCRAFTER_SSH_KEY:?Set SHADOWCRAFTER_SSH_KEY to the Vessl key path}"
REMOTE_HOST="${SHADOWCRAFTER_REMOTE_HOST:-root@capella.cloud.vessl.ai}"
REMOTE_PORT="${SHADOWCRAFTER_REMOTE_PORT:-31044}"
REMOTE_DIR="${SHADOWCRAFTER_REMOTE_DIR:-/root/ShadowCrafter}"

scripts/check-local-weight-custody.sh

ssh -i "$SSH_KEY_PATH" -p "$REMOTE_PORT" "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
# Runtime outputs and downloaded weights are managed separately. Excluding the
# whole tree prevents --delete from removing remote-only checkpoints.
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv*/' \
  --exclude '.DS_Store' \
  --exclude '.coverage' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude 'artifacts/' \
  --exclude 'data/raw/' \
  -e "ssh -i $SSH_KEY_PATH -p $REMOTE_PORT" \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"
