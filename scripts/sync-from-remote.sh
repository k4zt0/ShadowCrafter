#!/usr/bin/env bash
set -euo pipefail

SSH_KEY_PATH="${SHADOWCRAFTER_SSH_KEY:?Set SHADOWCRAFTER_SSH_KEY to the Vessl key path}"
REMOTE_HOST="${SHADOWCRAFTER_REMOTE_HOST:-root@capella.cloud.vessl.ai}"
REMOTE_PORT="${SHADOWCRAFTER_REMOTE_PORT:-31044}"
REMOTE_DIR="${SHADOWCRAFTER_REMOTE_DIR:-/root/ShadowCrafter}"

mkdir -p artifacts/manifests artifacts/preflight artifacts/environment data/processed reports
# The operator explicitly keeps model/base/checkpoint files off the local workstation.
# Pull only small audit evidence from artifacts; weights remain on approved remote storage.
for artifact_dir in manifests preflight environment; do
  rsync -az --partial --info=progress2 \
    --exclude '*.safetensors' \
    --exclude '*.bin' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.ckpt' \
    --exclude '*.gguf' \
    --exclude '*.onnx' \
    -e "ssh -i $SSH_KEY_PATH -p $REMOTE_PORT" \
    "$REMOTE_HOST:$REMOTE_DIR/artifacts/$artifact_dir/" "artifacts/$artifact_dir/"
done
rsync -az --partial --info=progress2 \
  -e "ssh -i $SSH_KEY_PATH -p $REMOTE_PORT" \
  "$REMOTE_HOST:$REMOTE_DIR/data/processed/" data/processed/
rsync -az --partial --info=progress2 \
  -e "ssh -i $SSH_KEY_PATH -p $REMOTE_PORT" \
  "$REMOTE_HOST:$REMOTE_DIR/reports/" reports/
