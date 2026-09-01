#!/usr/bin/env bash
set -euo pipefail

SSH_KEY_PATH="${SHADOWCRAFTER_SSH_KEY:?Set SHADOWCRAFTER_SSH_KEY to the training key path}"
REMOTE_HOST="${SHADOWCRAFTER_REMOTE_HOST:-root@capella.cloud.vessl.ai}"
REMOTE_PORT="${SHADOWCRAFTER_REMOTE_PORT:-31044}"
REMOTE_SOURCE_ROOT="${SHADOWCRAFTER_REMOTE_SOURCE_ROOT:-/root/ShadowCrafter-source}"
REVISION="${1:?Usage: scripts/deploy-source-snapshot.sh <full-git-revision>}"

if [[ ! "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "revision must be an exact lowercase 40-character Git SHA" >&2
  exit 2
fi
if [[ ! "$REMOTE_HOST" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$ ]]; then
  echo "remote host must have the form user@hostname" >&2
  exit 2
fi
if [[ ! "$REMOTE_SOURCE_ROOT" =~ ^/root/ShadowCrafter-source(/[A-Za-z0-9._-]+)*$ ]]; then
  echo "remote source root is outside the dedicated allowlist" >&2
  exit 2
fi

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"
HEAD_REVISION="$(git rev-parse HEAD)"
if [[ "$HEAD_REVISION" != "$REVISION" ]]; then
  echo "requested revision is not the current local HEAD" >&2
  exit 2
fi
if ! git diff --quiet -- || ! git diff --cached --quiet --; then
  echo "tracked source must be clean before creating a training snapshot" >&2
  exit 2
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "untracked non-ignored source files must be committed or removed" >&2
  exit 2
fi
scripts/check-local-weight-custody.sh >/dev/null

TEMPORARY_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMPORARY_DIR"' EXIT
BUNDLE_PATH="$TEMPORARY_DIR/shadowcrafter-$REVISION.bundle"
REMOTE_BUNDLE="$REMOTE_SOURCE_ROOT/.shadowcrafter-$REVISION.bundle"
REMOTE_TARGET="$REMOTE_SOURCE_ROOT/$REVISION"

git bundle create "$BUNDLE_PATH" HEAD
git bundle verify "$BUNDLE_PATH" >/dev/null
LOCAL_BUNDLE_SHA256="$(shasum -a 256 "$BUNDLE_PATH" | awk '{print $1}')"

ssh -o BatchMode=yes -i "$SSH_KEY_PATH" -p "$REMOTE_PORT" "$REMOTE_HOST" \
  "test ! -e '$REMOTE_TARGET' && test ! -e '$REMOTE_BUNDLE' && mkdir -p '$REMOTE_SOURCE_ROOT'"
scp -q -o BatchMode=yes -i "$SSH_KEY_PATH" -P "$REMOTE_PORT" \
  "$BUNDLE_PATH" "$REMOTE_HOST:$REMOTE_BUNDLE"

ssh -o BatchMode=yes -i "$SSH_KEY_PATH" -p "$REMOTE_PORT" "$REMOTE_HOST" \
  "test \"\$(sha256sum '$REMOTE_BUNDLE' | awk '{print \$1}')\" = '$LOCAL_BUNDLE_SHA256' && \
   git clone --quiet --no-hardlinks '$REMOTE_BUNDLE' '$REMOTE_TARGET' && \
   test \"\$(git -C '$REMOTE_TARGET' rev-parse HEAD)\" = '$REVISION' && \
   test -z \"\$(git -C '$REMOTE_TARGET' status --porcelain)\""

echo "deployed clean source snapshot: $REMOTE_TARGET"
echo "set PYTHONPATH=$REMOTE_TARGET/src when using an existing training environment"
