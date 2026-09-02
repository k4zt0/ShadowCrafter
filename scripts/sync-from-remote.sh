#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
SSH_KEY_PATH="${SHADOWCRAFTER_SSH_KEY:?Set SHADOWCRAFTER_SSH_KEY to the Vessl key path}"
REMOTE_HOST="${SHADOWCRAFTER_REMOTE_HOST:-root@capella.cloud.vessl.ai}"
REMOTE_PORT="${SHADOWCRAFTER_REMOTE_PORT:-31044}"
REMOTE_DIR="${SHADOWCRAFTER_REMOTE_DIR:-/root/ShadowCrafter}"
REMOTE_SOURCE_DIR="${SHADOWCRAFTER_REMOTE_SOURCE_DIR:-/root/ShadowCrafter-source}"
MIRROR_ROOT="$REPOSITORY_ROOT/local_mirror"
PROJECT_MIRROR="$MIRROR_ROOT/remote-project"
SOURCE_MIRROR="$MIRROR_ROOT/source-snapshots"
RESERVE_KIB=$((10 * 1024 * 1024))

if [[ "$REMOTE_HOST" != "root@capella.cloud.vessl.ai" \
  || "$REMOTE_PORT" != "31044" \
  || "$REMOTE_DIR" != "/root/ShadowCrafter" \
  || "$REMOTE_SOURCE_DIR" != "/root/ShadowCrafter-source" ]]; then
  echo "remote mirror target is outside the approved ShadowCrafter endpoint" >&2
  exit 2
fi
if [[ ! -f "$SSH_KEY_PATH" || -L "$SSH_KEY_PATH" ]]; then
  echo "SSH key must be a regular non-symlink file" >&2
  exit 2
fi

mkdir -p "$PROJECT_MIRROR" "$SOURCE_MIRROR"
SSH_OPTIONS=(
  -F /dev/null
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o ClearAllForwardings=yes
  -o ForwardAgent=no
  -o StrictHostKeyChecking=yes
  -i "$SSH_KEY_PATH"
  -p "$REMOTE_PORT"
)

# Refuse a broad mirror when a likely credential is present. `.env.example` is documentation.
SUSPECT_FILES="$(
  ssh "${SSH_OPTIONS[@]}" "$REMOTE_HOST" \
    "find '$REMOTE_DIR' -xdev \
      -path '$REMOTE_DIR/.venv' -prune -o \
      -type f \( -name '.env' -o -name '.env.*' -o -name '.netrc' \
      -o -name '.pypirc' -o -name '*.key' -o -name '*.p12' -o -name '*.pfx' \
      -o -name 'id_rsa' -o -name 'id_ed25519' \) \
      ! -name '.env.example' -print"
)"
if [[ -n "$SUSPECT_FILES" ]]; then
  echo "remote mirror refused because likely credential files exist:" >&2
  echo "$SUSPECT_FILES" >&2
  exit 2
fi

REMOTE_KIB="$(
  ssh "${SSH_OPTIONS[@]}" "$REMOTE_HOST" \
    "du -sk '$REMOTE_DIR' '$REMOTE_SOURCE_DIR'" | awk '{ total += $1 } END { print total + 0 }'
)"
LOCAL_KIB="$(du -sk "$PROJECT_MIRROR" "$SOURCE_MIRROR" | awk '{ total += $1 } END { print total + 0 }')"
AVAILABLE_KIB="$(df -Pk "$MIRROR_ROOT" | awk 'NR == 2 { print $4 }')"
INCREMENTAL_KIB=$((REMOTE_KIB > LOCAL_KIB ? REMOTE_KIB - LOCAL_KIB : 0))
if (( AVAILABLE_KIB < INCREMENTAL_KIB + RESERVE_KIB )); then
  echo "insufficient local space for remote mirror plus 10 GiB reserve" >&2
  exit 2
fi

RSYNC_RSH="ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes -o ClearAllForwardings=yes -o ForwardAgent=no -o StrictHostKeyChecking=yes -i \"$SSH_KEY_PATH\" -p $REMOTE_PORT"
RSYNC_OPTIONS=(
  --archive
  --partial
  --human-readable
  --info=stats2
  '--include=.env.example'
  '--exclude=.env'
  '--exclude=.env.*'
  '--exclude=.netrc'
  '--exclude=.pypirc'
  '--exclude=.ssh/'
  '--exclude=secrets/'
  --exclude='*.key'
  --exclude='*.p12'
  --exclude='*.pfx'
)

# Never use --delete: an interrupted or partial remote view must not erase a good local copy.
rsync "${RSYNC_OPTIONS[@]}" -e "$RSYNC_RSH" \
  "$REMOTE_HOST:$REMOTE_DIR/" "$PROJECT_MIRROR/"
rsync "${RSYNC_OPTIONS[@]}" -e "$RSYNC_RSH" \
  "$REMOTE_HOST:$REMOTE_SOURCE_DIR/" "$SOURCE_MIRROR/"

echo "remote project mirror completed: $PROJECT_MIRROR"
echo "immutable source mirror completed: $SOURCE_MIRROR"
