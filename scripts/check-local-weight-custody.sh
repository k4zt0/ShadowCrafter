#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"

FORBIDDEN_ROOTS=(
  artifacts/base_models
  artifacts/checkpoints
  artifacts/releases
  artifacts/cache
  checkpoints
  models
)

FOUND=0
for root in "${FORBIDDEN_ROOTS[@]}"; do
  [[ -d "$root" ]] || continue
  while IFS= read -r -d '' path; do
    case "${path##*/}" in
      .gitkeep | .DS_Store) continue ;;
    esac
    echo "model/checkpoint file escaped the ignored local_mirror boundary: $path" >&2
    FOUND=1
  done < <(find "$root" -type f -print0)
done

if [[ "$FOUND" -ne 0 ]]; then
  echo "store local model copies only below the gitignored local_mirror directory" >&2
  exit 1
fi

if ! git check-ignore -q local_mirror/remote-project/model.safetensors; then
  echo "local_mirror must remain excluded from Git" >&2
  exit 1
fi

echo "local model archive isolation check passed"
