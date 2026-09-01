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
    echo "local model/checkpoint custody violation: $path" >&2
    FOUND=1
  done < <(find "$root" -type f -print0)
done

if [[ "$FOUND" -ne 0 ]]; then
  echo "model and checkpoint files must remain on approved remote storage" >&2
  exit 1
fi

echo "local model/checkpoint custody check passed"
