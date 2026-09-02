# Remote Official Release manifest contract

This contract is intentionally narrow. It supports only this existing private model
repository:

- `KaztoRay/ShadowCrafter-9B`

The JSON manifest and its five approval records are workstation-local control/evidence files.
Model, base-model, adapter, checkpoint, and weight bytes remain on the approved remote host for
publication and may also be retained in the Git-ignored local mirror. Before invoking the publisher, freeze the manifest
itself with an independently retained lowercase SHA-256 and supply that value through
`--manifest-sha256`.

## Manifest

Unknown keys, duplicate JSON keys, unpinned manifests, symlinks, unsafe paths, inconsistent
counts, and mismatched model families fail closed. `files` is the exact complete upload allowlist,
sorted lexicographically by `path` with no duplicate path:

```json
{
  "schema_version": 1,
  "release_id": "<path-safe-immutable-id>",
  "repo_id": "KaztoRay/ShadowCrafter-9B",
  "release_tier": "Official Release",
  "visibility": "private",
  "commercial_release": false,
  "parent_commit": "<current-private-Hub-main-commit>",
  "candidate_checkpoint_sha256": "<64-lowercase-hex>",
  "remote_root": "/root/ShadowCrafter/artifacts/releases/shadowcrafter-9b/<release-id>",
  "ssh": {
    "host": "capella.cloud.vessl.ai",
    "port": 31044,
    "user": "root"
  },
  "files": [
    {"path": "README.md", "size": 1234, "sha256": "<64-lowercase-hex>"},
    {
      "path": "releases/<release-id>/adapter_config.json",
      "size": 1234,
      "sha256": "<64-lowercase-hex>"
    },
    {
      "path": "releases/<release-id>/adapter_model.safetensors",
      "size": 1234,
      "sha256": "<64-lowercase-hex>"
    }
  ],
  "total_bytes": 3702,
  "evaluation": {
    "status": "not-yet-evaluated",
    "reason": "Explicit reason that is reproduced in the model card."
  },
  "approvals": {
    "artifact_integrity": {"path": "reviews/integrity.json", "sha256": "<sha256>"},
    "provenance": {"path": "reviews/provenance.json", "sha256": "<sha256>"},
    "license": {"path": "reviews/license.json", "sha256": "<sha256>"},
    "privacy": {"path": "reviews/privacy.json", "sha256": "<sha256>"},
    "safety": {"path": "reviews/safety.json", "sha256": "<sha256>"}
  }
}
```

Every path other than root `README.md` must be below `releases/<release-id>/`.
The inventory requires at least one `.safetensors` file and a model or
adapter config. Executable/pickle formats are not accepted. A file is capped at 2 GiB, the total
at 4 GiB, the model card at 2 MiB, and the inventory at 256 files.

A measured release replaces the evaluation object with:

```json
{
  "status": "measured",
  "evidence_manifest_sha256": "<frozen-release-evidence-json-sha256>"
}
```

That evidence must pass the integrity/contamination/review gate and be bound to this repository
and checkpoint. Its measured accuracy, balanced accuracy, macro-F1, benchmark identity/revision/
hash, sample count, and `quality_target_met` must be reproduced exactly in the model card. A false
quality target is nonblocking for this private Official tier.

## Approval records

Compute `remote_inventory_sha256` as SHA-256 of UTF-8 compact canonical JSON made with sorted
object keys and separators `,` and `:` over exactly this object:

```json
{"files": ["<the manifest file objects, in order>"], "remote_root": "<remote_root>", "total_bytes": 3702}
```

Here `["<...>"]` denotes the actual file objects, not a string placeholder. Each referenced
approval JSON must be a regular non-symlink file inside the manifest directory and have this
shape (with its own exact `review` value):

```json
{
  "schema_version": 1,
  "review": "artifact_integrity",
  "passed": true,
  "repo_id": "KaztoRay/ShadowCrafter-9B",
  "release_id": "<release-id>",
  "candidate_checkpoint_sha256": "<candidate-sha256>",
  "remote_inventory_sha256": "<canonical-inventory-sha256>",
  "private_official_release_authorized": true,
  "public_release_authorized": false
}
```

The five `review` values are `artifact_integrity`, `provenance`, `license`, `privacy`, and
`safety`. The license record additionally requires
`"commercial_release_authorized": false` and
`"benchmark_material_sharing_authorized": false`. The publisher verifies each approval file's
manifest-pinned SHA-256 before reading any remote model byte.

## Model card fields

Root `README.md` must contain YAML front matter with `shadowcrafter_release` and
`shadowcrafter_evaluation`. The release object contains exact `status`, `visibility`,
`commercial_use`, `release_id`, `repository`, and `candidate_checkpoint_sha256` values. The body
must prominently say `Official Release`.

For `not-yet-evaluated`, accuracy, balanced accuracy, macro-F1, and `quality_target_met` are all
YAML null and the body includes the exact manifest reason. For `measured`, the evaluation object
contains exact `benchmark`, `revision`, `dataset_sha256`, `sample_count`, `accuracy`,
`balanced_accuracy`, `macro_f1`, and `quality_target_met` values from the frozen gate report.

See [`REMOTE_WORKFLOW.md`](REMOTE_WORKFLOW.md#9-mirror-a-gated-release-privately) for the command,
authentication, one-commit, post-upload verification, and recovery procedure.
