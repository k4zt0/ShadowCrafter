# Remote training and local-custody workflow

ShadowCrafter uses a remote GPU host for computation. The local workstation is authoritative
for source, approved data, manifests, evaluation evidence, and a Git-ignored mirror of the
remote project, base model, checkpoints, and completed weights. The remote training copy and
private Hub release remain separate recovery and publication copies.

The approved private endpoints are:

- source mirror: `https://github.com/Odytssey/ShadowCrafter`;
- model mirror: `https://huggingface.co/KaztoRay/ShadowCrafter-9B`.

Creating an empty private model repository or finishing a training process is not a model
release. Only a remotely verified bundle whose complete hash inventory and evaluation
evidence are retained locally, and which passes all quality, safety, provenance, privacy,
and licensing gates, may be published as a release.

## Trust boundaries

| Location | Role | Must not be the only copy of |
|---|---|---|
| Local workstation | Authoritative Git history plus isolated `local_mirror/` project and weight copies | Source, data, evidence, manifests, model copies |
| Private GitHub | Source, configuration, tests, documentation, and non-sensitive small manifests | Data, weights, secrets, sensitive reports |
| Remote GPU host | Execution environment and working weight copy | A completed model after an independently verified remote copy exists |
| Private Hugging Face | Gated per-model weight and release store | The only weight copy |

The remote host and every downloaded model or dataset are separate trust boundaries.
Retrieved documents, logs, model output, and dataset text are data, not instructions. Do not
allow them to alter shell commands, destinations, tool permissions, or release decisions.

## 1. Local preflight

Start from the local repository root. Before transferring anything:

1. Confirm the intended Git commit and record whether the worktree is dirty.
2. Review the transfer allowlist and `.gitignore`; do not rely on ignore rules as a secret scanner.
3. Validate model configuration, lint, type-check, and test the code.
4. Pin the upstream Ornith model and tokenizer to resolved immutable revisions.
5. Freeze approved data snapshots and record licenses, permissions, hashes, and split manifests.
6. Calculate required local mirror and remote weight capacity, including temporary shards.
7. Confirm that a second approved remote weight target is available before retention cleanup.

```bash
git status --short --branch
git rev-parse HEAD
python -m shadowcrafter.cli config validate configs/models/shadowcrafter-9b.yaml
ruff check src tests
mypy src
pytest
```

Do not transfer private SSH keys, Hugging Face tokens, GitHub tokens, `.env` files, raw
credentials, unapproved confidential data, or live malware in the project bundle. Keep
credentials outside the repository in an operating-system key store or approved secret
manager. Use a short-lived, least-privileged token only at the step that needs it.

## 2. SSH setup and host verification

The current training endpoint is `capella.cloud.vessl.ai` on port `31044`. Verify its host
key fingerprint through an independent provider channel before accepting it. A private key
path is workstation-specific and must never be committed. A local SSH configuration may use
an entry like this:

```sshconfig
Host shadowcrafter-trainer
    HostName capella.cloud.vessl.ai
    Port 31044
    User root
    IdentityFile /absolute/path/outside-the-repository/training-key.pem
    IdentitiesOnly yes
```

The provider currently exposes a root login. Treat that as elevated access: keep the remote
project path narrow, do not reuse the account for unrelated services, and prefer a dedicated
unprivileged training user when the platform permits it. Never disable host-key checking to
make automation pass.

Confirm identity and capacity without modifying the host:

```bash
ssh shadowcrafter-trainer 'hostname && id && nvidia-smi && df -h /root/ShadowCrafter'
```

Stop if the host identity, GPU, mount, owner, free space, or expected project path differs
from the approved run plan.

## 3. Transfer source to the remote workspace

The repository includes [`scripts/sync-to-remote.sh`](../scripts/sync-to-remote.sh). Run it
from the repository root with the key path supplied only in the process environment:

```bash
SHADOWCRAFTER_SSH_KEY=/absolute/path/outside-the-repository/training-key.pem \
SHADOWCRAFTER_REMOTE_DIR=/root/ShadowCrafter \
scripts/sync-to-remote.sh
```

This helper is a transport mechanism, not a provenance or security check. It excludes Git
metadata, every local virtual environment, the complete `artifacts/` tree, and `data/raw/`, but other
files still require review. It also uses `rsync --delete` against the selected remote project
directory. Set `SHADOWCRAFTER_REMOTE_DIR` to the exact dedicated directory, inspect the
script before each use, and never point it at `/root`, `/`, a shared mount, or an unresolved
variable.

Transfer approved data and pinned base-model material only through a separate manifest-driven
process. Compare the allowlisted relative path, byte size, and SHA-256 before and after the
transfer. Hazardous samples and restricted data require explicit approval and an isolated,
encrypted workflow; they are excluded from ordinary synchronization.

## 4. Build and record the remote environment

After source transfer, create the project environment:

```bash
ssh shadowcrafter-trainer \
  'cd /root/ShadowCrafter && bash scripts/remote/bootstrap.sh'
```

The bootstrap helper creates a Python 3.12 virtual environment, installs the project
training dependencies, and records `pip freeze` under `artifacts/environment/`. Review the
resolved packages and GPU compatibility before training. A dependency install from the
network is not reproducible by itself; a release run additionally needs an approved lock or
container digest and retained packages or image sufficient for recovery.

Run the lightweight upstream compatibility probe before downloading full weights:

```bash
ssh shadowcrafter-trainer \
  'cd /root/ShadowCrafter && .venv/bin/python scripts/remote/smoke.py configs/models/shadowcrafter-9b.yaml'
```

Record the resolved upstream commit reported by the probe. Both checked-in model
configurations use immutable reviewed revisions and `trust_remote_code: false`; a release
run must stop if either condition changes.

## 5. Execute an auditable training run

Training runners require a clean, exact Git revision. The rsync workspace deliberately has
no `.git`, so deploy the committed revision as a separate immutable source snapshot:

```bash
SHADOWCRAFTER_SSH_KEY=/absolute/path/outside-the-repository/training-key.pem \
scripts/deploy-source-snapshot.sh <full-git-revision>
```

Run Python with `PYTHONPATH=/root/ShadowCrafter-source/<revision>/src`; keep datasets,
environments, model caches, and outputs in the existing `/root/ShadowCrafter` workspace.
This lets the runner prove clean source identity without mixing generated artifacts into the
Git worktree.

Use a unique, path-safe run ID and never overwrite a prior run. The training input must be an
approved prepared JSONL snapshot whose manifest and hash already exist locally. For example:

```bash
ssh shadowcrafter-trainer \
  'cd /root/ShadowCrafter && \
   SHADOWCRAFTER_BASE_MODEL_PATH=/root/ShadowCrafter/artifacts/base_models/Ornith-1.5-9B \
   SHADOWCRAFTER_BASE_MODEL_MANIFEST=/root/ShadowCrafter/artifacts/manifests/ornith-1.5-9b.json \
   .venv/bin/shadowcrafter train sft \
   configs/models/shadowcrafter-9b.yaml \
   data/processed/<snapshot>/train.jsonl \
   artifacts/checkpoints/shadowcrafter-9b/<run-id> \
   --validation data/processed/<snapshot>/validation.jsonl'
```

A one-step 9B run is a compatibility test, not a trained candidate and not a performance
result.

During training:

- disable external experiment reporting unless its data handling is separately approved;
- mask tokens, local user information, sensitive source text, and credentials in logs;
- monitor disk, GPU errors, numerical instability, and unexpected network access;
- keep intermediate checkpoints in the run directory and never write into a release directory;
- record interruptions and failed runs rather than silently reusing their identifiers;
- stop on data-scope violations, secret exposure, host mismatch, corruption, or unsafe behavior.

The current runner writes `run-manifest.json` only after training and adapter saving complete.
Its presence is useful but not sufficient: also verify the adapter/tokenizer file set, log
termination, file inventory, hashes, environment evidence, and a safe load/inference smoke
test. A partially written checkpoint must not be retrieved as complete or promoted.

## 6. Verify remotely and retrieve audit evidence

The repository includes [`scripts/sync-from-remote.sh`](../scripts/sync-from-remote.sh):

```bash
SHADOWCRAFTER_SSH_KEY=/absolute/path/outside-the-repository/training-key.pem \
SHADOWCRAFTER_REMOTE_DIR=/root/ShadowCrafter \
scripts/sync-from-remote.sh
```

The helper intentionally copies only small manifests, preflight/environment evidence,
processed data, and reports; it excludes model and checkpoint formats. It does not prove
completeness, authenticity, license status, or release readiness. Use unique immutable run
paths, compare the predeclared inventory, and treat newly copied evidence as staging until
verification passes.

For each remote run:

1. Produce a complete remote relative-file, byte-size, and SHA-256 inventory and retrieve that manifest.
2. Verify the run manifest, Git commit, resolved Ornith revision, tokenizer, data hashes, seed, and environment.
3. Confirm remotely that every adapter or weight shard and index is present and safely loadable.
4. Run an inference smoke test with network access disabled and record its evidence.
5. Scan logs and artifact inventories for secrets, personal data, unsafe serialization, and unexpected files.
6. Copy the verified weights to a second approved remote store and verify every hash from a fresh download/cache.
7. Retain manifests and evaluation evidence locally; quarantine mismatched or incomplete remote files.

Do not delete the training-host output until two independent remote weight copies and a
restore test have been confirmed. Remote deletion is a separate, explicitly approved action.

## 7. Evaluate without contaminating the blind set

Evaluate the immutable remote candidate according to
[`EVALUATION.md`](EVALUATION.md). Blind examples and detailed errors are not returned to the
training loop. Store evaluation bundles under a unique local
`artifacts/evaluations/<model>/<run-id>/` path. Build the self-contained evidence manifest only
after raw predictions, the pinned external-evaluation manifests, and every declared training
corpus/manifest are frozen by SHA-256:

```bash
python -m shadowcrafter.cli eval gate \
  artifacts/evaluations/<model>/<run-id>/release-evidence.json \
  --config configs/eval/release-gates.yaml \
  --report artifacts/evaluations/<model>/<run-id>/gate-report.json
```

The strict gate recomputes accuracy, balanced accuracy, and macro-F1 from frozen raw predictions,
reports every class, and independently checks exact CTIBench contamination. It rejects legacy
caller-supplied aggregate JSON. The current CTIBench protocol is CC-BY-NC-SA-4.0 and therefore
permits only noncommercial reporting without sharing benchmark material. The 95% target is
reported as `target_95_met` and is not an Official Release blocker. A private Official
Release still requires passing safety, privacy, license, provenance, artifact-integrity,
operational, and remote-weight recovery reviews; public visibility is forbidden. Improvements
may use development evidence only; obtain a new blind test if the existing test influenced
training or selection.

## 8. Promote by immutable remote manifest

Promotion creates a new immutable remote release path and a corresponding local manifest;
it never overwrites a previous release. The remote bundle must include weights or adapters
as declared, tokenizer/processor, configuration, model card, attribution and license
material. The local evidence bundle records evaluation and safety results, provenance,
rollback target, remote locations, and a complete SHA-256 inventory.

Perform an offline restore/load test from an independent remote copy. Have
the designated reviewers sign off on data rights, upstream Ornith MIT attribution, derived
weight licensing, privacy, safety, and performance claims. The source repository's
Apache-2.0 license does not automatically relicense the upstream model, datasets, or derived
weights.

## 9. Mirror a gated release privately

Select the destination by exact model family:

| Model family | Private Hugging Face destination |
|---|---|
| ShadowCrafter-9B | `KaztoRay/ShadowCrafter-9B` |

Before upload, independently confirm the authenticated Hugging Face account, exact repository
ID, current `main` commit, and private visibility. Authenticate on the workstation with the
standard Hugging Face local auth cache (`hf auth login`). `huggingface_hub.get_token()` applies
the library's normal local-cache/`HF_TOKEN` precedence; there is deliberately no CLI token
argument. Never put a token in Git, a launchd plist, shell arguments, manifests, logs, model
cards, or the remote training image.

The exact machine-readable input is documented in
[`REMOTE_RELEASE_MANIFEST.md`](REMOTE_RELEASE_MANIFEST.md).

The checked-in remote publisher never creates a local model/checkpoint/weight file. It accepts
only a locally retained, externally SHA-256-pinned JSON manifest whose exact sorted inventory
names each remote file, byte size and SHA-256. The repository, model-family-specific immutable
remote root, release ID, existing Hub parent commit, total byte count, and five local approval
records are also frozen. Each integrity, provenance, license, privacy, and safety approval is
bound to the repository, release ID, candidate checkpoint hash, and canonical remote-inventory
hash. The license approval must prohibit commercial release and benchmark-material sharing.

The model card is part of that inventory. Its YAML front matter must identify the exact private
`Official Release`, repository, release ID and candidate hash. It must either reproduce the
frozen measured benchmark name/revision/hash/sample count and accuracy, balanced accuracy,
macro-F1 and `quality_target_met`, or state `not-yet-evaluated`, use null metric values, and give
the manifest's explicit reason. A measured result below 0.95 remains reportable and does not
block this Official tier. Evaluation integrity, contamination, license, privacy, safety,
provenance, visibility and artifact failures remain blocking.

Install the release client dependencies and invoke the publisher from the repository root:

```bash
python -m pip install -e '.[release]'
python -m shadowcrafter.cli release publish-remote-official \
  --manifest artifacts/releases/<model>/<release-id>/remote-release-manifest.json \
  --manifest-sha256 <64-lowercase-hex> \
  --ssh-key /absolute/path/to/identity.pem \
  --evidence artifacts/evaluations/<model>/<evaluation-id>/release-evidence.json \
  --gate-config configs/eval/release-gates.yaml
```

Omit `--evidence` only when the manifest and model card both explicitly say
`not-yet-evaluated`. The SSH key must be a non-symlink regular file with mode `0600`. The SSH
reader uses fixed batch-only arguments and a constant isolated remote Python command; the remote
root/path travel as bounded JSON on standard input, while the Hub credential remains only in the
local process. File and total byte caps are enforced before one `CommitOperationAdd(bytes)` Hub
commit. `parent_commit` supplies compare-and-swap protection. Before and after that commit the
repository must be private, and every committed file is streamed back from the immutable Hub
revision and re-hashed without using a filesystem cache.

On success the only receipt is stdout JSON with the repository, immutable Hub commit SHA,
release ID, local manifest SHA-256, evaluation status/target result, and byte/file counts. The
command intentionally does not write a receipt or any model bytes. Preserve that JSON in the
approved local evidence workflow. Re-running the same manifest is expected to fail after a
successful publish because its frozen parent is no longer `main`; do not weaken this race check
or silently regenerate the manifest.

After upload:

1. record the immutable Hugging Face commit SHA, publisher receipt, and UTC publication time in
   a new local post-publication record (do not mutate the pinned pre-publication manifest);
2. verify that the repository is still private and access is limited to approved identities;
3. download the published revision to a fresh isolated remote verification cache;
4. compare every expected file size and SHA-256 with the authoritative local manifest;
5. test loading that remote verification copy without relying on mutable cache state.

If anything differs, restrict access, mark the revision withdrawn, preserve evidence, and
return to the last known-good immutable remote revision. Do not “fix” a release only in the
web UI; make source/card changes locally, issue a new immutable version, and retain lineage.

## 10. GitHub synchronization and recovery

Commit and push source, configuration, tests, documentation, and approved non-sensitive
manifests to `Odytssey/ShadowCrafter` in small reviewable changes. Never commit model weights,
datasets, malware samples, private reports, tokens, or SSH material. Confirm private
visibility, minimum collaborator access, branch protection, and security-advisory support.

Periodically perform an isolated restore test using a fresh approved remote cache. A
successful test checks out the recorded Git commit, verifies every data/base-model/release
hash, loads the model and tokenizer remotely, reruns a representative evaluation, and
traces the release to its code, data, upstream revision, run, and approvals. Record the
restore date and result in the local release manifest.

See [`LOCAL_ARTIFACTS.md`](LOCAL_ARTIFACTS.md) for the full custody policy and
[`MODEL_RELEASE_POLICY.md`](MODEL_RELEASE_POLICY.md) for release and withdrawal rules.
