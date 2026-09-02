# Frozen evaluation and the 95% performance gate

ShadowCrafter does not accept a hand-written metrics JSON as release evidence. The strict gate
recomputes scores from raw, frozen predictions and an immutable external benchmark. Missing,
mutable, contaminated, malformed, unpinned, or aggregate-only evidence fails closed.

Neither ShadowCrafter model currently has a completed prediction bundle that passes this gate.
Accordingly, the project has not demonstrated a 95% result for ShadowCrafter-9B.

The checked-in protocol supports a **noncommercial, private Official Release** evaluation
against `AI4Sec/cti-bench`. CTIBench is licensed `CC-BY-NC-SA-4.0`; its material may not be used
as commercial-release evidence or shared through the model repository. A measured score may be
reported with the exact benchmark identity and limitations. Publication still requires separate
artifact-integrity, provenance, upstream/data-license, privacy, and safety approvals. Attribution
and ShareAlike review are required before benchmark-derived material is shared.

## Fixed CTIBench snapshot

[`configs/eval/release-gates.yaml`](../configs/eval/release-gates.yaml) pins all of the following:

- upstream revision `9237e1636ee3e168fbe5ebdcc1c571de0525e568`;
- snapshot-manifest, adapter-manifest, canonical case-file, and dataset SHA-256 values;
- the `eval-only-v2` post-filter total of 5,533 scorable cases;
- exact task counts: CTI-ATE 56, CTI-MCQ 2,488, CTI-RCM 995,
  CTI-RCM-2021 999, and CTI-VSP 995;
- the reviewed CTIBench repository identity, evaluation-only policy, and NC license.

The 5,533 count is intentionally lower than the upstream 5,560 ground-truth-bearing rows.
Twenty-seven inputs fail the repository's defensive content or provenance filters. Fifty
additional CTI-TAA rows have no ground truth and never enter scoring. Changing any reviewed
filter output requires a new adapter artifact and an explicit config/hash review; silently
evaluating a subset is not allowed.

## Evidence bundle

The gate accepts a bundle-local `release-evidence.json`. Every referenced path must be relative,
must remain inside the bundle, and must not traverse a symlink. Each file has a frozen SHA-256;
JSONL files also have an exact record count. The bundle contains only code/data/evaluation
evidence and manifests locally. Base-model and checkpoint weights are identified by immutable
manifests and hashes; the evaluation gate itself does not copy weights, while the separate
post-run mirror stores them below the Git-ignored `local_mirror/` boundary.

Required evidence includes:

- candidate/model family, exact Ornith model ID and commit, candidate checkpoint hash,
  checkpoint-manifest hash, training-run-manifest hash, ShadowCrafter Git commit, and clean-tree
  assertion;
- evaluator code revision, environment, prompt-template and decoding-config hashes, seed,
  timezone-aware run timestamps, deterministic decoding, retained raw outputs, and confirmation
  that the answer key was hidden from the model;
- CTIBench snapshot manifest, canonical adapter manifest and eval-case file with their pinned
  hashes and counts;
- one or more complete prepared training JSONL files plus their prepared-data manifests;
- an explicit NC/private-research license scope and local license-review record hash;
- the frozen raw-prediction JSONL and its hash/count.

The evidence manifest itself is hashed into the generated report. Evidence signing and reviewer
identity are organizational controls outside this numerical implementation; the report must stay
access-controlled because per-class labels can expose benchmark answer information.

## Frozen prediction contract

Each prediction line contains only the case identity and raw model output. Expected answers,
caller-computed scores, and caller-selected correctness flags are forbidden:

```json
{"schema_version":1,"case_id":"ctibench:cti-mcq:000000","raw_output":"B","raw_output_sha256":"<64 lowercase hex>"}
```

The file order must exactly match the frozen case order. Missing, reordered, duplicate, or extra
case IDs fail. Each raw-output hash and the whole-file hash are verified before scoring. A trusted
task-specific parser accepts only the declared CTIBench answer syntax; explanations and malformed
outputs become an invalid prediction rather than being leniently credited.

## Metric gate

The evaluator recomputes these three metrics from the frozen answer key and raw outputs:

| Metric | Threshold |
|---|---:|
| Accuracy | `>= 0.95` |
| Balanced accuracy | `>= 0.95` |
| Macro-F1 | `>= 0.95` |

All three targets are measured both on the complete 5,533-case benchmark and independently on
every scored CTIBench task. Missing one or all 95% targets does **not** block a private
`Official Release`; the report records `quality_target_met`/`target_95_met: false` and every
shortfall. It cannot be labelled a Qualified Release. The generated report includes, for every
scope, the sample count, reference/observed class counts, and per-class support, predicted count,
true positives, precision, recall, and F1.

This is an exact frozen-protocol statement, not a claim that ShadowCrafter has “95% security
accuracy” in general. It does not cover report quality, real-world vulnerability discovery,
malware verdicts, SIEM/SOAR safety, calibration, privacy, or exploit resistance. Those need
separate predeclared evaluations and reviewers.

## Contamination gate

The gate does not trust a declared contamination percentage. It loads every supplied prepared
training record, verifies its canonical provenance checksum and training-policy manifest, then
recomputes overlap against the CTIBench cases. It rejects:

- any training record sourced from CTIBench or marked as a benchmark holdout;
- exact normalized matches to the retained upstream-prompt, input, or rendered-input hashes;
- a complete normalized benchmark input embedded in a training message;
- duplicate training record IDs across declared corpora;
- a prepared-data manifest whose train hash/count or overall dataset fingerprint is inconsistent.

The required overlap count is exactly zero. This automated scan covers deterministic provenance,
exact normalized content, and full-input containment. Semantic near-duplicate review remains a
separate mandatory human/data-governance control; do not describe the automated scan as proof
that no conceptual leakage exists.

## Running the gate

From the repository root:

```bash
python -m shadowcrafter.cli eval gate \
  artifacts/evaluations/<model>/<evaluation-id>/release-evidence.json \
  --config configs/eval/release-gates.yaml \
  --report artifacts/evaluations/<model>/<evaluation-id>/gate-report.json
```

The report path is created exclusively and is never overwritten. Exit status zero means the
frozen evaluation, zero-contamination check, and mandatory integrity/provenance/license/privacy/
safety evidence passed. It does not mean 95% was reached: inspect `quality_target_met` and the
reported scores. Authorization is limited to a noncommercial **private Official Release**;
public visibility and commercial publication remain forbidden.

The workstation-local directory publisher is intentionally disabled because local base/model/
checkpoint files are transferred only by the isolated project-mirror workflow. The supported
`release publish-remote-official` path reads each exact manifest-allowlisted remote file into
bounded process memory, submits one private Hub commit, and streams that immutable revision back
for size/SHA-256 verification without a filesystem cache. It resolves Hugging Face credentials
only from standard local auth sources and never sends them over SSH. The manifest SHA pin,
parent-commit race check, private visibility, model-card reporting, approvals, provenance,
integrity, privacy, safety, and license constraints fail closed; the measured 0.95 target remains
nonblocking for this Official tier. See [`REMOTE_WORKFLOW.md`](REMOTE_WORKFLOW.md#9-mirror-a-gated-release-privately)
for its exact CLI and manifest contract. Publication continues to stream the immutable remote
release so a stale local mirror cannot alter what is uploaded.

If a blind example, answer, or detailed failure is used for training, prompt selection,
hyperparameter search, checkpoint selection, or error-driven retraining, the benchmark is no
longer blind for that candidate. Obtain a new independent evaluation protocol; do not update
hashes merely to make an existing candidate pass.
