---
language:
  - en
  - ko
license: other
base_model: ornith-ai/Ornith-1.5-9B
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
tags:
  - cybersecurity
  - defensive-security
  - qlora
  - research
---

# ShadowCrafter-9B

> **Status: unreleased research candidate.** No ShadowCrafter-9B checkpoint has yet
> passed the project's release gates. This card does not claim that 94% accuracy has
> been achieved, and it is not evidence that a deployable model exists.

ShadowCrafter-9B is a planned defensive-first cybersecurity assistant developed by
**Odytssey** as a fine-tune of
[`ornith-ai/Ornith-1.5-9B`](https://huggingface.co/ornith-ai/Ornith-1.5-9B).
Odytssey did **not** create the upstream Ornith model. The ShadowCrafter name identifies
the downstream training, evaluation, safety, and release work only.

The private Hugging Face release target is
[`KaztoRay/ShadowCrafter-9B`](https://huggingface.co/KaztoRay/ShadowCrafter-9B). The
canonical source repository is the private
[`Odytssey/ShadowCrafter`](https://github.com/Odytssey/ShadowCrafter) repository.

## Model details

- **Developer:** Odytssey
- **Upstream model:** `ornith-ai/Ornith-1.5-9B`
- **Relationship:** planned QLoRA fine-tune; not an independently trained foundation model
- **Intended languages:** English and Korean, subject to per-language evaluation
- **Intended access:** private, approved research and defensive operations only
- **Release revision:** not assigned
- **Training data snapshot:** not finalized
- **Evaluation snapshot:** not finalized

The project configuration records the upstream Ornith model as MIT-licensed. That is an
upstream attribution, not a blanket statement that every future ShadowCrafter weight,
dataset, or bundled artifact is MIT-licensed. Before a checkpoint is released, the exact
upstream revision, its license text, dataset obligations, notices, and the license for the
derived weights must be recorded in the immutable local release evidence manifest. The source code in
the ShadowCrafter GitHub repository is separately licensed under Apache-2.0.

## Intended use

Subject to authorization, data-handling controls, and human review, the model is intended
to assist with:

- vulnerability and configuration review on systems the operator owns or is authorized to test;
- CVE/CWE mapping, affected-version triage, and remediation research;
- defensive malware triage, behavior summarization, and IOC extraction in an isolated lab;
- evidence-grounded security-assessment and incident-report drafts;
- defensive knowledge-base search and synthesis;
- SIEM alert enrichment and draft SOAR playbooks that remain in dry-run or approval-pending mode.

Model output is advisory. It must not be the sole basis for a security, legal, employment,
or enforcement decision. Citations, affected versions, detection logic, and recommended
actions require verification by a qualified reviewer.

## Prohibited and restricted use

The model is not intended for unauthorized access, credential theft, phishing, persistence,
evasion, destructive activity, exploit weaponization, malware creation or deployment, or
targeting third-party systems. Dual-use evaluation is limited to explicitly authorized,
isolated sandboxes with a documented scope, stop conditions, logging, and human oversight.

The model must not autonomously execute shell commands or make changes in EDR, SIEM, SOAR,
identity, network, cloud, ticketing, or production systems. High-impact actions such as
blocking, isolation, deletion, account disablement, key rotation, and external notification
require explicit authenticated human approval. The complete policy is in the project's
[`ACCEPTABLE_USE.md`](https://github.com/Odytssey/ShadowCrafter/blob/main/ACCEPTABLE_USE.md).

## Training status and planned runs

The checked-in candidate configuration currently specifies QLoRA with 4-bit NF4 loading,
bfloat16 compute, a 4,096-token maximum sequence length, rank 32, alpha 64, dropout 0.05,
gradient checkpointing, and completion-only loss. Packing, padding-free batches, and the
template thinking path are disabled to preserve the audited prompt/completion boundary.
These remain candidate run parameters, not a statement that full training completed
successfully.
Every run must pin the base-model revision, tokenizer revision, code commit, environment,
approved data snapshot, seed, and artifact hashes.

Training data will be limited to sources that pass provenance, license, privacy, secret,
malware-handling, quality, deduplication, and evaluation-contamination reviews. Public
availability alone does not grant permission to train on or redistribute a source.

### Compatibility preflight

A one-optimizer-step QLoRA compatibility run completed on the private H100 training server
against upstream revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. It produced a
finite training loss of `3.349` and an evaluation loss of `3.565`; the resulting adapter
contained 496 finite LoRA tensors and no non-LoRA tensors. The adapter weight SHA-256 is
`73ea80c1ae519001bcc58d390a102872c1c040758e0dfa50c9427fadc52db1aa`, and the adapter
configuration SHA-256 is
`225694a7f46b387b9363bab28a2eb6eb7742ba9cc11ca0dc78f1394f4cf9fc28`.

This was a synthetic one-step engineering fixture. It demonstrates loader, adapter,
optimizer, save, and verification compatibility only. It is not a cybersecurity training
run, benchmark result, release checkpoint, or evidence of useful model quality.

## Evaluation and the 94% reporting target

No measured performance is reported yet.

The project's `0.94` value is a predeclared reporting target for named, leakage-free held-out
tasks; it is not a universal cybersecurity accuracy figure or a private-upload threshold. Current planned evaluations cover CVE
triage accuracy, CWE mapping macro-F1, malware-behavior accuracy, detection-rule schema
validity, and safety-policy accuracy. Minimum sample counts, confidence-bound treatment,
split hashes, zero-contamination requirements, slice results, and failure cases must
accompany any future result.

Below-target checkpoints may be uploaded as private **Experimental Releases** only when the
model card reports the actual scores, evaluation snapshot, and `94% target met: no`.
Generative report quality, factual grounding, calibration, useful handling of legitimate
defensive requests, and resistance to unsafe requests require separate evaluation. A model
that misses any mandatory safety, provenance, privacy, licensing, or artifact-integrity gate
must not be published. Accuracy alone is non-blocking. See the repository's
[`docs/EVALUATION.md`](https://github.com/Odytssey/ShadowCrafter/blob/main/docs/EVALUATION.md)
and
[`docs/MODEL_RELEASE_POLICY.md`](https://github.com/Odytssey/ShadowCrafter/blob/main/docs/MODEL_RELEASE_POLICY.md).

## Known limitations and risks

Because no gated release exists, task capabilities and production hardware requirements
are not yet established. Future candidates may hallucinate CVE identifiers, cite stale or inapplicable
advisories, misunderstand version ranges, produce invalid detection syntax, miss malicious
behavior, over-refuse benign work, or comply with unsafe requests. Performance may vary by
language, product, vulnerability class, time period, and data source. Model output can also
contain sensitive material supplied in context; operators must apply access control,
redaction, retention, and audit requirements.

Do not execute generated code, queries, rules, or remediation steps without independent
review and a safe test. Treat retrieved documents, logs, and tool output as untrusted data,
not instructions.

## Artifact custody and publication

At the operator's direction, base-model and checkpoint weight files are not retained on the
local workstation. Local custody is limited to source code, configurations, approved data,
evaluation evidence, and checksum/provenance manifests. The private training server retains
working base-model and checkpoint artifacts; approved release adapters will be mirrored to
the private Hugging Face repository and verified against their manifests. A manifest is not
a substitute for a recoverable weight copy, so server and Hub retention must be verified
before any remote artifact is removed.

Security issues must be reported through the private process described in
[`SECURITY.md`](https://github.com/Odytssey/ShadowCrafter/blob/main/SECURITY.md), without
posting credentials, personal data, live payloads, or malware samples.
