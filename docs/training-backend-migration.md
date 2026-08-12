# Training backend migration

This document maps Palinkle's completed Tinker experiments to the pinned Miles
and SGLang source trees. It is a migration contract, not evidence that the GPU
backend has reproduced a result yet.

## Decision

Palinkle remains the source of truth for task construction, split isolation,
agent workspaces, trajectories, rewards, remote TPU verification, profiling,
and frozen evaluation. The training runtime becomes replaceable.

- **Miles is the only active training runtime.** The pinned source has
  native Inkling and Inkling Small model code, LoRA, rendering, Megatron
  training, SGLang rollout, GRPO, and OPD.
- **SGLang is the only active rollout runtime.** The `sglang-miles` branch is
  the serving half of the Miles Inkling contract: it renders Inkling messages,
  serves the base and LoRA adapters, returns rollout log probabilities and MoE
  routed-expert IDs, and receives updated adapter tensors.
- **Laguna XS 2.1 is a parallel model arm.** The pinned SGLang source can serve
  its base checkpoint today. The pinned Miles source has no Laguna model or
  training plugin, so matched SFT, DAPT, GRPO, and OPD remain blocked on that
  port. Inkling Small remains the primary training arm until this gap closes.
- **Palinkle keeps its current harness and verifier.** Migrating the trainer
  does not authorize replacing the isolated Git workspace, patch snapshots,
  remote TPU verifier, or evidence schemas.

PRIME-RL and other Prime Intellect products are deferred. Their OPD/OPSD
implementations may be read later, but they are not dependencies, submodules,
or execution targets for this migration.

## Canonical Phase 1 foundation

The provider-neutral experiment boundary is implemented in
`opjax.pallas.experiment_foundation`. It defines the model arm, training
method, training backend, inference backend, and checkpoint-store protocols.
The concrete filesystem store works on local storage and mounted Modal
Volumes. A checkpoint package is accepted only when it contains one model or
adapter weight role, optimizer and scheduler state, RNG and data/rollout
cursors, an inference export, monotonic update and policy versions, immutable
base-model identity, and hashes for the experiment, configuration, data,
tokenizer, renderer, toolchain, and every payload file.

Publication has four recoverable boundaries: staging, immutable package
commit, validation, and `latest` promotion. The accepted pointer binds the
manifest, validation result, and acceptance record. Replacement requires the
declared parent to equal the current `latest`, and both the global update and
policy version must increase. A failed validation, interrupted publication,
payload mutation, metadata mutation, stale parent, version regression,
symlink, or overlapping artifact path cannot change `latest`.

The deterministic acceptance probe runs an uninterrupted two-update control
against save, reload, and the same second update. It compares the complete
model, optimizer, scheduler, RNG, cursor, loss, and canary-logit evidence. The
same checkpoint export is loaded through the inference-backend protocol and
must reproduce the frozen-prompt logits before promotion. Boundary fakes test
the orchestration; the live control below proves the real Miles and SGLang
paths.

The storage boundary was also exercised through two separate Modal containers
against the mounted `opjax-checkpoints-v2` Volume. Run
`ap-UEE3YCNbVK7DsA4HboYrFf` wrote, committed, reloaded, and independently read
checkpoint `modal-canary-d6ccdbe207b748f2af02f352498b8585`; its manifest,
validation, and acceptance hashes matched. This proves the mounted-Volume
publication path. It does not prove a Miles optimizer resume or SGLang model
logit match.

The generic experiment-foundation substrate and canonical Phase 1 live control
are complete. Modal run `ap-fnMquMUw7KgopdH9fsBXOf` trained a full-model
Qwen2.5-0.5B control through synchronous Miles lanes: uninterrupted two
updates, one update plus checkpoint and process exit, then a fresh-process
resume. The second loss was exactly `0.1679287552833557` in both paths. Run
`ap-A4ix8UGzBGULR1P53PTwg2` proved exact DCP model, optimizer, and RNG payloads,
semantic scheduler and iteration state, cursor state, and all 13 Hugging Face
export files. The checkpoint artifacts include a resumed Hugging Face model,
the native DCP package that also contains model and RNG shards, the next data
and rollout cursor, and a separate inference export.

Run `ap-AX5BJLP1EYQMkylzYfx9ZT` loaded that export into a fresh pinned SGLang
process. With float32 and native PyTorch fused operations, all 151,936 token
log-probabilities matched the Transformers reference within `2.29e-05`, and
the top-128 ordering was exact. This is full-vocabulary log-probability parity,
not bit-exact logits. Production bfloat16 did not meet the original stricter
`0.02` and exact-top-128 contract: the preserved diagnostic has exact top-1,
32/32 top-32 set overlap, and a maximum aligned error of `0.1651`. It is not
accepted as parity evidence.

Run `ap-ADzcuaqPhIKiHcNUEzhzU1` published the accepted checkpoint through the
provider-neutral store only after both live validations passed. The published
manifest is `6f9945363a033bad8b8113bf9b0a5cb50b666c45a593b993565684183c061dff`.
The validator hash-binds the resume evidence, float32 parity evidence, and the
rejected bfloat16 diagnostic. Canonical Phase 3 repeats this accepted lifecycle
for exact Inkling Small, then adds its agent trajectory and TPU grade. Laguna's
equivalent proof remains canonical Phase 4.

## Pinned inspection surfaces

| Surface | Revision | Relevant contract |
|---|---|---|
| Tinker SDK | `0.24.0` | Managed LoRA client, forward/backward, optimizer, sampler materialization, checkpoint state |
| Tinker Cookbook | `0.5.3` | Inkling tokenizer/renderer, supervised datum construction, rollout and RL recipes |
| Miles | `b1860dd264e17c96d5d92da96c957d88cfd3a1f8` | Inkling Small LoRA, SGLang rollout, Megatron trainer, GRPO, OPD |
| SGLang `sglang-miles` | `c80a38edcd2c7077c909a5ed925c9241e754c067` | Inkling and Laguna inference, Poolside reasoning parser, LoRA serving, routed-expert capture, dynamic adapter loading |
| Laguna XS 2.1 | `e9df9a59996d790b94b70f3fef343fe1d9e34bdf` | External 33B-total, 3B-active code-model arm; SGLang inference only at this boundary |

Run the contract audit after checkout, dependency changes, or submodule
updates:

```bash
git submodule update --init --recursive
uv run --no-default-groups --group tinker python scripts/audit_training_backends.py
```

The audit uses live Python reflection for Tinker and AST inspection for the
pinned source trees. It fails on Tinker version drift, missing git revisions,
missing files, or renamed critical symbols.

## Existing Tinker contract

| Palinkle stage | Current Tinker mechanism | Provider-neutral meaning |
|---|---|---|
| G4 SFT | Cookbook renderer and `conversation_to_datum`; rank-64 LoRA; cross-entropy; Adam | Render messages exactly, mask only intended assistant tokens, apply one optimizer update per frozen batch |
| G5 DAPT | Raw source tokenization; deterministic lane-aware packing; next-token cross-entropy; DAPT state continued through identical G4.2 SFT | Preserve token order, boundaries, EOS policy, loss mask, lane weights, optimizer state, and parent checkpoint identity |
| G6 GRPO | Materialize sampler weights; collect bounded agent trajectories; TPU-verify patches; construct behavior-logprob, token-mask, and advantage tensors; importance-sampling loss | Sample from the exact policy checkpoint, retain token-level behavior probabilities, assign group-relative reward only to sampled response tokens, then update the same policy |
| Evaluation | Separate immutable task package and remote TPU verifier | Keep evaluation outside the training provider and never return hidden benchmark feedback during a rollout |
| Evidence | Preparation hash, config hash, row IDs, token counts, per-step events, checkpoint identity, sampler identity, TPU artifacts | Every backend must emit enough data to reconstruct the update and attribute failures |

The Tinker-specific code remains in `src/opjax/pallas/training.py`,
`g5_training.py`, `g6_rollout.py`, and `g6_training.py`. These files define the
behavior to reproduce; they are not the abstraction boundary for a new
backend.

## Miles translation

| Palinkle object or operation | Miles surface | Required adaptation |
|---|---|---|
| Inkling Small model and rank-64 LoRA | `miles_plugins/models/inkling/model.py`, `lora.py`, and `scripts/run_inkling.py` | Convert the official checkpoint, set rank and alpha explicitly, and prove which linear modules correspond to Tinker's attention, MLP, and unembedding targets |
| `tml_v0` rendered messages | `render_inkling_messages_to_ids` | Compare token IDs and supervised masks for a frozen corpus; semantic similarity is insufficient |
| SFT `Datum` | Miles `Sample` plus `sft_loss_function` and SFT rollout | Map prompt/response tokens and loss mask without re-rendering or truncation |
| Packed DAPT sequence | Miles `Sample` with a full next-token loss mask | Add a raw-token data source that bypasses chat rendering and preserves G5 packing exactly |
| Tinker sampler materialization | `RolloutManager` and SGLang rollout | Record the actor version served by every rollout and reject stale weight versions outside the frozen policy-lag rule |
| G6 trainable turn | Miles `Sample` | Map tokens, response length, reward, loss mask, rollout log probabilities, and metadata one-for-one |
| GRPO advantage and update | `compute_advantages` and `policy_loss_function` | Freeze group construction, normalization, clipping, KL, and token aggregation before comparison |
| Gate 7 OPD | `on_policy_distillation.py` and OPD loss | Add the teacher endpoint, teacher-token log probabilities, top-k policy, and reverse-KL settings to the run manifest |
| Checkpoint and sampler export | Megatron checkpoint plus Inkling LoRA export | Hash both training state and served adapter; prove a reload produces identical logits on canary prompts |

The pinned Miles script states that full Inkling Small and its LoRA mode were
validated upstream on 32 H200 GPUs. This is an upstream capability claim, not
a Palinkle result. The four-layer CI checkpoint is also not proof that the
official full checkpoint converts or trains correctly in our environment.

## SGLang translation

| Palinkle object or operation | SGLang surface | Required adaptation |
|---|---|---|
| `tml_v0` generation prompt | `render_inkling_messages` and `InklingTokenizer` | Compare exact token IDs, reasoning-effort framing, tool-call encoding, assistant prefix, and stop behavior |
| Inkling Small base policy | `InklingForConditionalGeneration` | Verify checkpoint identity and base logits before enabling an adapter |
| Rank-64 policy adapter | `LoRAAdapter`, `LoRAManager`, and Inkling-specific LoRA layers | Load the exact Miles-exported names and shapes; reject partial adapter loads |
| G6 generation request | `GenerateReqInput` | Request token IDs, log probabilities, routed experts, and the exact adapter version for every sampled turn |
| MoE routing replay | routed-expert capturer and Miles `rollout_routed_experts` | Validate row count against the rendered and media-expanded sequence; retain the complete top-k expert trace |
| Policy update | Miles dynamic adapter loading through the SGLang engine | Pause generation, switch one complete adapter version, flush affected caches, and prove the served version before resuming |
| Rollout evidence | token output and response metadata | Preserve prompt tokens, response tokens, stop reason, behavior log probabilities, route trace, adapter version, and server configuration |

The joint [SGLang and Miles Inkling report](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support/)
explains why ordinary rollout parity is insufficient for this MoE. Their
contract adds training-side kernels aligned with SGLang arithmetic, Rollout
Routing Replay for selected expert IDs, and adapter-only synchronization. The
reported train-rollout KL near `1e-3` is a useful diagnostic target, not a pass
condition copied into Palinkle without measurement.

## Laguna XS 2.1 baseline

The [model card](https://huggingface.co/poolside/Laguna-XS-2.1) and
[technical report](https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf)
describe a 33B-total, 3B-active mixture-of-experts code model. Inkling Small is
276B total and 12B active. At equal precision, Laguna therefore has about
`8.36x` less weight residency and `4x` less active forward compute. This is a
model-capacity estimate, not an end-to-end throughput claim.

The estimate held at the deployment boundary. The exact BF16 Laguna revision
served with tensor parallelism 1 on one H200. After CUDA graph capture,
`nvidia-smi` reported 106,044 MiB used out of 143,771 MiB. The frozen SGLang
configuration used a 32,768-token context and the `poolside_v1` reasoning
parser.

The first frozen benchmark used the same 16 near-held-out tasks, seed 0, three
model calls, temperature 0.2, top-p 0.95, hidden verifier, and TPU evidence
contract as the existing model arms. It produced:

| Result | Count |
|---|---:|
| Profile-verified | 0/16 |
| Candidate failures | 16/16 |
| Infrastructure failures | 0/16 |
| Non-empty patches | 0/16 |

The paired six-call run captured true snapshots after calls 3 and 6. Both
horizons scored 0/16, all 32 verifier units were candidate failures at
`artifact_contract`, and no infrastructure or recovery event occurred. All
turn-3 and turn-6 patches were empty, producing 16 fail-to-fail transitions.

Every trajectory used calls 1–5 to read `instruction.md`, `PALLAS_API.md`,
`kernel.py`, `dev_check.py`, and list the workspace. Ten used call 6 to reread
the instruction; the others performed another repository search, inspected Git
history, or emitted one malformed action. No trajectory edited or submitted.
The first three actions and patch matched the original three-call run for all
16 tasks, although only four textual reasoning prefixes were byte-identical.
This exposes residual SGLang generation nondeterminism while confirming that
the observable agent behavior and submitted artifacts were stable.

This is a valid result for both frozen call limits. It shows that additional
calls alone do not correct the model-harness interaction. It does not isolate
Laguna's underlying Pallas ability from its serial inspection policy. The
result remains a baseline; changing the initial observation or action protocol
would define a separate harness experiment.

Prime Intellect's
[Laguna Jacobian-lens artifact](https://huggingface.co/PrimeIntellect/Laguna-XS.2-jlens)
adds a useful diagnostics surface for layer convergence, representation
similarity, and module-level probes. It is not a stronger generation
checkpoint. Any use must first establish exact checkpoint and corpus alignment
with the model arm under test.

## Conformance gates

No result is compared with Tinker until the applicable lower gate passes.

1. **Checkout audit:** both Miles and SGLang gitlinks resolve to the revisions
   above; the source-contract audit passes with clean submodules.
2. **Renderer parity:** frozen prompts produce identical token IDs, stop rules,
   assistant loss masks, and truncation decisions.
3. **Forward parity:** the converted base checkpoint produces sufficiently
   close logits and next-token rankings on frozen short and long canaries.
4. **SFT parity:** one rank-64 LoRA update on one frozen batch matches trainable
   parameter coverage, token-weighted loss, optimizer settings, and direction
   of logit change.
5. **DAPT parity:** the frozen G5 packs conserve tokens and masks; one update
   reduces held-out DAPT NLL without chat rendering.
6. **Rollout parity:** the same checkpoint and seed preserve task visibility,
   turn accounting, action parsing, patch hashes, behavior log probabilities,
   and TPU reward classification.
7. **GRPO reproduction:** rerun the G6 S0 lane and its frozen 16-task
   evaluation before changing the algorithm.
8. **Distillation:** run Miles OPD from the reproduced S0 state. OPSD and RMSD
   remain separate future algorithm ports rather than dependencies of this
   migration.

Each gate must produce a machine-readable manifest that identifies the source
revisions, environment, model conversion, renderer, tokenizer, data release,
optimizer, rollout policy, verifier, and checkpoint hashes.

## Known gaps

- Miles has OPD but no native OPSD symbol at this revision.
- Neither Miles nor SGLang exposes RMSD as a named algorithm.
- Miles' Inkling LoRA target set is not yet proven equivalent to Tinker's
  `train_mlp=True`, `train_attn=True`, and `train_unembed=True` contract.
- Tinker's managed checkpoint and sampler identities do not directly map to a
  Megatron checkpoint plus SGLang adapter; reload and logit parity are required.
- SGLang's routed-expert trace must be proven aligned with Miles after token
  rendering, packing, and any media expansion.
- The official full Inkling Small checkpoint conversion has not been run in
  this repository.
- Miles has no Laguna XS 2.1 model, checkpoint conversion, LoRA, or optimizer
  plugin at the pinned revision. SGLang inference support alone does not prove
  train-inference parity for Laguna.

Canonical Phase 3 is therefore the next model-specific boundary: Miles
renderer and base-logit parity for Inkling Small, followed by a one-batch SFT
canary, exact continuation, export, SGLang reload, one agent trajectory, and a
TPU grade. Laguna's six-call inference baseline is complete; canonical Phase 4
starts with its Miles model-plugin port. Once both arms pass the same checks,
the experiment factory can cross the model arm with the existing SFT, DAPT,
GRPO, and OPD recipes without changing the data, harness, reward, or
evaluation.
