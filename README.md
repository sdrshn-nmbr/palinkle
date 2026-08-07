# Palinkle

Palinkle trains Inkling Small to write correct and fast
[JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernels for TPUs.
The name combines Pallas and Inkling.

The project has one rule: a result only counts when the generated kernel is
correct, runs through real Pallas lowering on a TPU, and leaves enough evidence
for someone else to check it.

## Sources

[`config/pallas/sources.json`](config/pallas/sources.json) pins each code and
data source to a revision. A source listed here is not automatically approved
for training.

### Models

- **Inkling:** [release](https://thinkingmachines.ai/news/introducing-inkling/),
  [model card](https://thinkingmachines.ai/model-card/inkling/), and
  [weights](https://huggingface.co/thinkingmachines/Inkling). This was the
  original base model and is now a historical comparison.
- **Inkling Small:**
  [release](https://thinkingmachines.ai/news/inkling-small/),
  [model card](https://thinkingmachines.ai/model-card/inkling-small/), and
  [weights](https://huggingface.co/thinkingmachines/Inkling-Small). This is the
  current base model.

### Training sources

- **JAX and Pallas:**
  [documentation](https://docs.jax.dev/en/latest/pallas/index.html) and
  [pinned source](https://github.com/jax-ml/jax/tree/aaf50c6a71d3bde4188c1836323f3a0ae9cb9e7f).
  Only approved documentation, implementation, and test paths may enter the
  training data.
- **Tokamax:**
  [pinned repository](https://github.com/openxla/tokamax/tree/b33bdfa64a78cc16193f3c77dd223bb040aeebf4),
  used for approved Pallas and Mosaic kernel examples.
- **MaxText:**
  [pinned kernel directory](https://github.com/AI-Hypercomputer/maxtext/tree/17c7172720ca813b05e5ea248dedd78a0c64612e/src/maxtext/kernels),
  used for approved production kernel examples.
- **Hugging Face:** the data pipeline records each row's license, source
  revision, duplicate status, and split. The broad-kernel dataset contains 830
  approved rows from 95 repositories. Its
  [manifest](data/pallas/runs/g3-hub-dapt-admission/manifest.json) defines the
  dataset.

### Evaluation-only sources

- **JAXBench:**
  [pinned benchmark](https://github.com/AI-Hypercomputer/accelerator-agents/tree/6b6c44293c43976032ba12d2f72d6bebeaf2394f/JAXBench).
  Its implementations are held out from training.
- **PallasBench:**
  [pinned repository](https://github.com/Tyronita/PallasBench/tree/30a6ee07fd4923f3877906a94002d994e972d6fe)
  and
  [pinned dataset](https://huggingface.co/datasets/EvanOLeary/pallasbench-unified/tree/b0c928c21101a96ddee17682d897b8897fa27740).
  It is used for evaluation and data discovery, not training.

The coding harness uses
[DeepSWE](https://github.com/datacurve-ai/deep-swe),
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent),
[Chex](https://github.com/google-deepmind/chex), and
[Tinker](https://tinker-docs.thinkingmachines.ai/). The multi-turn training
reference is [Kevin](https://arxiv.org/abs/2507.11948), with a local extraction
in [`kevin32b.md`](kevin32b.md). `composer2.md` contains the Composer 2 notes.

The open training-backend migration uses pinned git submodules for
[Miles](references/miles) and its [SGLang](references/sglang) rollout runtime.
Miles provides the Inkling Small model, LoRA, Megatron training, GRPO, and OPD;
the `sglang-miles` branch provides Inkling rendering, inference, routed-expert
capture, and adapter serving. See the
[`training backend migration`](docs/training-backend-migration.md) for the
mapping and conformance gates.

## Current result

G4.2 is the first checkpoint that clearly improved Pallas kernel generation.
It is still a small result, not a general or fast kernel-writing agent.

| Model | Valid after 3 calls | Valid after 6 calls |
|---|---:|---:|
| Inkling Small base | 0/12 | 4/12 |
| G4.1 SFT | 0/12 | 3/12 |
| G4.2 repair SFT | **7/12** | **7/12** |

The 12 cases are four tasks—add, matrix multiplication, RMSNorm, and row
sum—sampled with three model seeds. They are not 12 independent tasks.

Across the same task, seed, and call-limit pairs, G4.2 beat the base model in
10 of 24 cases and lost none. No task family became worse after six calls.
Speed did not meaningfully improve:

- median speedup was `1.0039x` after three calls and `0.9998x` after six;
- the best valid sample reached `1.0825x`;
- row sum remained unsolved; and
- the extra three calls did not fix any failed G4.2 runs.

The result shows better Pallas syntax, API use, and solution structure. It does
not yet show broad transfer, learned repair, or kernel optimization skill. The
full result is in
[`g42-final-results.json`](data/pallas/runs/g42-final-results.json).

Gate 5 tested domain-adaptive LoRA on the corrected 854-row broad-kernel
corpus. It improved held-out corpus likelihood but hurt the frozen agent task:

| Model | Profile-verified kernels on 16 unseen tasks |
|---|---:|
| Inkling Small base | 0/16 |
| G4.2 SFT (S0) | **8/16** |
| DAPT only (D0) | 0/16 |
| DAPT then identical SFT (S1) | 1/16 |

D0 reduced held-out mean NLL from `1.10265` to `0.68093` (`-38.25%`), so the
domain-modeling intervention worked on its own objective. It did not transfer
to three-call kernel generation. S1's only valid kernel was binary add at
`0.99842x` XLA. The full result is in
[`g5-results.json`](data/pallas/runs/g5-results.json).

Gate 6 then ran matched online GRPO from S0 and S1. Each lane used 32
non-held-out repair tasks, 16 trajectories per task, four calls per trajectory,
and eight optimizer updates. Training-time reward improved, but the primary
held-out result did not:

| Model | Profile-verified kernels on 16 unseen tasks |
|---|---:|
| G4.2 SFT (S0) | **8/16** |
| GRPO from S0 (R0) | 7/16 |
| DAPT then SFT (S1) | 1/16 |
| GRPO from S1 (R1) | 2/16 |

R0 preserved seven S0 wins, lost softmax, and gained no task. R1 preserved
binary add and added SiLU gate, but remained five tasks behind R0. Neither
checkpoint reached `1.05x` XLA: median verified speedup was `0.98945x` for R0
and `1.01017x` for R1. The full result is in
[`g6-results.json`](data/pallas/runs/g6-results.json).

Laguna XS 2.1 now exists as an external base-model arm through SGLang. Its
paired six-call run captured and graded true prefixes at calls 3 and 6:

| Horizon | Profile-verified | Non-empty patches | Infrastructure failures |
|---|---:|---:|---:|
| k=3 | 0/16 | 0/16 | 0/16 |
| k=6 | 0/16 | 0/16 | 0/16 |

Every trajectory read the instruction, API guide, starter kernel, and public
check in separate calls. Most then listed files and reread the instruction.
None edited or submitted. This is a real agent-policy result for both horizons,
but it does not isolate Pallas competence from Laguna's one-file-per-call
inspection policy. The preserved result is in
[`laguna-xs-21-k6-result.json`](data/pallas/runs/laguna-xs-21-k6-result.json).

## What counts as a valid kernel

A generated kernel must:

1. match an independently written numerical specification;
2. use Pallas instead of falling back to ordinary JAX;
3. lower normally on a real TPU without `interpret=True`;
4. run safely at the full declared shape;
5. leave compiler and profiler evidence; and
6. be compared with XLA only after it passes the first five checks.

The model works in a temporary Git repository and submits a patch. It cannot
see the hidden verifier or reference solution. Training tasks may return one
short failure message. Evaluation tasks return no hidden feedback while the
model is working.

The verifier returns:

- `1` for a valid Pallas kernel;
- `0` when the candidate fails; and
- `-1` when the test system fails for a reason unrelated to the candidate.

Speed is recorded separately. A fast wrong answer gets no credit.

## Data rules

- JAXBench code never enters training data.
- PallasBench code never enters training data.
- The G4 SFT dataset has 32 TPU-verified kernels across eight operation
  families. This was enough to run one small experiment, not enough to claim
  broad coverage.
- The G4.2 dataset has 32 verified six-action runs and 192 training rows. The
  192 rows are prefixes of those 32 runs, not independent repairs.
- The corrected broad-kernel dataset has 854 approved rows: 179 clean
  JAX/Tokamax/MaxText rows and 675 admitted Triton rows. The earlier 830-row
  count used a superseded base that still contained 46 forbidden PallasBench
  rows.

Every SFT kernel must pass full-shape correctness on fixed seeds, real TPU
lowering, profiling, license checks, duplicate checks, and JAXBench overlap
checks.

## Repository map

| Path | Contents |
|---|---|
| [`config/pallas`](config/pallas) | Experiment, source, split, harness, and evaluation settings |
| [`src/opjax/pallas`](src/opjax/pallas) | Data, training, agent, verifier, and evaluation code |
| [`tests/pallas`](tests/pallas) | Local and adversarial tests |
| [`tests_tinker/pallas`](tests_tinker/pallas) | Tinker and agent integration tests |
| [`environments/pallas-eval`](environments/pallas-eval) | Isolated TPU test environment |
| [`data/pallas/runs`](data/pallas/runs) | Run manifests and evidence |
| [`references/miles`](references/miles) | Pinned Miles runtime, Inkling Small model, LoRA, rollout, GRPO, and OPD source |
| [`references/sglang`](references/sglang) | Pinned `sglang-miles` rollout, Inkling and Laguna inference, Poolside reasoning parser, LoRA, and route-capture source |
| [`docs/model-factory`](docs/model-factory) | Earlier model-training experiments |
| [`archive`](archive) | Old plans, references, and work logs kept for provenance |

The current code is in `src/opjax/pallas`. The broad plan in
[`archive/opjax.md`](archive/opjax.md), `composer2.md`, and the model-factory
documents is historical context, not the current plan.

## Run locally

Palinkle uses Python 3.12 and `uv`.

```bash
uv sync
uv run pytest -q
uv run opjax-pallas validate-contracts
```

Useful commands:

```bash
uv run opjax-pallas --help
uv run opjax-pallas validate-corpus --help
uv run opjax-pallas-g42-agent --help
uv run opjax-pallas-g42-experiment --help
uv run opjax-pallas-g5-corpus --help
uv run opjax-pallas-g5-experiment --help
uv run --no-default-groups --group tinker python scripts/audit_training_backends.py
```

Local tests do not prove that a kernel works on a TPU. TPU runs use the pinned
cloud environment and produce evidence manifests.

## What we learned

- Passing a JAX correctness test does not prove Pallas skill. A model can copy
  the baseline or return ordinary JAX.
- Showing the reference implementation encourages copying. When references
  were removed and copied answers lost credit, the earlier LoRA advantage
  disappeared.
- `interpret=True` runs the kernel body as a JAX loop. It does not prove normal
  Pallas lowering on a TPU. Five of the first six reported successes used it.
- Lower training loss does not prove executable code. The first Pallas SFT
  model produced Pallas-looking code with reversed `BlockSpec` arguments and
  incomplete kernels.
- A failed run is not always the model's fault. One TPU failure came from a
  stale lock in the evaluator.
- Row count is not the same as data diversity. The 32-row threshold allowed a
  small test; it did not prove that the dataset was large enough.
- Multi-turn training helped, but the current runs were built around known
  solutions. The result does not yet separate real feedback-driven repair from
  seeing the solution pattern and learning the shell format.
- Add and dense matrix multiplication are useful basic tests but weak speed
  tests because XLA already handles them well.
- Lower domain validation loss is not agent capability. Gate 5 reduced DAPT
  validation NLL by 38.25%, while D0 remained at 0/16 and S1 regressed from
  the S0 control's 8/16 to 1/16.
- High online reward is not held-out improvement. Both Gate 6 lanes solved most
  training repairs after feedback, but R0 fell from S0's 8/16 to 7/16 and R1
  only recovered from S1's 1/16 to 2/16.
- For small elementwise kernels, end-to-end latency can hide device behavior.
  Gate 5 binary add spent about 5.54 microseconds in the TPU program but about
  53.21 microseconds in the blocking region, so its 0.99842x end-to-end result
  is dispatch and synchronization dominated.
- Historical G4.2 and G4.3 scores used a lossy action parser and underspecified
  task prompts. They remain useful legacy diagnostics, but they are not valid
  capability baselines for the repaired evaluator.

The main lesson is simple: keep the claim, the training change, and the
evidence separate. If any one changes, score the result again.

## Next step

Phase 1 repaired the evaluator boundary. Phase 2 is a new DeepSWE-style
benchmark release with exact task contracts and JAXBench-like, independently
specified tasks, weighted toward compound kernels with measured XLA headroom.
Legacy G4.2 and G4.3 scores will not be relabeled; all model arms must run again
through the new contract.

Provider-neutral checkpoint storage, Miles resume parity, and SGLang logit
parity move to Phase 3. The backend mapping remains in
[`training-backend-migration.md`](docs/training-backend-migration.md).

The JAXBench v5e check found one possible performance task: a corrected
Megablox grouped matrix multiplication ran at `1.147x` XLA speed across three
profiled runs. The other seven optimized references did not run as fair
default-shape comparisons in one shared environment. This result remains
provisional until the runtime and test setup are frozen. See the
[`headroom manifest`](data/pallas/runs/jaxbench-v5e-headroom/manifest.json).

## Worklog

This table is the human-readable project log. New entries are appended. If a
result is overturned, a later entry records the correction. The manifests in
[`data/pallas/runs`](data/pallas/runs) are the source of truth.

| Date | Work | Result |
|---|---|---|
| 2026-07-15 | Started with a personalized Inkling coding model trained on exported agent sessions. The scope grew into a general model factory, RL, serving, teacher transfer, and memory research. | The broader research was useful, but it mixed separate claims and lacked stable pass/fail rules. |
| 2026-07-16 | Added data rights, retention, scrubbing, upload checks, sealed splits, agent-session curation, and controlled Tinker training. | The first rank-64 Inkling LoRA passed 4/4 small mechanical tasks. This showed narrow task compliance, not general coding skill. |
| 2026-07-22 | Ran a small GRPO experiment on the harder eight-task set. | The score stayed 7/8 and the same task failed. Training stopped because the small task set had saturated. |
| 2026-07-23 to 2026-07-28 | Compared the trace LoRA with base Inkling on JAXBench, fixed unequal prompts and extraction, hid references, and rejected copied answers. | The apparent LoRA advantage reversed. Base Inkling tried Pallas more often, so the project shifted to correct, real, fast Pallas kernels. |
| 2026-07-29 | Froze the first contracts, built the evaluator, and tested 50 JAXBench tasks with three model seeds. | The first report claimed six correct Pallas kernels. Five used `interpret=True`; the corrected result was one normally lowered kernel out of 150 and no speed win. |
| 2026-07-30 | Added real lowering checks, isolated TPU processes, compiler markers, Perfetto traces, Chex assertions, and strict evidence checks. | The evaluator could now separate normal Pallas execution from interpreted or ordinary JAX code. |
| 2026-07-30 to 2026-08-02 | Built GitHub and Hugging Face data discovery, scanned about 982,000 datasets, fixed source-classification bugs, and verified the SFT data on TPUs. | The SFT dataset reached 32 kernels across eight families. The broad DAPT dataset reached 830 approved rows; DAPT remained untested. |
| 2026-08-02 | Switched training to Inkling Small and searched for a faster JAXBench matrix multiplication kernel. | The best Pallas kernel reached `0.9993x`, effectively equal to XLA. JAXPR was too high-level to explain the final TPU speed difference. |
| 2026-08-04 | Trained direct Pallas SFT on the 32 verified kernels. | Training finished, but all three TPU checks failed. The model learned the shape of Pallas code, not working kernels. |
| 2026-08-04 | Fixed hidden interface details in the training prompts and allowed up to three feedback attempts in G4.1. | G4.1 recovered some kernels after feedback but did not beat the base model: 1/4 versus 1/4. |
| 2026-08-04 to 2026-08-05 | Built the G4.2 patch-based agent test, hidden verifier, 32-task repair dataset, six-action runs, and matched three-model comparison. | G4.2 reached 7/12 after both three and six calls, versus base at 0/12 and 4/12. This was the first positive checkpoint result, but speed stayed near XLA and the test covered only four tasks. |
| 2026-08-05 | Tested all eight optimized JAXBench references on a v5e before choosing speed tasks. | None worked as a strict shared-environment comparison. A corrected setup found stable `1.147x` Megablox headroom, which remains provisional. |
| 2026-08-05 | Ran a three-seed 8/16/32-trajectory SFT learning curve on 16 unseen tasks. | The curve was non-monotonic and seed-sensitive. G4.2 remained strongest at 8/16; the best new arms reached 7/16. |
| 2026-08-05 | Corrected the DAPT corpus to 854 rows, trained D0, continued it through the identical G4.2 SFT recipe as S1, and ran the frozen 16-task TPU benchmark. | DAPT reduced validation NLL by 38.25%, but D0 scored 0/16 and S1 scored 1/16 versus S0 at 8/16. Gate 5 closed negative and Gate 6 GRPO became active. |
| 2026-08-06 | Ran matched four-turn online GRPO from S0 and S1 with real compiler, correctness, runtime, profile, and timing feedback on eight TPU workers. | R0 scored 7/16 versus S0 at 8/16; R1 scored 2/16 versus S1 at 1/16 but remained far below R0. Online reward rose in both lanes, so the result separates training-task repair from held-out weight improvement. |
| 2026-08-06 | Inspected the pinned Tinker SDK and Cookbook through the project `uv` environment, then imported Miles and its `sglang-miles` rollout branch as pinned submodules and added an executable source-contract audit. | Miles has native Inkling Small, LoRA, GRPO, and OPD support. SGLang supplies Inkling rendering, inference, routed-expert capture, and adapter serving. Gate 7 paused at renderer, checkpoint, SFT, DAPT, rollout, and G6 reproduction canaries; no open-backend training result exists yet. |
| 2026-08-06 | Added Laguna XS 2.1 as an exact-revision SGLang model arm and ran the frozen 16-task, three-call baseline through the authoritative TPU verifier. | BF16 tensor-parallel-1 inference fit on one H200. Laguna scored 0/16 with zero infrastructure failures because every trajectory used all three calls for inspection, emitted an empty patch, and failed the artifact contract. Miles training support remains unimplemented. |
| 2026-08-06 | Extended the Laguna baseline to six calls while preserving immutable call-3 and call-6 snapshots from each trajectory. | Both horizons scored 0/16 with no infrastructure failures. Calls 4–6 continued inspection, no patch was created, and all 16 tasks remained fail-to-fail. The action-level k=3 prefix matched the original run on all tasks. |
| 2026-08-06 to 2026-08-07 | Closed Phase 0 by preserving all five loadable Inkling Small checkpoints, their training manifests, exact Hub revisions, LFS hashes, Tinker resume URIs, frozen evaluation results, and the Laguna k=3/k=6 controls. | The private model repositories hold 84,507,051,680 bytes of verified adapter weights. The private `sdrshn-nmbr/opjax-checkpoints` Bucket holds the canonical index and separately validated `best` and `latest` pointers. The Bucket could not duplicate a 16.9 GB adapter under its storage limit, so weight identity is the immutable repo revision plus LFS SHA-256. Phase 0 is complete; cross-provider optimizer resume and SGLang logit parity remain Phase 1. |
| 2026-08-07 | Completed the recalibrated Phase 1 evaluator repair with native TML parsing, schema-bound exact task semantics, honest stage accounting, isolated candidate execution, strict compiler and Perfetto admission, and balanced interleaved uncertainty-aware timing. | The authoritative v5e verifier gave the reference reward 1 across seeds 0, 1, and 2. It gave a Chex-tampering kernel reward 0 at correctness and a SIGABRT kernel reward 0 at runtime safety with recovery required; the reference then passed again. The reference measured 0.98788x XLA with a 95% interval of 0.98100x to 0.99940x, so it has no performance headroom. Historical G4.2/G4.3 scores remain legacy diagnostics and Phase 2 requires a fresh benchmark. |
