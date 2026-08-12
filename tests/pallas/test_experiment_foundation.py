import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from opjax.pallas.experiment_foundation import (
    ArtifactRole,
    CheckpointDraft,
    CheckpointIdentity,
    CheckpointValidation,
    FilesystemCheckpointStore,
    InferenceBackend,
    ModelArm,
    RuntimeCursor,
    RuntimeBinding,
    TrainingBackend,
    TrainingMethod,
    UpdateEvidence,
    prove_resume_and_inference_parity,
)


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class DeterministicTrainingBackend(TrainingBackend):
    def __init__(
        self,
        *,
        corrupt_resume: bool = False,
        parent_checkpoint_id: str | None = None,
        version_offset: int = 0,
    ) -> None:
        self.weight = 0
        self.momentum = 0
        self.scheduler = 0
        self.rng = 7
        self.cursor = RuntimeCursor(data=0, rollout=0)
        self.corrupt_resume = corrupt_resume
        self.parent_checkpoint_id = parent_checkpoint_id
        self.version_offset = version_offset

    @property
    def name(self) -> str:
        return "deterministic-training"

    def initialize(self, *, seed: int) -> None:
        self.weight = seed
        self.momentum = 0
        self.scheduler = 0
        self.rng = seed + 7
        self.cursor = RuntimeCursor(data=0, rollout=0)

    def update(self, batch_id: str) -> UpdateEvidence:
        batch = int(batch_id)
        self.rng = (self.rng * 17 + 3) % 101
        self.momentum = self.momentum * 2 + batch + self.rng
        self.weight += self.momentum
        self.scheduler += 1
        self.cursor = RuntimeCursor(
            data=self.cursor.data + 1,
            rollout=self.cursor.rollout + batch,
        )
        return self._evidence()

    def save(self, directory: Path) -> CheckpointDraft:
        (directory / "adapter").mkdir(parents=True)
        (directory / "training").mkdir()
        (directory / "runtime").mkdir()
        (directory / "export").mkdir()
        (directory / "adapter" / "weights.json").write_text(
            json.dumps({"weight": self.weight})
        )
        (directory / "training" / "state.json").write_text(
            json.dumps(
                {"momentum": self.momentum, "scheduler": self.scheduler}
            )
        )
        (directory / "runtime" / "state.json").write_text(
            json.dumps(
                {
                    "rng": self.rng,
                    "data": self.cursor.data,
                    "rollout": self.cursor.rollout,
                }
            )
        )
        (directory / "export" / "adapter.json").write_text(
            json.dumps({"weight": self.weight})
        )
        return CheckpointDraft(
            root=directory,
            identity=CheckpointIdentity(
                checkpoint_id=(
                    f"control-{self.version_offset + self.scheduler}-{self.weight}"
                ),
                model_arm=ModelArm.INKLING_SMALL,
                training_method=TrainingMethod.SFT,
                training_backend=self.name,
                global_update=self.version_offset + self.scheduler,
                policy_version=self.version_offset + self.scheduler,
                base_model_id="control/base",
                base_model_revision="a" * 40,
                base_model_sha256="4" * 64,
                parent_checkpoint_id=self.parent_checkpoint_id,
            ),
            cursor=self.cursor,
            experiment_sha256="1" * 64,
            config_sha256="2" * 64,
            data_sha256="3" * 64,
            rng_sha256=_sha(self.rng),
            runtime=RuntimeBinding(
                python_version="3.12.11",
                training_backend_revision="miles-control",
                inference_backend_revision="sglang-control",
                tokenizer_sha256="5" * 64,
                renderer_sha256="6" * 64,
                toolchain_sha256="7" * 64,
                precision="fp32",
                quantization="none",
            ),
            artifacts={
                ArtifactRole.ADAPTER_WEIGHTS: Path("adapter"),
                ArtifactRole.OPTIMIZER_SCHEDULER: Path("training"),
                ArtifactRole.RUNTIME_STATE: Path("runtime/state.json"),
                ArtifactRole.INFERENCE_EXPORT: Path("export"),
            },
        )

    def resume(self, directory: Path) -> None:
        weights = json.loads((directory / "adapter" / "weights.json").read_text())
        training = json.loads((directory / "training" / "state.json").read_text())
        runtime = json.loads((directory / "runtime" / "state.json").read_text())
        self.weight = weights["weight"] + int(self.corrupt_resume)
        self.momentum = training["momentum"]
        self.scheduler = training["scheduler"]
        self.rng = runtime["rng"]
        self.cursor = RuntimeCursor(
            data=runtime["data"], rollout=runtime["rollout"]
        )

    def logits(self, prompt_token_ids: tuple[int, ...]) -> tuple[float, ...]:
        return tuple(float(self.weight + token) for token in prompt_token_ids)

    def _evidence(self) -> UpdateEvidence:
        return UpdateEvidence(
            loss=float(self.weight),
            model_sha256=_sha(self.weight),
            optimizer_sha256=_sha(self.momentum),
            scheduler_sha256=_sha(self.scheduler),
            rng_sha256=_sha(self.rng),
            cursor=self.cursor,
            logits_sha256=_sha(self.logits((2, 3))),
        )


class DeterministicInferenceBackend(InferenceBackend):
    def __init__(self, *, corrupt_load: bool = False) -> None:
        self.weight = 0
        self.corrupt_load = corrupt_load

    @property
    def name(self) -> str:
        return "deterministic-inference"

    def load_export(self, directory: Path) -> None:
        value = json.loads((directory / "adapter.json").read_text())["weight"]
        self.weight = value + int(self.corrupt_load)

    def logits(self, prompt_token_ids: tuple[int, ...]) -> tuple[float, ...]:
        return tuple(float(self.weight + token) for token in prompt_token_ids)


def _run_proof(
    tmp_path: Path,
    *,
    corrupt_resume: bool = False,
    corrupt_load: bool = False,
    fault=None,
):
    store = FilesystemCheckpointStore(tmp_path / "store", fault=fault)
    return prove_resume_and_inference_parity(
        store=store,
        training_factory=lambda: DeterministicTrainingBackend(
            corrupt_resume=corrupt_resume
        ),
        inference_factory=lambda: DeterministicInferenceBackend(
            corrupt_load=corrupt_load
        ),
        seed=11,
        first_batch_id="5",
        next_batch_id="9",
        prompt_token_ids=(2, 3),
        workspace=tmp_path / "work",
    )


def test_acceptance_resume_export_and_atomic_latest(tmp_path: Path) -> None:
    result = _run_proof(tmp_path)

    assert result.resume_exact is True
    assert result.inference_parity_passed is True
    assert result.validation.passed is True
    latest = FilesystemCheckpointStore(tmp_path / "store").latest()
    assert latest.identity.checkpoint_id == result.checkpoint.identity.checkpoint_id
    assert latest.validation == result.validation


@pytest.mark.parametrize(
    ("corrupt_resume", "corrupt_load", "expected_error"),
    [
        (True, False, "CHECKPOINT_RESUME_PARITY_FAILED"),
        (False, True, "CHECKPOINT_INFERENCE_PARITY_FAILED"),
    ],
)
def test_invalid_checkpoint_never_becomes_latest(
    tmp_path: Path,
    corrupt_resume: bool,
    corrupt_load: bool,
    expected_error: str,
) -> None:
    with pytest.raises(RuntimeError, match=expected_error):
        _run_proof(
            tmp_path,
            corrupt_resume=corrupt_resume,
            corrupt_load=corrupt_load,
        )

    assert FilesystemCheckpointStore(tmp_path / "store").latest_or_none() is None


@pytest.mark.parametrize(
    "fault_point",
    [
        "after_staging",
        "after_package_commit",
        "after_validation",
        "before_latest_commit",
    ],
)
def test_kill_windows_never_publish_unvalidated_latest(
    tmp_path: Path, fault_point: str
) -> None:
    def fault(point: str) -> None:
        if point == fault_point:
            raise KeyboardInterrupt(point)

    with pytest.raises(KeyboardInterrupt, match=fault_point):
        _run_proof(tmp_path, fault=fault)

    assert FilesystemCheckpointStore(tmp_path / "store").latest_or_none() is None


def test_failed_new_checkpoint_does_not_replace_valid_latest(tmp_path: Path) -> None:
    accepted = _run_proof(tmp_path)

    with pytest.raises(RuntimeError, match="CHECKPOINT_INFERENCE_PARITY_FAILED"):
        prove_resume_and_inference_parity(
            store=FilesystemCheckpointStore(tmp_path / "store"),
            training_factory=lambda: DeterministicTrainingBackend(
                parent_checkpoint_id=accepted.checkpoint.identity.checkpoint_id,
                version_offset=1,
            ),
            inference_factory=lambda: DeterministicInferenceBackend(
                corrupt_load=True
            ),
            seed=12,
            first_batch_id="5",
            next_batch_id="9",
            prompt_token_ids=(2, 3),
            workspace=tmp_path / "failed-work",
        )

    latest = FilesystemCheckpointStore(tmp_path / "store").latest()
    assert latest.identity.checkpoint_id == accepted.checkpoint.identity.checkpoint_id


def test_tamper_and_missing_role_fail_closed(tmp_path: Path) -> None:
    result = _run_proof(tmp_path)
    checkpoint = result.checkpoint.path
    (checkpoint / "payload" / "adapter" / "weights.json").write_text("tampered")

    with pytest.raises(RuntimeError, match="CHECKPOINT_ARTIFACT_HASH_MISMATCH"):
        FilesystemCheckpointStore(tmp_path / "store").latest()

    backend = DeterministicTrainingBackend()
    backend.initialize(seed=1)
    draft = backend.save(tmp_path / "incomplete")
    draft = replace(
        draft,
        artifacts={
            role: path
            for role, path in draft.artifacts.items()
            if role is not ArtifactRole.RUNTIME_STATE
        },
        identity=replace(draft.identity, checkpoint_id="incomplete"),
    )
    with pytest.raises(ValueError, match="CHECKPOINT_REQUIRED_ROLE_MISSING"):
        FilesystemCheckpointStore(tmp_path / "other").publish(
            draft,
            lambda _: CheckpointValidation(
                resume_exact=True,
                inference_parity_passed=True,
                inference_parity_mode="full_logit",
                details={},
            ),
        )


def test_symlink_payload_is_rejected(tmp_path: Path) -> None:
    backend = DeterministicTrainingBackend()
    backend.initialize(seed=1)
    draft = backend.save(tmp_path / "draft")
    external = tmp_path / "secret"
    external.write_text("secret")
    (draft.root / "adapter" / "link").symlink_to(external)

    with pytest.raises(ValueError, match="CHECKPOINT_SYMLINK_FORBIDDEN"):
        FilesystemCheckpointStore(tmp_path / "store").publish(
            draft,
            lambda _: CheckpointValidation(
                resume_exact=True,
                inference_parity_passed=True,
                inference_parity_mode="full_logit",
                details={},
            ),
        )


def test_full_vocabulary_logprob_is_an_explicit_parity_mode() -> None:
    validation = CheckpointValidation(
        resume_exact=True,
        inference_parity_passed=True,
        inference_parity_mode="full_vocabulary_logprob",
        details={"vocabulary_token_ids_exact": True},
    )

    assert validation.passed


@pytest.mark.parametrize(
    "fault_point",
    [
        "after_staging",
        "after_package_commit",
        "after_validation",
        "before_latest_commit",
    ],
)
def test_retry_after_kill_recovers_same_checkpoint(
    tmp_path: Path, fault_point: str
) -> None:
    armed = True

    def fault(point: str) -> None:
        nonlocal armed
        if armed and point == fault_point:
            armed = False
            raise KeyboardInterrupt(point)

    with pytest.raises(KeyboardInterrupt, match=fault_point):
        _run_proof(tmp_path, fault=fault)

    recovered = prove_resume_and_inference_parity(
        store=FilesystemCheckpointStore(tmp_path / "store"),
        training_factory=DeterministicTrainingBackend,
        inference_factory=DeterministicInferenceBackend,
        seed=11,
        first_batch_id="5",
        next_batch_id="9",
        prompt_token_ids=(2, 3),
        workspace=tmp_path / "recovery-work",
    )
    assert recovered.validation.passed is True
    assert FilesystemCheckpointStore(tmp_path / "store").latest().identity == (
        recovered.checkpoint.identity
    )


@pytest.mark.parametrize("filename", ["validation.json", "acceptance.json"])
def test_latest_binds_validation_and_acceptance(
    tmp_path: Path, filename: str
) -> None:
    result = _run_proof(tmp_path)
    (result.checkpoint.path / filename).write_text("{}\n")

    with pytest.raises(RuntimeError):
        FilesystemCheckpointStore(tmp_path / "store").latest()


def test_artifact_paths_must_not_overlap(tmp_path: Path) -> None:
    backend = DeterministicTrainingBackend()
    backend.initialize(seed=1)
    draft = backend.save(tmp_path / "draft-overlap")
    (draft.root / "training" / "runtime.json").write_text("{}")
    draft = replace(
        draft,
        artifacts={
            **draft.artifacts,
            ArtifactRole.RUNTIME_STATE: Path("training/runtime.json"),
        },
    )

    with pytest.raises(ValueError, match="CHECKPOINT_ARTIFACT_PATH_OVERLAP"):
        FilesystemCheckpointStore(tmp_path / "store").publish(
            draft,
            lambda _: CheckpointValidation(True, True, "full_logit", {}),
        )


@pytest.mark.parametrize(
    ("parent", "offset", "expected_error"),
    [
        ("wrong-parent", 1, "CHECKPOINT_PARENT_NOT_LATEST"),
        (None, 1, "CHECKPOINT_PARENT_NOT_LATEST"),
        ("CURRENT", 0, "CHECKPOINT_GLOBAL_UPDATE_REGRESSION"),
    ],
)
def test_latest_lineage_and_update_are_monotonic(
    tmp_path: Path,
    parent: str | None,
    offset: int,
    expected_error: str,
) -> None:
    accepted = _run_proof(tmp_path)
    current = accepted.checkpoint.identity.checkpoint_id
    declared_parent = current if parent == "CURRENT" else parent
    backend = DeterministicTrainingBackend(
        parent_checkpoint_id=declared_parent,
        version_offset=offset,
    )
    backend.initialize(seed=12)
    backend.update("5")
    draft = backend.save(tmp_path / f"lineage-{offset}-{parent}")

    with pytest.raises(RuntimeError, match=expected_error):
        FilesystemCheckpointStore(tmp_path / "store").publish(
            draft,
            lambda _: CheckpointValidation(True, True, "full_logit", {}),
        )


def test_policy_version_must_increase(tmp_path: Path) -> None:
    accepted = _run_proof(tmp_path)
    backend = DeterministicTrainingBackend(
        parent_checkpoint_id=accepted.checkpoint.identity.checkpoint_id,
        version_offset=1,
    )
    backend.initialize(seed=12)
    backend.update("5")
    draft = backend.save(tmp_path / "policy-regression")
    draft = replace(
        draft,
        identity=replace(draft.identity, policy_version=1),
    )

    with pytest.raises(RuntimeError, match="CHECKPOINT_POLICY_VERSION_REGRESSION"):
        FilesystemCheckpointStore(tmp_path / "store").publish(
            draft,
            lambda _: CheckpointValidation(True, True, "full_logit", {}),
        )


def test_validator_cannot_mutate_payload_before_promotion(tmp_path: Path) -> None:
    backend = DeterministicTrainingBackend()
    backend.initialize(seed=1)
    backend.update("5")
    draft = backend.save(tmp_path / "validator-mutation")

    def mutate(payload: Path) -> CheckpointValidation:
        (payload / "adapter" / "weights.json").write_text("mutated")
        return CheckpointValidation(True, True, "full_logit", {})

    store = FilesystemCheckpointStore(tmp_path / "store")
    with pytest.raises(RuntimeError, match="CHECKPOINT_VALIDATOR_MUTATED_PAYLOAD"):
        store.publish(draft, mutate)
    assert store.latest_or_none() is None
