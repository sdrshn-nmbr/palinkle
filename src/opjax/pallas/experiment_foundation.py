"""Provider-neutral experiment and checkpoint contracts.

The filesystem store is suitable for a local disk or a mounted durable volume.
Its deterministic fault model covers process death between staging, package
commit, validation, and latest-pointer commit. It does not model filesystem or
object-store implementations that violate atomic same-filesystem rename.
"""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable


class ModelArm(StrEnum):
    CONTROL_QWEN25_05B = "control_qwen25_05b"
    INKLING_SMALL = "inkling_small"
    LAGUNA_XS_21 = "laguna_xs_21"


class TrainingMethod(StrEnum):
    SFT = "sft"
    DAPT = "dapt"
    GRPO = "grpo"
    PPO = "ppo"
    OPD = "opd"
    OPSD = "opsd"
    RMSD = "rmsd"


class ArtifactRole(StrEnum):
    MODEL_WEIGHTS = "model_weights"
    ADAPTER_WEIGHTS = "adapter_weights"
    OPTIMIZER_SCHEDULER = "optimizer_scheduler"
    RUNTIME_STATE = "runtime_state"
    INFERENCE_EXPORT = "inference_export"


@dataclass(frozen=True)
class RuntimeCursor:
    data: int
    rollout: int

    def __post_init__(self) -> None:
        if self.data < 0 or self.rollout < 0:
            raise ValueError("CHECKPOINT_CURSOR_NEGATIVE")


@dataclass(frozen=True)
class CheckpointIdentity:
    checkpoint_id: str
    model_arm: ModelArm
    training_method: TrainingMethod
    training_backend: str
    global_update: int
    policy_version: int
    base_model_id: str
    base_model_revision: str
    base_model_sha256: str
    parent_checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_id or "/" in self.checkpoint_id or ".." in self.checkpoint_id:
            raise ValueError("CHECKPOINT_ID_INVALID")
        if self.global_update < 0 or self.policy_version < 0:
            raise ValueError("CHECKPOINT_VERSION_NEGATIVE")
        if not self.base_model_id or not self.base_model_revision:
            raise ValueError("CHECKPOINT_BASE_MODEL_UNBOUND")
        _validate_sha256("base_model", self.base_model_sha256)


@dataclass(frozen=True)
class RuntimeBinding:
    python_version: str
    training_backend_revision: str
    inference_backend_revision: str
    tokenizer_sha256: str
    renderer_sha256: str
    toolchain_sha256: str
    precision: str
    quantization: str

    def __post_init__(self) -> None:
        if not self.python_version or not self.precision or not self.quantization:
            raise ValueError("CHECKPOINT_RUNTIME_BINDING_INCOMPLETE")
        if not self.training_backend_revision or not self.inference_backend_revision:
            raise ValueError("CHECKPOINT_BACKEND_REVISION_MISSING")
        for label, value in (
            ("tokenizer", self.tokenizer_sha256),
            ("renderer", self.renderer_sha256),
            ("toolchain", self.toolchain_sha256),
        ):
            _validate_sha256(label, value)


@dataclass(frozen=True)
class CheckpointDraft:
    root: Path
    identity: CheckpointIdentity
    cursor: RuntimeCursor
    experiment_sha256: str
    config_sha256: str
    data_sha256: str
    rng_sha256: str
    runtime: RuntimeBinding
    artifacts: Mapping[ArtifactRole, Path]


@dataclass(frozen=True)
class CheckpointValidation:
    resume_exact: bool
    inference_parity_passed: bool
    inference_parity_mode: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.inference_parity_mode not in {
            "full_logit",
            "top_k_logprob",
            "full_vocabulary_logprob",
        }:
            raise ValueError("CHECKPOINT_INFERENCE_PARITY_MODE_INVALID")

    @property
    def passed(self) -> bool:
        return self.resume_exact and self.inference_parity_passed


@dataclass(frozen=True)
class PublishedCheckpoint:
    path: Path
    identity: CheckpointIdentity
    cursor: RuntimeCursor
    runtime: RuntimeBinding
    artifacts: Mapping[ArtifactRole, Path]
    artifact_sha256: Mapping[ArtifactRole, str]
    manifest_sha256: str
    validation_sha256: str
    acceptance_sha256: str
    validation: CheckpointValidation

    def artifact_path(self, role: ArtifactRole) -> Path:
        return self.path / "payload" / self.artifacts[role]


@dataclass(frozen=True)
class UpdateEvidence:
    loss: float
    model_sha256: str
    optimizer_sha256: str
    scheduler_sha256: str
    rng_sha256: str
    cursor: RuntimeCursor
    logits_sha256: str


@dataclass(frozen=True)
class FoundationProof:
    checkpoint: PublishedCheckpoint
    validation: CheckpointValidation
    control_next_update: UpdateEvidence
    resumed_next_update: UpdateEvidence
    resume_exact: bool
    inference_parity_passed: bool


@runtime_checkable
class TrainingBackend(Protocol):
    @property
    def name(self) -> str: ...

    def initialize(self, *, seed: int) -> None: ...

    def update(self, batch_id: str) -> UpdateEvidence: ...

    def save(self, directory: Path) -> CheckpointDraft: ...

    def resume(self, directory: Path) -> None: ...

    def logits(self, prompt_token_ids: tuple[int, ...]) -> tuple[float, ...]: ...


@runtime_checkable
class InferenceBackend(Protocol):
    @property
    def name(self) -> str: ...

    def load_export(self, directory: Path) -> None: ...

    def logits(self, prompt_token_ids: tuple[int, ...]) -> tuple[float, ...]: ...


CheckpointValidator = Callable[[Path], CheckpointValidation]
FaultInjector = Callable[[str], None]


@runtime_checkable
class CheckpointStore(Protocol):
    def publish(
        self,
        draft: CheckpointDraft,
        validator: CheckpointValidator,
    ) -> PublishedCheckpoint: ...

    def latest(self) -> PublishedCheckpoint: ...

    def latest_or_none(self) -> PublishedCheckpoint | None: ...


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_sha256(label: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"CHECKPOINT_HASH_INVALID: {label}")


class FilesystemCheckpointStore:
    """Atomic checkpoint store for local disks and mounted Modal Volumes."""

    def __init__(self, root: Path, *, fault: FaultInjector | None = None) -> None:
        self.root = root
        self.fault = fault or (lambda _point: None)
        self.checkpoints = root / "checkpoints"
        self.staging = root / ".staging"
        self.latest_path = root / "latest.json"
        self.lock_path = root / ".publish.lock"

    def publish(
        self,
        draft: CheckpointDraft,
        validator: CheckpointValidator,
    ) -> PublishedCheckpoint:
        self._validate_draft(draft)
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir(exist_ok=True)
        self.staging.mkdir(exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._publish_locked(draft, validator)

    def latest_or_none(self) -> PublishedCheckpoint | None:
        if not self.latest_path.is_file():
            return None
        return self.latest()

    def latest(self) -> PublishedCheckpoint:
        try:
            pointer = json.loads(self.latest_path.read_text())
            checkpoint_id = pointer["checkpoint_id"]
            expected_manifest_hash = pointer["manifest_sha256"]
            expected_validation_hash = pointer["validation_sha256"]
            expected_acceptance_hash = pointer["acceptance_sha256"]
        except (KeyError, json.JSONDecodeError) as error:
            raise RuntimeError("CHECKPOINT_LATEST_POINTER_INVALID") from error
        if not isinstance(checkpoint_id, str) or "/" in checkpoint_id or ".." in checkpoint_id:
            raise RuntimeError("CHECKPOINT_LATEST_POINTER_INVALID")
        checkpoint = self._load(self.checkpoints / checkpoint_id)
        if checkpoint.manifest_sha256 != expected_manifest_hash:
            raise RuntimeError("CHECKPOINT_LATEST_MANIFEST_MISMATCH")
        if checkpoint.validation_sha256 != expected_validation_hash:
            raise RuntimeError("CHECKPOINT_LATEST_VALIDATION_MISMATCH")
        if checkpoint.acceptance_sha256 != expected_acceptance_hash:
            raise RuntimeError("CHECKPOINT_LATEST_ACCEPTANCE_MISMATCH")
        return checkpoint

    def _publish_locked(
        self,
        draft: CheckpointDraft,
        validator: CheckpointValidator,
    ) -> PublishedCheckpoint:
        current = self.latest_or_none()
        if current is None:
            if draft.identity.parent_checkpoint_id is not None:
                raise RuntimeError("CHECKPOINT_PARENT_WITHOUT_LATEST")
        elif current.identity.checkpoint_id != draft.identity.checkpoint_id:
            if draft.identity.parent_checkpoint_id != current.identity.checkpoint_id:
                raise RuntimeError("CHECKPOINT_PARENT_NOT_LATEST")
            if draft.identity.global_update <= current.identity.global_update:
                raise RuntimeError("CHECKPOINT_GLOBAL_UPDATE_REGRESSION")
            if draft.identity.policy_version <= current.identity.policy_version:
                raise RuntimeError("CHECKPOINT_POLICY_VERSION_REGRESSION")

        stage = self.staging / f"{draft.identity.checkpoint_id}.{uuid.uuid4().hex}"
        payload = stage / "payload"
        payload.mkdir(parents=True)
        artifact_paths: dict[ArtifactRole, Path] = {}
        for role, relative in draft.artifacts.items():
            source = draft.root / relative
            destination = payload / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            artifact_paths[role] = relative

        files = self._inventory(payload)
        manifest = {
            "schema_version": 1,
            "identity": {
                **asdict(draft.identity),
                "model_arm": draft.identity.model_arm.value,
                "training_method": draft.identity.training_method.value,
            },
            "cursor": asdict(draft.cursor),
            "experiment_sha256": draft.experiment_sha256,
            "config_sha256": draft.config_sha256,
            "data_sha256": draft.data_sha256,
            "rng_sha256": draft.rng_sha256,
            "runtime": asdict(draft.runtime),
            "artifacts": {
                role.value: relative.as_posix()
                for role, relative in sorted(
                    artifact_paths.items(), key=lambda item: item[0].value
                )
            },
            "files": files,
        }
        manifest["artifact_sha256"] = {
            role.value: self._artifact_digest(payload / relative)
            for role, relative in sorted(
                artifact_paths.items(), key=lambda item: item[0].value
            )
        }
        manifest_bytes = _canonical_json(manifest) + b"\n"
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        for file_path in sorted(stage.rglob("*")):
            if file_path.is_file():
                _fsync_file(file_path)
        _fsync_directory(payload)
        _fsync_directory(stage)
        self.fault("after_staging")

        target = self.checkpoints / draft.identity.checkpoint_id
        if target.exists():
            existing_manifest = target / "manifest.json"
            if not existing_manifest.is_file() or existing_manifest.read_bytes() != manifest_bytes:
                raise RuntimeError(f"CHECKPOINT_ID_COLLISION: {draft.identity.checkpoint_id}")
            shutil.rmtree(stage)
        else:
            os.replace(stage, target)
            _fsync_directory(self.checkpoints)
        self.fault("after_package_commit")

        validation = validator(target / "payload")
        if not validation.passed:
            raise RuntimeError("CHECKPOINT_VALIDATION_FAILED")
        if self._inventory(target / "payload") != files:
            raise RuntimeError("CHECKPOINT_VALIDATOR_MUTATED_PAYLOAD")
        if (target / "manifest.json").read_bytes() != manifest_bytes:
            raise RuntimeError("CHECKPOINT_VALIDATOR_MUTATED_MANIFEST")
        validation_value = asdict(validation)
        _atomic_json(target / "validation.json", validation_value)
        validation_sha256 = _sha256_bytes(_canonical_json(validation_value) + b"\n")
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        acceptance = {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "validation_sha256": validation_sha256,
        }
        _atomic_json(target / "acceptance.json", acceptance)
        acceptance_sha256 = _sha256_bytes(_canonical_json(acceptance) + b"\n")
        accepted_checkpoint = self._load(target)
        self.fault("after_validation")

        pointer = {
            "schema_version": 1,
            "checkpoint_id": draft.identity.checkpoint_id,
            "manifest_sha256": manifest_sha256,
            "validation_sha256": validation_sha256,
            "acceptance_sha256": acceptance_sha256,
        }
        pointer_temp = self.root / f".latest.{uuid.uuid4().hex}.tmp"
        pointer_temp.write_bytes(_canonical_json(pointer) + b"\n")
        _fsync_file(pointer_temp)
        self.fault("before_latest_commit")
        os.replace(pointer_temp, self.latest_path)
        _fsync_directory(self.root)
        for stale_stage in self.staging.glob(
            f"{draft.identity.checkpoint_id}.*"
        ):
            if stale_stage.is_dir():
                shutil.rmtree(stale_stage)
            else:
                stale_stage.unlink()
        return accepted_checkpoint

    def _validate_draft(self, draft: CheckpointDraft) -> None:
        for label, value in (
            ("experiment", draft.experiment_sha256),
            ("config", draft.config_sha256),
            ("data", draft.data_sha256),
            ("rng", draft.rng_sha256),
        ):
            _validate_sha256(label, value)
        roles = set(draft.artifacts)
        weight_roles = roles & {
            ArtifactRole.MODEL_WEIGHTS,
            ArtifactRole.ADAPTER_WEIGHTS,
        }
        if len(weight_roles) != 1:
            raise ValueError("CHECKPOINT_WEIGHT_ROLE_INVALID")
        required = {
            ArtifactRole.OPTIMIZER_SCHEDULER,
            ArtifactRole.RUNTIME_STATE,
            ArtifactRole.INFERENCE_EXPORT,
        }
        missing = required - roles
        if missing:
            raise ValueError(
                "CHECKPOINT_REQUIRED_ROLE_MISSING: "
                + ",".join(sorted(role.value for role in missing))
            )
        root = draft.root.resolve(strict=True)
        normalized_paths: dict[ArtifactRole, PurePosixPath] = {}
        for role, relative in draft.artifacts.items():
            pure = PurePosixPath(relative.as_posix())
            if relative.is_absolute() or ".." in pure.parts or pure == PurePosixPath("."):
                raise ValueError(f"CHECKPOINT_ARTIFACT_PATH_INVALID: {role.value}")
            source = draft.root / relative
            if not source.exists():
                raise ValueError(f"CHECKPOINT_ARTIFACT_MISSING: {role.value}")
            descendants = list(source.rglob("*")) if source.is_dir() else []
            for path in [source, *descendants]:
                if path.is_symlink():
                    raise ValueError(f"CHECKPOINT_SYMLINK_FORBIDDEN: {path}")
            if not source.resolve().is_relative_to(root):
                raise ValueError(f"CHECKPOINT_ARTIFACT_OUTSIDE_ROOT: {role.value}")
            normalized_paths[role] = pure
        for left_role, left in normalized_paths.items():
            for right_role, right in normalized_paths.items():
                if left_role >= right_role:
                    continue
                if left == right or left in right.parents or right in left.parents:
                    raise ValueError(
                        "CHECKPOINT_ARTIFACT_PATH_OVERLAP: "
                        f"{left_role.value}={left} {right_role.value}={right}"
                    )

    @staticmethod
    def _inventory(payload: Path) -> list[dict[str, Any]]:
        files = []
        for path in sorted(payload.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"CHECKPOINT_SYMLINK_FORBIDDEN: {path}")
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(payload).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
        if not files:
            raise ValueError("CHECKPOINT_PAYLOAD_EMPTY")
        return files

    @classmethod
    def _artifact_digest(cls, path: Path) -> str:
        if path.is_file():
            inventory = [
                {
                    "path": path.name,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            ]
        else:
            inventory = cls._inventory(path)
        return _sha256_bytes(_canonical_json(inventory))

    def _load(self, path: Path) -> PublishedCheckpoint:
        try:
            manifest_bytes = (path / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            validation_bytes = (path / "validation.json").read_bytes()
            validation_raw = json.loads(validation_bytes)
            acceptance_bytes = (path / "acceptance.json").read_bytes()
            acceptance = json.loads(acceptance_bytes)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise RuntimeError("CHECKPOINT_PACKAGE_INCOMPLETE") from error
        payload = path / "payload"
        actual_files = self._inventory(payload)
        if actual_files != manifest.get("files"):
            raise RuntimeError("CHECKPOINT_ARTIFACT_HASH_MISMATCH")
        identity_raw = manifest["identity"]
        identity = CheckpointIdentity(
            **{
                **identity_raw,
                "model_arm": ModelArm(identity_raw["model_arm"]),
                "training_method": TrainingMethod(identity_raw["training_method"]),
            }
        )
        try:
            validation = CheckpointValidation(**validation_raw)
        except (TypeError, KeyError) as error:
            raise RuntimeError("CHECKPOINT_VALIDATION_INVALID") from error
        if not validation.passed:
            raise RuntimeError("CHECKPOINT_VALIDATION_FAILED")
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        validation_sha256 = _sha256_bytes(validation_bytes)
        if acceptance != {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "validation_sha256": validation_sha256,
        }:
            raise RuntimeError("CHECKPOINT_ACCEPTANCE_BINDING_INVALID")
        return PublishedCheckpoint(
            path=path,
            identity=identity,
            cursor=RuntimeCursor(**manifest["cursor"]),
            runtime=RuntimeBinding(**manifest["runtime"]),
            artifacts={
                ArtifactRole(role): Path(relative)
                for role, relative in manifest["artifacts"].items()
            },
            artifact_sha256={
                ArtifactRole(role): digest
                for role, digest in manifest["artifact_sha256"].items()
            },
            manifest_sha256=manifest_sha256,
            validation_sha256=validation_sha256,
            acceptance_sha256=_sha256_bytes(acceptance_bytes),
            validation=validation,
        )


def prove_resume_and_inference_parity(
    *,
    store: CheckpointStore,
    training_factory: Callable[[], TrainingBackend],
    inference_factory: Callable[[], InferenceBackend],
    seed: int,
    first_batch_id: str,
    next_batch_id: str,
    prompt_token_ids: tuple[int, ...],
    workspace: Path,
) -> FoundationProof:
    """Prove exact continuation and export parity before promoting latest."""

    if not prompt_token_ids:
        raise ValueError("CHECKPOINT_PARITY_PROMPT_EMPTY")
    workspace.mkdir(parents=True, exist_ok=True)

    control = training_factory()
    control.initialize(seed=seed)
    control.update(first_batch_id)
    control_next = control.update(next_batch_id)

    interrupted = training_factory()
    interrupted.initialize(seed=seed)
    interrupted.update(first_batch_id)
    draft = interrupted.save(workspace / "draft")
    training_logits = interrupted.logits(prompt_token_ids)
    captured: dict[str, Any] = {}

    def validate(payload: Path) -> CheckpointValidation:
        resumed = training_factory()
        resumed.resume(payload)
        resumed_next = resumed.update(next_batch_id)
        resume_exact = resumed_next == control_next
        captured["resumed_next"] = resumed_next
        if not resume_exact:
            raise RuntimeError("CHECKPOINT_RESUME_PARITY_FAILED")

        inference = inference_factory()
        inference.load_export(payload / draft.artifacts[ArtifactRole.INFERENCE_EXPORT])
        inference_logits = inference.logits(prompt_token_ids)
        logits_exact = inference_logits == training_logits
        if not logits_exact:
            raise RuntimeError("CHECKPOINT_INFERENCE_PARITY_FAILED")
        validation = CheckpointValidation(
            resume_exact=True,
            inference_parity_passed=True,
            inference_parity_mode="full_logit",
            details={
                "training_backend": interrupted.name,
                "inference_backend": inference.name,
                "prompt_token_ids_sha256": _sha256_bytes(
                    _canonical_json(prompt_token_ids)
                ),
                "training_logits_sha256": _sha256_bytes(
                    _canonical_json(training_logits)
                ),
                "inference_logits_sha256": _sha256_bytes(
                    _canonical_json(inference_logits)
                ),
                "next_update_sha256": _sha256_bytes(
                    _canonical_json(asdict(control_next))
                ),
            },
        )
        captured["validation"] = validation
        return validation

    checkpoint = store.publish(draft, validate)
    validation = captured["validation"]
    resumed_next = captured["resumed_next"]
    return FoundationProof(
        checkpoint=checkpoint,
        validation=validation,
        control_next_update=control_next,
        resumed_next_update=resumed_next,
        resume_exact=True,
        inference_parity_passed=True,
    )
