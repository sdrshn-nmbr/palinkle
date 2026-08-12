"""Live Modal Volume canary for the Phase 1 checkpoint store."""

import hashlib
import json
import tempfile
import uuid
from pathlib import Path

import modal

from opjax.pallas.experiment_foundation import (
    ArtifactRole,
    CheckpointDraft,
    CheckpointIdentity,
    CheckpointValidation,
    FilesystemCheckpointStore,
    ModelArm,
    RuntimeCursor,
    RuntimeBinding,
    TrainingMethod,
)


APP_NAME = "opjax-phase1-foundation"
VOLUME_NAME = "opjax-checkpoints-v2"
MOUNT = Path("/checkpoints")
CANARY_ROOT = MOUNT / "phase1-foundation-canary-v2"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "opjax"
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=120)
def write_canary(checkpoint_id: str) -> dict[str, object]:
    store = FilesystemCheckpointStore(CANARY_ROOT)
    current = store.latest_or_none()
    version = current.identity.global_update + 1 if current else 1
    parent = current.identity.checkpoint_id if current else None
    with tempfile.TemporaryDirectory(prefix="opjax-phase1-") as temporary:
        root = Path(temporary)
        for directory in ("model", "training", "runtime", "export"):
            (root / directory).mkdir()
        payload = b"phase1-modal-volume-canary\n"
        (root / "model" / "weights.bin").write_bytes(payload)
        (root / "training" / "optimizer-scheduler.json").write_text(
            json.dumps({"optimizer": 1, "scheduler": 1})
        )
        (root / "runtime" / "state.json").write_text(
            json.dumps({"rng": 7, "data": 1, "rollout": 1})
        )
        (root / "export" / "weights.bin").write_bytes(payload)
        draft = CheckpointDraft(
            root=root,
            identity=CheckpointIdentity(
                checkpoint_id=checkpoint_id,
                model_arm=ModelArm.INKLING_SMALL,
                training_method=TrainingMethod.SFT,
                training_backend="modal-volume-canary",
                global_update=version,
                policy_version=version,
                base_model_id="phase1/canary",
                base_model_revision="canary",
                base_model_sha256=_sha(payload),
                parent_checkpoint_id=parent,
            ),
            cursor=RuntimeCursor(data=1, rollout=1),
            experiment_sha256=_sha(b"experiment"),
            config_sha256=_sha(b"config"),
            data_sha256=_sha(b"data"),
            rng_sha256=_sha(b"rng-7"),
            runtime=RuntimeBinding(
                python_version="3.12",
                training_backend_revision="modal-canary",
                inference_backend_revision="modal-canary",
                tokenizer_sha256=_sha(b"tokenizer"),
                renderer_sha256=_sha(b"renderer"),
                toolchain_sha256=_sha(b"modal-1.4.2"),
                precision="bytes",
                quantization="none",
            ),
            artifacts={
                ArtifactRole.MODEL_WEIGHTS: Path("model"),
                ArtifactRole.OPTIMIZER_SCHEDULER: Path("training"),
                ArtifactRole.RUNTIME_STATE: Path("runtime/state.json"),
                ArtifactRole.INFERENCE_EXPORT: Path("export"),
            },
        )

        def validate(staged: Path) -> CheckpointValidation:
            source = (staged / "model" / "weights.bin").read_bytes()
            exported = (staged / "export" / "weights.bin").read_bytes()
            exact = source == exported == payload
            return CheckpointValidation(
                resume_exact=exact,
                inference_parity_passed=exact,
                inference_parity_mode="full_logit",
                details={"canary_payload_sha256": _sha(payload)},
            )

        checkpoint = store.publish(draft, validate)
        volume.commit()
        return {
            "checkpoint_id": checkpoint.identity.checkpoint_id,
            "manifest_sha256": checkpoint.manifest_sha256,
            "validation_sha256": checkpoint.validation_sha256,
            "acceptance_sha256": checkpoint.acceptance_sha256,
            "validation": checkpoint.validation.passed,
        }


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=120)
def read_canary() -> dict[str, object]:
    volume.reload()
    checkpoint = FilesystemCheckpointStore(CANARY_ROOT).latest()
    return {
        "checkpoint_id": checkpoint.identity.checkpoint_id,
        "manifest_sha256": checkpoint.manifest_sha256,
        "validation_sha256": checkpoint.validation_sha256,
        "acceptance_sha256": checkpoint.acceptance_sha256,
        "validation": checkpoint.validation.passed,
    }


@app.local_entrypoint()
def main() -> None:
    checkpoint_id = f"modal-canary-{uuid.uuid4().hex}"
    written = write_canary.remote(checkpoint_id)
    observed = read_canary.remote()
    if written != observed:
        raise RuntimeError(
            "PHASE1_MODAL_VOLUME_PERSISTENCE_FAILED: "
            f"written={written} observed={observed}"
        )
    print(json.dumps(observed, indent=2, sort_keys=True))
