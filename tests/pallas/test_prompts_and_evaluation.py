from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from opjax.pallas.evaluation import (
    EvaluationError,
    SampleCandidate,
    _assert_evaluation_runtime,
    _assert_tpu_runtime,
    _evaluate_workload,
    _load_or_create_manifest,
    _oracle_summary,
    _parse_json_output,
    _rescore_result_rows,
    _result_compiled,
    _run_jaxbench_once,
    probe_runtime_hardware,
    validate_sample_run,
)
from opjax.pallas.contracts import load_contracts
from opjax.pallas.prompts import extract_code, render_prompt, spec_only
from opjax.pallas.scoring import PromptContext

BASELINE = """\
import jax.numpy as jnp

CONFIG = {"size": 8}

def create_inputs():
    return (jnp.ones((CONFIG["size"],)),)

def workload(x):
    \"\"\"Square each element.\"\"\"
    return jnp.square(x)
"""
CONFIG_ROOT = Path(__file__).parents[2] / "config" / "pallas"


def test_spec_prompt_withholds_reference_body() -> None:
    specification = spec_only(BASELINE)
    prompt = render_prompt(
        workload="square",
        baseline_source=BASELINE,
        prompt_context="spec",
    )

    assert "return jnp.square(x)" not in specification
    assert "return jnp.square(x)" not in prompt
    assert "CONFIG" in prompt
    assert "create_inputs" not in prompt
    assert "jax.numpy" not in prompt
    assert "def workload(x)" in prompt


def test_baseline_prompt_is_explicitly_diagnostic() -> None:
    prompt = render_prompt(
        workload="square",
        baseline_source=BASELINE,
        prompt_context="baseline",
    )

    assert "diagnostic context only" in prompt
    assert "return jnp.square(x)" in prompt


def test_extractor_prefers_complete_parseable_workload() -> None:
    completion = """analysis
```python
def workload(x):
    return x
```
"""

    assert extract_code(completion) == "def workload(x):\n    return x\n"


def test_resume_requires_identical_fingerprint(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    original = {"contract": "a", "kernels": {"x": "1"}}
    _load_or_create_manifest(out_dir=out_dir, fingerprint=original, resume=False)

    resumed = _load_or_create_manifest(
        out_dir=out_dir,
        fingerprint=original,
        resume=True,
    )
    assert resumed["fingerprint"] == original

    with pytest.raises(EvaluationError, match="RESUME_FINGERPRINT_MISMATCH"):
        _load_or_create_manifest(
            out_dir=out_dir,
            fingerprint={"contract": "b"},
            resume=True,
        )


def test_existing_run_requires_explicit_resume(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    _load_or_create_manifest(out_dir=out_dir, fingerprint={"x": 1}, resume=False)

    with pytest.raises(EvaluationError, match="RUN_ALREADY_EXISTS"):
        _load_or_create_manifest(
            out_dir=out_dir,
            fingerprint={"x": 1},
            resume=False,
        )


def test_jaxbench_json_parser_accepts_log_prefix() -> None:
    payload = {"workload": "square", "status": "correct"}
    output = f"compiler log\n{json.dumps(payload)}\n"

    assert _parse_json_output(output) == payload


def test_runtime_hardware_must_match_declared_tpu_generation() -> None:
    metadata = {
        "device_count": 1,
        "process_count": 1,
        "process_index": 0,
    }
    _assert_tpu_runtime(
        {
            **metadata,
            "platforms": ["tpu"],
            "device_kinds": ["TPU v5 lite"],
        },
        {"hardware": "v5e"},
    )

    with pytest.raises(EvaluationError, match="HARDWARE_TARGET_MISMATCH"):
        _assert_tpu_runtime(
            {
                **metadata,
                "platforms": ["cpu"],
                "device_kinds": ["Apple M3"],
            },
            {"hardware": "v5e"},
        )

    with pytest.raises(EvaluationError, match="HARDWARE_TARGET_MISMATCH"):
        _assert_tpu_runtime(
            {
                **metadata,
                "platforms": ["tpu"],
                "device_kinds": ["TPU v4"],
            },
            {"hardware": "v5e"},
        )

    for invalid_kind in (
        "TPU v5p",
        "not-v5e-but-contains-5",
        "TPU v4 and v5 lite",
    ):
        with pytest.raises(EvaluationError, match="HARDWARE_TARGET_MISMATCH"):
            _assert_tpu_runtime(
                {
                    **metadata,
                    "platforms": ["tpu"],
                    "device_kinds": [invalid_kind],
                },
                {"hardware": "v5e"},
            )


def test_evaluation_runtime_must_match_frozen_packages() -> None:
    expected = {
        "python": "3.10.12",
        "chex": "0.1.90",
        "jax": "0.6.2",
        "jaxlib": "0.6.2",
        "libtpu": "0.0.17",
    }
    fingerprint = {
        "python": "3.10.12",
        "packages": {
            "chex": "0.1.90",
            "jax": "0.6.2",
            "jaxlib": "0.6.2",
            "libtpu": "0.0.17",
        },
    }

    _assert_evaluation_runtime(fingerprint, expected)

    fingerprint["packages"]["jax"] = "0.10.0"
    with pytest.raises(EvaluationError, match="EVALUATION_RUNTIME_MISMATCH"):
        _assert_evaluation_runtime(fingerprint, expected)


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {
            "platforms": ["tpu"],
            "device_kinds": ["TPU v5 lite"],
            "device_count": 0,
            "process_count": 1,
            "process_index": 0,
        },
        {
            "platforms": ["tpu"],
            "device_kinds": [],
            "device_count": 1,
            "process_count": 1,
            "process_index": 0,
        },
        {
            "platforms": ["tpu"],
            "device_kinds": ["TPU v5 lite"],
            "device_count": 1,
            "process_count": 1,
            "process_index": 1,
        },
    ],
)
def test_runtime_hardware_rejects_inconsistent_probe_metadata(
    invalid_metadata: dict[str, object],
) -> None:
    with pytest.raises(EvaluationError, match="HARDWARE_PROBE_INVALID"):
        _assert_tpu_runtime(invalid_metadata, {"hardware": "v5e"})


def test_runtime_probe_uses_chex_tpu_availability_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_command: list[str] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        observed_command.extend(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "platforms": ["tpu"],
                    "device_kinds": ["TPU v5 lite"],
                    "device_count": 1,
                    "process_count": 1,
                    "process_index": 0,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("opjax.pallas.evaluation.subprocess.run", run)

    result = probe_runtime_hardware()

    assert result["platforms"] == ["tpu"]
    assert (
        "chex.assert_devices_available(1, 'tpu', not_less_than=True)"
        in observed_command[2]
    )


def test_runtime_probe_rejects_failed_chex_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr="AssertionError: [Chex] No 1 TPUs available",
        )

    monkeypatch.setattr("opjax.pallas.evaluation.subprocess.run", run)

    with pytest.raises(EvaluationError, match="HARDWARE_PROBE_FAILED"):
        probe_runtime_hardware()


def test_evaluation_binds_kernel_to_completed_sample_manifest(tmp_path: Path) -> None:
    bundle = load_contracts(CONFIG_ROOT)
    sample_run = tmp_path / "sample"
    kernels = sample_run / "kernels" / "seed-0"
    kernels.mkdir(parents=True)
    source = "def workload(x):\n    return x\n"
    kernel = kernels / "1p_Flash_Attention.py"
    kernel.write_text(source, encoding="utf-8")
    code_sha256 = hashlib.sha256(source.encode()).hexdigest()
    fingerprint = {
        "sha256": "a" * 64,
        "contract_sha256": bundle.sha256,
        "jaxbench_revision": next(
            source["revision"]
            for source in bundle.sources["sources"]
            if source["id"] == "jaxbench"
        ),
        "arm": "A",
        "prompt_context": "spec",
        "model_path": None,
        "request": {
            "sample_ids": ["1p_Flash_Attention::seed=0"],
            "workloads": ["1p_Flash_Attention"],
            "seeds": [0],
        },
    }
    (sample_run / "manifest.json").write_text(
        json.dumps({"status": "sampled", "fingerprint": fingerprint}),
        encoding="utf-8",
    )
    (sample_run / "samples.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "sample_id": "1p_Flash_Attention::seed=0",
                "workload": "1p_Flash_Attention",
                "seed": 0,
                "kernel_path": "kernels/seed-0/1p_Flash_Attention.py",
                "code_sha256": code_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    observed = validate_sample_run(
        bundle=bundle,
        sample_run=sample_run,
        model_id="thinkingmachines/Inkling-Small",
        arm="A",
        prompt_context=PromptContext.SPEC,
    )

    assert observed.fingerprint_sha256 == "a" * 64
    assert observed.candidates[0].sample_id == "1p_Flash_Attention::seed=0"

    kernel.write_text(source + "# changed\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="SAMPLE_KERNEL_HASH_MISMATCH"):
        validate_sample_run(
            bundle=bundle,
            sample_run=sample_run,
            model_id="thinkingmachines/Inkling-Small",
            arm="A",
            prompt_context=PromptContext.SPEC,
        )


def test_compilation_is_separate_from_correctness() -> None:
    assert _result_compiled({"status": "correct"}) is True
    assert _result_compiled({"status": "incorrect"}) is True
    assert _result_compiled({"status": "compile_error"}) is False
    assert _result_compiled({"status": "runtime_error"}) is False


def test_jaxbench_retries_transient_tpu_runtime_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        subprocess.CompletedProcess(
            [],
            1,
                stdout=json.dumps(
                    {
                        "hardware": None,
                        "result": {
                            "status": "error",
                            "error": "TPU is already in use by process with pid 123",
                        },
                    }
                ),
            stderr="",
        ),
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {
                    "hardware": {
                        "platforms": ["tpu"],
                        "device_kinds": ["TPU v5 lite"],
                        "device_count": 1,
                        "process_count": 1,
                        "process_index": 0,
                    },
                    "result": {"status": "incorrect"},
                }
            ),
            stderr="",
        ),
    ]
    sleeps: list[float] = []
    monkeypatch.setattr(
        "opjax.pallas.evaluation.subprocess.run",
        lambda *_args, **_kwargs: responses.pop(0),
    )
    monkeypatch.setattr(
        "opjax.pallas.evaluation.time.sleep",
        lambda duration: sleeps.append(duration),
    )

    result = _run_jaxbench_once(
        jaxbench_root=tmp_path,
        workload="8p_GEMM",
        kernel=tmp_path / "kernel.py",
        tpu="v5e",
        num_warmup=1,
        num_iters=1,
        timeout_seconds=30,
    )

    assert result["result"]["status"] == "incorrect"
    assert len(result["transient_attempts"]) == 1
    assert sleeps == [1.0]


def test_rescore_removes_interpret_mode_pallas_credit(tmp_path: Path) -> None:
    baseline_dir = (
        tmp_path
        / "jaxbench"
        / "JAXBench"
        / "benchmark"
        / "1p_Flash_Attention"
    )
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline.py").write_text(BASELINE, encoding="utf-8")
    interpreted = """\
import jax
from jax.experimental import pallas as pl

def workload(x):
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pl.pallas_call(kernel, out_shape=shape, interpret=True)(x)
"""
    lowered = interpreted.replace(", interpret=True", "")
    candidates: list[SampleCandidate] = []
    rows: list[dict[str, object]] = []
    for seed, source in enumerate((interpreted, lowered)):
        kernel = tmp_path / f"{seed}.py"
        kernel.write_text(source, encoding="utf-8")
        sample_id = f"1p_Flash_Attention::seed={seed}"
        candidates.append(
            SampleCandidate(
                sample_id=sample_id,
                workload="1p_Flash_Attention",
                seed=seed,
                kernel=kernel,
                sample={"status": "sampled"},
            )
        )
        rows.append(
            {
                "sample_id": sample_id,
                "workload": "1p_Flash_Attention",
                "seed": seed,
                "kernel_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "compiled": True,
                "correct": True,
                "prompt_context": "spec",
                "inspection": {"authentic": True},
                "timing": {"stable": True},
                "speedup": 2.0,
                "pallas_credited": True,
                "headline_credited": True,
            }
        )

    rescored = _rescore_result_rows(
        candidates=tuple(candidates),
        rows=rows,
        jaxbench_root=tmp_path / "jaxbench",
        prompt_context=PromptContext.SPEC,
        headline_speedup_threshold=1.05,
    )

    assert rescored[0]["correct"] is True
    assert rescored[0]["inspection"]["authentic"] is False
    assert rescored[0]["pallas_credited"] is False
    assert rescored[0]["headline_credited"] is False
    assert "PALLAS_INTERPRET_MODE" in rescored[0]["no_credit_reasons"]
    assert rescored[1]["inspection"]["authentic"] is True
    assert rescored[1]["pallas_credited"] is True
    assert rescored[1]["headline_credited"] is True


def test_oracle_summary_quantifies_seed_variation(tmp_path: Path) -> None:
    bundle = load_contracts(CONFIG_ROOT)
    candidates = tuple(
        SampleCandidate(
            sample_id=f"1p_Flash_Attention::seed={seed}",
            workload="1p_Flash_Attention",
            seed=seed,
            kernel=tmp_path / f"{seed}.py",
            sample={
                "status": "sampled",
                "inspection": {"authentic": seed != 1},
                "attempts": [{"attempt": 0}],
            },
        )
        for seed in (0, 1, 2)
    )
    rows = [
        {
            "sample_id": candidate.sample_id,
            "compiled": candidate.seed != 1,
            "correct": candidate.seed == 0,
            "pallas_credited": candidate.seed == 0,
            "headline_credited": False,
            "timing": {"stable": candidate.seed == 0},
            "speedup": 0.9 if candidate.seed == 0 else None,
        }
        for candidate in candidates
    ]

    summary = _oracle_summary(
        bundle=bundle,
        candidates=candidates,
        rows=rows,
    )

    assert summary["n_samples"] == 3
    assert summary["parse_rate"] == 1.0
    assert summary["compilation_rate"] == round(2 / 3, 6)
    assert summary["correctness_rate"] == round(1 / 3, 6)
    assert summary["seed_rate_ranges"]["correctness_rate"] == 1.0
    assert summary["seed_consistency"]["n_workloads_with_any_correct"] == 1
    assert summary["seed_consistency"]["n_workloads_with_all_seeds_correct"] == 0


def test_static_rejection_does_not_launch_jaxbench(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = load_contracts(CONFIG_ROOT)
    kernel = tmp_path / "candidate.py"
    kernel.write_text("this is not python", encoding="utf-8")
    baseline_dir = (
        tmp_path
        / "jaxbench"
        / "JAXBench"
        / "benchmark"
        / "1p_Flash_Attention"
    )
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline.py").write_text(BASELINE, encoding="utf-8")
    candidate = SampleCandidate(
        sample_id="1p_Flash_Attention::seed=0",
        workload="1p_Flash_Attention",
        seed=0,
        kernel=kernel,
        sample={},
    )

    def fail_if_executed(**_: object) -> dict[str, object]:
        raise AssertionError("JAXBench must not execute a static rejection")

    monkeypatch.setattr(
        "opjax.pallas.evaluation._run_jaxbench_once",
        fail_if_executed,
    )

    result = _evaluate_workload(
        bundle=bundle,
        jaxbench_root=tmp_path / "jaxbench",
        candidate=candidate,
        prompt_context=PromptContext.SPEC,
        lowering_calibration=tmp_path / "calibration",
        lowering_evidence_root=tmp_path / "lowering",
        timeout_seconds=1,
    )

    assert result["execution_status"] == "STATIC_REJECTED"
    assert result["raw_runs"] == []
    assert result["compiled"] is False
    assert result["correct"] is False
