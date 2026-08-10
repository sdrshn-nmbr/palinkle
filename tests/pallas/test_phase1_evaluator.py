from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from opjax.pallas.benchmarking import (
    BenchmarkingError,
    measure_interleaved,
    validate_timing_result,
)
from opjax.pallas.g42_harness import (
    G42HarnessError,
    parse_model_action,
    write_verifier_artifacts,
)
from opjax.pallas.lowering import LoweringEvidenceError, validate_execution_evidence
from opjax.pallas.environment_runner import (
    classify_missing_candidate_result,
    classify_seed_failure,
    classify_worker_failure,
    evaluate_task,
    validate_candidate_output,
)
from opjax.pallas.environment import verify_static
from opjax.pallas.task_semantics import operation_specification, render_task_instruction


def test_native_tool_action_is_preserved_without_rerendering() -> None:
    message = {
        "role": "assistant",
        "content": "I will inspect the task.",
        "tool_calls": [
            {
                "type": "function",
                "id": "call-1",
                "function": {
                    "name": "mswea_bash_command",
                    "arguments": {"command": "sed -n '1,160p' instruction.md"},
                },
            }
        ],
    }

    assert parse_model_action(message) == {
        "command": "sed -n '1,160p' instruction.md"
    }


def test_mixed_native_and_fenced_actions_fail_closed() -> None:
    message = {
        "role": "assistant",
        "content": "```mswea_bash_command\nls\n```",
        "tool_calls": [
            {
                "type": "function",
                "id": None,
                "function": {
                    "name": "mswea_bash_command",
                    "arguments": '{"command":"pwd"}',
                },
            }
        ],
    }

    with pytest.raises(G42HarnessError, match="ACTION_COUNT_INVALID"):
        parse_model_action(message)


def test_tinker_adapter_uses_renderer_parse_response_for_native_action() -> None:
    tinker_cookbook = pytest.importorskip("tinker_cookbook.renderers.base")
    from opjax.pallas.g42_agent import TinkerMiniSWEModel

    class Sequence:
        def __init__(self) -> None:
            self.tokens = [11, 22]
            self.stop_reason = "stop"

    class Response:
        def __init__(self) -> None:
            self.sequences = [Sequence()]

    class Future:
        def result(self) -> Response:
            return Response()

    class Client:
        def sample(self, **_: object) -> Future:
            return Future()

    class Renderer:
        parsed = False

        class ToolCall:
            def model_dump(self, *, mode: str) -> dict[str, object]:
                assert mode == "json"
                return {
                    "type": "function",
                    "id": "call-1",
                    "function": {
                        "name": "mswea_bash_command",
                        "arguments": {"command": "pwd"},
                    },
                }

        def build_generation_prompt(self, _: object) -> object:
            return object()

        def get_stop_sequences(self) -> list[int]:
            return []

        def parse_response(self, tokens: list[int]) -> tuple[dict[str, object], object]:
            assert tokens == [11, 22]
            self.parsed = True
            return (
                {
                    "role": "assistant",
                    "content": "inspection",
                    "tool_calls": [self.ToolCall()],
                },
                tinker_cookbook.ParseTermination.STOP_SEQUENCE,
            )

    class Tokenizer:
        def decode(self, _: list[int]) -> str:
            raise AssertionError("raw token decoding must not parse TML output")

    renderer = Renderer()
    model = TinkerMiniSWEModel(
        client=Client(),
        renderer=renderer,
        tokenizer=Tokenizer(),
        checkpoint=None,
        seed=0,
        max_tokens=128,
        temperature=0.2,
        top_p=0.95,
    )

    message = model.query([{"role": "user", "content": "inspect"}])

    assert renderer.parsed is True
    assert message["extra"]["actions"] == [{"command": "pwd"}]
    assert message["tool_calls"][0]["function"]["name"] == "mswea_bash_command"


def test_real_tml_renderer_round_trips_native_shell_action() -> None:
    base = pytest.importorskip("tinker_cookbook.renderers.base")
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.renderers.base import TrainOnWhat
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    model_id = "thinkingmachines/Inkling-Small"
    tokenizer = get_tokenizer(model_id)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(model_id),
        tokenizer,
        model_name=model_id,
    )
    model_input, _ = renderer.build_supervised_example(
        [
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call-1",
                        "function": {
                            "name": "mswea_bash_command",
                            "arguments": {"command": "pwd"},
                        },
                    }
                ],
            }
        ],
        TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    all_tokens = model_input.to_ints()
    message_model = tokenizer.tml_tokenizer.encode_special("message_model")
    tokens = all_tokens[all_tokens.index(message_model) :]

    message, termination = renderer.parse_response(tokens)

    assert termination is base.ParseTermination.STOP_SEQUENCE
    assert parse_model_action(dict(message)) == {"command": "pwd"}


@pytest.mark.parametrize(
    ("task", "required"),
    [
        (
            {
                "operation": "safe_divide",
                "input_shapes": [[640, 512], [640, 512]],
                "input_dtypes": ["float32", "float32"],
            },
            ("x0 / (abs(x1) + 0.25)", "shape [640, 512]"),
        ),
        (
            {
                "operation": "rmsnorm",
                "input_shapes": [[192, 384]],
                "input_dtypes": ["float32"],
            },
            ("axis -1", "epsilon 1e-05", "shape [192, 384]"),
        ),
        (
            {
                "operation": "row_sum",
                "input_shapes": [[192, 384]],
                "input_dtypes": ["float32"],
            },
            ("axis -1", "broadcast", "shape [192, 384]"),
        ),
    ],
)
def test_visible_task_spec_states_exact_hidden_semantics(
    task: dict[str, object], required: tuple[str, ...]
) -> None:
    specification = operation_specification(task)
    instruction = render_task_instruction(task, repair=None)

    assert specification["output_shape"] == task["input_shapes"][0]
    for phrase in required:
        assert phrase in instruction


def _profile_case(tmp_path: Path, *, marker: int, execute_count: int) -> Path:
    case_dir = tmp_path / "candidate"
    trace = case_dir / "trace" / "perfetto_trace.json.gz"
    trace.parent.mkdir(parents=True)
    stablehlo = case_dir / "stablehlo.mlir"
    executable = case_dir / "executable.hlo.txt"
    stablehlo.write_text("stablehlo", encoding="utf-8")
    executable.write_text("executable", encoding="utf-8")
    trace.write_bytes(b"trace")
    import hashlib

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = {
        "label": "candidate",
        "repetitions": 3,
        "correctness_verified": True,
        "compiler": {
            "stablehlo_sha256": sha(stablehlo),
            "executable_hlo_sha256": sha(executable),
            "stablehlo_markers": {"tpu_custom_call": marker},
            "executable_hlo_markers": {"tpu_custom_call": marker},
        },
        "trace": {
            "perfetto_relative_path": "trace/perfetto_trace.json.gz",
            "perfetto_sha256": sha(trace),
            "top_duration_event_names": [
                {"name": "tpu::System::Execute=>Done", "count": execute_count}
            ],
        },
    }
    (case_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    return case_dir


def test_profile_admission_requires_compiler_and_execute_evidence(tmp_path: Path) -> None:
    accepted = validate_execution_evidence(_profile_case(tmp_path / "ok", marker=1, execute_count=3))
    assert accepted["verified"] is True

    with pytest.raises(LoweringEvidenceError, match="TPU_CUSTOM_CALL_MISSING"):
        validate_execution_evidence(_profile_case(tmp_path / "marker", marker=0, execute_count=3))
    with pytest.raises(LoweringEvidenceError, match="TRACE_EXECUTION_MISSING"):
        validate_execution_evidence(_profile_case(tmp_path / "trace", marker=1, execute_count=0))


def test_artifact_stage_is_not_precredited_before_static_validation(
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text("this is not python", encoding="utf-8")

    result = evaluate_task(task={}, kernel_path=kernel)

    assert result["stage"] == "artifact_contract"
    assert result["stages"]["artifact_contract"] is False


def test_later_seed_compile_failure_remains_a_compile_failure() -> None:
    assert classify_seed_failure(
        phase="compile", error=RuntimeError("seed 2 lowering rejected")
    ) == "tpu_compile"
    assert classify_seed_failure(
        phase="execute", error=AssertionError("values differ")
    ) == "full_shape_correctness"
    assert classify_seed_failure(
        phase="execute", error=RuntimeError("Core halted during DMA")
    ) == "runtime_safety"


def test_candidate_output_requires_exact_shape_and_dtype() -> None:
    import numpy as np

    expected = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(AssertionError):
        validate_candidate_output(
            actual=np.ones((1, 3), dtype=np.float32),
            expected=expected,
            rtol=1e-3,
            atol=1e-3,
        )
    with pytest.raises(AssertionError):
        validate_candidate_output(
            actual=np.ones((2, 3), dtype=np.float16),
            expected=expected,
            rtol=1e-3,
            atol=1e-3,
        )


def test_candidate_process_abort_is_not_mislabeled_as_infrastructure() -> None:
    result = classify_missing_candidate_result(returncode=134, stderr="Aborted")

    assert result["phase"] == "execute"
    assert result["worker_recovery_required"] is True
    assert classify_worker_failure(result) == "runtime_safety"


def test_candidate_module_mutation_is_rejected_by_runtime_import_policy(
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "import chex\n"
        "chex.assert_trees_all_close = lambda *args, **kwargs: None\n"
        "def workload(x):\n"
        "    return x * 0\n",
        encoding="utf-8",
    )
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "operation": "relu",
                "input_shapes": [[4]],
                "input_dtypes": ["float32"],
                "correctness_seeds": [0, 1, 2],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "worker"
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "opjax.pallas.candidate_worker",
            "--task",
            str(task),
            "--kernel",
            str(kernel),
            "--output-dir",
            str(output),
            "--allow-cpu-test",
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
        check=False,
    )

    assert process.returncode == 2, process.stderr
    result = json.loads(process.stdout.splitlines()[-1])
    assert result["phase"] == "compile"
    assert "CANDIDATE_IMPORT_NOT_ALLOWED:chex" in result["error"]


def test_adversarial_pallas_fixture_reaches_dynamic_isolation_probe() -> None:
    fixture = Path(__file__).parent / "fixtures" / "malicious_chex_pallas.py"
    verdict = verify_static(f"```python\n{fixture.read_text(encoding='utf-8')}\n```")

    assert verdict.passed is True


def test_aborting_pallas_fixture_reaches_process_recovery_probe() -> None:
    fixture = Path(__file__).parent / "fixtures" / "aborting_pallas.py"
    verdict = verify_static(f"```python\n{fixture.read_text(encoding='utf-8')}\n```")

    assert verdict.passed is True


def test_interleaved_timing_randomizes_order_and_requires_material_lower_bound() -> None:
    calls: list[str] = []
    candidate = iter((0.90, 0.91, 0.89, 0.90, 0.90, 0.91, 0.89, 0.90, 0.90))
    baseline = iter((1.00, 1.01, 0.99, 1.00, 1.00, 1.01, 0.99, 1.00, 1.00))

    result = measure_interleaved(
        candidate=lambda: calls.append("candidate") or next(candidate),
        baseline=lambda: calls.append("baseline") or next(baseline),
        rounds=9,
        seed=7,
        material_speedup=1.05,
    )

    pairs = [calls[index : index + 2] for index in range(0, len(calls), 2)]
    assert {tuple(pair) for pair in pairs} == {
        ("candidate", "baseline"),
        ("baseline", "candidate"),
    }
    counts = {order: pairs.count(list(order)) for order in {tuple(pair) for pair in pairs}}
    assert max(counts.values()) - min(counts.values()) == 1
    assert result["speedup"] > 1.05
    assert result["speedup_ci95"][0] > 1.05
    assert result["materially_beats_xla"] is True


def test_interleaved_timing_rejects_nominal_but_uncertain_win() -> None:
    candidate = iter((0.70, 1.30, 0.70, 1.30, 0.70, 1.30, 0.70, 1.30, 0.70))
    baseline = iter((1.0,) * 9)

    result = measure_interleaved(
        candidate=lambda: next(candidate),
        baseline=lambda: next(baseline),
        rounds=9,
        seed=2,
        material_speedup=1.05,
    )

    assert result["unstable"] is True
    assert result["materially_beats_xla"] is False


def test_timing_artifact_recomputation_rejects_tampered_speedup() -> None:
    candidate = iter((0.90,) * 9)
    baseline = iter((1.00,) * 9)
    result = measure_interleaved(
        candidate=lambda: next(candidate),
        baseline=lambda: next(baseline),
        rounds=9,
        seed=0,
    )
    result["speedup"] = 9.0

    with pytest.raises(BenchmarkingError, match="TIMING_RESULT_MISMATCH"):
        validate_timing_result(result, seed=0)


def test_timing_artifact_rejects_a_forged_measurement_order() -> None:
    candidate = iter((0.90,) * 9)
    baseline = iter((1.00,) * 9)
    result = measure_interleaved(
        candidate=lambda: next(candidate),
        baseline=lambda: next(baseline),
        rounds=9,
        seed=0,
    )
    result["measurement_orders"] = [
        ["candidate", "baseline"],
        ["baseline", "candidate"],
    ] * 4 + [["candidate", "baseline"]]

    with pytest.raises(BenchmarkingError, match="TIMING_ORDER_SCHEDULE_INVALID"):
        validate_timing_result(result, seed=0)


def test_reward_artifact_does_not_infer_profile_or_speed_win(tmp_path: Path) -> None:
    payload = write_verifier_artifacts(
        result={
            "passed": False,
            "stage": "profile",
            "stages": {"artifact_contract": True, "profile": False},
            "profile": {
                "speedup": 1.20,
                "timing": {
                    "speedup": 1.20,
                    "speedup_ci95": [1.01, 1.30],
                    "materially_beats_xla": False,
                },
            },
        },
        output_dir=tmp_path,
        task_id="task",
        kernel_sha256="a" * 64,
    )

    assert payload["profiled"] is False
    assert payload["beats_xla"] is False
