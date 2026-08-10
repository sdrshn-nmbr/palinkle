from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from litellm import ModelResponse

from opjax.pallas.g42_harness import canonical_sha256, file_sha256
from opjax.pallas.phase31_conformance import run_two_turn_conformance
from opjax.pallas.sglang_agent import SGLangEndpointModel
from opjax.remote.laguna_baseline import summarize_baseline


def _response(command: str, call_id: str) -> ModelResponse:
    return ModelResponse(
        model="test-model",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": command}),
                            },
                        }
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _model() -> SGLangEndpointModel:
    return SGLangEndpointModel(
        base_url="https://example.invalid",
        api_key="secret-key",
        proxy_headers={"Modal-Secret": "secret-proxy"},
        model_id="poolside/Laguna-XS-2.1",
        model_revision="model-revision",
        runtime_revision="runtime-revision",
        precision="bfloat16",
        seed=7,
        max_tokens=1024,
        temperature=0.2,
        top_p=0.95,
        chat_template_kwargs={"enable_thinking": True},
    )


def _stub_responses(
    model: SGLangEndpointModel, responses: Iterator[ModelResponse]
) -> list[list[dict]]:
    calls: list[list[dict]] = []

    def query(messages: list[dict], **_: object) -> ModelResponse:
        calls.append(messages)
        return next(responses)

    model._query = query
    return calls


def test_stock_model_preserves_tool_call_and_linked_observation() -> None:
    model = _model()
    calls = _stub_responses(model, iter([_response("pwd", "call-1")]))

    message = model.query([{"role": "user", "content": "task"}])
    observation = model.format_observation_messages(
        message,
        [{"returncode": 0, "output": "/workspace", "exception_info": ""}],
    )[0]

    assert calls == [[{"role": "user", "content": "task"}]]
    assert message["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "command": "pwd"
    }
    assert message["extra"]["actions"] == [
        {"command": "pwd", "tool_call_id": "call-1"}
    ]
    assert observation["role"] == "tool"
    assert observation["tool_call_id"] == "call-1"
    assert observation["extra"]["returncode"] == 0


def test_stock_model_sends_complete_native_history_unchanged() -> None:
    model = _model()
    calls = _stub_responses(
        model,
        iter(
            [
                _response("printf PROTOCOL_ONE", "call-1"),
                _response("printf PROTOCOL_TWO", "call-2"),
            ]
        ),
    )

    result = run_two_turn_conformance(
        model=model,
        provider="sglang_openai",
        model_identity={"model_id": "test", "model_revision": "revision"},
    )

    assert result["passed"] is True
    second_request = calls[1]
    assert second_request[2]["role"] == "assistant"
    assert second_request[2]["tool_calls"][0]["id"] == "call-1"
    assert second_request[3]["role"] == "tool"
    assert second_request[3]["tool_call_id"] == "call-1"
    assert "PROTOCOL_ONE" in second_request[3]["content"]


def test_litellm_openai_transport_declares_tool_and_round_trips_history() -> None:
    requests: list[dict] = []
    responses = iter(
        [
            _response("printf PROTOCOL_ONE", "call-1").model_dump(mode="json"),
            _response("printf PROTOCOL_TWO", "call-2").model_dump(mode="json"),
        ]
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            body = json.dumps(next(responses)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = SGLangEndpointModel(
            base_url=f"http://127.0.0.1:{server.server_port}",
            api_key="EMPTY",
            model_id="test-model",
            model_revision="revision",
            runtime_revision="runtime",
            precision="bfloat16",
            seed=0,
            max_tokens=512,
            temperature=0.2,
            top_p=0.95,
        )
        result = run_two_turn_conformance(
            model=model,
            provider="sglang_openai",
            model_identity={"model_id": "test-model"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["passed"] is True
    assert requests[0]["tools"][0]["function"]["name"] == "bash"
    assert requests[1]["messages"][2]["tool_calls"][0]["id"] == "call-1"
    assert requests[1]["messages"][3]["role"] == "tool"
    assert requests[1]["messages"][3]["tool_call_id"] == "call-1"


def test_endpoint_configuration_uses_litellm_without_serializing_credentials() -> None:
    model = _model()

    assert model.config.model_name == "openai/poolside/Laguna-XS-2.1"
    assert model.config.model_kwargs["api_base"] == "https://example.invalid/v1"
    assert model.config.model_kwargs["seed"] == 7
    assert model.config.model_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    serialized = json.dumps(model.serialize())
    assert "sglang_openai" in serialized
    assert "secret-key" not in serialized
    assert "secret-proxy" not in serialized


def test_laguna_baseline_summary_validates_authoritative_evidence(
    tmp_path: Path,
) -> None:
    unit_id = "laguna--task--seed-0--turn-3"
    result_dir = tmp_path / "results" / unit_id
    result_dir.mkdir(parents=True)
    reward_path = result_dir / "reward.json"
    reward_path.write_text(
        json.dumps({"reward": 0, "failure_stage": "artifact_contract"}) + "\n"
    )
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "g42": {"submitted": False},
                "messages": [
                    {
                        "role": "assistant",
                        "content": "inspect",
                        "extra": {"actions": [{"command": "cat kernel.py"}]},
                    }
                ],
            }
        )
        + "\n"
    )
    manifest = {
        "schema_version": 1,
        "records": [
            {
                "unit_id": unit_id,
                "task_id": "task",
                "family": "elementwise_binary",
                "turn": 3,
                "trajectory_sha256": file_sha256(trajectory_path),
                "patch_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }
        ],
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    verification = {
        "input_release_sha256": manifest["release_sha256"],
        "counts": {"infrastructure_failures": 0},
        "records": [
            {
                "unit_id": unit_id,
                "reward": 0,
                "artifacts": {"reward.json": file_sha256(reward_path)},
            }
        ],
    }
    verification["release_sha256"] = canonical_sha256(verification)
    (tmp_path / "verification.json").write_text(json.dumps(verification) + "\n")
    unit_root = tmp_path / "units" / unit_id
    unit_root.mkdir(parents=True)
    (unit_root / "trajectory.json").write_bytes(trajectory_path.read_bytes())

    output_path = tmp_path / "summary.json"
    result = summarize_baseline(verifier_root=tmp_path, out_path=output_path)

    assert result["counts"] == {
        "tasks": 1,
        "units": 1,
        "profile_verified": 0,
        "candidate_failures": 1,
        "infrastructure_failures": 0,
        "nonempty_patches": 0,
    }
    assert result["horizons"] == {
        "k3": {
            "units": 1,
            "profile_verified": 0,
            "candidate_failures": 1,
            "infrastructure_failures": 0,
            "nonempty_patches": 0,
        }
    }
    assert result["turn_3_to_6_transitions"] is None
    assert result["agent_behavior"] == {
        "trajectories": 1,
        "model_calls": 1,
        "format_errors": 0,
        "submitted": 0,
        "commands_by_call": {"1": {"cat kernel.py": 1}},
    }
    assert result["failure_stages"] == {"artifact_contract": 1}
    assert (
        json.loads(output_path.read_text())["result_sha256"]
        == result["result_sha256"]
    )
