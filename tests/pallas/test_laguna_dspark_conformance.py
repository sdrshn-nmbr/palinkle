from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from opjax.pallas.laguna_dspark_conformance import (
    BOUNDARY_ORDER,
    ConformanceError,
    DFLASH_BOUNDARIES,
    build_dflash_conformance_report,
    build_conformance_report,
    build_target_feature_conformance_report,
    canonical_sha256,
    finalize_conformance,
    validate_conformance_report,
    validate_dflash_conformance_report,
    validate_target_feature_conformance_report,
)
from opjax.remote.laguna_dspark_capture import (
    begin_capture_round,
    capture_is_configured,
    capture_step,
    capture_tensor,
    load_target_feature_override,
)
from opjax.remote.laguna_vllm_conformance import (
    _canonicalize_capture,
    _disable_adaptive_verification,
)


EVIDENCE_ROOT = (
    Path(__file__).parents[2]
    / "data"
    / "pallas"
    / "runs"
    / "laguna-dspark-conformance-v1"
)
TRAINED_EVIDENCE_ROOT = (
    Path(__file__).parents[2]
    / "data"
    / "pallas"
    / "runs"
    / "laguna-speculator-training-v1"
    / "conformance"
)


def _write_array(root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    path = root / f"{name}.npy"
    np.save(path, value)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _capture(root: Path, *, token_delta: int = 0) -> dict[str, object]:
    root.mkdir(parents=True)
    tensors: dict[str, object] = {}
    for index, name in enumerate(BOUNDARY_ORDER[:-1]):
        values = np.arange(12, dtype=np.float32).reshape(3, 4) + index
        tensors[name] = _write_array(root, name, values)
    tokens = np.arange(15, dtype=np.int64) + token_delta
    tensors[BOUNDARY_ORDER[-1]] = _write_array(root, BOUNDARY_ORDER[-1], tokens)
    trace = root / "trace.json"
    trace.write_text('{"traceEvents":[{"name":"draft_round"}]}', encoding="utf-8")
    return {
        "implementation": root.name,
        "provenance": {"revision": "frozen", "source_sha256": "a" * 64},
        "prompt_token_ids": [1, 2, 3],
        "boundaries": tensors,
        "trace": _write_array_bytes(trace),
    }


def _write_array_bytes(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def test_exact_differential_capture_passes(tmp_path: Path) -> None:
    source_root = tmp_path / "deepspec"
    adapter_root = tmp_path / "vllm"
    source = _capture(source_root)
    adapter = _capture(adapter_root)
    report = build_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
        mutation_controls={
            "markov_matrix_swap": {"detected": True, "failed_boundary": "markov_bias"}
        },
    )
    validate_conformance_report(report, root=tmp_path)
    assert report["passed"] is True
    assert report["report_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def test_dflash_requires_exact_proposal_tokens(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    adapter_root = tmp_path / "adapter"
    source = _capture(source_root)
    adapter = _capture(adapter_root)
    source["boundaries"] = {
        name: source["boundaries"][name] for name in DFLASH_BOUNDARIES
    }
    adapter["boundaries"] = {
        name: adapter["boundaries"][name] for name in DFLASH_BOUNDARIES
    }
    source["manifest_sha256"] = "a" * 64
    adapter["manifest_sha256"] = "b" * 64
    report = build_dflash_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
    )
    assert report["passed"] is True
    proposal = adapter_root / "proposal_token_ids.npy"
    values = np.load(proposal)
    np.save(proposal, values + 1)
    adapter["boundaries"]["proposal_token_ids"]["sha256"] = (
        hashlib.sha256(proposal.read_bytes()).hexdigest()
    )
    report = build_dflash_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
    )
    assert report["passed"] is False


def test_dflash_validator_binds_profiles_and_manifests(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    adapter_root = tmp_path / "adapter"
    source = _capture(source_root)
    adapter = _capture(adapter_root)
    source["boundaries"] = {
        name: source["boundaries"][name] for name in DFLASH_BOUNDARIES
    }
    adapter["boundaries"] = {
        name: adapter["boundaries"][name] for name in DFLASH_BOUNDARIES
    }
    profile = adapter_root / "profile.json.gz"
    profile.write_bytes(b"profile")
    adapter["profiles"] = [_write_array_bytes(profile)]
    for root, manifest in ((source_root, source), (adapter_root, adapter)):
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        (root / "manifest.json").write_text(json.dumps(manifest))
    report = build_dflash_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
    )
    validate_dflash_conformance_report(report, root=tmp_path)
    profile.write_bytes(b"drift")
    with pytest.raises(ConformanceError, match="CAPTURE_ARTIFACT_HASH_MISMATCH"):
        validate_dflash_conformance_report(report, root=tmp_path)


def test_token_mismatch_fails_closed(tmp_path: Path) -> None:
    source_root = tmp_path / "deepspec"
    adapter_root = tmp_path / "vllm"
    report = build_conformance_report(
        source_root=source_root,
        source_capture=_capture(source_root),
        adapter_root=adapter_root,
        adapter_capture=_capture(adapter_root, token_delta=1),
        mutation_controls={
            "markov_matrix_swap": {"detected": True, "failed_boundary": "markov_bias"}
        },
    )
    assert report["passed"] is False
    with pytest.raises(ConformanceError, match="CONFORMANCE_FAILED"):
        validate_conformance_report(report, root=tmp_path)


def test_bfloat16_close_logits_allow_near_tie_but_final_tokens_remain_exact(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "deepspec"
    adapter_root = tmp_path / "vllm"
    source = _capture(source_root)
    adapter = _capture(adapter_root)
    source_base_path = source_root / "base_logits.npy"
    source_base = np.load(source_base_path)
    source_base[0] = [2.0, 2.01, 0.0, 0.0]
    np.save(source_base_path, source_base)
    source["boundaries"]["base_logits"]["sha256"] = (
        hashlib.sha256(source_base_path.read_bytes()).hexdigest()
    )
    base_path = adapter_root / "base_logits.npy"
    base = np.load(base_path)
    base[0] = [2.01, 2.0, 0.0, 0.0]
    np.save(base_path, base)
    adapter["boundaries"]["base_logits"]["sha256"] = (
        hashlib.sha256(base_path.read_bytes()).hexdigest()
    )
    report = build_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
        mutation_controls={
            "markov_matrix_swap": {
                "detected": True,
                "failed_boundary": "markov_bias",
            }
        },
    )
    assert report["comparisons"]["base_logits"]["argmax_match"] is False
    assert report["comparisons"]["base_logits"]["passed"] is True
    assert report["comparisons"]["proposal_token_ids"]["exact_match"] is True
    assert report["passed"] is True


def test_missing_profile_or_mutation_discrimination_fails(tmp_path: Path) -> None:
    source_root = tmp_path / "deepspec"
    adapter_root = tmp_path / "vllm"
    source = _capture(source_root)
    adapter = _capture(adapter_root)
    adapter.pop("trace")
    with pytest.raises(ConformanceError, match="CAPTURE_TRACE_MISSING"):
        build_conformance_report(
            source_root=source_root,
            source_capture=source,
            adapter_root=adapter_root,
            adapter_capture=adapter,
            mutation_controls={
                "markov_matrix_swap": {
                    "detected": False,
                    "failed_boundary": None,
                }
            },
        )


def test_artifact_drift_is_detected(tmp_path: Path) -> None:
    source_root = tmp_path / "deepspec"
    adapter_root = tmp_path / "vllm"
    source = _capture(source_root)
    adapter = _capture(adapter_root)
    report = build_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
        mutation_controls={
            "markov_matrix_swap": {"detected": True, "failed_boundary": "markov_bias"}
        },
    )
    (adapter_root / "base_logits.npy").write_bytes(b"drift")
    with pytest.raises(ConformanceError, match="ARTIFACT_HASH_MISMATCH"):
        validate_conformance_report(report, root=tmp_path)


def test_report_json_round_trip(tmp_path: Path) -> None:
    source_root = tmp_path / "deepspec"
    adapter_root = tmp_path / "vllm"
    report = build_conformance_report(
        source_root=source_root,
        source_capture=_capture(source_root),
        adapter_root=adapter_root,
        adapter_capture=_capture(adapter_root),
        mutation_controls={
            "markov_matrix_swap": {"detected": True, "failed_boundary": "markov_bias"}
        },
    )
    assert json.loads(json.dumps(report)) == report


def test_finalizer_binds_manifests_and_validates_artifacts(tmp_path: Path) -> None:
    source_root = tmp_path / "deepspec"
    adapter_root = tmp_path / "vllm"
    source = _capture(source_root)
    adapter = _capture(adapter_root)
    source["mutation_controls"] = {
        "markov_matrix_swap": {
            "detected": True,
            "failed_boundary": "markov_bias",
        }
    }
    for root, manifest in ((source_root, source), (adapter_root, adapter)):
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"
    report = finalize_conformance(
        source_root=source_root,
        adapter_root=adapter_root,
        output_path=output,
    )
    assert report["passed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_frozen_hardware_conformance_evidence_is_bound() -> None:
    report = json.loads((EVIDENCE_ROOT / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["report_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    assert report["comparisons"]["proposal_token_ids"]["exact_match"] is True
    assert all(comparison["passed"] for comparison in report["comparisons"].values())
    assert report["mutation_controls"]["markov_matrix_swap"]["detected"] is True

    remote = json.loads((EVIDENCE_ROOT / "remote.json").read_text(encoding="utf-8"))
    index_path = EVIDENCE_ROOT / "artifact-index.json"
    assert (
        remote["hf_artifact_index_sha256"]
        == hashlib.sha256(index_path.read_bytes()).hexdigest()
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert remote["hf_file_count"] == len(index["files"]) + 1
    assert remote["modal_runs"] == index["modal_runs"]


def test_trained_conformance_package_has_immutable_remote_locator() -> None:
    remote = json.loads(
        (TRAINED_EVIDENCE_ROOT / "remote-evidence.json").read_text(encoding="utf-8")
    )
    expected = canonical_sha256(
        {key: value for key, value in remote.items() if key != "sha256"}
    )
    assert remote["sha256"] == expected
    assert len(remote["hub"]["revision"]) == 40
    assert remote["hub"]["files"] == 504
    assert remote["clean_download_validation"] == {
        "conditional_adapter_report_valid": True,
        "content_tree_match": True,
        "live_target_feature_reports_valid": True,
    }
    conditional = json.loads(
        (TRAINED_EVIDENCE_ROOT / "dspark-step120-v1" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    first_live = json.loads(
        (
            TRAINED_EVIDENCE_ROOT
            / "dspark-step120-v1"
            / "target-feature-report.json"
        ).read_text(encoding="utf-8")
    )
    second_live = json.loads(
        (
            TRAINED_EVIDENCE_ROOT
            / "dspark-step120-live-validation-v1"
            / "target-feature-report.json"
        ).read_text(encoding="utf-8")
    )
    assert remote["reports"] == {
        "conditional_adapter_conformance": conditional["report_sha256"],
        "live_target_features_prompt_1": first_live["report_sha256"],
        "live_target_features_prompt_2": second_live["report_sha256"],
    }
    assert conditional["passed"] is True
    assert first_live["passed"] is False
    assert second_live["passed"] is False


def test_runtime_capture_is_explicit_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "capture"
    assert capture_is_configured() is False
    monkeypatch.setenv("OPJAX_DSPARK_CAPTURE_ROOT", str(capture_root))
    assert capture_is_configured() is True
    capture_tensor("inactive", torch.tensor([1], dtype=torch.int64))
    assert not capture_root.exists()

    capture_root.mkdir()
    (capture_root / "active.json").write_text(
        json.dumps({"session": "case-0"}), encoding="utf-8"
    )
    begin_capture_round()
    capture_tensor("hidden", torch.tensor([1.0], dtype=torch.bfloat16))
    output = capture_root / "case-0" / "hidden-000.npy"
    assert np.load(output, allow_pickle=False).dtype == np.float32
    ledger = json.loads(
        (capture_root / "case-0" / "ledger.jsonl").read_text(encoding="utf-8")
    )
    assert ledger["source_dtype"] == "torch.bfloat16"
    assert (
        ledger["sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )


def test_vllm_capture_reconstructs_the_exact_markov_chain(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    static = tmp_path / "static"
    output = tmp_path / "canonical"
    raw.mkdir()
    static.mkdir()
    output.mkdir()
    records: list[dict[str, object]] = []

    def record(name: str, index: int, value: np.ndarray) -> None:
        path = raw / f"{name}-{index:03d}.npy"
        np.save(path, value)
        records.append(
            {
                "name": name,
                "index": index,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    record("combined_target_feature", 0, np.zeros((3, 4), dtype=np.float32))
    record("raw_target_features", 0, np.zeros((3, 20), dtype=np.float32))
    record("draft_backbone_hidden_state", 0, np.zeros((15, 4), dtype=np.float32))
    record("draft_input_ids", 0, np.arange(15, dtype=np.int64))
    record("draft_positions", 0, np.arange(15, dtype=np.int64))
    record("draft_input_embeddings", 0, np.zeros((15, 4), dtype=np.float32))
    for layer_id in range(5):
        record(
            f"draft_layer_{layer_id}_output",
            0,
            np.zeros((15, 4), dtype=np.float32),
        )
    for name in (
        "layer0_input_norm",
        "layer0_qkv_projection",
        "layer0_q_norm",
        "layer0_k_norm",
        "layer0_gate_projection",
        "layer0_attention_output",
        "layer0_gated_attention",
        "layer0_post_attention_norm",
        "layer0_mlp_output",
        "layer0_query_q_after_rope",
        "layer0_query_k_after_rope",
        "layer0_query_v",
        "layer0_context_k_before_rope",
        "layer0_context_v",
    ):
        record(name, 0, np.zeros((15, 4), dtype=np.float32))
    base = np.zeros((15, 32), dtype=np.float32)
    record("base_logits", 0, base)
    previous = 7
    for step in range(15):
        bias = np.zeros((1, 32), dtype=np.float32)
        bias[0, step] = 1.0
        record("markov_input_token_ids", step, np.array([previous], dtype=np.int64))
        record("markov_embedding", step, np.zeros((1, 4), dtype=np.float32))
        record("markov_bias", step, bias)
        record("corrected_logits_runtime", step, bias)
        record(
            "proposal_token_ids_runtime",
            step,
            np.array([step], dtype=np.int64),
        )
        record(
            "confidence_logits_instrumented",
            step,
            np.zeros((1,), dtype=np.float32),
        )
        previous = step
    (raw / "ledger.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in records), encoding="utf-8"
    )
    static_records: list[dict[str, object]] = []
    for name, value in {
        "confidence_head_proj_weight": np.zeros((1, 8), dtype=np.float32),
        "confidence_head_proj_bias": np.zeros((1,), dtype=np.float32),
    }.items():
        path = static / f"{name}-000.npy"
        np.save(path, value)
        static_records.append(
            {
                "name": name,
                "index": 0,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (static / "ledger.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in static_records),
        encoding="utf-8",
    )
    result = _canonicalize_capture(raw, static, output)
    assert np.load(output / result["raw_target_features"]["path"]).shape == (3, 20)
    tokens = np.load(output / result["proposal_token_ids"]["path"])
    assert tokens.tolist() == list(range(15))


def test_eager_differential_lane_disables_only_adaptive_scheduling() -> None:
    command = [
        "vllm",
        "--speculative-config",
        json.dumps(
            {
                "method": "dspark",
                "num_speculative_tokens": 15,
                "enable_adaptive_verification": True,
            }
        ),
    ]
    _disable_adaptive_verification(command)
    config = json.loads(command[-1])
    assert config == {
        "method": "dspark",
        "num_speculative_tokens": 15,
        "enable_adaptive_verification": False,
    }


def test_target_feature_override_normalizes_the_source_batch_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((1, 52, 10240), dtype=np.float32))
    monkeypatch.setenv("OPJAX_DSPARK_TARGET_FEATURE_OVERRIDE", str(path))
    value = load_target_feature_override()
    assert value is not None
    assert tuple(value.shape) == (52, 10240)


def test_target_feature_override_tracks_proposal_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.full((1, 2, 4), 1.0, dtype=np.float32))
    np.save(second, np.full((1, 2, 4), 2.0, dtype=np.float32))
    monkeypatch.setenv("OPJAX_DSPARK_CAPTURE_ROOT", str(capture_root))
    (capture_root / "active.json").write_text(
        json.dumps(
            {
                "session": "round-overrides",
                "target_feature_overrides": [str(first), str(second)],
            }
        ),
        encoding="utf-8",
    )
    assert begin_capture_round() == 0
    assert torch.equal(load_target_feature_override(), torch.ones((2, 4)))
    assert begin_capture_round() == 1
    assert torch.equal(load_target_feature_override(), torch.full((2, 4), 2.0))


def test_target_feature_override_exhaustion_is_explicitly_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    feature = tmp_path / "feature.npy"
    np.save(feature, np.ones((1, 2, 4), dtype=np.float32))
    monkeypatch.setenv("OPJAX_DSPARK_CAPTURE_ROOT", str(capture_root))
    control = {
        "session": "exhaustion-policy",
        "target_feature_overrides": [str(feature)],
    }
    (capture_root / "active.json").write_text(
        json.dumps(control), encoding="utf-8"
    )
    assert begin_capture_round() == 0
    assert load_target_feature_override() is not None
    assert begin_capture_round() == 1
    with pytest.raises(RuntimeError, match="DSPARK_TARGET_FEATURE_OVERRIDE_EXHAUSTED"):
        load_target_feature_override()

    control["allow_native_after_override_exhaustion"] = True
    (capture_root / "active.json").write_text(
        json.dumps(control), encoding="utf-8"
    )
    assert load_target_feature_override() is None


def test_capture_step_supports_runtime_batched_and_unbatched_layouts() -> None:
    unbatched = torch.arange(12).reshape(3, 4)
    batched = torch.arange(24).reshape(2, 3, 4)
    assert torch.equal(capture_step(unbatched, 1), unbatched[1:2, :])
    assert torch.equal(capture_step(batched, 1), batched[:, 1, :])
    with pytest.raises(RuntimeError, match="LAGUNA_CAPTURE_STEP_RANGE"):
        capture_step(unbatched, 3)
    with pytest.raises(RuntimeError, match="LAGUNA_CAPTURE_STEP_RANGE"):
        capture_step(unbatched, -1)
    with pytest.raises(RuntimeError, match="LAGUNA_CAPTURE_STEP_RANK"):
        capture_step(torch.zeros(4), 0)


def test_target_feature_conformance_accepts_numeric_drift_and_preserves_order(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    adapter_root = tmp_path / "adapter-live"
    source_root.mkdir()
    adapter_root.mkdir()
    rng = np.random.default_rng(0)
    source_value = rng.normal(size=(1, 7, 20)).astype(np.float32)
    adapter_value = source_value.reshape(7, 20) * np.float32(1.001)
    source = {
        "manifest_sha256": "a" * 64,
        "prompt_token_ids": [1, 2],
        "boundaries": {
            "raw_target_features": _write_array(
                source_root, "raw_target_features", source_value
            )
        },
    }
    adapter = {
        "manifest_sha256": "b" * 64,
        "prompt_token_ids": [1, 2],
        "boundaries": {
            "raw_target_features": _write_array(
                adapter_root, "raw_target_features", adapter_value
            )
        },
    }

    report = build_target_feature_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
    )

    assert report["passed"] is True
    assert [layer["best_adapter_layer"] for layer in report["layers"]] == list(
        range(5)
    )
    validate_target_feature_conformance_report(report, root=tmp_path)


def test_target_feature_conformance_rejects_reordered_layers(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    adapter_root = tmp_path / "adapter-live"
    source_root.mkdir()
    adapter_root.mkdir()
    rng = np.random.default_rng(1)
    source_value = rng.normal(size=(6, 20)).astype(np.float32)
    adapter_value = source_value.reshape(6, 5, 4)[:, [1, 0, 2, 3, 4], :].reshape(6, 20)
    source = {
        "manifest_sha256": "a" * 64,
        "prompt_token_ids": [1, 2],
        "boundaries": {
            "raw_target_features": _write_array(
                source_root, "raw_target_features", source_value
            )
        },
    }
    adapter = {
        "manifest_sha256": "b" * 64,
        "prompt_token_ids": [1, 2],
        "boundaries": {
            "raw_target_features": _write_array(
                adapter_root, "raw_target_features", adapter_value
            )
        },
    }

    report = build_target_feature_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
    )

    assert report["passed"] is False
    validate_target_feature_conformance_report(
        report, root=tmp_path, require_pass=False
    )
