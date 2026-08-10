import json
from pathlib import Path

import pytest

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256
from opjax.pallas.phase3_baseline import INKLING_MODEL_ID, LAGUNA_MODEL_ID
from opjax.pallas.phase31_experiment import load_contract
from opjax.pallas.phase32_experiment import build_experiment


REPO_ROOT = Path(__file__).parents[2]


def _conformance(path: Path, model_id: str) -> None:
    value = {
        "schema_version": 1,
        "kind": "opjax_phase31_provider_protocol_conformance",
        "provider": "sglang_openai",
        "model": {"model_id": model_id},
        "passed": True,
        "model_calls": 2,
        "repair_turns": 0,
        "messages": [],
        "outputs": [],
    }
    value["conformance_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _contract():
    return load_contract(
        release_root=REPO_ROOT / "data/pallas/benchmarks/jaxbench-phase31",
        validity_path=REPO_ROOT
        / "data/pallas/runs/phase31-oracle-validity/manifest.json",
        calibration_path=REPO_ROOT
        / "data/pallas/runs/phase31-positive-control-calibration/manifest.json",
    )


def test_gate32_freezes_standard_protocol_for_both_models(tmp_path: Path) -> None:
    inkling = tmp_path / "inkling.json"
    laguna = tmp_path / "laguna.json"
    _conformance(inkling, INKLING_MODEL_ID)
    _conformance(laguna, LAGUNA_MODEL_ID)
    release = json.loads(
        (
            REPO_ROOT / "data/pallas/benchmarks/jaxbench-phase31/manifest.json"
        ).read_text()
    )

    experiment = build_experiment(
        contract=_contract(),
        release=release,
        inkling_conformance=inkling,
        laguna_conformance=laguna,
    )

    assert experiment["phase"] == "3.2"
    assert experiment["harness"]["driver"] == "mini-swe-agent==2.4.6"
    assert experiment["harness"]["transport"] == "openai_chat_completions"
    assert {model["provider"] for model in experiment["models"]} == {
        "sglang_openai_inkling",
        "sglang_openai_laguna",
    }
    assert experiment["counts"]["trajectories"] == len(
        experiment["valid_task_ids"]
    ) * 3 * 2


def test_gate32_rejects_unproved_provider(tmp_path: Path) -> None:
    inkling = tmp_path / "inkling.json"
    laguna = tmp_path / "laguna.json"
    _conformance(inkling, INKLING_MODEL_ID)
    _conformance(laguna, LAGUNA_MODEL_ID)
    value = json.loads(laguna.read_text())
    value["passed"] = False
    value["conformance_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "conformance_sha256"}
    )
    laguna.write_text(json.dumps(value))
    release = json.loads(
        (
            REPO_ROOT / "data/pallas/benchmarks/jaxbench-phase31/manifest.json"
        ).read_text()
    )

    with pytest.raises(G42HarnessError, match="PHASE32_CONFORMANCE_INVALID"):
        build_experiment(
            contract=_contract(),
            release=release,
            inkling_conformance=inkling,
            laguna_conformance=laguna,
        )
