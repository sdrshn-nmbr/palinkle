"""Local orchestration for Inkling Small through SGLang's OpenAI API."""

from __future__ import annotations

import json
from pathlib import Path

from opjax.pallas.phase31_conformance import run_two_turn_conformance
from opjax.pallas.phase31_experiment import load_contract
from opjax.pallas.phase32_experiment import validate_experiment
from opjax.pallas.phase3_sampling import sample_sglang_matrix
from opjax.pallas.sglang_agent import SGLangEndpointModel
from opjax.remote.config import modal_proxy_headers
from opjax.remote.inkling_small_sglang import (
    ENDPOINT_URL,
    MODEL_ID,
    MODEL_REVISION,
    PRECISION,
    SGLANG_REVISION,
    app,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@app.local_entrypoint()
def protocol_canary(
    out_path: str = "data/pallas/runs/phase32-provider-conformance/inkling-small.json",
) -> None:
    base_url = ENDPOINT_URL
    model = SGLangEndpointModel(
        base_url=base_url,
        api_key="EMPTY",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        runtime_revision=SGLANG_REVISION,
        precision=PRECISION,
        seed=0,
        max_tokens=512,
        temperature=0.2,
        top_p=0.95,
        proxy_headers=modal_proxy_headers(),
        reasoning_effort="high",
        chat_template_kwargs={"thinking": True},
    )
    result = run_two_turn_conformance(
        model=model,
        provider="sglang_openai",
        model_identity={
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_revision": SGLANG_REVISION,
            "precision": PRECISION,
            "endpoint": base_url,
            "transport": "openai_chat_completions",
        },
    )
    _write(Path(out_path).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def phase32(
    release_root: str = "data/pallas/benchmarks/jaxbench-phase31",
    validity_path: str = "data/pallas/runs/phase31-oracle-validity/manifest.json",
    calibration_path: str = "data/pallas/runs/phase31-positive-control-calibration/manifest.json",
    experiment_path: str = "data/pallas/runs/phase32-base-capability/experiment.json",
    out_dir: str = "data/pallas/runs/phase32-base-capability/inkling-samples",
    task_ids: str = "",
    seeds: str = "0,1,2",
    max_concurrency: int = 4,
) -> None:
    contract = load_contract(
        release_root=Path(release_root).resolve(),
        validity_path=Path(validity_path).resolve(),
        calibration_path=Path(calibration_path).resolve(),
    )
    experiment = validate_experiment(
        value=json.loads(Path(experiment_path).resolve().read_text(encoding="utf-8")),
        contract=contract,
    )
    base_url = ENDPOINT_URL
    manifest = sample_sglang_matrix(
        contract=contract,
        experiment=experiment,
        provider="sglang_openai_inkling",
        output_root=Path(out_dir).resolve(),
        base_url=base_url,
        api_key="EMPTY",
        runtime_revision=SGLANG_REVISION,
        precision=PRECISION,
        task_ids={value for value in task_ids.split(",") if value} or None,
        seeds={int(value) for value in seeds.split(",") if value},
        max_concurrency=max_concurrency,
        proxy_headers=modal_proxy_headers(),
        reasoning_effort="high",
        chat_template_kwargs={"thinking": True},
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
