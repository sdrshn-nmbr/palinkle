"""Resumable base-model sampling for the frozen Phase 3 JAXBench matrix."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.g42_agent import TinkerMiniSWEModel, _close_service_holder
from opjax.pallas.g42_harness import (
    AGENT_IMAGE,
    G42HarnessError,
    canonical_sha256,
    file_sha256,
)
from opjax.pallas.jaxbench_agent import load_agent_task, run_jaxbench_agent
from opjax.pallas.phase3_baseline import (
    INKLING_MODEL_ID,
    Phase3Contract,
    load_phase3_contract,
    validate_sample_matrix,
)
from opjax.pallas.sampling import _sampling_client
from opjax.pallas.sglang_agent import SGLangEndpointModel

def select_cells(
    *,
    experiment: dict[str, Any],
    provider: str,
    task_ids: set[str] | None = None,
    seeds: set[int] | None = None,
) -> list[dict[str, Any]]:
    cells = [
        cell
        for cell in experiment["cells"]
        if cell["provider"] == provider
        and (task_ids is None or cell["task_id"] in task_ids)
        and (seeds is None or cell["seed"] in seeds)
    ]
    if task_ids is not None:
        observed = {cell["task_id"] for cell in cells}
        if observed != task_ids:
            raise G42HarnessError(
                f"PHASE3_TASK_FILTER_INVALID:{sorted(task_ids - observed)}"
            )
    if seeds is not None:
        observed_seeds = {cell["seed"] for cell in cells}
        if observed_seeds != seeds:
            raise G42HarnessError(
                f"PHASE3_SEED_FILTER_INVALID:{sorted(seeds - observed_seeds)}"
            )
    return cells


def cell_run_id(cell: dict[str, Any]) -> str:
    model = (
        "inkling-small-base"
        if cell["model_id"] == INKLING_MODEL_ID
        else "laguna-xs-21-base"
    )
    return f"{model}--{cell['task_id']}--seed-{cell['seed']}"


def validate_completed_run(
    *, path: Path, cell: dict[str, Any], experiment: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    trajectory_path = path / "trajectory.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise G42HarnessError(f"PHASE3_RUN_MANIFEST_INVALID:{path}") from exc
    if (
        manifest.get("kind") != "opjax_phase3_jaxbench_agent_run"
        or manifest.get("task_id") != cell["task_id"]
        or manifest.get("task_sha256") != cell["task_sha256"]
        or manifest.get("seed") != cell["seed"]
        or manifest.get("model", {}).get("model_id") != cell["model_id"]
        or manifest.get("model", {}).get("model_revision")
        != cell["model_revision"]
        or manifest.get("turn_limit") != 6
        or manifest.get("snapshot_turns") != [3, 6]
        or not trajectory_path.is_file()
        or manifest.get("trajectory_sha256") != file_sha256(trajectory_path)
    ):
        raise G42HarnessError(f"PHASE3_RUN_MANIFEST_MISMATCH:{path}")
    if experiment is not None and experiment.get("kind") in {
        "opjax_phase31_base_capability_experiment",
        "opjax_phase32_base_capability_experiment",
    }:
        identity = manifest.get("experiment_identity")
        if (
            not isinstance(identity, dict)
            or identity.get("experiment_sha256") != experiment.get("experiment_sha256")
            or manifest.get("agent_image") != experiment.get("harness", {}).get("agent_image")
        ):
            raise G42HarnessError(f"PHASE31_RUN_CONTRACT_MISMATCH:{path}")
    snapshots = manifest.get("snapshots", {})
    for turn in (3, 6):
        record = snapshots.get(str(turn), snapshots.get(turn))
        patch_path = path / "snapshots" / f"turn-{turn}.patch"
        kernel_path = path / "snapshots" / f"turn-{turn}-kernel.py"
        if (
            not isinstance(record, dict)
            or not patch_path.is_file()
            or not kernel_path.is_file()
            or record.get("patch_sha256") != file_sha256(patch_path)
            or record.get("kernel_sha256") != file_sha256(kernel_path)
        ):
            raise G42HarnessError(f"PHASE3_RUN_SNAPSHOT_INVALID:{path}:turn-{turn}")
    return manifest


def assemble_sample_manifest(
    *,
    experiment: dict[str, Any],
    cells: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    records = []
    for cell in cells:
        run_id = cell_run_id(cell)
        run_root = output_root / "runs" / run_id
        manifest = validate_completed_run(path=run_root, cell=cell, experiment=experiment)
        records.append(
            {
                **cell,
                "run_id": run_id,
                "run_path": f"runs/{run_id}",
                "run_manifest_sha256": file_sha256(run_root / "manifest.json"),
                "trajectory_sha256": manifest["trajectory_sha256"],
                "submitted": manifest["submitted"],
                "snapshots": manifest["snapshots"],
            }
        )
    result = {
        "schema_version": 1,
        "kind": "opjax_phase3_sample_matrix",
        "experiment_sha256": experiment["experiment_sha256"],
        "provider": cells[0]["provider"] if cells else None,
        "counts": {
            "runs": len(records),
            "snapshots": len(records) * 2,
            "submitted": sum(record["submitted"] for record in records),
        },
        "records": records,
    }
    result["release_sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_experiment(*, path: Path, contract: Phase3Contract) -> dict[str, Any]:
    validate_sample_matrix(path=path, contract=contract)
    return json.loads(path.read_text(encoding="utf-8"))


async def sample_tinker_matrix(
    *,
    contract: Phase3Contract,
    experiment: dict[str, Any],
    output_root: Path,
    task_ids: set[str] | None,
    seeds: set[int] | None,
    max_concurrency: int,
) -> dict[str, Any]:
    cells = select_cells(
        experiment=experiment,
        provider="tinker",
        task_ids=task_ids,
        seeds=seeds,
    )
    if max_concurrency < 1:
        raise G42HarnessError("PHASE3_CONCURRENCY_INVALID")
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(INKLING_MODEL_ID)
    renderer_name = model_info.get_recommended_renderer_name(INKLING_MODEL_ID)
    renderer = renderers.get_renderer(
        renderer_name, tokenizer, model_name=INKLING_MODEL_ID
    )
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0), follow_redirects=True
    )
    service = tinker.ServiceClient(http_client=http_client, max_retries=0)
    client = await _sampling_client(
        service=service,
        base_model=INKLING_MODEL_ID,
        model_path=None,
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    failures: list[str] = []

    async def run_cell(cell: dict[str, Any]) -> None:
        run_root = output_root / "runs" / cell_run_id(cell)
        if run_root.exists():
            validate_completed_run(path=run_root, cell=cell, experiment=experiment)
            return
        async with semaphore:
            model = TinkerMiniSWEModel(
                client=client,
                renderer=renderer,
                tokenizer=tokenizer,
                checkpoint=None,
                seed=cell["seed"],
                max_tokens=experiment["sampling"]["max_tokens"],
                temperature=experiment["sampling"]["temperature"],
                top_p=experiment["sampling"]["top_p"],
            )
            task = load_agent_task(
                release_root=contract.release_root,
                task_id=cell["task_id"],
            )
            identity = {
                "model_id": cell["model_id"],
                "model_revision": cell["model_revision"],
                "provider": "tinker",
                "provider_checkpoint": None,
                "tinker_sdk_version": version("tinker"),
                "renderer": renderer_name,
                "weight_identity": "provider_managed_base_bound_to_public_revision",
            }
            try:
                await asyncio.to_thread(
                    run_jaxbench_agent,
                    task=task,
                    output_dir=run_root,
                    model=model,
                    model_identity=identity,
                    seed=cell["seed"],
                    turn_limit=6,
                    snapshot_turns=(3, 6),
                    agent_image=experiment.get("harness", {}).get("agent_image", AGENT_IMAGE),
                    experiment_identity=(
                        {"experiment_sha256": experiment["experiment_sha256"]}
                        if experiment.get("kind") == "opjax_phase31_base_capability_experiment"
                        else None
                    ),
                )
            except Exception as exc:
                failures.append(f"{cell_run_id(cell)}:{type(exc).__name__}:{exc}")
                raise

    try:
        await asyncio.gather(*(run_cell(cell) for cell in cells))
    finally:
        try:
            _close_service_holder(service)
        finally:
            await http_client.aclose()
    if failures:
        raise G42HarnessError("PHASE3_TINKER_CELLS_FAILED:" + "|".join(failures))
    manifest = assemble_sample_manifest(
        experiment=experiment,
        cells=cells,
        output_root=output_root,
    )
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def sample_sglang_matrix(
    *,
    contract: Phase3Contract,
    experiment: dict[str, Any],
    provider: str,
    output_root: Path,
    base_url: str,
    api_key: str,
    runtime_revision: str,
    precision: str,
    task_ids: set[str] | None,
    seeds: set[int] | None,
    max_concurrency: int = 1,
    proxy_headers: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cells = select_cells(
        experiment=experiment,
        provider=provider,
        task_ids=task_ids,
        seeds=seeds,
    )
    if max_concurrency < 1:
        raise G42HarnessError("PHASE3_CONCURRENCY_INVALID")
    output_root.mkdir(parents=True, exist_ok=True)

    def run_cell(index: int, cell: dict[str, Any]) -> None:
        run_root = output_root / "runs" / cell_run_id(cell)
        if run_root.exists():
            validate_completed_run(path=run_root, cell=cell, experiment=experiment)
            return
        model = SGLangEndpointModel(
            base_url=base_url,
            api_key=api_key,
            model_id=cell["model_id"],
            model_revision=cell["model_revision"],
            runtime_revision=runtime_revision,
            precision=precision,
            seed=cell["seed"],
            max_tokens=experiment["sampling"]["max_tokens"],
            temperature=experiment["sampling"]["temperature"],
            top_p=experiment["sampling"]["top_p"],
            proxy_headers=proxy_headers,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
        )
        task = load_agent_task(
            release_root=contract.release_root,
            task_id=cell["task_id"],
        )
        identity = {
            "model_id": cell["model_id"],
            "model_revision": cell["model_revision"],
            "provider": provider,
            "transport": "openai_chat_completions",
            "endpoint": base_url,
            "runtime_revision": runtime_revision,
            "precision": precision,
            "weight_identity": "exact_hugging_face_revision",
        }
        run_jaxbench_agent(
            task=task,
            output_dir=run_root,
            model=model,
            model_identity=identity,
            seed=cell["seed"],
            turn_limit=6,
            snapshot_turns=(3, 6),
            agent_image=experiment.get("harness", {}).get("agent_image", AGENT_IMAGE),
            experiment_identity=(
                {"experiment_sha256": experiment["experiment_sha256"]}
                if experiment.get("kind")
                in {
                    "opjax_phase31_base_capability_experiment",
                    "opjax_phase32_base_capability_experiment",
                }
                else None
            ),
        )
        print(
            f"PHASE3_SGLANG_SAMPLE completed={index}/{len(cells)} "
            f"cell={cell_run_id(cell)}",
            flush=True,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_concurrency
    ) as executor:
        futures = [
            executor.submit(run_cell, index, cell)
            for index, cell in enumerate(cells, start=1)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    manifest = assemble_sample_manifest(
        experiment=experiment,
        cells=cells,
        output_root=output_root,
    )
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def _csv_set(value: str) -> set[str] | None:
    return {item for item in value.split(",") if item} or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase3-sample")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/pallas/benchmarks/jaxbench-v1"),
    )
    parser.add_argument(
        "--closeout-root",
        type=Path,
        default=Path("data/pallas/runs/jaxbench-full-v1-closeout"),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("data/pallas/runs/phase3-base-capability/experiment.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        contract = load_phase3_contract(
            release_root=args.release_root,
            closeout_root=args.closeout_root,
        )
        experiment = _load_experiment(path=args.experiment, contract=contract)
        task_ids = _csv_set(args.task_ids)
        seeds = {int(value) for value in args.seeds.split(",") if value}
        result = asyncio.run(
            sample_tinker_matrix(
                contract=contract,
                experiment=experiment,
                output_root=args.output_root,
                task_ids=task_ids,
                seeds=seeds,
                max_concurrency=args.max_concurrency,
            )
        )
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"PHASE3_SAMPLING_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
