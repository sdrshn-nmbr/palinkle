from __future__ import annotations

import argparse
import json
from pathlib import Path

from opjax.pallas.phase31_experiment import load_contract
from opjax.pallas.phase32_experiment import validate_experiment
from opjax.pallas.phase3_sampling import sample_sglang_matrix
from opjax.pallas.laguna_speculative import validate_bound_replay_result
from opjax.remote.config import modal_proxy_headers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--serving-evidence", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/pallas/benchmarks/jaxbench-phase31"),
    )
    parser.add_argument(
        "--validity-path",
        type=Path,
        default=Path("data/pallas/runs/phase31-oracle-validity/manifest.json"),
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=Path(
            "data/pallas/runs/phase31-positive-control-calibration/manifest.json"
        ),
    )
    parser.add_argument(
        "--experiment-path",
        type=Path,
        default=Path("data/pallas/runs/phase32-base-capability/experiment.json"),
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    serving_result = json.loads(args.serving_evidence.read_text(encoding="utf-8"))
    runtime_evidence = validate_bound_replay_result(
        result=serving_result,
        selection=selection,
    )
    contract = load_contract(
        release_root=args.release_root.resolve(),
        validity_path=args.validity_path.resolve(),
        calibration_path=args.calibration_path.resolve(),
    )
    experiment = validate_experiment(
        value=json.loads(args.experiment_path.read_text(encoding="utf-8")),
        contract=contract,
    )
    result = sample_sglang_matrix(
        contract=contract,
        experiment=experiment,
        provider="sglang_openai_laguna",
        output_root=args.output_root.resolve(),
        base_url=args.endpoint,
        api_key="EMPTY",
        runtime_revision=runtime_evidence["runtime_sha256"],
        precision="bfloat16",
        task_ids={item for item in args.task_ids.split(",") if item} or None,
        seeds={int(item) for item in args.seeds.split(",") if item},
        max_concurrency=args.max_concurrency,
        proxy_headers=modal_proxy_headers(),
        chat_template_kwargs={"enable_thinking": True},
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
