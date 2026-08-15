from __future__ import annotations

import argparse
import json
from pathlib import Path

from opjax.pallas.laguna_live_public_gate import run_live_public_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("data/pallas/runs/phase32-base-capability/experiment.json"),
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/pallas/benchmarks/jaxbench-phase31"),
    )
    parser.add_argument(
        "--dflash-samples",
        type=Path,
        default=Path(
            "data/pallas/runs/laguna-speculator-training-v1/live-k6/dflash/samples"
        ),
    )
    parser.add_argument(
        "--dspark-samples",
        type=Path,
        default=Path(
            "data/pallas/runs/laguna-speculator-training-v1/live-k6/dspark/samples"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/pallas/runs/laguna-speculator-training-v1/live-k6/public-gate"
        ),
    )
    parser.add_argument(
        "--canary-workspace",
        type=Path,
        default=Path(
            "data/pallas/runs/g5-verifier/units/"
            "g5-s1--g43-benchmark-row-max-320x512--seed-0--turn-3/"
            "materialized-workspace"
        ),
    )
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()
    result = run_live_public_gate(
        experiment_path=args.experiment.resolve(),
        release_root=args.release_root.resolve(),
        sample_roots={
            "dflash": args.dflash_samples.resolve(),
            "dspark": args.dspark_samples.resolve(),
        },
        output_root=args.output_root.resolve(),
        canary_workspace=args.canary_workspace.resolve(),
        max_concurrency=args.max_concurrency,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
