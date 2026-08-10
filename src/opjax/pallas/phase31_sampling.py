"""Sampling entry point for the frozen Phase 3.1 matched-base experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.phase31_experiment import load_contract, validate_experiment
from opjax.pallas.phase3_sampling import sample_tinker_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase31-sample")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        contract = load_contract(
            release_root=args.release_root,
            validity_path=args.validity,
            calibration_path=args.calibration,
        )
        experiment = json.loads(args.experiment.read_text())
        validate_experiment(value=experiment, contract=contract)
        task_ids = {value for value in args.task_ids.split(",") if value} or None
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
        print(f"PHASE31_SAMPLING_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
