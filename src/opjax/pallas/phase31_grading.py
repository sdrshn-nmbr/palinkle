"""Authoritative grading entry point for Phase 3.1 snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.phase31_experiment import load_contract, validate_experiment
from opjax.pallas.phase3_grading import grade_sample_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase31-grade")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--zone", default="us-west4-a")
    parser.add_argument("--max-concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        contract = load_contract(
            release_root=args.release_root,
            validity_path=args.validity,
            calibration_path=args.calibration,
        )
        experiment = json.loads(args.experiment.read_text())
        validate_experiment(value=experiment, contract=contract)
        result = grade_sample_matrix(
            release_root=args.release_root.resolve(),
            experiment=experiment,
            sample_root=args.sample_root.resolve(),
            output_root=args.output_root.resolve(),
            service_account=args.service_account,
            zone=args.zone,
            max_concurrency=args.max_concurrency,
            phase31=True,
        )
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"PHASE31_GRADING_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
