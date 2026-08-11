"""Authoritative grading entry point for frozen Gate 3.2 snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.phase31_experiment import load_contract
from opjax.pallas.phase32_experiment import validate_experiment
from opjax.pallas.phase3_grading import grade_sample_matrix


def remove_empty_unit_roots(output_root: Path) -> list[str]:
    results_root = output_root / "results"
    if not results_root.exists():
        return []
    removed = []
    for path in sorted(results_root.iterdir()):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            removed.append(path.name)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m opjax.pallas.phase32_grading")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/pallas/benchmarks/jaxbench-phase31"),
    )
    parser.add_argument(
        "--validity",
        type=Path,
        default=Path("data/pallas/runs/phase31-oracle-validity/manifest.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            "data/pallas/runs/phase31-positive-control-calibration/manifest.json"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("data/pallas/runs/phase32-base-capability/experiment.json"),
    )
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
        experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
        validate_experiment(value=experiment, contract=contract)
        removed = remove_empty_unit_roots(args.output_root.resolve())
        for unit_id in removed:
            print(f"PHASE32_GRADING_RESUME_REMOVED_EMPTY unit={unit_id}", file=sys.stderr)
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
        print(f"PHASE32_GRADING_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
