from __future__ import annotations

import argparse
import json
from pathlib import Path

from opjax.pallas.dspark_artifacts import analyze_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-events", type=int, default=25)
    args = parser.parse_args()

    report = analyze_run(args.run_root, top_n=args.top_events)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
