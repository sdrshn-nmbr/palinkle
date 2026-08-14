from __future__ import annotations

import argparse
import json
from pathlib import Path

from opjax.pallas.laguna_depth_selection import select_depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--arm", choices=("dflash", "dspark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_depth(args.summary, args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
