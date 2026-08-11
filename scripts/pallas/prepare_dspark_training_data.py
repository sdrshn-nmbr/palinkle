from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


SOURCE = "https://huggingface.co/datasets/Avesed/Qwen3.6-27B-DSpark-data/resolve/main/pb_pool94k_clean.jsonl"
OUTPUT = Path("/tmp/opjax-dspark-training-data-20260810.jsonl")
LIMIT = 10_000


def valid(row: dict) -> bool:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return False
    if conversations[0].get("role") != "user":
        return False
    if conversations[-1].get("role") != "assistant":
        return False
    total_chars = sum(len(str(turn.get("content", ""))) for turn in conversations)
    return 64 <= total_chars <= 16_000


def main() -> None:
    count = 0
    digest = hashlib.sha256()
    with requests.get(SOURCE, stream=True, timeout=120) as response:
        response.raise_for_status()
        with OUTPUT.open("wb") as destination:
            for line in response.iter_lines():
                if not line:
                    continue
                row = json.loads(line)
                if not valid(row):
                    continue
                normalized = json.dumps(
                    {
                        "id": f"pb-{row['id']}",
                        "conversations": row["conversations"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode() + b"\n"
                destination.write(normalized)
                digest.update(normalized)
                count += 1
                if count == LIMIT:
                    break
    if count != LIMIT:
        raise RuntimeError(f"expected {LIMIT} rows, wrote {count}")
    print(json.dumps({"rows": count, "sha256": digest.hexdigest(), "path": str(OUTPUT)}))


if __name__ == "__main__":
    main()

