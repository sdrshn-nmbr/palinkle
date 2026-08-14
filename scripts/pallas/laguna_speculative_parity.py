from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from opjax.pallas.laguna_speculative import (
    ARMS,
    build_replay_corpus,
    canonical_sha256,
    run_replay_benchmark,
    select_parity_panel,
)
from opjax.remote.config import modal_proxy_headers

ENDPOINTS = {
    arm: f"https://conway--opjax-laguna-speculative-v1-{arm}.modal.run"
    for arm in ARMS
}
SAMPLE_ROOT = Path("data/pallas/runs/phase32-base-capability/laguna-samples")
OUTPUT_ROOT = Path("data/pallas/runs/laguna-speculative-v1/parity")


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    corpus = build_replay_corpus(sample_root=SAMPLE_ROOT.resolve())
    panel = select_parity_panel(corpus=corpus, size=48)
    _write(OUTPUT_ROOT / "panel.json", panel)

    def run_arm(arm: str) -> tuple[str, dict[str, Any]]:
        result = run_replay_benchmark(
            arm=arm,
            base_url=ENDPOINTS[arm],
            headers=modal_proxy_headers(),
            corpus=panel,
            concurrency=1,
            max_tokens=8192,
        )
        _write(OUTPUT_ROOT / f"{arm}.json", result)
        return arm, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = dict(executor.map(run_arm, ARMS))
    records = {
        arm: {row["prompt_id"]: row for row in results[arm]["records"]}
        for arm in ARMS
    }
    prompt_ids = sorted(records["plain"])
    token_matches = {
        arm: sum(
            records["plain"][prompt_id]["completion_token_ids"]
            == records[arm][prompt_id]["completion_token_ids"]
            for prompt_id in prompt_ids
        )
        for arm in ("dflash", "dspark")
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_speculative_parity_result",
        "panel_sha256": panel["release_sha256"],
        "prompts": len(prompt_ids),
        "max_tokens": 8192,
        "exact_token_matches_with_plain": token_matches,
        "pure_acceleration_gate": {
            arm: token_matches[arm] == len(prompt_ids)
            for arm in ("dflash", "dspark")
        },
        "results": {
            arm: {
                "result_sha256": results[arm]["result_sha256"],
                "output_tps": results[arm]["output_tps"],
                "completion_tokens": results[arm]["completion_tokens"],
                "finish_reasons": {
                    reason: sum(
                        row["finish_reason"] == reason for row in results[arm]["records"]
                    )
                    for reason in sorted(
                        {row["finish_reason"] for row in results[arm]["records"]}
                    )
                },
            }
            for arm in ARMS
        },
    }
    summary["sha256"] = canonical_sha256(summary)
    _write(OUTPUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
