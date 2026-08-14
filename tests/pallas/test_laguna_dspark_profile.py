from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from opjax.pallas.laguna_dspark_profile import (
    build_profile_analysis,
    prometheus_values,
)


def test_prometheus_parser_preserves_positions_and_finish_reasons() -> None:
    values = prometheus_values(
        '\n'.join(
            [
                'vllm:spec_decode_num_accepted_tokens_total 7',
                'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="2"} 3',
                'vllm:request_success_total{finished_reason="length"} 1',
            ]
        )
    )
    assert values == {
        "vllm:request_success_total.finished_length": 1.0,
        "vllm:spec_decode_num_accepted_tokens_per_pos_total.position_2": 3.0,
        "vllm:spec_decode_num_accepted_tokens_total": 7.0,
    }


def test_profile_analysis_is_hash_bound(tmp_path: Path) -> None:
    trace = tmp_path / "dp_rank0.trace.json.gz"
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "traceEvents": [
                    {"name": "cudaGraphLaunch", "dur": 5},
                    {"name": "_combine_sampled_and_draft_tokens_kernel", "dur": 2},
                ]
            },
            handle,
        )
    before = "vllm:spec_decode_num_drafts_total 1\n"
    after = "\n".join(
        [
            "vllm:spec_decode_num_drafts_total 3",
            "vllm:spec_decode_num_draft_tokens_total 31",
            "vllm:spec_decode_num_accepted_tokens_total 7",
        ]
    )
    (tmp_path / "metrics-before.prom").write_text(before)
    (tmp_path / "metrics-after.prom").write_text(after)
    manifest = {
        "manifest_sha256": "a" * 64,
        "elapsed_seconds": 1.0,
        "response": {"metrics": {"tokens_per_second": 10.0}},
        "trace_files": [
            {
                "path": trace.name,
                "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    analysis = build_profile_analysis(tmp_path)
    assert analysis["speculation"] == {
        "draft_rounds": 2.0,
        "drafted_tokens": 31.0,
        "accepted_tokens": 7.0,
        "acceptance_rate": 7 / 31,
        "accepted_tokens_per_round": 3.5,
    }
    assert analysis["trace"]["cudaGraphLaunch"]["calls"] == 1
