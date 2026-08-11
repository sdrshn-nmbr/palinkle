from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request

from opjax.remote.config import modal_proxy_headers


MODEL = "thinkingmachines/Inkling-Small-NVFP4"
INSTRUCTIONS = [
    "Implement an authentic normally lowered Pallas kernel in kernel.py for matmul_relu with float32 inputs [256,384] and [384,512]. Do not use interpret mode or a plain-JAX fallback.",
    "Implement an authentic normally lowered Pallas kernel in kernel.py for layernorm with float32 input [320,512]. Do not use interpret mode or a plain-JAX fallback.",
    "Implement an authentic normally lowered Pallas kernel in kernel.py for log_softmax with float32 input [320,256]. Do not use interpret mode or a plain-JAX fallback.",
    "Implement an authentic normally lowered Pallas kernel in kernel.py for transpose_square with float32 input [256,512]. Do not use interpret mode or a plain-JAX fallback.",
]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def request(url: str, model: str, index: int, max_tokens: int) -> dict[str, float | int]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a JAX and Pallas kernel engineer. Think carefully and produce concrete code-oriented work.",
            },
            {"role": "user", "content": INSTRUCTIONS[index % len(INSTRUCTIONS)]},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **modal_proxy_headers()},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as response:
        payload = json.load(response)
    elapsed = time.perf_counter() - started
    usage = payload["usage"]
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
    }


def run(url: str, model: str, concurrency: int, requests: int, max_tokens: int) -> dict[str, object]:
    request(url, model, 0, min(32, max_tokens))
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda i: request(url, model, i, max_tokens), range(requests)))
    wall_s = time.perf_counter() - started
    completion_tokens = sum(int(row["completion_tokens"]) for row in rows)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    latencies = [float(row["elapsed_s"]) for row in rows]
    return {
        "url": url,
        "concurrency": concurrency,
        "requests": requests,
        "max_tokens": max_tokens,
        "wall_s": wall_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_tps": completion_tokens / wall_s,
        "input_tps": prompt_tokens / wall_s,
        "request_latency_s": {
            "mean": statistics.mean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.model, args.concurrency, args.requests, args.max_tokens), indent=2))


if __name__ == "__main__":
    main()
