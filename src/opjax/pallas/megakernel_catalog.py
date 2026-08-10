"""Pinned source catalog for the real-megakernel benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256

SGLANG_REVISION = "ea706a305497897b4a5d3a25844f168185ddcbcf"
TPU_INFERENCE_REVISION = "a4cfff0f10cee7525295236ce7cd13eba230633e"

EXPECTED_FAMILIES = {
    "causal_convolution",
    "collective_matmul",
    "collective_reduce_scatter",
    "deepseek_v4_pipeline",
    "flash_attention",
    "fused_moe",
    "gated_delta_net",
    "grouped_matmul",
    "keyed_delta_attention",
    "multi_latent_attention",
    "quantized_matmul",
    "ragged_paged_attention",
    "simple_gla",
    "speculative_decoding",
    "structured_sparse_matmul",
}


@dataclass(frozen=True)
class SourceRepository:
    name: str
    url: str
    local_path: str
    revision: str


@dataclass(frozen=True)
class MegakernelTask:
    task_id: str
    family: str
    repository: str
    implementation_files: tuple[str, ...]
    oracle_files: tuple[str, ...]
    fusion_stages: int
    minimum_devices: int
    required_collectives: tuple[str, ...]
    output_names: tuple[str, ...]
    mutable_inputs: tuple[str, ...]
    admission_status: str
    source_sha256: dict[str, str]
    oracle_sha256: dict[str, str]


@dataclass(frozen=True)
class MegakernelCatalog:
    repositories: tuple[SourceRepository, ...]
    tasks: tuple[MegakernelTask, ...]

    @property
    def scored_tasks(self) -> tuple[MegakernelTask, ...]:
        return tuple(task for task in self.tasks if task.admission_status == "admitted")

    @property
    def registered_tasks(self) -> tuple[MegakernelTask, ...]:
        return tuple(task for task in self.tasks if task.admission_status == "registered")


REPOSITORIES = (
    SourceRepository(
        name="sglang-jax",
        url="https://github.com/sgl-project/sglang-jax",
        local_path="references/sglang-jax",
        revision=SGLANG_REVISION,
    ),
    SourceRepository(
        name="tpu-inference",
        url="https://github.com/vllm-project/tpu-inference",
        local_path="references/tpu-inference",
        revision=TPU_INFERENCE_REVISION,
    ),
)


def _task(
    task_id: str,
    family: str,
    repository: str,
    implementation_files: tuple[str, ...],
    oracle_files: tuple[str, ...],
    *,
    fusion_stages: int,
    minimum_devices: int = 1,
    required_collectives: tuple[str, ...] = (),
    output_names: tuple[str, ...] = ("output",),
    mutable_inputs: tuple[str, ...] = (),
    admitted: bool = False,
) -> MegakernelTask:
    return MegakernelTask(
        task_id=task_id,
        family=family,
        repository=repository,
        implementation_files=implementation_files,
        oracle_files=oracle_files,
        fusion_stages=fusion_stages,
        minimum_devices=minimum_devices,
        required_collectives=required_collectives,
        output_names=output_names,
        mutable_inputs=mutable_inputs,
        admission_status="admitted" if admitted else "registered",
        source_sha256={},
        oracle_sha256={},
    )


TASKS = (
    _task(
        "sglang-kda-32k-varlen",
        "keyed_delta_attention",
        "sglang-jax",
        ("python/sgl_jax/srt/kernels/kda/kda.py",),
        (
            "python/sgl_jax/srt/kernels/kda/naive.py",
            "python/sgl_jax/test/kernels/kda_test.py",
        ),
        fusion_stages=4,
        output_names=("output", "final_state"),
        admitted=True,
    ),
    _task(
        "sglang-fused-ep-moe-v1",
        "fused_moe",
        "sglang-jax",
        ("python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py",),
        ("python/sgl_jax/test/kernels/fused_moe_v1_test.py",),
        fusion_stages=6,
        minimum_devices=4,
        required_collectives=("all_to_all",),
    ),
    _task(
        "sglang-fused-ep-moe-v2",
        "fused_moe",
        "sglang-jax",
        ("python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py",),
        ("python/sgl_jax/test/kernels/fused_moe_v2_test.py",),
        fusion_stages=6,
        minimum_devices=4,
        required_collectives=("all_to_all",),
    ),
    _task(
        "sglang-rpa-v1",
        "ragged_paged_attention",
        "sglang-jax",
        ("python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py",),
        ("python/sgl_jax/test/flashattention_common.py",),
        fusion_stages=6,
        output_names=("attention_output", "updated_kv_cache"),
        mutable_inputs=("kv_cache",),
    ),
    _task(
        "sglang-rpa-v3",
        "ragged_paged_attention",
        "sglang-jax",
        ("python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py",),
        ("python/sgl_jax/test/flashattention_common.py",),
        fusion_stages=6,
        output_names=("attention_output", "updated_kv_cache"),
        mutable_inputs=("kv_cache",),
    ),
    _task(
        "sglang-mla-v2",
        "multi_latent_attention",
        "sglang-jax",
        ("python/sgl_jax/srt/kernels/mla/v2/kernel.py",),
        ("python/sgl_jax/test/test_mla_attention.py",),
        fusion_stages=5,
        output_names=("attention_output", "updated_kv_cache"),
        mutable_inputs=("kv_cache",),
    ),
    _task(
        "sglang-simple-gla",
        "simple_gla",
        "sglang-jax",
        (
            "python/sgl_jax/srt/kernels/simple_gla/simple_gla.py",
            "python/sgl_jax/srt/kernels/simple_gla/simple_gla_fused.py",
        ),
        (
            "python/sgl_jax/srt/kernels/simple_gla/native.py",
            "python/sgl_jax/test/kernels/simple_gla_fused_test.py",
        ),
        fusion_stages=3,
        output_names=("output", "final_state"),
    ),
    _task(
        "sglang-speculative-tree-pipeline",
        "speculative_decoding",
        "sglang-jax",
        (
            "python/sgl_jax/srt/kernels/speculative/build_eagle_tree_structure_kernel.py",
            "python/sgl_jax/srt/kernels/speculative/verify_tree_greedy_kernel.py",
            "python/sgl_jax/srt/kernels/speculative/tree_speculative_sampling_target_only_kernel.py",
        ),
        (
            "python/sgl_jax/srt/kernels/speculative/kernel.py",
            "python/sgl_jax/test/speculative/test_eagle_utils.py",
        ),
        fusion_stages=3,
        output_names=("tree", "accepted_tokens", "sampling_state"),
    ),
    _task(
        "vllm-gdn-v1",
        "gated_delta_net",
        "tpu-inference",
        ("tpu_inference/kernels/gdn/v1/fused_gdn_decode_kernel.py",),
        (
            "tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py",
            "tests/kernels/fused_gdn_kernel_test.py",
        ),
        fusion_stages=4,
        output_names=("output", "final_state"),
    ),
    _task(
        "vllm-gdn-v2",
        "gated_delta_net",
        "tpu-inference",
        ("tpu_inference/kernels/gdn/v2/gdn_decode_kernel.py",),
        (
            "tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py",
            "tests/kernels/gdn_decode_kernel_test.py",
        ),
        fusion_stages=4,
        output_names=("output", "final_state"),
    ),
    _task(
        "vllm-gdn-v3",
        "gated_delta_net",
        "tpu-inference",
        ("tpu_inference/kernels/gdn/v3/wrapper.py",),
        (
            "tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py",
            "tests/kernels/gdn_attention_v3_test.py",
        ),
        fusion_stages=4,
        output_names=("output", "final_state"),
    ),
    _task(
        "vllm-all-gather-matmul",
        "collective_matmul",
        "tpu-inference",
        ("tpu_inference/kernels/collectives/all_gather_matmul.py",),
        ("tests/kernels/collectives/all_gather_matmul_kernel_test.py",),
        fusion_stages=2,
        minimum_devices=8,
        required_collectives=("all_gather",),
    ),
    _task(
        "vllm-hierarchical-reduce-scatter",
        "collective_reduce_scatter",
        "tpu-inference",
        (
            "tpu_inference/kernels/collectives/hierrs_sc/kernel.py",
            "tpu_inference/kernels/collectives/hierrs_sc/dma_pipeline.py",
            "tpu_inference/kernels/collectives/hierrs_sc/wrapper.py",
        ),
        ("tpu_inference/kernels/collectives/hierrs_sc/wrapper.py",),
        fusion_stages=3,
        minimum_devices=8,
        required_collectives=("reduce_scatter", "d2d", "c2c"),
    ),
    _task(
        "vllm-stacked-rpa",
        "ragged_paged_attention",
        "tpu-inference",
        ("tpu_inference/kernels/experimental/stacked_rpa/kernel.py",),
        ("tests/kernels/stacked_rpa/stacked_rpa_test.py",),
        fusion_stages=7,
        output_names=("attention_output", "updated_kv_cache"),
        mutable_inputs=("kv_cache",),
    ),
    _task(
        "vllm-batched-rpa",
        "ragged_paged_attention",
        "tpu-inference",
        ("tpu_inference/kernels/experimental/batched_rpa/kernel.py",),
        ("tests/kernels/experimental/batched_rpa/test_batched_rpa_tuned_params.py",),
        fusion_stages=7,
        output_names=("attention_output", "updated_kv_cache"),
        mutable_inputs=("kv_cache",),
    ),
    _task(
        "vllm-dsv4-compress-store",
        "deepseek_v4_pipeline",
        "tpu-inference",
        ("tpu_inference/kernels/experimental/deepseek_v4/compress_and_store/kernel.py",),
        (
            "tpu_inference/kernels/experimental/deepseek_v4/compress_and_store/compress_store_ref.py",
            "tests/kernels/deepseek_v4/compress_store_test.py",
        ),
        fusion_stages=5,
        output_names=("compressed_state", "updated_cache"),
        mutable_inputs=("cache",),
    ),
    _task(
        "vllm-dsv4-mla",
        "deepseek_v4_pipeline",
        "tpu-inference",
        ("tpu_inference/kernels/experimental/deepseek_v4/core_attention/mla.py",),
        ("tests/kernels/deepseek_v4/mla_test.py",),
        fusion_stages=4,
    ),
    _task(
        "vllm-dsv4-mla-swa",
        "deepseek_v4_pipeline",
        "tpu-inference",
        ("tpu_inference/kernels/experimental/deepseek_v4/core_attention/mla_swa.py",),
        ("tests/kernels/deepseek_v4/mla_swa_test.py",),
        fusion_stages=5,
    ),
    _task(
        "vllm-dsv4-sparse-mla",
        "deepseek_v4_pipeline",
        "tpu-inference",
        ("tpu_inference/kernels/experimental/deepseek_v4/core_attention/sparse_mla.py",),
        ("tests/kernels/deepseek_v4/sparse_mla_test.py",),
        fusion_stages=5,
    ),
    _task(
        "vllm-dsv4-csa-gather",
        "deepseek_v4_pipeline",
        "tpu-inference",
        ("tpu_inference/kernels/experimental/deepseek_v4/core_attention/csa_gather.py",),
        ("tests/kernels/deepseek_v4/csa_gather_test.py",),
        fusion_stages=2,
    ),
    _task(
        "vllm-dsv4-streamindex-topk",
        "deepseek_v4_pipeline",
        "tpu-inference",
        ("tpu_inference/kernels/experimental/deepseek_v4/indexer/streamindex_topk.py",),
        ("tests/kernels/deepseek_v4/test_streamindex_topk.py",),
        fusion_stages=3,
        output_names=("indices", "scores"),
    ),
    _task(
        "vllm-dsv4-rope-o-projection",
        "deepseek_v4_pipeline",
        "tpu-inference",
        (
            "tpu_inference/kernels/experimental/deepseek_v4/rope.py",
            "tpu_inference/kernels/experimental/deepseek_v4/o_projection.py",
        ),
        (
            "tests/kernels/deepseek_v4/rope_test.py",
            "tests/kernels/deepseek_v4/o_projection_test.py",
        ),
        fusion_stages=4,
        output_names=("projected_output", "quantization_scale"),
    ),
    _task(
        "vllm-fused-ep-moe-v1",
        "fused_moe",
        "tpu-inference",
        ("tpu_inference/kernels/fused_moe/v1/kernel.py",),
        ("tests/kernels/fused_moe_v1_test.py",),
        fusion_stages=6,
        minimum_devices=4,
        required_collectives=("all_to_all",),
    ),
    _task(
        "vllm-causal-conv1d",
        "causal_convolution",
        "tpu-inference",
        ("tpu_inference/kernels/causal_conv1d/causal_conv1d.py",),
        ("tests/kernels/causal_conv1d_test.py",),
        fusion_stages=3,
        output_names=("output", "updated_conv_state"),
        mutable_inputs=("conv_state",),
    ),
    _task(
        "vllm-flash-attention",
        "flash_attention",
        "tpu-inference",
        ("tpu_inference/kernels/flash_attention/kernel.py",),
        ("tests/kernels/flash_attention_kernel_test.py",),
        fusion_stages=4,
    ),
    _task(
        "vllm-megablox-gmm-v1",
        "grouped_matmul",
        "tpu-inference",
        ("tpu_inference/kernels/megablox/gmm.py",),
        ("tests/kernels/gmm_test.py",),
        fusion_stages=2,
    ),
    _task(
        "vllm-megablox-gmm-v2",
        "grouped_matmul",
        "tpu-inference",
        ("tpu_inference/kernels/megablox/gmm_v2.py",),
        ("tests/kernels/gmm_test.py",),
        fusion_stages=3,
    ),
    _task(
        "vllm-mla-v2",
        "multi_latent_attention",
        "tpu-inference",
        ("tpu_inference/kernels/mla/v2/kernel.py",),
        ("tests/kernels/mla_v2_test.py",),
        fusion_stages=5,
        output_names=("attention_output", "updated_kv_cache"),
        mutable_inputs=("kv_cache",),
    ),
    _task(
        "vllm-quantized-matmul",
        "quantized_matmul",
        "tpu-inference",
        ("tpu_inference/kernels/quantized_matmul/__init__.py",),
        ("tests/kernels/quantized_matmul_kernel_test.py",),
        fusion_stages=2,
    ),
    _task(
        "vllm-rpa-v2",
        "ragged_paged_attention",
        "tpu-inference",
        ("tpu_inference/kernels/ragged_paged_attention/v2/kernel.py",),
        ("tests/kernels/ragged_paged_attention_kernel_v2_test.py",),
        fusion_stages=6,
        output_names=("attention_output", "updated_kv_cache"),
        mutable_inputs=("kv_cache",),
    ),
    _task(
        "vllm-rpa-v3",
        "ragged_paged_attention",
        "tpu-inference",
        ("tpu_inference/kernels/ragged_paged_attention/v3/kernel.py",),
        ("tests/kernels/ragged_paged_attention_kernel_v3_test.py",),
        fusion_stages=6,
        output_names=("attention_output", "updated_kv_cache"),
        mutable_inputs=("kv_cache",),
    ),
    _task(
        "vllm-structured-sparse-matmul-v1",
        "structured_sparse_matmul",
        "tpu-inference",
        ("tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py",),
        ("tests/kernels/spmm_v1_test.py",),
        fusion_stages=2,
    ),
)


def _git_revision(path: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise G42HarnessError(f"MEGAKERNEL_REPOSITORY_MISSING:{path}")
    return process.stdout.strip()


def load_megakernel_catalog(workspace_root: Path) -> MegakernelCatalog:
    repositories = {repository.name: repository for repository in REPOSITORIES}
    roots: dict[str, Path] = {}
    for repository in REPOSITORIES:
        root = workspace_root / repository.local_path
        revision = _git_revision(root)
        if revision != repository.revision:
            raise G42HarnessError(
                f"MEGAKERNEL_SOURCE_REVISION_DRIFT:{repository.name}:{revision}"
            )
        roots[repository.name] = root

    bound_tasks: list[MegakernelTask] = []
    for task in TASKS:
        if task.repository not in repositories:
            raise G42HarnessError(f"MEGAKERNEL_REPOSITORY_UNKNOWN:{task.repository}")
        if task.family not in EXPECTED_FAMILIES:
            raise G42HarnessError(f"MEGAKERNEL_FAMILY_UNKNOWN:{task.family}")
        root = roots[task.repository]
        all_files = task.implementation_files + task.oracle_files
        missing = [name for name in all_files if not (root / name).is_file()]
        if missing:
            raise G42HarnessError(
                f"MEGAKERNEL_SOURCE_FILE_MISSING:{task.task_id}:{missing[0]}"
            )
        hashes = {name: file_sha256(root / name) for name in task.implementation_files}
        oracle_hashes = {name: file_sha256(root / name) for name in task.oracle_files}
        bound_tasks.append(
            replace(task, source_sha256=hashes, oracle_sha256=oracle_hashes)
        )

    task_ids = [task.task_id for task in bound_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise G42HarnessError("MEGAKERNEL_TASK_ID_DUPLICATE")
    return MegakernelCatalog(
        repositories=REPOSITORIES,
        tasks=tuple(bound_tasks),
    )


def build_catalog_manifest(workspace_root: Path) -> dict[str, object]:
    catalog = load_megakernel_catalog(workspace_root)
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "opjax_megakernel_catalog",
        "contract_capabilities": {
            "exact_topology": True,
            "logical_mesh": True,
            "collective_attestation": True,
            "mutable_state": True,
            "multiple_outputs": True,
            "per_output_correctness": True,
            "pallas_output_ownership": True,
        },
        "repositories": [repository.__dict__ for repository in catalog.repositories],
        "tasks": [task.__dict__ for task in catalog.tasks],
        "counts": {
            "total": len(catalog.tasks),
            "admitted": len(catalog.scored_tasks),
            "registered": len(catalog.registered_tasks),
        },
    }
    payload["release_sha256"] = canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-megakernel-catalog")
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        manifest = build_catalog_manifest(args.workspace_root)
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"MEGAKERNEL_CATALOG_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
