from __future__ import annotations

import ast
import importlib.metadata
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import conversation_to_datum


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pallas" / "training-backends.json"

SOURCE_CONTRACTS = {
    "miles": {
        "scripts/run_inkling.py": {
            "ScriptArgs",
            "_get_parallel_config",
            "_train",
        },
        "miles/utils/types.py": {"Sample"},
        "miles/ray/rollout/rollout_manager.py": {"RolloutManager"},
        "miles/rollout/sglang_rollout.py": {"generate", "generate_rollout"},
        "miles/backends/training_utils/loss_hub/advantages.py": {
            "compute_advantages"
        },
        "miles/backends/training_utils/loss_hub/losses.py": {
            "policy_loss_function",
            "sft_loss_function",
        },
        "miles/backends/megatron_utils/checkpoint.py": {
            "load_checkpoint",
            "save_checkpoint_with_lora",
        },
        "miles/backends/megatron_utils/lora_utils.py": {
            "load_lora_adapter",
            "save_lora_checkpoint",
        },
        "miles/rollout/on_policy_distillation.py": {
            "post_process_rewards",
            "reward_func",
        },
        "miles_plugins/models/inkling/model.py": {
            "InklingGPTModel",
            "inkling_model_provider",
        },
        "miles_plugins/models/inkling/lora.py": {
            "InklingLoRAAdapter",
            "apply_inkling_lora",
            "export_inkling_lora_hf_named",
        },
        "miles_plugins/models/inkling/mm_processor.py": {
            "render_inkling_messages_to_ids"
        },
    },
    "sglang": {
        "python/sglang/srt/configs/laguna.py": {"LagunaConfig"},
        "python/sglang/srt/models/laguna.py": {"LagunaForCausalLM"},
        "python/sglang/srt/models/inkling.py": {
            "InklingCausalLLM",
            "InklingForConditionalGeneration",
        },
        "python/sglang/srt/parser/inkling_renderer.py": {
            "render_inkling_messages"
        },
        "python/sglang/srt/parser/inkling_tokenizer.py": {"InklingTokenizer"},
        "python/sglang/srt/parser/reasoning_parser.py": {"_PoolsideV1Detector"},
        "python/sglang/srt/lora/lora.py": {"LoRAAdapter", "LoRALayer"},
        "python/sglang/srt/lora/lora_manager.py": {"LoRAManager"},
        "python/sglang/srt/managers/io_struct.py": {
            "BatchTokenIDOutput",
            "GenerateReqInput",
            "LoadLoRAAdapterReqInput",
            "LoRAUpdateOutput",
        },
        "python/sglang/srt/entrypoints/http_server.py": {
            "load_lora_adapter",
        },
        "python/sglang/srt/state_capturer/routed_experts.py": {
            "RoutedExpertsCapturer",
            "get_global_experts_capturer",
        },
    },
}


def _run_git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"BACKEND_AUDIT_GIT_FAILED: {directory}: {detail}")
    return result.stdout.strip()


def _top_level_symbols(path: Path) -> set[str]:
    if not path.is_file():
        raise RuntimeError(f"BACKEND_AUDIT_SOURCE_MISSING: {path}")
    try:
        module = ast.parse(path.read_text())
    except SyntaxError as error:
        raise RuntimeError(
            f"BACKEND_AUDIT_PARSE_FAILED: {path}:{error.lineno}: {error.msg}"
        ) from error
    return {
        node.name
        for node in module.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef))
    }


def _audit_source_contract(
    name: str, contract: dict[str, set[str]], expected_revision: str
) -> dict[str, Any]:
    directory = ROOT / "references" / name
    revision = _run_git(directory, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise RuntimeError(
            "BACKEND_AUDIT_REVISION_DRIFT: "
            f"{name}: expected={expected_revision} actual={revision}"
        )
    dirty = bool(_run_git(directory, "status", "--short"))
    if dirty:
        raise RuntimeError(f"BACKEND_AUDIT_SUBMODULE_DIRTY: {name}")
    checked_files: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}

    for relative_path, expected_symbols in contract.items():
        symbols = _top_level_symbols(directory / relative_path)
        absent = sorted(expected_symbols - symbols)
        checked_files[relative_path] = sorted(expected_symbols)
        if absent:
            missing[relative_path] = absent

    if missing:
        raise RuntimeError(
            f"BACKEND_AUDIT_CONTRACT_DRIFT: {name}: {json.dumps(missing, sort_keys=True)}"
        )

    result: dict[str, Any] = {
        "revision": revision,
        "dirty": dirty,
        "checked_files": checked_files,
    }
    return result


def _audit_tinker(config: dict[str, Any]) -> dict[str, Any]:
    expected_versions = {
        "tinker": config["sdk_version"],
        "tinker-cookbook": config["cookbook_version"],
    }
    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in expected_versions
    }
    if versions != expected_versions:
        raise RuntimeError(
            "BACKEND_AUDIT_TINKER_VERSION_DRIFT: "
            f"expected={expected_versions} actual={versions}"
        )

    required_service_methods = {
        "create_lora_training_client",
        "create_sampling_client",
        "create_training_client_from_state",
        "get_server_capabilities",
    }
    missing = sorted(required_service_methods - set(dir(tinker.ServiceClient)))
    if missing:
        raise RuntimeError(f"BACKEND_AUDIT_TINKER_API_DRIFT: missing={missing}")

    return {
        "versions": versions,
        "signatures": {
            "ServiceClient": str(inspect.signature(tinker.ServiceClient)),
            "ServiceClient.create_lora_training_client": str(
                inspect.signature(tinker.ServiceClient.create_lora_training_client)
            ),
            "SamplingClient.sample": str(
                inspect.signature(tinker.SamplingClient.sample)
            ),
            "TrainingClient.forward_backward": str(
                inspect.signature(tinker.TrainingClient.forward_backward)
            ),
            "TrainingClient.optim_step": str(
                inspect.signature(tinker.TrainingClient.optim_step)
            ),
            "TrainingClient.save_state": str(
                inspect.signature(tinker.TrainingClient.save_state)
            ),
            "conversation_to_datum": str(inspect.signature(conversation_to_datum)),
            "get_renderer": str(inspect.signature(renderers.get_renderer)),
        },
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    if config.get("schema_version") != 1:
        raise RuntimeError("BACKEND_AUDIT_CONFIG_SCHEMA_INVALID")
    report = {
        "schema_version": 1,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "tinker": _audit_tinker(config["tinker"]),
        "backends": {
            name: _audit_source_contract(
                name,
                contract,
                config["backends"][name]["revision"],
            )
            for name, contract in SOURCE_CONTRACTS.items()
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
