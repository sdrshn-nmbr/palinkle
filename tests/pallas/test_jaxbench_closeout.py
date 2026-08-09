from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.jaxbench_closeout import (
    JaxBenchCloseoutError,
    validate_closeout,
)


REPO_ROOT = Path(__file__).parents[2]
CLOSEOUT_ROOT = REPO_ROOT / "data/pallas/runs/jaxbench-full-v1-closeout"


def test_phase2_full_jaxbench_closeout_is_hash_bound() -> None:
    result = validate_closeout(repo_root=REPO_ROOT, closeout_root=CLOSEOUT_ROOT)

    assert result["status"] == "completed"
    assert result["shape_policy"] == "original_unmodified"
    assert result["capability_task_count"] == 50
    assert result["scoreable_task_count"] == 47
    assert result["excluded_task_count"] == 3
    assert result["reference"]["task_id"] == "8p_GEMM"
    assert result["reference"]["worker_destroyed_at"]
    assert result["adversarial_wrong"]["task_id"] == "8p_GEMM"
    assert result["adversarial_wrong"]["worker_destroyed_at"]
    assert result["adversarial_mixed"]["task_id"] == "8p_GEMM"
    assert result["adversarial_mixed"]["worker_destroyed_at"]
    assert result["adversarial_xla_custom_call"]["task_id"] == "1p_Flash_Attention"
    assert result["adversarial_xla_custom_call"]["worker_destroyed_at"]
    assert result["scoreability"]["unscoreable"] == {
        "11p_Megablox_GMM": "lower_compile",
        "16p_Mamba2_SSD": "lower_compile",
        "2p_GQA_Attention": "lower_compile",
    }


def test_closeout_rejects_evidence_drift(tmp_path: Path) -> None:
    manifest = json.loads((CLOSEOUT_ROOT / "manifest.json").read_text())
    copied = tmp_path / "closeout"
    copied.mkdir()
    manifest["scoreable_task_count"] = 50
    (copied / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(JaxBenchCloseoutError, match="CLOSEOUT_MANIFEST_INVALID"):
        validate_closeout(repo_root=REPO_ROOT, closeout_root=copied)
