from pathlib import Path

import pytest

from opjax.remote.phase1_miles_sglang_control import _tree_sha256


def test_tree_hash_rejects_missing_and_empty_roots(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="PHASE1_TREE_ROOT_MISSING"):
        _tree_sha256(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="PHASE1_TREE_EMPTY"):
        _tree_sha256(empty)


def test_tree_hash_binds_relative_paths_and_content(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"weights")
    original = _tree_sha256(root)

    (root / "weights.bin").write_bytes(b"changed")

    assert _tree_sha256(root) != original
