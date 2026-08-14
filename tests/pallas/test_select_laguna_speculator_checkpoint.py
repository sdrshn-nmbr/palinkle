from __future__ import annotations

import json
from pathlib import Path

from opjax.pallas.laguna_checkpoint_selection import select_checkpoint


def _write(root: Path, step: int, tau: float, cross_entropy: float) -> None:
    path = root / "dflash" / "raw" / f"step_{step}" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "arm": "dflash",
                "split": "calibration",
                "step": step,
                "probabilistic_tau": tau,
                "greedy_tau": tau - 0.1,
                "loss": {"cross_entropy": cross_entropy},
            }
        )
    )


def test_selects_tau_then_loss_then_earlier_step(tmp_path: Path) -> None:
    _write(tmp_path, 13, 2.0, 1.0)
    _write(tmp_path, 26, 2.1, 1.2)
    _write(tmp_path, 39, 2.1, 1.1)
    _write(tmp_path, 52, 2.1, 1.1)
    result = select_checkpoint(tmp_path, "dflash")
    assert result["selected_step"] == 39
