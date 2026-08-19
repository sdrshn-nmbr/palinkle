"""Periodically commit a Modal volume for a long-lived server process."""

from __future__ import annotations

import os
import sys
import time

import modal


def main() -> None:
    volume = modal.Volume.from_name(
        os.environ["OPJAX_SPEC_ARTIFACT_VOLUME"],
        environment_name=os.environ["OPJAX_SPEC_MODAL_ENVIRONMENT"],
        version=1,
    )
    while True:
        time.sleep(60)
        try:
            volume.commit()
        except Exception as error:
            print(
                f"LAGUNA_ARTIFACT_COMMIT_FAILED:{type(error).__name__}:{error}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
