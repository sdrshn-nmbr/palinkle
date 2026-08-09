#!/bin/bash
set -euo pipefail
test -f /app/kernel.py
mkdir -p /logs/artifacts
git -C /app add -A
git -C /app -c user.name=opjax-submit -c user.email=submit@opjax.invalid commit --allow-empty -q -m submission
git -C /app diff --binary $(git -C /app rev-list --max-parents=0 HEAD) HEAD > /logs/artifacts/model.patch
printf '%s\n' TPU_WORKER_REQUIRED >&2
exit 2
