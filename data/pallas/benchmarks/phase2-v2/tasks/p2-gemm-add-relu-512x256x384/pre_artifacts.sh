#!/bin/bash
set -euo pipefail
mkdir -p /logs/artifacts
git add -A
git -c user.name=opjax-submit -c user.email=submit@opjax.invalid commit --allow-empty -q -m submission
git diff --binary $(git rev-list --max-parents=0 HEAD) HEAD > /logs/artifacts/model.patch
