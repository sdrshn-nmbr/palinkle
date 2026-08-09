#!/bin/bash
set -uo pipefail
mkdir -p /logs/artifacts /logs/verifier
git config --global --add safe.directory /app
git -C /app add -A
git -C /app -c user.name=opjax-submit -c user.email=submit@opjax.invalid commit --allow-empty -q -m submission
git -C /app diff --binary $(git -C /app rev-list --max-parents=0 HEAD) HEAD > /logs/artifacts/model.patch
if [ -f /app/kernel.py ]; then
  stage=tpu_worker_required
  contract=1.0
  message=TPU_WORKER_REQUIRED
else
  stage=artifact_contract
  contract=0.0
  message=KERNEL_MISSING
fi
printf '%s\n' "$message" > /logs/verifier/run.log
printf '%s\n' "$message" > /logs/verifier/test-stdout.txt
printf '{\"passed\":false,\"stage\":\"%s\",\"error\":\"%s\",\"infrastructure_error\":false}\n' "$stage" "$message" > /logs/verifier/result.json
printf '{\"reward\":0,\"infrastructure_error\":0.0,\"stage_artifact_contract\":%s}\n' "$contract" > /logs/verifier/reward.json
printf '{\"tests\":[{\"name\":\"artifact_contract\",\"status\":\"%s\"}]}\n' "$([ "$contract" = 1.0 ] && printf passed || printf failed)" > /logs/verifier/ctrf.json
cp /logs/verifier/result.json /logs/verifier/score.json
exit 0
