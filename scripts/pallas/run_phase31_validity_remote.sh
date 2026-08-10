#!/usr/bin/env bash
set -euo pipefail

release_root=$1
evidence_root=$2
log_path=$3
worker_hostname=$4
shift 4

mkdir -p "$evidence_root"
for task_id in "$@"; do
  env \
    TPU_ACCELERATOR_TYPE=v5litepod-1 \
    TPU_WORKER_ID=0 \
    TPU_WORKER_HOSTNAMES="$worker_hostname" \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
    TPU_HOST_BOUNDS=1,1,1 \
    LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536 \
    TPU_SKIP_MDS_QUERY=1 \
    JAX_PLATFORMS=tpu \
    PYTHONPATH=/tmp/opjax-phase31-validity/input \
    /tmp/opjax-phase2-worker-venv-final12/bin/python \
      -m opjax.pallas.phase31_validity task \
      --release-root "$release_root" \
      --task-id "$task_id" \
      --out-path "$evidence_root/$task_id.json" >>"$log_path" 2>&1
done
