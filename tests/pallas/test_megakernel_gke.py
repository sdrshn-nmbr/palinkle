import pytest

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.megakernel_gke import (
    GKEJobSetConfig,
    IronwoodProvisioningConfig,
    build_jobset_manifest,
    build_jobset_for_contract,
    build_provisioning_plan,
    ironwood_slice,
)
from opjax.pallas.megakernel_contract import (
    OutputContract,
    TaskExecutionContract,
    TensorContract,
    TopologyContract,
)


def test_v7x_16_is_four_hosts_in_a_2x2x4_physical_topology() -> None:
    shape = ironwood_slice(16)

    assert shape.machine_type == "tpu7x-standard-4t"
    assert shape.physical_topology == "2x2x4"
    assert shape.host_count == 4
    assert shape.chips_per_host == 4


def test_unsupported_ironwood_slice_fails_closed() -> None:
    with pytest.raises(G42HarnessError, match="IRONWOOD_SLICE_UNSUPPORTED"):
        ironwood_slice(12)


def test_jobset_preserves_multi_host_topology_and_isolation() -> None:
    manifest = build_jobset_manifest(
        GKEJobSetConfig(
            name="opjax-mega-rpa-001",
            namespace="opjax-mega-rpa-001",
            task_id="vllm-stacked-rpa",
            image="us-docker.pkg.dev/opjax/bench/runner@sha256:" + "a" * 64,
            command=("python", "-m", "opjax.pallas.megakernel_worker"),
            placement_policy="opjax-tpu7x-2x2x4",
            chip_count=16,
        )
    )

    jobset = manifest["items"][1]
    job = jobset["spec"]["replicatedJobs"][0]["template"]["spec"]
    pod = job["template"]["spec"]
    container = pod["containers"][0]
    assert job["parallelism"] == job["completions"] == 4
    assert pod["nodeSelector"] == {
        "cloud.google.com/gke-tpu-accelerator": "tpu7x-standard-4t",
        "cloud.google.com/gke-tpu-topology": "2x2x4",
        "cloud.google.com/placement-policy-name": "opjax-tpu7x-2x2x4",
    }
    assert container["resources"]["limits"]["google.com/tpu"] == 4
    assert pod["automountServiceAccountToken"] is False
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"] == {"drop": ["ALL"]}


def test_jobset_requires_digest_pinned_image() -> None:
    with pytest.raises(G42HarnessError, match="GKE_IMAGE_DIGEST_REQUIRED"):
        build_jobset_manifest(
            GKEJobSetConfig(
                name="opjax-mega-rpa-001",
                namespace="opjax-mega-rpa-001",
                task_id="vllm-stacked-rpa",
                image="us-docker.pkg.dev/opjax/bench/runner:latest",
                command=("python", "worker.py"),
                placement_policy="opjax-tpu7x-2x2x4",
                chip_count=16,
            )
        )


def test_v7x_16_provisioning_plan_uses_flex_start_and_four_nodes() -> None:
    plan = build_provisioning_plan(
        IronwoodProvisioningConfig(
            project="astral-medley-465922-b2",
            cluster="opjax-megakernel",
            region="us-central1",
            zone="us-central1-c",
            node_pool="opjax-v7x-16",
            placement_policy="opjax-tpu7x-2x2x4",
            chip_count=16,
        )
    )

    assert plan["physical_topology"] == "2x2x4"
    assert plan["host_count"] == 4
    assert plan["create_policy"] == [
        "gcloud",
        "compute",
        "resource-policies",
        "create",
        "workload-policy",
        "opjax-tpu7x-2x2x4",
        "--type=HIGH_THROUGHPUT",
        "--accelerator-topology=2x2x4",
        "--project=astral-medley-465922-b2",
        "--region=us-central1",
    ]
    assert "--machine-type=tpu7x-standard-4t" in plan["create_node_pool"]
    assert "--max-nodes=4" in plan["create_node_pool"]
    assert "--flex-start" in plan["create_node_pool"]
    assert "--placement-policy=opjax-tpu7x-2x2x4" in plan["create_node_pool"]


def test_jobset_topology_is_derived_from_task_contract() -> None:
    contract = TaskExecutionContract(
        task_id="vllm-all-gather-matmul",
        topology=TopologyContract(
            accelerator_family="v7x",
            device_count=16,
            host_count=4,
            chips_per_host=4,
            physical_topology=(2, 2, 4),
            mesh=(("tensor", 16),),
            required_collectives=("all_gather",),
        ),
        inputs=(TensorContract("lhs", (16384, 4096), "bfloat16", ("tensor", None)),),
        outputs=(
            OutputContract(
                "output",
                (16384, 4096),
                "bfloat16",
                ("tensor", None),
                rtol=0.02,
                atol=0.01,
            ),
        ),
        mutable_state=(),
        correctness_seeds=(0, 1, 2),
    )

    manifest = build_jobset_for_contract(
        contract=contract,
        name="opjax-mega-agmm-001",
        namespace="opjax-mega-agmm-001",
        image="us-docker.pkg.dev/opjax/bench/runner@sha256:" + "a" * 64,
        command=("python", "-m", "opjax.pallas.megakernel_worker"),
        placement_policy="opjax-tpu7x-2x2x4",
    )

    job = manifest["items"][1]["spec"]["replicatedJobs"][0]["template"]["spec"]
    assert job["parallelism"] == 4
