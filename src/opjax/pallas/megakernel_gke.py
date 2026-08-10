"""GKE JobSet contracts for isolated multi-host Ironwood benchmark workers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.megakernel_contract import TaskExecutionContract

_DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_DIGEST_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class IronwoodSlice:
    chip_count: int
    physical_topology: str
    host_count: int
    chips_per_host: int = 4
    machine_type: str = "tpu7x-standard-4t"


@dataclass(frozen=True)
class GKEJobSetConfig:
    name: str
    namespace: str
    task_id: str
    image: str
    command: tuple[str, ...]
    placement_policy: str
    chip_count: int


@dataclass(frozen=True)
class IronwoodProvisioningConfig:
    project: str
    cluster: str
    region: str
    zone: str
    node_pool: str
    placement_policy: str
    chip_count: int


_IRONWOOD_SLICES = {
    4: IronwoodSlice(4, "2x2x1", 1),
    8: IronwoodSlice(8, "2x2x2", 2),
    16: IronwoodSlice(16, "2x2x4", 4),
}


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise G42HarnessError(f"{code}:{detail}")


def ironwood_slice(chip_count: int) -> IronwoodSlice:
    try:
        return _IRONWOOD_SLICES[chip_count]
    except KeyError as exc:
        raise G42HarnessError(f"IRONWOOD_SLICE_UNSUPPORTED:{chip_count}") from exc


def _validate_config(config: GKEJobSetConfig) -> IronwoodSlice:
    for label, value in (("name", config.name), ("namespace", config.namespace)):
        _require(
            _DNS_LABEL.fullmatch(value) is not None,
            "GKE_DNS_LABEL_INVALID",
            f"{label}={value}",
        )
    _require(bool(config.task_id), "GKE_TASK_ID_INVALID", repr(config.task_id))
    _require(
        _DIGEST_IMAGE.fullmatch(config.image) is not None,
        "GKE_IMAGE_DIGEST_REQUIRED",
        config.image,
    )
    _require(bool(config.command), "GKE_COMMAND_INVALID", repr(config.command))
    _require(
        _DNS_LABEL.fullmatch(config.placement_policy) is not None,
        "GKE_PLACEMENT_POLICY_INVALID",
        config.placement_policy,
    )
    return ironwood_slice(config.chip_count)


def build_jobset_manifest(config: GKEJobSetConfig) -> dict[str, Any]:
    shape = _validate_config(config)
    labels = {"app.kubernetes.io/name": config.name, "opjax/task-id": config.task_id}
    pod_selector = {
        "cloud.google.com/gke-tpu-accelerator": shape.machine_type,
        "cloud.google.com/gke-tpu-topology": shape.physical_topology,
        "cloud.google.com/placement-policy-name": config.placement_policy,
    }
    container = {
        "name": "candidate",
        "image": config.image,
        "imagePullPolicy": "IfNotPresent",
        "command": list(config.command),
        "env": [
            {"name": "OPJAX_TASK_ID", "value": config.task_id},
            {"name": "OPJAX_ACCELERATOR_FAMILY", "value": "v7x"},
            {"name": "OPJAX_PHYSICAL_TOPOLOGY", "value": shape.physical_topology},
            {"name": "OPJAX_GLOBAL_DEVICE_COUNT", "value": str(shape.chip_count)},
        ],
        "ports": [
            {"name": "jax-coordinator", "containerPort": 8471},
            {"name": "tpu-metrics", "containerPort": 8431},
        ],
        "resources": {
            "requests": {"google.com/tpu": shape.chips_per_host},
            "limits": {"google.com/tpu": shape.chips_per_host},
        },
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
            "readOnlyRootFilesystem": True,
        },
        "volumeMounts": [
            {"name": "scratch", "mountPath": "/tmp"},
            {"name": "artifacts", "mountPath": "/artifacts"},
        ],
    }
    jobset = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {"name": config.name, "namespace": config.namespace, "labels": labels},
        "spec": {
            "failurePolicy": {"maxRestarts": 0},
            "replicatedJobs": [
                {
                    "name": "slice",
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "parallelism": shape.host_count,
                            "completions": shape.host_count,
                            "completionMode": "Indexed",
                            "backoffLimit": 0,
                            "template": {
                                "metadata": {"labels": labels},
                                "spec": {
                                    "restartPolicy": "Never",
                                    "subdomain": config.name,
                                    "automountServiceAccountToken": False,
                                    "nodeSelector": pod_selector,
                                    "containers": [container],
                                    "volumes": [
                                        {"name": "scratch", "emptyDir": {}},
                                        {"name": "artifacts", "emptyDir": {}},
                                    ],
                                },
                            },
                        }
                    },
                }
            ],
        },
    }
    network_policy = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": config.name, "namespace": config.namespace},
        "spec": {
            "podSelector": {"matchLabels": labels},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [
                {
                    "from": [{"podSelector": {"matchLabels": labels}}],
                    "ports": [
                        {"protocol": "TCP", "port": 8471},
                        {"protocol": "TCP", "port": 8431},
                    ],
                }
            ],
            "egress": [
                {
                    "to": [{"podSelector": {"matchLabels": labels}}],
                    "ports": [
                        {"protocol": "TCP", "port": 8471},
                        {"protocol": "TCP", "port": 8431},
                    ],
                },
                {
                    "to": [
                        {
                            "namespaceSelector": {
                                "matchLabels": {"kubernetes.io/metadata.name": "kube-system"}
                            }
                        }
                    ],
                    "ports": [
                        {"protocol": "UDP", "port": 53},
                        {"protocol": "TCP", "port": 53},
                    ],
                },
            ],
        },
    }
    namespace = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": config.namespace, "labels": {"opjax/isolation": "megakernel"}},
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [namespace, jobset, network_policy],
    }


def build_jobset_for_contract(
    *,
    contract: TaskExecutionContract,
    name: str,
    namespace: str,
    image: str,
    command: tuple[str, ...],
    placement_policy: str,
) -> dict[str, Any]:
    topology = contract.topology
    _require(
        topology.accelerator_family == "v7x",
        "GKE_CONTRACT_ACCELERATOR_UNSUPPORTED",
        topology.accelerator_family,
    )
    shape = ironwood_slice(topology.device_count)
    expected_physical = tuple(int(value) for value in shape.physical_topology.split("x"))
    _require(
        topology.physical_topology == expected_physical
        and topology.host_count == shape.host_count
        and topology.chips_per_host == shape.chips_per_host,
        "GKE_CONTRACT_TOPOLOGY_MISMATCH",
        contract.task_id,
    )
    return build_jobset_manifest(
        GKEJobSetConfig(
            name=name,
            namespace=namespace,
            task_id=contract.task_id,
            image=image,
            command=command,
            placement_policy=placement_policy,
            chip_count=topology.device_count,
        )
    )


def build_provisioning_plan(config: IronwoodProvisioningConfig) -> dict[str, Any]:
    shape = ironwood_slice(config.chip_count)
    for label, value in (
        ("cluster", config.cluster),
        ("node_pool", config.node_pool),
        ("placement_policy", config.placement_policy),
    ):
        _require(
            _DNS_LABEL.fullmatch(value) is not None,
            "GKE_DNS_LABEL_INVALID",
            f"{label}={value}",
        )
    _require(bool(config.project), "GKE_PROJECT_INVALID", repr(config.project))
    _require(config.region == "us-central1", "GKE_IRONWOOD_REGION_INVALID", config.region)
    _require(config.zone == "us-central1-c", "GKE_IRONWOOD_ZONE_INVALID", config.zone)
    create_policy = [
        "gcloud",
        "compute",
        "resource-policies",
        "create",
        "workload-policy",
        config.placement_policy,
        "--type=HIGH_THROUGHPUT",
        f"--accelerator-topology={shape.physical_topology}",
        f"--project={config.project}",
        f"--region={config.region}",
    ]
    create_node_pool = [
        "gcloud",
        "container",
        "node-pools",
        "create",
        config.node_pool,
        f"--project={config.project}",
        f"--cluster={config.cluster}",
        f"--location={config.region}",
        f"--node-locations={config.zone}",
        f"--machine-type={shape.machine_type}",
        "--reservation-affinity=none",
        "--enable-autoscaling",
        "--num-nodes=0",
        "--min-nodes=0",
        f"--max-nodes={shape.host_count}",
        "--flex-start",
        f"--placement-policy={config.placement_policy}",
        "--consolidation-delay=60s",
    ]
    return {
        "machine_type": shape.machine_type,
        "physical_topology": shape.physical_topology,
        "chip_count": shape.chip_count,
        "host_count": shape.host_count,
        "chips_per_host": shape.chips_per_host,
        "create_policy": create_policy,
        "create_node_pool": create_node_pool,
        "delete_node_pool": [
            "gcloud",
            "container",
            "node-pools",
            "delete",
            config.node_pool,
            f"--project={config.project}",
            f"--cluster={config.cluster}",
            f"--location={config.region}",
            "--quiet",
        ],
    }
