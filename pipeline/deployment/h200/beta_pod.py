"""Scoped reserved allocation API used by beta round automation.

Run with the operator's existing approved Lepton login. No credentials are
returned, copied, or accepted in JSON. Planning never calls a mutating API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout

RESERVATION = "cosmos-solution-architect-xcwtffiq"
WORKSPACE = "b5k2m9x7"
# Explicit GPU-workload policy, not the CPU access pod's inherited defaults.
# These are the reviewed account-available levels; arbitrary/system priorities
# and out-of-reservation burst are never accepted by this allocator.
DEFAULT_GPU_PRIORITY_CLASS = "high-9000"
GPU_PRIORITY_CLASSES = ("high-7000", "high-9000")
SCALED_EVALUATION_RUN_ID = "dagger-leo-eval-scale8-20261024"


def gpu_queue_policy(priority_class=DEFAULT_GPU_PRIORITY_CLASS):
    if not isinstance(priority_class, str) or priority_class not in GPU_PRIORITY_CLASSES:
        raise ValueError("GPU priority must be one of the reviewed high-7000/high-9000 levels")
    return {"priority_class": priority_class, "can_preempt": True, "can_be_preempted": False}


def fresh_gpu_queue_config(source_spec, priority_class=DEFAULT_GPU_PRIORITY_CLASS):
    """Build a new typed queue model; inherit no CPU scheduling values.

    The typed SDK template normally supplies QueueConfig's class. An absent
    queue_config still uses the SDK QueueConfig, not a null/default policy.
    The lazy import keeps read-only inspection of old owners SDK-independent.
    """
    policy = gpu_queue_policy(priority_class)
    prototype = getattr(source_spec, "queue_config", None)
    if prototype is None:
        from leptonai.api.v1.types.deployment import QueueConfig
        queue_type = QueueConfig
    else:
        queue_type = type(prototype)
    return queue_type(**policy)


def error_status(exc):
    value = getattr(getattr(exc, "response", None), "status_code", None)
    if value is None:
        value = getattr(exc, "status_code", None)
    return value if type(value) is int and 100 <= value <= 599 else None


def transient_read(exc):
    if isinstance(exc, (RequestsConnectionError, RequestsTimeout, TimeoutError, json.JSONDecodeError)):
        return True
    if error_status(exc) in {408,429,500,502,503,504}:
        return True
    # The installed SDK discards the response object when its typed JSON
    # decoder fails. Recognize that precise SDK failure, never arbitrary
    # RuntimeError/ownership/auth errors, and never log its response body.
    return (type(exc) is RuntimeError and str(exc).startswith("You encountered a programming error. Please report this")
            and re.search(r"response returned status 2\d\d, but the content cannot be decoded as", str(exc)) is not None)


class AllocationReadError(RuntimeError):
    def __init__(self, stage, cause):
        self.stage, self.status_code = stage, error_status(cause)
        self.retryable, self.source_error_type = True, type(cause).__name__
        super().__init__("transient cloud allocation read exhausted bounded attempts")


class AllocationInvariantError(RuntimeError):
    """A bounded diagnostic code, never an SDK payload or credential string."""
    def __init__(self, reason_code, stage):
        self.reason_code, self.stage = reason_code, stage
        self.retryable = False
        super().__init__(reason_code)


def workspace_client(record=None):
    """Use the same explicit Beta-owned v2 login as the cloud tunnel.

    Never ask the obsolete v1/default-workspace path to rediscover a login or
    send a credential to an unexpected API origin. Bound every cloud call.
    """
    os.environ["LOGURU_LEVEL"] = "ERROR"
    if record is None:
        from leptonai.api.v2.workspace_record import WorkspaceRecord
        record = WorkspaceRecord
    client = record.client(WORKSPACE)
    url = urlsplit(client.url)
    if (url.scheme != "https" or url.netloc != "gateway.dgxc-lepton.nvidia.com"
            or url.path.rstrip("/") != f"/api/v2/workspaces/{WORKSPACE}"
            or url.query or url.fragment):
        raise AllocationInvariantError("workspace_api_origin_mismatch", "workspace_admission")
    client._timeout = 12
    client._header["Origin"] = "https://dashboard.dgxc-lepton.nvidia.com"
    return client


def name_for(run_id, round_id):
    if not re.fullmatch(r"dagger-(?:beta|leo)-[a-z0-9-]{1,64}", run_id) or type(round_id) is not int or round_id not in range(1, 5):
        raise ValueError("invalid campaign allocation identity")
    role = "leo" if run_id.startswith("dagger-leo-") else "beta"
    return f"leo-rr-{role}-" + hashlib.sha256(run_id.encode()).hexdigest()[:12] + f"-r{round_id}"


def allocation_name(run_id, round_id, gpus, existing_name=None):
    default = name_for(run_id, round_id)
    if existing_name is None:
        return default
    if (existing_name != "isaac-sim-render-leo" or gpus != 1 or round_id != 1
            or run_id != "dagger-leo-smoke-prebc-named-20261013"):
        raise ValueError("named allocation adoption is restricted to the user-owned Leo BC smoke")
    return existing_name


def attest(deployment, run_id, round_id, gpus=4, existing_name=None, infrastructure_attempt=None):
    spec = deployment.spec
    env = {e.name: e.value for e in spec.envs}
    if (deployment.metadata.id_ != allocation_name(run_id, round_id, gpus, existing_name)
            or spec.resource_requirement.resource_shape != f"custom.gpu.{gpus}xh200"
            or spec.reservation_config.reservation_id != RESERVATION
            or spec.reservation_config.allow_burst_to_other_reservations not in (None, False)
            or env.get("ROBORENDER_BETA_RUN_ID") != run_id
            or env.get("ROBORENDER_BETA_ROUND_ID") != str(round_id)
            or env.get("ROBORENDER_INFRASTRUCTURE_ATTEMPT") != infrastructure_attempt):
        raise AllocationInvariantError("owner_shape_reservation_mismatch", "deployment_attestation")
    if gpus == 8:
        queue = getattr(spec, 'queue_config', None)
        if (run_id != SCALED_EVALUATION_RUN_ID or round_id != 1
                or spec.resource_requirement.min_replicas != 1
                or spec.resource_requirement.max_replicas != 1
                or getattr(spec, 'auto_scaler', None) is not None
                or any(getattr(queue, key, None) != value for key, value in gpu_queue_policy().items())):
            raise AllocationInvariantError('scaled_evaluation_retention_policy_mismatch', 'deployment_attestation')


def read_deployment(client, name, *, sleep=time.sleep):
    """Read one exact owner; tolerate deletion races and bounded API outages.

    Only a real HTTP 404 means absent. Authentication/ownership failures are
    never retried or converted to absence. This never retries a mutation.
    """
    for attempt in range(3):
        try:
            return client.deployment.get(name)
        except Exception as exc:
            status = error_status(exc)
            if status == 404:
                return None
            if not transient_read(exc):
                raise
            if attempt == 2:
                raise AllocationReadError("deployment_read", exc) from None
            sleep(2 ** attempt)


def inspect_replicas(client, name, *, sleep=time.sleep):
    """Read raw readiness: older SDK replica models discard scheduler issues.

    Return only public placement identities/reason, not envs, API headers,
    credentials, arbitrary scheduler messages, or the full response body.
    """
    for attempt in range(3):
        try:
            response = client._get(f"/deployments/{name}/replicas", allow_redirects=False)
            response.raise_for_status()
            values = response.json()
            break
        except Exception as exc:
            if not transient_read(exc):
                raise
            if attempt == 2:
                raise AllocationReadError("replica_read", exc) from None
            sleep(2 ** attempt)
    if not isinstance(values, list):
        raise AllocationInvariantError("replica_response_not_list", "replica_read")
    if len(values) > 1:
        raise AllocationInvariantError("multiple_replica_response", "replica_read")
    result = []
    for value in values:
        status = value.get("status") or {}
        identity = (value.get("metadata") or {}).get("id") or value.get("id")
        node = status.get("node") or {}
        node_id = node.get("id") or node.get("name")
        reason = (status.get("readiness_issue") or {}).get("reason", "Unknown")
        if (not isinstance(identity, str) or not identity.startswith(name + "-")
                or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,200}", identity)
                or (node_id is not None and (not isinstance(node_id, str)
                    or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,200}", node_id)))
                or not isinstance(reason, str) or not re.fullmatch(r"[a-zA-Z0-9_. -]{1,128}", reason)):
            raise AllocationInvariantError("replica_placement_identity_invalid", "replica_read")
        result.append({"replica_id": identity, "node_id": node_id, "reason": reason})
    return result


def operate(client, action, run_id, round_id, public_keys=(), *, gpus=4, existing_name=None,
            infrastructure_attempt=None, priority_class=DEFAULT_GPU_PRIORITY_CLASS,
            confirm_release=None):
    queue_policy = gpu_queue_policy(priority_class)
    if infrastructure_attempt is not None and (not isinstance(infrastructure_attempt, str)
            or not re.fullmatch(r"[a-f0-9]{32}", infrastructure_attempt)):
        raise ValueError("invalid infrastructure attempt")
    # Offline rendering/training can use one H200; it must not reserve the
    # four-worker online pool merely to satisfy an old launcher's shape gate.
    # Keep a distinct owner namespace and round 1 for this concrete workload.
    one_gpu_bc = (gpus == 1 and round_id == 1 and run_id.startswith((
        "dagger-beta-smoke-prebc-", "dagger-leo-smoke-prebc-", "dagger-leo-bc-")))
    eight_gpu_evaluation = (type(gpus) is int and gpus == 8 and round_id == 1
        and run_id == SCALED_EVALUATION_RUN_ID and existing_name is None
        and infrastructure_attempt is None and priority_class == DEFAULT_GPU_PRIORITY_CLASS)
    if run_id == SCALED_EVALUATION_RUN_ID and not eight_gpu_evaluation:
        raise ValueError('scaled evaluation namespace admits exactly one high-9000 eight-H200 request in round 1')
    if gpus != 4 and not one_gpu_bc and not eight_gpu_evaluation:
        raise ValueError("one H200 requires an isolated BC workload identity in round 1")
    name = allocation_name(run_id, round_id, gpus, existing_name)
    if confirm_release is not None and (action != "delete" or confirm_release != name):
        raise ValueError("release confirmation must name this exact owned pod and accompany delete")
    dep = read_deployment(client, name)
    if dep is not None:
        attest(dep, run_id, round_id, gpus, existing_name, infrastructure_attempt)
        if action == "create":
            return {"name": name, "present": True, "idempotent": True}
        if action == "inspect":
            return {"name": name, "present": True, "replicas": inspect_replicas(client, name)}
        if action == "delete":
            if confirm_release != name:
                return {"name": name, "present": True, "retained": True,
                        "delete_requested": False, "reason": "explicit_user_release_required"}
            try:
                client.deployment.delete(name)
            except Exception as exc:
                if getattr(getattr(exc, "response", None), "status_code", None) != 404:
                    raise
                return {"name": name, "present": False}
            return {"name": name, "delete_requested": True}
        return {"name": name, "present": True}
    if action in {"delete", "status", "inspect"}:
        return {"name": name, "present": False}
    if existing_name is not None:
        raise RuntimeError("the explicitly adopted named allocation no longer exists")
    if len(public_keys) != 2 or any(not k.startswith("ssh-ed25519 ") or "\n" in k for k in public_keys):
        raise ValueError("exactly two approved Ed25519 public keys required")
    source = client.deployment.get("leo-cosmos-h200-access")
    spec = source.spec.model_copy(deep=True)
    spec.resource_requirement.resource_shape = f"custom.gpu.{gpus}xh200"
    spec.resource_requirement.min_replicas = spec.resource_requirement.max_replicas = 1
    spec.reservation_config.reservation_id = RESERVATION
    spec.reservation_config.allow_burst_to_other_reservations = False
    spec.queue_config = fresh_gpu_queue_config(source.spec, priority_class)
    spec.container.command = ["/bin/bash", "/workspace/project/roborender/pipeline/deployment/h200/beta_pod_entrypoint.sh"]
    env_type = type(source.spec.envs[0])
    spec.envs = [env_type(name="ROBORENDER_BETA_PUBLIC_KEYS", value="\n".join(public_keys)),
                 env_type(name="ROBORENDER_BETA_RUN_ID", value=run_id),
                 env_type(name="ROBORENDER_BETA_ROUND_ID", value=str(round_id))]
    if infrastructure_attempt is not None:
        spec.envs.append(env_type(name="ROBORENDER_INFRASTRUCTURE_ATTEMPT", value=infrastructure_attempt))
    spec.auto_scaler = None
    spec.api_tokens = None
    spec.is_pod = True
    request = type(source)(metadata=type(source.metadata)(name=name), spec=spec)
    if action == "create":
        client.deployment.create(request)
    elif action != "plan":
        raise ValueError("unknown operation")
    return {"name": name, "created": action == "create", "shape": f"custom.gpu.{gpus}xh200",
            "reservation": RESERVATION, "allow_burst": False, "queue_config": queue_policy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "create", "status", "inspect", "delete"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--public-key", action="append", default=[])
    parser.add_argument("--gpus", type=int, choices=[1, 4, 8], default=4)
    parser.add_argument("--existing-name", help="Adopt only the explicitly user-owned Leo BC smoke allocation")
    parser.add_argument("--infrastructure-attempt")
    parser.add_argument("--priority-class", choices=GPU_PRIORITY_CLASSES, default=DEFAULT_GPU_PRIORITY_CLASS,
                        help="Scheduling policy for a new owned GPU allocation; never modifies an existing owner")
    parser.add_argument("--confirm-release", metavar="EXACT_POD_NAME",
                        help="Only with an explicit user release request: authorize deleting this exact owned pod")
    args = parser.parse_args()
    try:
        client = workspace_client()
        print(json.dumps(operate(client, args.action, args.run_id, args.round_id, args.public_key,
                                 gpus=args.gpus, existing_name=args.existing_name,
                                 infrastructure_attempt=args.infrastructure_attempt,
                                 priority_class=args.priority_class,
                                 confirm_release=args.confirm_release)))
    except Exception as exc:
        # Preserve useful failure identity without SDK response text/headers.
        print(json.dumps({"schema": "roborender.allocation-error.v1", "action": args.action,
            "error_type": type(exc).__name__,
            "status_code": error_status(exc), "stage": getattr(exc, "stage", "unclassified"),
            "retryable": getattr(exc, "retryable", False),
            "reason_code": getattr(exc, "reason_code", None),
            "source_error_type": getattr(exc, "source_error_type", type(exc).__name__)}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
