"""Create an explicitly sized reserved migration workload; never change other pods."""
import argparse
import json
import re
from leptonai.api.v1.workspace_record import WorkspaceRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--public-key", action="append", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Build the plan locally; never POST or allocate")
    parser.add_argument("--gpus", type=int, choices=[1, 4], default=1,
                        help="Explicit reserved workload size; four is for the bounded throughput sweep")
    args = parser.parse_args()
    if len(args.name) > 36 or not re.fullmatch(r"leo-roborender-beta-smoke-[a-z0-9-]*[a-z0-9]", args.name):
        raise ValueError("unexpected workload name")
    for key in args.public_key:
        if not key.startswith("ssh-ed25519 ") or "\n" in key:
            raise ValueError("only single-line Ed25519 public keys are accepted")
    client = WorkspaceRecord.client()
    # DGXC's gateway requires the dashboard Origin for authenticated mutations.
    # This matches the existing approved launcher.
    client._header["Origin"] = "https://dashboard.dgxc-lepton.nvidia.com"
    if any(d.metadata.id_ == args.name for d in client.deployment.list_all()):
        raise RuntimeError("pod already exists; refusing to overwrite or duplicate it")
    source = client.deployment.get("leo-cosmos-h200-access")
    spec = source.spec.model_copy(deep=True)
    spec.resource_requirement.resource_shape = f"custom.gpu.{args.gpus}xh200"
    spec.resource_requirement.min_replicas = 1
    spec.resource_requirement.max_replicas = 1
    spec.reservation_config.reservation_id = "cosmos-solution-architect-xcwtffiq"
    spec.reservation_config.allow_burst_to_other_reservations = False
    spec.container.command = [
        "/bin/bash",
        "/workspace/project/roborender/pipeline/deployment/h200/smoke_pod_entrypoint.sh",
    ]
    env_type = type(source.spec.envs[0])
    spec.envs = [env_type(name="ROBORENDER_SMOKE_PUBLIC_KEYS", value="\n".join(args.public_key)),
                 env_type(name="ROBORENDER_SMOKE_ID", value=args.name)]
    spec.auto_scaler = None
    spec.api_tokens = None
    spec.is_pod = True
    request = type(source)(metadata=type(source.metadata)(name=args.name), spec=spec)
    # This API ignores dryrun=true on POST /deployments. Never send a mutating
    # request in dry-run mode, even if the PATCH API supports server dry runs.
    if not args.dry_run:
        client.deployment.create(request)
    print(json.dumps({"pod": args.name, "dry_run": args.dry_run, "shape": spec.resource_requirement.resource_shape,
                      "reservation": spec.reservation_config.reservation_id,
                      "allow_burst": False, "codex_started": False}))


if __name__ == "__main__":
    main()
