"""One bounded offline BC workload in the approved reserved H200 pool.

Orchestration stays on the CPU pod. This shares the existing pinned SSH,
idempotent allocation and owned-process machinery, without constructing a
fictional DAgger run/config just to render or train offline BC data.
Bounded workload cleanup retains the allocation; only an explicit exact-owner
release confirmation permits deletion after user approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import signal
import time

from pipeline.deployment.beta_site import Site, ROOT, validate_site, site_host
from pipeline.deployment.beta_operations import atomic
from pipeline.deployment.h200.beta_pod import name_for


class BCWorkload(Site):
    def __init__(self, site, run_id, *, gpus=1):
        self.site = validate_site(site)
        if (site_host(site) != "leo-rtx-6k" or gpus not in (1, 4)
                or not re.fullmatch(r"dagger-leo-bc-[a-z0-9-]{1,48}", run_id)):
            raise ValueError("offline BC requires its explicit Leo owner and 1/4 H200")
        self.run, self.rid, self.role, self.gpus = run_id, 1, "leo", gpus
        self.root = ROOT / "runtime/bc-workloads" / run_id
        self.round = self.root / "round-1"
        self.round.mkdir(parents=True, exist_ok=True)
        self.remote = Path(self.site["rtx_repo"]) / "runtime/bc-workloads" / run_id
        self.source = Path(self.site["rtx_source_root"])
        self.pod = name_for(run_id, 1)
        self.gpu_ssh = self.round / "h200-ssh.conf"
        self.attempts = {}

    def allocation(self, action, keys=(), *, confirm_release=None):
        return self.json([self.site["broker_python"], "-", action,
            "--run-id", self.run, "--round-id", "1", "--gpus", str(self.gpus),
            *(["--confirm-release", confirm_release] if confirm_release is not None else []),
            *[v for key in keys for v in ("--public-key", key)]], "broker",
            data=(ROOT / "pipeline/deployment/h200/beta_pod.py").read_bytes())

    def acquire(self):
        if self.command(["/bin/hostname"], "beta").strip() != b"leo-rtx-6k":
            raise RuntimeError("wrong cloud broker host")
        cpu_key = Path(self.site["cpu_identity_file"] + ".pub").read_text().strip()
        leo_key = self.command(["/bin/cat", self.site["rtx_identity_file"] + ".pub"], "beta").decode().strip()
        self.allocation("create", [cpu_key, leo_key])
        host_key_path = ROOT / "runtime/leo-pods" / self.run / "round-1/host-key.pub"
        self.await_allocation(host_key_path)
        key = host_key_path.read_text().strip()
        if not key.startswith("ssh-ed25519 ") or "\n" in key:
            raise RuntimeError("invalid owned workload host key")
        self.publish(self.round / "known_hosts", (self.pod + " " + key + "\n").encode())
        conf = (f"Host {self.pod}\n HostName {self.pod}\n HostKeyAlias {self.pod}\n User root\n"
                f" IdentityFile {self.site['cpu_identity_file']}\n IdentitiesOnly yes\n BatchMode yes\n"
                f" StrictHostKeyChecking yes\n UserKnownHostsFile {self.round / 'known_hosts'}\n"
                f" ConnectTimeout 12\n ProxyCommand {shlex.join(self.gpu_proxy('local'))}\n")
        self.publish(self.gpu_ssh, conf.encode())
        inventory = self.command(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"], "gpu")
        lines = inventory.decode().strip().splitlines()
        if len(lines) != self.gpus or any("H200" not in line for line in lines):
            raise RuntimeError("allocated device count/model differs from offline workload")
        atomic(self.round / "device-admission.json", {"gpus": self.gpus, "devices": lines,
            "run_id": self.run, "reservation": "cosmos-solution-architect-xcwtffiq"})

    def release(self, *, confirm_release=None):
        self.validate_release_confirmation(confirm_release)
        if not self.allocation_present():
            return self.finish_allocation_cleanup(confirm_release=confirm_release)
        if self.gpu_ssh.exists():
            self.stop_process("bc-workload", "gpu")
            for _ in range(3):
                clients = self.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], "gpu")
                if clients.strip():
                    raise RuntimeError("GPU clients remain; reconcile before releasing the owned BC allocation")
                time.sleep(1)
        return self.finish_allocation_cleanup(confirm_release=confirm_release)

    def execute(self, argv, *, env=None, ttl=3600):
        # Publish the exact workload before any GPU reservation. A different
        # command may never take over an existing owner/runtime identity.
        self.publish(self.root / "workload.json", json.dumps({"argv": argv, "env": env or {},
            "ttl": ttl, "gpus": self.gpus}, sort_keys=True, separators=(",", ":")).encode())
        from pipeline.deployment.owned_process import read_spec
        spec = self.root / "workload-spec.json"
        self.publish(spec, json.dumps({"argv": argv, "env": env or {}, "ttl": ttl,
            "cwd": str(ROOT), "owner": self.run}, sort_keys=True).encode())
        read_spec(spec)  # Invalid commands/credential-bearing env fail before allocation.
        try:
            self.acquire()
            self.process("bc-workload", "start", host="gpu", argv=argv, env=env, ttl=ttl)
            result = self.await_process("bc-workload", host="gpu", timeout=ttl + 60)
            atomic(self.root / "complete.json", {"run_id": self.run, "result": result})
            return result
        finally:
            self.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", type=int, choices=[1, 4], default=1)
    parser.add_argument("--workload", type=Path, required=True,
                        help="JSON with exact argv/env/ttl for a concrete render or training job")
    args = parser.parse_args()
    task = json.loads(args.workload.read_bytes())
    if set(task) != {"argv", "env", "ttl"}:
        raise ValueError("explicit argv/env/ttl workload required")
    def stop(signum, frame):
        raise KeyboardInterrupt("offline BC workload stop requested")
    signal.signal(signal.SIGTERM, stop)
    workload = BCWorkload(json.loads(args.site.read_bytes()), args.run_id, gpus=args.gpus)
    print(json.dumps(workload.execute(**task)), flush=True)


if __name__ == "__main__":
    main()
