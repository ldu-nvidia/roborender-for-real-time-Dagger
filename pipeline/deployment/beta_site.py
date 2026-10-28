"""Concrete CPU-access-pod deployment operations for a beta DAgger campaign.

All credentials are references to approved host-local files. The site file
names the existing SSH/broker routes explicitly; no route/key discovery or
unreserved allocation fallback occurs. This program never runs by import.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.deployment.beta_operations import atomic, round_runtime_path
from pipeline.deployment.allocation_wait import wait_for_host_key
from pipeline.deployment.h200.beta_pod import name_for
from pipeline.deployment.source_snapshot import verify
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import canonical, validate_config, actor_ids, deployment_host, host_role
from pipeline.dagger.services.contracts import file_sha256
from roborender.infrastructure.source_locations import is_source_release_location, LEGACY_RELEASE_ROOT

PATH_FIELDS = {"repository_root", "source_manifest", "rtx_source_root", "rtx_source_manifest", "rtx_repo",
               "cpu_python", "rtx_python", "ssh_config", "broker_python", "broker_proxy",
               "cpu_identity_file", "rtx_identity_file", "rtx_proxy_ssh_config", "netrc_file"}
ALIAS_FIELDS = {"rtx_ssh_host", "broker_ssh_host", "rtx_proxy_ssh_host"}
TRANSPORT_RECOVERY_SECONDS = 360


def site_host(site):
    return "leo-rtx-6k" if site.get("schema") == "roborender.leo-site.v1" else "465d6b0-lcedt"


def validate_site(site):
    version = site.get("schema")
    if version not in {"roborender.beta-site.v1", "roborender.beta-site.v2", "roborender.leo-site.v1"}:
        raise ValueError("explicit versioned beta site required")
    legacy = version == "roborender.beta-site.v1"
    paths = PATH_FIELDS if legacy else PATH_FIELDS - {"rtx_proxy_ssh_config"}
    aliases = ALIAS_FIELDS if legacy else ALIAS_FIELDS - {"rtx_proxy_ssh_host"}
    if set(site) != paths | aliases | {"schema"}:
        raise ValueError("explicit beta site fields required")
    if any(not isinstance(site[k], str) or not Path(site[k]).is_absolute() for k in paths):
        raise ValueError("all site paths must be absolute")
    if any(not isinstance(site[k], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", site[k]) for k in aliases):
        raise ValueError("only configured SSH aliases are accepted")
    if site["repository_root"] != "/workspace/project/roborender" or site["rtx_repo"] != "/home/ldu/roborender":
        raise ValueError("site must bind the existing shared repository and beta repo")
    if (not is_source_release_location(site["rtx_source_root"])
            or version != "roborender.leo-site.v1"
            and not Path(site["rtx_source_root"]).is_relative_to(LEGACY_RELEASE_ROOT)):
        raise ValueError("beta code must be a frozen runtime source release")
    if not legacy:
        role = "leo" if version == "roborender.leo-site.v1" else "beta"
        if (site["broker_ssh_host"] != f"roborender-{role}-direct" or site["rtx_ssh_host"] != f"roborender-{role}-direct"
                or site["ssh_config"] != f"/workspace/.ssh/roborender_{role}_access_config"
                or site["broker_python"] != "/home/ldu/.local/share/roborender-cloud/venv-compatible/bin/python"
                or site["broker_proxy"] != str(Path(site["rtx_source_root"]) / "pipeline/deployment/connectivity/cloud_proxy.py")):
            raise ValueError("v2 requires the Beta-owned route and frozen local cloud proxy")
    return site


class Site:
    def __init__(self, site, request):
        self.site = validate_site(site)
        self.request, self.config = request, validate_config(request["config"])
        if deployment_host(self.config) != site_host(site):
            raise ValueError("campaign host and site host differ")
        self.role = host_role(self.config)
        self.gpus = [a["physical_gpu_index"] for a in self.config["rtx_actors"]]
        self.run, self.rid = request["run_id"], request["round_id"]
        if self.config["run_id"] != self.run or self.config["round_id"] != self.rid:
            raise ValueError("operation and round identity differ")
        self.root = Path(request["campaign_root"]).resolve()
        # Code may execute from an immutable, verified source release. Assets
        # and campaign ledgers still belong to the canonical shared repository.
        runtime_root = Path(self.site["repository_root"]).resolve() / "runtime"
        if not self.root.is_relative_to(runtime_root) or self.root == runtime_root:
            raise ValueError("campaign root must be scoped shared runtime storage")
        self.attempts = request.get("infrastructure_attempts", {})
        self.round = round_runtime_path(self.root, self.rid, self.attempts)
        self.round.mkdir(parents=True, exist_ok=True)
        self.remote = round_runtime_path(Path(site["rtx_repo"]) / f"runtime/{self.role}-campaigns" / self.run,
                                         self.rid, self.attempts)
        self.source = Path(site["rtx_source_root"])
        self.pod = name_for(self.run, self.rid)
        self.state_identity = hashlib.sha256((self.run + ":beta-service-ledger-v1").encode()).hexdigest()
        self.token = self.root / "services.token"
        self.gpu_ssh = self.round / "h200-ssh.conf"
        self.watchdog_command = None

    def ssh(self, host="beta"):
        if host == "gpu":
            config, alias = self.gpu_ssh, self.pod
        else:
            config, alias = self.site["ssh_config"], self.site["rtx_ssh_host" if host == "beta" else "broker_ssh_host"]
        return ["ssh", "-F", str(config), "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                "-o", "ControlMaster=no", "-o", "ControlPath=none", alias]

    def gpu_proxy(self, host):
        cloud = [self.site["broker_python"], self.site["broker_proxy"], self.pod]
        if host == "local":
            return self.ssh("broker") + cloud
        if host != "beta":
            raise ValueError("proxy host must be local or beta")
        if self.site["schema"] in {"roborender.beta-site.v2", "roborender.leo-site.v1"}:
            # Beta owns the cloud login; no Alpha hop and no SSH back to itself.
            return cloud
        return ["ssh", "-F", self.site["rtx_proxy_ssh_config"], "-T", "-o", "BatchMode=yes",
                self.site["rtx_proxy_ssh_host"], *cloud]

    def command(self, argv, host="local", *, timeout=45, data=None):
        cmd = list(map(str, argv)) if host == "local" else self.ssh(host) + [shlex.join(list(map(str, argv)))]
        # Only pass public code or a scoped file payload as stdin. Never print
        # command output on failure: third-party errors may include secrets.
        started = time.monotonic()
        deadline = started + max(timeout, TRANSPORT_RECOVERY_SECONDS)
        attempt = 0
        while True:
            attempt += 1
            try:
                result = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        timeout=min(timeout, max(.1, deadline-time.monotonic())), cwd=ROOT)
            except subprocess.TimeoutExpired:
                if host == "local" or time.monotonic() >= deadline:
                    raise
            else:
                if result.returncode == 0:
                    if attempt > 1:
                        print(json.dumps({"phase": "TRANSPORT_RECOVERED", "host": host,
                            "attempts": attempt, "elapsed_seconds": time.monotonic()-started}), flush=True)
                    return result.stdout
                security_failure = any(reason in result.stderr.lower() for reason in (
                    b"permission denied", b"host key verification failed", b"remote host identification has changed"))
                if result.returncode != 255 or host == "local" or security_failure:
                    if host == "broker" and len(result.stdout) < 4096:
                        try:
                            detail = json.loads(result.stdout)
                        except (ValueError, UnicodeError):
                            detail = {}
                        if (isinstance(detail, dict) and detail.get("schema") == "roborender.allocation-error.v1"
                                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", str(detail.get("error_type", "")))
                                and (detail.get("status_code") is None or type(detail["status_code"]) is int)):
                            if (detail.get("action") in {"status", "inspect"} and detail.get("retryable") is True
                                    and detail.get("stage") in {"deployment_read", "replica_read"}
                                    and time.monotonic() < deadline):
                                remaining = deadline-time.monotonic()
                                print(json.dumps({"phase": "H200_STATUS_RECOVERING", "stage": detail["stage"],
                                    "http_status": detail.get("status_code"), "attempt": attempt,
                                    "elapsed_seconds": time.monotonic()-started,
                                    "native_replay_allowed": False}), flush=True)
                                time.sleep(min(2 ** min(attempt-1, 3), remaining))
                                continue
                            # Preserve only our bounded invariant code. SDK
                            # exception bodies, messages and headers remain
                            # private; a generic RuntimeError hid the exact
                            # admission gate in an otherwise clean queue wait.
                            reason = detail.get("reason_code")
                            suffix = (f" [{reason}]" if isinstance(reason, str)
                                      and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", reason) else "")
                            raise RuntimeError(f"broker allocation failed: {detail['error_type']} "
                                               f"(HTTP {detail.get('status_code')}){suffix}")
                    raise RuntimeError(f"{host} operation failed (exit={result.returncode})")
            # Remote calls here are immutable writes, read-only probes or
            # identity-idempotent process/allocation controls, never env.step.
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{host} transport recovery deadline exceeded; no native restart")
            print(json.dumps({"phase": "TRANSPORT_RECOVERING", "host": host, "attempt": attempt,
                "elapsed_seconds": time.monotonic()-started, "native_replay_allowed": False}), flush=True)
            time.sleep(min(2 ** min(attempt-1, 3), remaining))

    def json(self, argv, host="local", **kwargs):
        raw = self.command(argv, host, **kwargs)
        if len(raw) > 4 << 20:
            raise RuntimeError("oversized metadata response")
        return json.loads(raw)

    def read(self, path, host="local"):
        if host == "local":
            return json.loads(Path(path).read_bytes())
        return self.json(["python3", "-c", "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())", path], host)

    def publish(self, path, data, host="local", mode=0o600):
        """Immutable small deployment metadata, including an owner-only token."""
        if host == "local":
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.is_symlink() or path.read_bytes() != data:
                    raise RuntimeError("refusing to overwrite existing deployment metadata")
            else:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
                with os.fdopen(fd, "wb") as output:
                    output.write(data)
            return
        code = ("import os,pathlib,sys; p=pathlib.Path(sys.argv[1]); b=sys.stdin.buffer.read(); "
                "p.parent.mkdir(parents=True,exist_ok=True); "
                "existing=p.exists(); "
                "assert not p.is_symlink(); "
                "assert not existing or p.read_bytes()==b; "
                "fd=None if existing else os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,int(sys.argv[2])); "
                "f=None if fd is None else os.fdopen(fd,'wb'); "
                "None if f is None else (f.write(b),f.close())")
        self.command(["python3", "-c", code, path, str(mode)], host, data=data)

    def process(self, label, action, *, host="local", argv=None, ttl=24000, env=None):
        directory = (self.remote if host == "beta" else self.round) / "processes" / label
        code_root = self.source if host == "beta" else ROOT
        python = self.site["rtx_python"] if host == "beta" else self.site["cpu_python"]
        if action == "start":
            spec = {"argv": list(map(str, argv)), "cwd": str(self.site["rtx_repo"] if host == "beta" else ROOT),
                    "ttl": ttl, "env": env or {}, "owner": self.run}
            self.publish(directory.with_suffix(".json"), canonical(spec), host)
        cmd = [python, code_root / "pipeline/deployment/owned_process.py", action, "--directory", directory]
        if action == "start": cmd += ["--spec", directory.with_suffix(".json")]
        return self.json(cmd, host)

    def await_process(self, label, *, host="local", timeout=21600):
        deadline, disconnected = time.monotonic() + timeout, None
        while time.monotonic() < deadline:
            try:
                value = self.process(label, "status", host=host)
                disconnected = None
            except (RuntimeError, subprocess.TimeoutExpired):
                disconnected = disconnected or time.monotonic()
                if time.monotonic() - disconnected > 75:
                    raise RuntimeError("management route lost beyond the recovery budget; no native restart")
                time.sleep(5)
                continue
            if value["state"] == "COMPLETE": return value
            if value["state"] not in {"STARTING", "RUNNING"}:
                raise RuntimeError("owned workload failed or became uncertain")
            time.sleep(5)
        raise TimeoutError("owned workload deadline; reconcile before any restart")

    def stop_process(self, label, host="local"):
        value = self.process(label, "status", host=host)
        # A detached supervisor may not have published its PID yet. Wait for
        # its identity before stopping; ABSENT and STARTING are not equivalent.
        started = time.monotonic()
        while value["state"] == "STARTING" and time.monotonic() - started < 10:
            time.sleep(.1)
            value = self.process(label, "status", host=host)
        if value["state"] == "ABSENT": return
        self.process(label, "stop", host=host)
        deadline = time.monotonic() + 70
        while time.monotonic() < deadline:
            value = self.process(label, "status", host=host)
            if value["state"] in {"STOPPED", "COMPLETE", "FAILED"}: return
            if value["state"] == "UNCERTAIN":
                raise RuntimeError("owned process uncertain; cannot assert cleanup")
            time.sleep(1)
        raise TimeoutError("owned process did not stop")

    def allocation(self, action, keys=(), *, confirm_release=None):
        attempt = self.attempts.get(str(self.rid))
        return self.json([self.site["broker_python"], "-", action, "--run-id", self.run,
                          "--round-id", str(self.rid),
                          *(["--infrastructure-attempt", attempt] if attempt else []),
                          *(["--confirm-release", confirm_release] if confirm_release is not None else []),
                          *[item for key in keys for item in ("--public-key", key)]],
                         "broker", data=(ROOT / "pipeline/deployment/h200/beta_pod.py").read_bytes())

    def await_allocation(self, key_file, inspect=None):
        last_phase = [None]
        def report(value):
            value = {"schema": "roborender.allocation-readiness.v1", "run_id": self.run,
                     "round_id": self.rid, "updated_at_ns": time.time_ns(), **value}
            atomic(self.round / "allocation-readiness.json", value)
            if value["phase"] != last_phase[0]:
                print(json.dumps(value), flush=True)
                last_phase[0] = value["phase"]
        return wait_for_host_key(inspect or (lambda: self.allocation("inspect")), key_file, report)

    def ready_bundle(self, host, mode):
        output = self.round / (mode + "-services")
        argv = [self.site["cpu_python"], ROOT / "pipeline/deployment/h200/beta_service_host.py",
                "--source", ROOT, "--state", self.root / "services", "--token", self.token,
                "--output", output, "--mode", mode, "--run-id", self.run,
                "--state-identity", self.state_identity, "--ttl", "24000"]
        if mode == "gpu": argv += ["--netrc", self.site["netrc_file"]]
        self.process(mode + "-services", "start", host=host, argv=argv)
        deadline = time.monotonic() + 1320
        while time.monotonic() < deadline:
            if (output / "ready.json").is_file():
                value = self.read(output / "ready.json")
                if value["run_id"] != self.run or value["state_identity"] != self.state_identity:
                    raise RuntimeError("service bundle identity changed")
                if self.process(mode + "-services", "status", host=host)["state"] != "RUNNING":
                    raise RuntimeError("service bundle exited after readiness")
                return value["services"]
            if self.process(mode + "-services", "status", host=host)["state"] not in {"STARTING", "RUNNING"}:
                raise RuntimeError("service bundle failed before readiness")
            time.sleep(2)
        raise TimeoutError("genuine service initialization deadline")

    def supervisor(self, host, ready, ssh_config, known):
        base = self.remote if host == "beta" else self.round
        ports = [18610, 18620, 18621, 18622, 18623] if host == "beta" else [18610, 18612, 18613, 18614]
        config = {"schema": "roborender.ssh-supervisor.v1", "run_id": self.run,
                  "ssh_config": str(ssh_config), "known_hosts": str(known), "ssh_host": self.pod,
                  "lifetime_seconds": 24000, "forwards": [
                      {"local_port": port, "remote_port": port, "expected_ready": {
                          k: v for k, v in ready[str(port)].items()
                          if k in {"ok", "role", "schema", "pid", "gateway_instance_id"}}} for port in ports]}
        self.publish(base / "supervisor.json", canonical(config), host)
        python, source = (self.site["rtx_python"], self.source) if host == "beta" else (self.site["cpu_python"], ROOT)
        self.process("tunnels", "start", host=host, argv=[python, source / "pipeline/deployment/rtx_actor/tunnel_supervisor.py",
                     "--config", base / "supervisor.json", "--root", base / "supervisor"], ttl=24000)
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline:
            try:
                if self.read(base / "supervisor/status.json", host).get("state") == "ready": return
            except (RuntimeError, FileNotFoundError): pass
            time.sleep(1)
        raise TimeoutError("identity-pinned forwards did not become ready")

    def verify_sources(self):
        manifest = json.loads(Path(self.site["source_manifest"]).read_bytes())
        if manifest["source_sha256"] != self.config["source_sha256"]:
            raise ValueError("site release and campaign source differ")
        verify(ROOT, manifest)
        return manifest

    def prepare(self):
        manifest = self.verify_sources()
        for path in (self.site["netrc_file"], self.site["cpu_identity_file"]):
            if Path(path).stat().st_mode & 0o077:
                raise ValueError("credential reference must be owner-only")
        if file_sha256(Path(self.config["student"]["initial_checkpoint"])) != self.config["student"]["initial_checkpoint_sha256"]:
            raise ValueError("round parent checkpoint unavailable")
        # No GPU is allocated until both existing host routes respond.
        if self.command(["/bin/hostname"], "beta").strip().decode() != site_host(self.site):
            raise RuntimeError("wrong RTX host")
        self.command(["/bin/hostname"], "broker")
        source_check = """import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);m=json.loads(pathlib.Path(sys.argv[2]).read_bytes())
assert m['source_sha256']==sys.argv[3]
sys.path.insert(0,str(r))
from pipeline.deployment.source_snapshot import file_record,canonical
rows=[file_record(r,x['path']) for x in m['files']]
assert rows==m['files'] and hashlib.sha256(canonical(rows)).hexdigest()==m['source_sha256']
print('ok')
"""
        self.command([self.site["rtx_python"], "-c", source_check, self.source,
                      self.site["rtx_source_manifest"], manifest["source_sha256"]], "beta")
        if self.watchdog_command is None:
            raise RuntimeError("independent CPU allocation watchdog must be bound")
        self.process("lease-watchdog", "start", argv=self.watchdog_command, ttl=24000)
        if self.rid > 1:
            # The previous round's CPU ledger owner must exit before this pod opens it.
            previous = copy_request(self.request, round_id=self.rid - 1)
            Site(self.site, previous).stop_process("cpu-services")
        if not self.token.exists(): self.publish(self.token, (secrets.token_hex(32) + "\n").encode())
        cpu_key = Path(self.site["cpu_identity_file"] + ".pub").read_text().strip()
        rtx_key = self.command(["python3", "-c", "from pathlib import Path;import sys;print(Path(sys.argv[1]).read_text().strip())",
                                self.site["rtx_identity_file"] + ".pub"], "beta").decode().strip()
        self.allocation("create", [cpu_key, rtx_key])
        key_file = round_runtime_path(ROOT / f"runtime/{self.role}-pods" / self.run,
                                      self.rid, self.attempts) / "host-key.pub"
        self.await_allocation(key_file)
        host_key = key_file.read_text().strip()
        if not host_key.startswith("ssh-ed25519 ") or "\n" in host_key:
            raise ValueError("unexpected workload public host key")
        for host, base, identity, proxy in [
            ("local", self.round, self.site["cpu_identity_file"], self.gpu_proxy("local")),
            ("beta", self.remote, self.site["rtx_identity_file"], self.gpu_proxy("beta"))]:
            self.publish(base / "known_hosts", (self.pod + " " + host_key + "\n").encode(), host)
            conf = (f"Host {self.pod}\n HostName {self.pod}\n HostKeyAlias {self.pod}\n User root\n"
                    f" IdentityFile {identity}\n IdentitiesOnly yes\n BatchMode yes\n StrictHostKeyChecking yes\n"
                    f" UserKnownHostsFile {base / 'known_hosts'}\n ConnectTimeout 12\n"
                    f" ProxyCommand {shlex.join(proxy)}\n")
            self.publish(base / "h200-ssh.conf", conf.encode(), host)
        ready = self.ready_bundle("gpu", "gpu")
        # The bundle starts only the genuine pool; check its live worker argv
        # and exact per-GPU ports rather than merely trusting HTTP readiness.
        pids = [ready[str(p)]["pid"] for p in range(18620, 18624)]
        code = "import json,pathlib,sys;print(json.dumps([pathlib.Path('/proc/'+p+'/cmdline').read_bytes().decode().split(chr(0)) for p in sys.argv[1:]]))"
        commands = self.json(["python3", "-c", code, *map(str, pids)], "gpu")
        for gpu, argv in enumerate(commands):
            if "--fake" in argv or "--devices" not in argv or argv[argv.index("--devices") + 1] != f"cuda:{gpu}":
                raise RuntimeError("genuine per-GPU gateway identity differs")
        self.supervisor("local", ready, self.gpu_ssh, self.round / "known_hosts")
        self.publish(self.remote / "services.token", self.token.read_bytes(), "beta")
        self.supervisor("beta", ready, self.remote / "h200-ssh.conf", self.remote / "known_hosts")
        return {"source_sha256": manifest["source_sha256"], "resource_shape": "custom.gpu.4xh200",
                "reservation": "cosmos-solution-architect-xcwtffiq", "allow_burst": False,
                "genuine_backends": True, "owned_control_plane": True, "gateway_pids": pids,
                "service_state_identity": self.state_identity, "token_file": str(self.token)}

    def collect(self):
        self.publish(self.remote / "config.json", Path(self.request["round_config_path"]).read_bytes(), "beta", 0o444)
        argv = [self.site["rtx_python"], self.source / "pipeline/deployment/rtx_actor/run_throughput_pair.py",
                "--run-id", self.run, "--code-root", self.source / "pipeline/dagger/rtx/code",
                "--deployment-host", site_host(self.site),
                "--mode", "beta-production", "--num-envs", "6", "--rtx-gpus", *map(str, self.gpus), "--fanout", "4",
                "--memory-gib", "21", "--compact-host-heap", "--timeout", "21600", "--seed", str(self.config["seed"]),
                "--token-file", self.remote / "services.token", "--qualification-config", self.remote / "config.json",
                "--gateway-pids", ",".join(map(str, self.config["h200"]["gateway_pids"])),
                "--source-manifest", self.site["rtx_source_manifest"],
                "--supervisor-config", self.remote / "supervisor.json", "--supervisor-status-dir", self.remote / "supervisor"]
        self.process("actors", "start", host="beta", argv=argv, ttl=21700)
        self.await_process("actors", host="beta", timeout=21800)
        output = Path(self.site["rtx_repo"]) / f"runtime/{self.role}-production" / self.run / f"round-{self.rid}"
        result = self.read(output / "pair-result.json", "beta")
        if result.get("passed") is not True or result.get("config_sha256") != hashlib.sha256(canonical(self.config)).hexdigest():
            raise RuntimeError("native pair did not complete this frozen round")
        return {"actors": [self.read(output / f"gpu{gpu}/beta-round-state.json", "beta") for gpu in self.gpus],
                "actor_ids": actor_ids(self.config), "owned_actors_stopped": self.verify_actors_stopped()}

    def verify_actors_stopped(self):
        # Inspect only names owned by this run, not GPU0 or unrelated containers.
        names = [self.run + f"-r{self.rid}-g{gpu}" for gpu in self.gpus]
        return self.verify_containers_stopped(names)

    def verify_containers_stopped(self, names, *, stop=False):
        code = ("import json,subprocess,sys;\n"
                "known=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()\n"
                "for n in sys.argv[2:]:\n"
                " if n in known:\n"
                "  d=json.loads(subprocess.check_output(['docker','inspect',n]))[0];assert d['Config']['Labels'].get('org.roborender.qualification')==n\n"
                "  if d['State']['Running']:\n"
                "   assert sys.argv[1]=='stop';subprocess.run(['docker','stop','--time','15',n],check=True,timeout=30)\n"
                "print('true')")
        return self.json(["python3", "-c", code, "stop" if stop else "check", *names], "beta", timeout=90) is True

    def validate_release_confirmation(self, confirm_release):
        if confirm_release is not None and confirm_release != self.pod:
            raise ValueError("release confirmation must name this exact owned pod")

    def allocation_present(self):
        result = self.allocation("status")
        if not isinstance(result, dict) or type(result.get("present")) is not bool:
            raise RuntimeError("owned allocation presence is unconfirmed")
        return result["present"]

    def finish_allocation_cleanup(self, *, confirm_release=None):
        """Retain the node unless the caller carries an explicit user release.

        Workload cleanup must already be complete. Retention is not a release
        proof, and does not authorize a new round/node or implicit pod adoption.
        """
        self.validate_release_confirmation(confirm_release)
        present = self.allocation_present()
        if present and confirm_release is not None:
            result = self.allocation("delete", confirm_release=confirm_release)
            if not (result.get("delete_requested") is True or result.get("present") is False):
                raise RuntimeError("explicit owned allocation release was not accepted")
            deadline = time.monotonic() + 90
            while self.allocation_present():
                if time.monotonic() > deadline:
                    raise TimeoutError("owned H200 allocation release unconfirmed")
                time.sleep(3)
            present = False
        receipt = {"run_id": self.run, "round_id": self.rid, "allocation_name": self.pod,
                   "released": not present, "retained": present,
                   "release_policy": "explicit_user_request_only",
                   "release_confirmation": confirm_release}
        atomic(self.round / ("retained.json" if present else "released.json"), receipt)
        return {"h200_allocation_released": not present, "h200_allocation_retained": present,
                "allocation_name": self.pod, "release_policy": "explicit_user_request_only"}

    def release(self, *, confirm_release=None):
        self.validate_release_confirmation(confirm_release)
        self.verify_actors_stopped()
        self.stop_process("tunnels", "beta")
        self.stop_process("tunnels")
        self.stop_process("gpu-services", "gpu")
        # Three consecutive samples; an unknown client is never killed.
        for _ in range(3):
            if self.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], "gpu").strip():
                raise RuntimeError("GPU clients remain; inspect ownership before releasing the pod")
            time.sleep(1)
        return {"owned_actors_stopped": True,
                **self.finish_allocation_cleanup(confirm_release=confirm_release)}

    def copy_checkpoint(self, source, digest):
        destination = self.remote / "assets" / (digest + ".pt")
        from pipeline.deployment.checkpoint_transfer import transfer
        def request(argv, data, timeout):
            result = subprocess.run(self.ssh("beta") + [shlex.join(argv)], input=data,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            if result.returncode:
                # Host-key/auth checks remain SSH-owned. Never log raw command,
                # payload or stderr; bounded safe checkpoint retries run no GPU work.
                if result.returncode == 255:
                    raise ConnectionError("checkpoint SSH stream interrupted")
                raise RuntimeError("checkpoint receiver process failed")
            if len(result.stdout)>4096:raise RuntimeError("checkpoint receiver response oversized")
            return json.loads(result.stdout)
        receipt = transfer(source, destination, digest, request)
        atomic(self.round / ("checkpoint-transfer-" + digest + ".json"), receipt)
        return destination

    def evaluate(self):
        raise RuntimeError(
            "Legacy ray-traced student success execution is disabled. Use the "
            "rendered fixed295/smoke resident evaluate_explicit_panel path with "
            "a hash-pinned rendered_observation_config; historical results remain readable."
        )

    def control_only(self):
        self.ready_bundle("local", "cpu")
        return {"token_file": str(self.token), "h200_gpu_count": 0, "exclusive_ledger_owner": True,
                "service_state_identity": self.state_identity}

    def abort(self, *, confirm_release=None):
        # Deadline/transport loss never authorizes replay or an unscoped kill.
        self.validate_release_confirmation(confirm_release)
        errors = []
        for name in ("actors", "evaluation"):
            try:
                self.stop_process(name, "beta")
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                errors.append(type(exc).__name__)
        self.verify_containers_stopped([self.run + f"-r{self.rid}-g{gpu}" for gpu in self.gpus], stop=True)
        self.verify_actors_stopped()
        evaluation = f"success-{self.role}-panel-" + hashlib.sha256(self.run.encode()).hexdigest()[:16] + f"-r{self.rid}"
        self.verify_containers_stopped([f"roborender-{evaluation}-{arm}" for arm in ("bc", "dagger", "warm")], stop=True)
        self.stop_process("tunnels", "beta")
        self.stop_process("tunnels")
        self.stop_process("cpu-services")
        if self.allocation_present():
            if self.gpu_ssh.exists():
                self.stop_process("gpu-services", "gpu")
                if self.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], "gpu").strip():
                    raise RuntimeError("owned allocation has unexplained GPU clients")
        return {"owned_actors_stopped": True,
                **self.finish_allocation_cleanup(confirm_release=confirm_release),
                "actor_process_reconciliation_warnings": errors}


def copy_request(request, round_id):
    """Find a completed prior round owner without inventing new science config."""
    previous = json.loads((Path(request["campaign_root"]) / f"round-{round_id}/config.json").read_bytes())
    return {**request, "round_id": round_id, "config": previous}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--site-sha256", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--confirm-release", metavar="EXACT_POD_NAME",
                        help="Only after explicit user approval, permit this release/abort to delete the exact pod")
    args = parser.parse_args()
    if file_sha256(args.site) != args.site_sha256:
        raise ValueError("site binding changed after admission")
    request = json.loads(args.request.read_bytes())
    phase = request["phase"]
    if phase not in {"prepare", "collect", "release", "evaluate", "control_only", "abort"}:
        raise ValueError("unknown deployment phase")
    if args.confirm_release is not None and phase not in {"release", "abort"}:
        raise ValueError("release confirmation is only valid for release/abort cleanup")
    site = Site(json.loads(args.site.read_bytes()), request)
    if phase == "prepare":
        # Start the independent CPU guard before any GPU allocation. It will
        # stop owned workloads if the scheduler disappears; retain the node.
        site.watchdog_command = [site.site["cpu_python"],
            ROOT / "pipeline/deployment/beta_lease_watchdog.py", "--site", args.site,
            "--site-sha256", args.site_sha256, "--request", args.request]
    result = getattr(site, phase)(**({"confirm_release": args.confirm_release}
                                    if args.confirm_release is not None else {}))
    atomic(args.receipt, {"ok": True, "operation_id": request["operation_id"], "run_id": request["run_id"],
                          "round_id": request["round_id"], "phase": phase, **result})


if __name__ == "__main__":
    main()
