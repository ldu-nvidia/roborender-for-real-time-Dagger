"""Bounded control client for a standing RTX owner; no Isaac imports/restarts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class NativeCaptureProgressGuard:
    """Bound an unchanged in-command camera capture, not checkpoint waiting.

    Uses local monotonic time and an authenticated owner's progress identity.
    Failure ends observation only: no restart, signal, replay or allocation API.
    A responsive /ready endpoint alone does not prove native progress.
    """
    def __init__(self, command_id, seconds=900):
        if seconds <= 0:
            raise ValueError('positive capture observation budget required')
        self.command_id, self.seconds = command_id, seconds
        self.identity = self.since = None

    def observe(self, ready, now):
        row = ready.get('native_progress') or {}
        if (ready.get('active_command_id') != self.command_id
                or ready.get('phase') != 'RUNNING'
                or row.get('phase') != 'capture_teacher'
                or row.get('event') != 'start'):
            self.identity = self.since = None
            return
        identity = (ready.get('worker_instance_id'), row.get('run_id'),
                    row.get('round_id'), row.get('vector_step'), row.get('sequence'))
        if identity != self.identity:
            self.identity, self.since = identity, now
        elif now - self.since >= self.seconds:
            raise RuntimeError('native camera capture progress stalled; inspect owner; replay forbidden')


class ResidentClient:
    def __init__(self, port, token_file, *, source_sha256, gpu_uuid, timeout=15):
        if type(port) is not int or not 1024 <= port < 65536:
            raise ValueError("invalid loopback port")
        path = Path(token_file)
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise ValueError("resident token must be owner-only")
        self.token = path.read_text().strip()
        self.base = f"http://127.0.0.1:{port}"
        self.timeout = timeout
        self.source, self.gpu = source_sha256, gpu_uuid
        self.instance = None

    def _request(self, route, body=None):
        raw = canonical(body) if body is not None else None
        request = Request(self.base+route, data=raw, headers={"Authorization": "Bearer "+self.token,
                          "Content-Type": "application/json"})
        deadline = time.monotonic()+120
        while True:
            self._check_observer_cancelled()
            try:
                with urlopen(request, timeout=min(self.timeout, max(.1, deadline-time.monotonic()))) as response:
                    data = response.read((1 << 20)+1)
                    if len(data) > 1 << 20:
                        raise RuntimeError("resident response size limit")
                    return json.loads(data)
            except HTTPError:
                raise  # Auth, identity/conflict and uncertain owner are terminal.
            except (URLError, TimeoutError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise
                self._observer_sleep(min(2, max(0, deadline-time.monotonic())))

    def _check_observer_cancelled(self):
        event = getattr(self, 'observer_cancel', None)
        if event is not None and event.is_set():
            raise RuntimeError('paired observer cancelled after peer failure; native replay forbidden')

    def _observer_sleep(self, seconds):
        event = getattr(self, 'observer_cancel', None)
        if event is None:
            time.sleep(seconds)
        else:
            event.wait(seconds)
            self._check_observer_cancelled()

    def ready(self):
        result = self._request("/ready")
        if (result.get("role") != "isaac_resident" or result.get("schema") != "roborender.isaac-resident.v1"
                or result.get("ok") is not True or result.get("source_sha256") != self.source
                or result.get("gpu_uuid") != self.gpu or result.get("native_initializations") != 1):
            raise RuntimeError("resident native/source/GPU identity mismatch")
        if self.instance is not None and self.instance != result["worker_instance_id"]:
            raise RuntimeError("resident owner restarted; native state cannot be restored")
        self.instance = result["worker_instance_id"]
        return result

    def submit(self, command_id, operation, payload):
        self.ready()
        body = {"schema": "roborender.isaac-resident.v1", "worker_instance_id": self.instance,
                "command_id": command_id, "operation": operation, "payload": payload}
        result = self._request("/v1/commands", body)
        if (result.get("command_id") != command_id or result.get("worker_instance_id") != self.instance
                or result.get("status") != "accepted"
                or result.get("payload_sha256") != hashlib.sha256(canonical(body)).hexdigest()):
            raise RuntimeError("resident command receipt differs")
        return result

    def wait(self, command_id, *, wall_budget=18000, interval=2):
        deadline = time.monotonic()+wall_budget
        capture_progress = NativeCaptureProgressGuard(command_id)
        while time.monotonic() < deadline:
            self._check_observer_cancelled()
            try:
                ready = self.ready()
                capture_progress.observe(ready, time.monotonic())
                result = self._request("/v1/commands/"+command_id)
            except HTTPError:
                # Authentication, missing command, conflicts and explicit
                # server rejection require reconciliation, not transport retry.
                raise
            except (URLError, TimeoutError, ConnectionError):
                # A bounded _request outage does not mean the native command
                # stopped. Re-attest the SAME owner and read the SAME command
                # again within this observer budget; never submit or step here.
                self._observer_sleep(min(interval, max(0, deadline-time.monotonic())))
                continue
            if result.get("status") == "COMPLETE":
                return result
            if result.get("status") not in {"QUEUED", "RUNNING"}:
                raise RuntimeError("resident phase failed/uncertain; replay forbidden")
            self._observer_sleep(min(interval, max(0, deadline-time.monotonic())))
        # Timing out the observer does not kill or re-execute a native phase.
        raise TimeoutError("resident phase still running; inspect its command receipt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--command", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    client = ResidentClient(args.port, args.token_file, source_sha256=args.source_sha256, gpu_uuid=args.gpu_uuid)
    print(json.dumps(client.ready(), sort_keys=True), flush=True)
    if args.command:
        command = json.loads(args.command.read_bytes())
        print(json.dumps(client.submit(command["command_id"], command["operation"], command["payload"]), sort_keys=True), flush=True)
        if args.wait:
            print(json.dumps(client.wait(command["command_id"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
