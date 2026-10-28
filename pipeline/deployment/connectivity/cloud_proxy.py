"""Alpha-free SSH transport through an authenticated Lepton pod shell.

Uses ONLY this host's approved SDK login. The WebSocket is TLS verified; the
inner SSH connection separately verifies its pinned host key. No token enters
argv, stdout, a child environment, or a repository file.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import secrets
import shlex
import sys
import threading
import time
from urllib.parse import quote, urlsplit, urlunsplit

WORKSPACE = "b5k2m9x7"
CPU_POD = "leo-cosmos-h200-access"
ORIGIN = "https://dashboard.dgxc-lepton.nvidia.com"


def checked_pod(pod):
    if pod not in {CPU_POD, "isaac-sim-render-leo"} and not re.fullmatch(r"leo-rr-(?:beta|leo)-[a-f0-9]{12}-r[1-4]", pod):
        raise ValueError("pod is outside the RoboRender management/workload scope")
    return pod


def websocket_url(api_url, pod, replica):
    checked_pod(pod)
    parsed = urlsplit(api_url)
    if (parsed.scheme != "https" or parsed.netloc != "gateway.dgxc-lepton.nvidia.com"
            or parsed.path.rstrip("/") != f"/api/v2/workspaces/{WORKSPACE}"
            or parsed.query or parsed.fragment or not re.fullmatch(r"[A-Za-z0-9_.-]+", replica)):
        raise ValueError("unexpected cloud workspace or replica")
    return urlunsplit(("wss", parsed.netloc, parsed.path.rstrip("/")
                       + f"/deployments/{quote(pod, safe='')}/replicas/{quote(replica, safe='')}/shell", "", ""))


def open_shell(pod):
    checked_pod(pod)
    os.environ["LOGURU_LEVEL"] = "ERROR"
    from leptonai.api.v2.workspace_record import WorkspaceRecord
    import websocket
    client = WorkspaceRecord.client(WORKSPACE)
    # Validate the destination BEFORE sending the bearer header.
    websocket_url(client.url, pod, "preflight")
    client._timeout = 12
    client._header["Origin"] = ORIGIN
    response = client._get(f"/deployments/{pod}/replicas", allow_redirects=False)
    if response.status_code != 200:
        raise RuntimeError(f"cloud replica lookup HTTP {response.status_code}")
    replicas = response.json()
    if not isinstance(replicas, list) or len(replicas) != 1:
        raise RuntimeError("expected exactly one running pod replica")
    replica = replicas[0].get("metadata", {}).get("id")
    if not isinstance(replica, str):
        raise RuntimeError("replica identity missing")
    connection = websocket.create_connection(
        websocket_url(client.url, pod, replica),
        header={"Authorization": client._header["Authorization"], "User-Agent": "roborender-beta-access/1"},
        origin=ORIGIN, subprotocols=["v4.channel.k8s.io"],
        enable_multithread=True, timeout=20, redirect_limit=0,
    )
    return connection


def channel_payload(frame):
    if isinstance(frame, str):
        frame = frame.encode("utf-8")
    if not frame:
        raise EOFError("cloud shell closed")
    if frame[0] == 3:
        raise RuntimeError("remote shell exited")
    return frame[1:] if frame[0] == 1 else b""


def read_until(ws, marker, *, deadline=25, limit=1 << 20):
    data = bytearray()
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        ws.settimeout(max(.1, min(10, end - time.monotonic())))
        data.extend(channel_payload(ws.recv()))
        where = data.find(marker)
        if where >= 0:
            return bytes(data[:where]), bytes(data[where + len(marker):])
        if len(data) > limit:
            raise RuntimeError("unexpected oversized shell preamble")
    raise TimeoutError("cloud shell handshake deadline")


def marker_command(marker):
    # The command echo contains escaped octal, not the literal marker. This
    # prevents accidentally mistaking echoed input for executed output.
    return "printf " + shlex.quote("".join(f"\\{v:03o}" for v in marker))


def run_code(code):
    """Bounded one-time provisioning on the existing CPU pod, never a GPU."""
    marker = b"\x1eRR_" + secrets.token_hex(12).encode() + b"\x1f"
    finish = b"\x1eEND_" + secrets.token_hex(12).encode() + b"\x1f"
    program = "import base64;exec(base64.b64decode(" + repr(base64.b64encode(code.encode()).decode()) + "))"
    command = ("stty -echo; " + marker_command(marker) + "; python3 -c " + shlex.quote(program)
               + "; " + marker_command(finish) + "; exit\n")
    ws = open_shell(CPU_POD)
    try:
        ws.send_binary(b"\x00" + command.encode())
        _, remainder = read_until(ws, marker)
        # Shell frames may combine start marker, output and finish marker.
        output = bytearray(remainder)
        deadline = time.monotonic() + 45
        while finish not in output:
            if time.monotonic() >= deadline:
                raise TimeoutError("cloud provisioning deadline")
            ws.settimeout(15)
            output.extend(channel_payload(ws.recv()))
            if len(output) > 1 << 20:
                raise RuntimeError("oversized provisioning result")
        return bytes(output).split(finish, 1)[0].strip()
    finally:
        ws.close()


def write_all(fd, data):
    data = memoryview(data)
    while data:
        count = os.write(fd, data)
        if count <= 0:
            raise BrokenPipeError()
        data = data[count:]


def proxy(pod, port=2222):
    if port != 2222 and not (pod == CPU_POD and port == 2233):
        raise ValueError("only the pinned pod sshd port is allowed")
    ws = open_shell(pod)
    marker = b"\x1eSSH_" + secrets.token_hex(12).encode() + b"\x1f"
    # Raw mode is essential: SSH binary must never traverse cooked TTY mode.
    tcp = f"exec 3<>/dev/tcp/127.0.0.1/{port}; cat <&3 & exec cat >&3"
    command = "stty raw -echo; " + marker_command(marker) + "; exec bash -c " + shlex.quote(tcp) + "\n"
    stop = threading.Event()
    try:
        ws.send_binary(b"\x00" + command.encode())
        _, initial = read_until(ws, marker)
        ws.settimeout(None)
        if initial:
            write_all(sys.stdout.fileno(), initial)

        def upstream():
            try:
                while not stop.is_set():
                    data = os.read(sys.stdin.fileno(), 65536)
                    if not data:
                        break
                    ws.send_binary(b"\x00" + data)
            except Exception:
                pass
            finally:
                stop.set()
                ws.close()

        def keepalive():
            while not stop.wait(30):
                try:
                    ws.send_binary(b"\x00")
                except Exception:
                    return

        threading.Thread(target=upstream, daemon=True).start()
        threading.Thread(target=keepalive, daemon=True).start()
        while not stop.is_set():
            data = channel_payload(ws.recv())
            if data:
                write_all(sys.stdout.fileno(), data)
    finally:
        stop.set()
        ws.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pod")
    args = parser.parse_args()
    try:
        proxy(args.pod)
    except (EOFError, BrokenPipeError):
        return 0
    except Exception as error:
        # Exception bodies from SDK/HTTP clients can contain sensitive data.
        print("roborender-cloud-proxy failed: " + type(error).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
