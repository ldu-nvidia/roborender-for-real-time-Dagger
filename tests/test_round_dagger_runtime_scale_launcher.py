"""Fail-closed shell gates for the bounded shadow scale launcher."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts/round_dagger/run_runtime_adapter_shadow_scale_docker.sh"
ACK = "bounded-runtime-adapter-shadow-scale-v1"
IMAGE_SHA = "a" * 64


class RuntimeScaleLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="runtime_scale_launcher_")
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "sealed"
        self.launcher = self.root / "scripts/round_dagger/run_runtime_adapter_shadow_scale_docker.sh"
        self.launcher.parent.mkdir(parents=True)
        self.launcher.write_bytes(SOURCE.read_bytes())
        self.launcher.chmod(0o755)
        self.plan = self.base / "plan.json"
        self.plan.write_text("{}\n", encoding="utf-8")
        self.plan_sha = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        self.state = self.base / "state.json"
        self.state.write_text(json.dumps({"2": "empty"}), encoding="utf-8")
        self.log = self.base / "events.jsonl"
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.fake_python = self.bin / "python3"
        self.fake_docker = self.bin / "docker"
        self._write_fake_python()
        self._write_fake_docker()
        self.fake_python.chmod(0o755)
        self.fake_docker.chmod(0o755)
        self.root.chmod(0o555)

    def tearDown(self) -> None:
        self.root.chmod(0o755)
        self.temporary.cleanup()

    def _write_fake_python(self) -> None:
        self.fake_python.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/python3
                import json, os, sys
                if sys.argv[1:2] == ["-c"]:
                    os.execv("/usr/bin/python3", ["python3", *sys.argv[1:]])
                args = sys.argv[1:]
                state_path = os.environ["SCALE_STATE"]
                log = os.environ["SCALE_LOG"]
                def event(name, **values):
                    with open(log, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({{"event": name, **values}}, sort_keys=True) + "\\n")
                operation = args[2]
                state = json.load(open(state_path, encoding="utf-8"))
                if operation == "validate":
                    print(json.dumps({{
                        "source": {{"repository_root": {str(self.root)!r}}},
                        "actor_image": "actor:test",
                        "actor_image_id_sha256": {IMAGE_SHA!r},
                        "rungs": {{"2": {{"state": state["2"]}}}},
                    }}))
                elif operation == "environment":
                    rung = args[args.index("--rung") + 1]
                    values = {{
                        "ROUND_DAGGER_ACTOR_DOCKER_GPU_ID": "0",
                        "ROUND_DAGGER_REPOSITORY_ROOT": {str(self.root)!r},
                        "ROUND_DAGGER_RUNTIME_SCALE_PLAN": {str(self.plan)!r},
                        "ROUND_DAGGER_RUNTIME_SCALE_PLAN_SHA256": {self.plan_sha!r},
                        "ROUND_DAGGER_RUNTIME_SCALE_RUNG_CONTRACT": {str(self.base / 'rung.json')!r},
                        "ROUND_DAGGER_RUNTIME_SCALE_RUNG_CONTRACT_SHA256": "b" * 64,
                        "ROUND_DAGGER_RUNTIME_SCALE_EVIDENCE": {str(self.base / 'run/ranks/rank_0000/evidence.json')!r},
                        "ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE": {str(self.base / 'run/ranks/rank_0000/evidence.json')!r},
                        "ROUND_DAGGER_RUNTIME_ADAPTER_MODE": "shadow",
                        "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK": "metadata-only-v1",
                        "ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK": "shadow-scale-2step-v1",
                        "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE": "live-isaac-v1",
                        "ROUND_DAGGER_VALIDATION_RANKS": "1",
                        "ROUND_DAGGER_VALIDATION_ENVS_PER_RANK": rung,
                        "ROUND_DAGGER_VALIDATION_STEPS": "2",
                        "ROUND_DAGGER_VALIDATION_PURPOSE": "train",
                        "ROUND_DAGGER_VALIDATION_TRAIN_ACK": "bounded-train-replay",
                        "ROUND_DAGGER_VALIDATION_ACTOR_RANK": "0",
                        "TORCH_HOME": "/cache",
                    }}
                    for name in sorted(values):
                        sys.stdout.buffer.write(name.encode() + b"\\0" + values[name].encode() + b"\\0")
                elif operation == "authorize":
                    event("authorize", argv=args)
                elif operation == "seal":
                    state["2"] = "sealed"
                    with open(state_path, "w", encoding="utf-8") as handle: json.dump(state, handle)
                    event("seal", argv=args)
                    print("{{}}")
                elif operation == "verify":
                    event("verify", argv=args)
                    print('{{"status":"PASS"}}')
                else:
                    raise SystemExit(f"unexpected operation: {{operation}}")
                """
            ),
            encoding="utf-8",
        )

    def _write_fake_docker(self) -> None:
        self.fake_docker.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/python3
                import json, os, sys
                args = sys.argv[1:]
                with open(os.environ["SCALE_LOG"], "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({{"event": "docker", "argv": args}}, sort_keys=True) + "\\n")
                if args[:2] == ["image", "inspect"]:
                    print("sha256:{IMAGE_SHA}")
                elif args[:1] != ["run"]:
                    raise SystemExit(2)
                """
            ),
            encoding="utf-8",
        )

    def _environment(self, acknowledgement=ACK):
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin}:" + environment["PATH"],
                "ROUND_DAGGER_DOCKER_BIN": str(self.fake_docker),
                "ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS": str(self.base),
                "ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE": "16g",
                "ROUND_DAGGER_ACTOR_DOCKER_IPC": "private",
                "SCALE_STATE": str(self.state),
                "SCALE_LOG": str(self.log),
            }
        )
        environment.pop("ROUND_DAGGER_RUNTIME_SCALE_LAUNCH_ACK", None)
        if acknowledgement is not None:
            environment["ROUND_DAGGER_RUNTIME_SCALE_LAUNCH_ACK"] = acknowledgement
        return environment

    def _run(self, acknowledgement=ACK):
        self.log.unlink(missing_ok=True)
        return subprocess.run(
            ["bash", str(self.launcher), str(self.plan), "2"],
            cwd=self.base,
            env=self._environment(acknowledgement),
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    def _events(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_requires_exact_ack_before_python_or_docker(self) -> None:
        for acknowledgement in (None, "wrong"):
            result = self._run(acknowledgement)
            self.assertEqual(result.returncode, 2)
            self.assertIn("scale launch requires", result.stderr)
            self.assertEqual(self._events(), [])

    def test_one_rung_uses_read_only_source_exact_env_then_seals(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self._events()
        docker = [event for event in events if event["event"] == "docker"]
        self.assertEqual(len(docker), 2)  # image inspect, then one actor run
        run = docker[1]["argv"]
        self.assertIn(f"type=bind,src={self.root},dst={self.root},readonly", run)
        self.assertIn("TORCH_HOME", run)
        self.assertIn("sha256:" + IMAGE_SHA, run)
        self.assertEqual(
            [event["event"] for event in events if event["event"] != "docker"],
            ["authorize", "seal", "verify"],
        )


if __name__ == "__main__":
    unittest.main()
