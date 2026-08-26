"""Fail-closed launcher gates for the bounded runtime-adapter A/B pair."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/round_dagger/run_runtime_adapter_ab_docker.sh"
ACTOR_VALIDATION = ROOT / "scripts/round_dagger/run_actor_validation.sh"
ACK = "bounded-runtime-adapter-ab-v1"
IMAGE_SHA256 = "d" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeABDockerLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="runtime_ab_launcher_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.plan = self.root / "plan.json"
        self.plan.write_bytes(b'{"fixture":"runtime-ab-launcher"}')
        self.plan_sha256 = _sha256(self.plan)
        self.state = self.root / "state.json"
        self.state.write_text(
            json.dumps(
                {"warmup": "pending", "legacy": "empty", "shadow": "empty"}
            ),
            encoding="utf-8",
        )
        self.log = self.root / "events.jsonl"
        self.control_path = self.root / "control.json"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.fake_python = self.bin / "python3"
        self.fake_docker = self.root / "fake-docker"
        self._write_fake_python()
        self._write_fake_docker()

        common = {
            "DRY_RUN": "0",
            "IMAGE3F_ENDPOINT_MANIFEST": str(self.root / "endpoints.txt"),
            "ROUND_DAGGER_ACTOR_DOCKER_GPU_ID": "3",
            "ROUND_DAGGER_IMAGE3F_SEED": "42",
            "ROUND_DAGGER_REPOSITORY_ROOT": str(ROOT),
            "ROUND_DAGGER_RTX_IMAGE": "actor:test",
            "ROUND_DAGGER_RTX_IMAGE_ID_SHA256": IMAGE_SHA256,
            "ROUND_DAGGER_RUNTIME_AB_PLAN": str(self.plan),
            "ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256": self.plan_sha256,
            "ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK": "paired-1x2-v1",
            "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE": "live-isaac-v1",
            "ROUND_DAGGER_SIMULATOR_SEED": "0",
            "ROUND_DAGGER_VALIDATION_ENVS_PER_RANK": "1",
            "ROUND_DAGGER_VALIDATION_PURPOSE": "train",
            "ROUND_DAGGER_VALIDATION_RANKS": "1",
            "ROUND_DAGGER_VALIDATION_STEPS": "2",
            "ROUND_DAGGER_VALIDATION_TRAIN_ACK": "bounded-train-replay",
        }
        environments: dict[str, dict[str, str]] = {}
        for arm in ("legacy", "shadow"):
            environment = {
                **common,
                "IMAGE3F_RUN_ID": f"runtime-ab-{arm}-seed42",
                "ROUND_DAGGER_RUNTIME_AB_ARM": arm,
                "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT": str(
                    self.root / f"{arm}.json"
                ),
                "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT_SHA256": (
                    "1" * 64 if arm == "legacy" else "2" * 64
                ),
                "ROUND_DAGGER_RUNTIME_AB_EVIDENCE": str(
                    self.root / f"{arm}-comparison.json"
                ),
                "ROUND_DAGGER_RUNTIME_AB_WARMUP_EVIDENCE": str(
                    self.root / "warmup-evidence.json"
                ),
                "ROUND_DAGGER_RUNTIME_AB_WARMUP_EVIDENCE_SHA256": "9" * 64,
                "ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE": str(
                    self.root / f"{arm}-runtime.json"
                ),
                "ROUND_DAGGER_RUNTIME_ADAPTER_MODE": arm,
                "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK": (
                    "" if arm == "legacy" else "metadata-only-v1"
                ),
            }
            environments[arm] = environment
        self.control = {
            "environments": environments,
            "inspect_image_sha256": IMAGE_SHA256,
            "no_seal_arms": [],
            "skip_warmup_evidence": False,
            "source_repository_root": str(ROOT),
            "warmup": {
                "schema": "fast_roborender.round_dagger.runtime_adapter_ab_warmup.v1",
                "endpoint": str(self.root / "renderer.txt"),
                "endpoint_sha256": "3" * 64,
                "anchor": str(self.root / "anchor.npz"),
                "anchor_sha256": "4" * 64,
                "output": str(self.root / "warmup-evidence.json"),
                "renderer_sha256": "5" * 64,
                "stream_id": "runtime-ab-launcher-warmup",
                "seed": 42,
                "reference_lag": 256,
                "actor_image": "actor:test",
                "actor_image_id_sha256": IMAGE_SHA256,
                "gr00t_python": "/opt/round-dagger-rtx/bin/python",
                "evidence_sha256": None,
                "inference_ms": None,
                "client_wall_ms": None,
            },
        }
        self._save_control()

    def _save_control(self) -> None:
        self.control_path.write_text(
            json.dumps(self.control, sort_keys=True), encoding="utf-8"
        )

    def _set_states(
        self, *, legacy: str, shadow: str, warmup: str = "pending"
    ) -> None:
        self.state.write_text(
            json.dumps(
                {"warmup": warmup, "legacy": legacy, "shadow": shadow}
            ),
            encoding="utf-8",
        )

    def _write_fake_python(self) -> None:
        self.fake_python.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/python3
                import json
                import os
                import sys

                control_path = os.environ["FAKE_RUNTIME_AB_CONTROL"]
                state_path = os.environ["FAKE_RUNTIME_AB_STATE"]
                log_path = os.environ["FAKE_RUNTIME_AB_LOG"]

                def load(path):
                    with open(path, "r", encoding="utf-8") as handle:
                        return json.load(handle)

                def event(name, **fields):
                    with open(log_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"event": name, **fields}, sort_keys=True) + "\\n")

                args = sys.argv[1:]
                if args and args[0] == "-c":
                    report = json.load(sys.stdin)
                    if "repository_root" in args[1] and "source" in args[1]:
                        print(report["source"]["repository_root"])
                    elif "arms" in report:
                        arm = args[-1] if len(args) > 2 else "legacy"
                        print(report["arms"][arm]["state"])
                    else:
                        for name in (
                            "state", "endpoint", "anchor", "output",
                            "renderer_sha256", "stream_id", "seed",
                            "reference_lag", "actor_image",
                            "actor_image_id_sha256", "gr00t_python",
                        ):
                            sys.stdout.buffer.write(
                                str(report[name]).encode("utf-8") + b"\\0"
                            )
                    raise SystemExit(0)
                if args[:2] != ["-m", "round_dagger.runtime_ab"] or len(args) < 3:
                    raise SystemExit(f"unexpected fake-python invocation: {args!r}")
                operation = args[2]
                control = load(control_path)
                states = load(state_path)
                warmup = dict(control["warmup"])
                warmup["state"] = states["warmup"]
                if states["warmup"] == "validated":
                    warmup["evidence_sha256"] = "9" * 64
                    warmup["inference_ms"] = 12.5
                    warmup["client_wall_ms"] = 14.0
                if operation == "validate":
                    event("validate", states=states)
                    print(json.dumps({
                        "source": {
                            "repository_root": control["source_repository_root"]
                        },
                        "warmup": warmup,
                        "arms": {
                            "legacy": {"state": states["legacy"]},
                            "shadow": {"state": states["shadow"]},
                        }
                    }))
                elif operation == "warmup":
                    event("warmup_report", state=states["warmup"])
                    print(json.dumps(warmup))
                elif operation == "environment":
                    arm = args[args.index("--arm") + 1]
                    if states["warmup"] != "validated":
                        raise SystemExit(2)
                    environment = control["environments"][arm]
                    event("environment", arm=arm, names=sorted(environment))
                    for name in sorted(environment):
                        sys.stdout.buffer.write(name.encode("ascii") + b"\\0")
                        sys.stdout.buffer.write(environment[name].encode("utf-8") + b"\\0")
                elif operation == "authorize":
                    arm = args[args.index("--arm") + 1]
                    expected = control["environments"][arm]
                    mismatches = {
                        name: [value, os.environ.get(name)]
                        for name, value in expected.items()
                        if os.environ.get(name) != value
                    }
                    event("authorize", arm=arm, mismatches=mismatches)
                    if mismatches:
                        raise SystemExit(2)
                    print(json.dumps({"status": "authorized", "arm": arm}))
                elif operation == "compare":
                    event("compare", states=states)
                    if (
                        states["warmup"] != "validated"
                        or states["legacy"] != "sealed"
                        or states["shadow"] != "sealed"
                    ):
                        raise SystemExit(2)
                    print(json.dumps({"status": "PASS"}))
                else:
                    raise SystemExit(f"unsupported runtime-ab operation: {operation}")
                """
            ),
            encoding="utf-8",
        )
        self.fake_python.chmod(0o755)

    def _write_fake_docker(self) -> None:
        self.fake_docker.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/python3
                import json
                import os
                import sys

                control_path = os.environ["FAKE_RUNTIME_AB_CONTROL"]
                state_path = os.environ["FAKE_RUNTIME_AB_STATE"]
                log_path = os.environ["FAKE_RUNTIME_AB_LOG"]

                def load(path):
                    with open(path, "r", encoding="utf-8") as handle:
                        return json.load(handle)

                def event(name, **fields):
                    with open(log_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"event": name, **fields}, sort_keys=True) + "\\n")

                args = sys.argv[1:]
                control = load(control_path)
                if args[:2] == ["image", "inspect"]:
                    event("image_inspect", argv=args)
                    print("sha256:" + control["inspect_image_sha256"])
                    raise SystemExit(0)
                if not args or args[0] != "run":
                    raise SystemExit(f"unexpected fake-docker invocation: {args!r}")
                warmup_module = "scripts.round_dagger.warm_image3f_renderer"
                if warmup_module in args:
                    module_index = args.index(warmup_module)
                    event(
                        "warmup_run",
                        argv=args,
                        command=args[module_index - 2:],
                        image=args[module_index - 3],
                    )
                    if not control.get("skip_warmup_evidence", False):
                        states = load(state_path)
                        states["warmup"] = "validated"
                        with open(state_path, "w", encoding="utf-8") as handle:
                            json.dump(states, handle)
                    raise SystemExit(0)
                environment_names = []
                fixed_environment = {}
                index = 0
                while index < len(args):
                    if args[index] == "--env":
                        entry = args[index + 1]
                        if "=" in entry:
                            name, value = entry.split("=", 1)
                            fixed_environment[name] = value
                        else:
                            environment_names.append(entry)
                        index += 2
                    else:
                        index += 1
                command_index = args.index("/usr/bin/env")
                image = args[command_index - 1]
                values = {name: os.environ.get(name) for name in environment_names}
                arm = values["ROUND_DAGGER_RUNTIME_AB_ARM"]
                event(
                    "docker_run",
                    arm=arm,
                    argv=args,
                    command=args[command_index:],
                    environment_names=environment_names,
                    environment_values=values,
                    fixed_environment=fixed_environment,
                    image=image,
                )
                if arm not in control.get("no_seal_arms", []):
                    states = load(state_path)
                    states[arm] = "sealed"
                    with open(state_path, "w", encoding="utf-8") as handle:
                        json.dump(states, handle)
                raise SystemExit(0)
                """
            ),
            encoding="utf-8",
        )
        self.fake_docker.chmod(0o755)

    def _environment(self, *, acknowledgement: str | None = ACK) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin}:{environment['PATH']}",
                "FAKE_RUNTIME_AB_CONTROL": str(self.control_path),
                "FAKE_RUNTIME_AB_STATE": str(self.state),
                "FAKE_RUNTIME_AB_LOG": str(self.log),
                "ROUND_DAGGER_DOCKER_BIN": str(self.fake_docker),
                "ROUND_DAGGER_ACTOR_DOCKER_IPC": "private",
                "ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE": "16g",
                "ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS": str(self.root),
                # These hostile ambient values must never become Docker --env
                # entries. Only the dynamic environment operation is trusted.
                "ROUND_DAGGER_RUNTIME_AB_HOST_ONLY": "must-not-forward",
                "ROUND_DAGGER_RUNTIME_ADAPTER_HOST_ONLY": "must-not-forward",
                "WANDB_API_KEY": "must-not-forward-secret",
                "RANK": "99",
                "WORLD_SIZE": "100",
            }
        )
        environment.pop("ROUND_DAGGER_RUNTIME_AB_LAUNCH_ACK", None)
        if acknowledgement is not None:
            environment["ROUND_DAGGER_RUNTIME_AB_LAUNCH_ACK"] = acknowledgement
        return environment

    def _run(
        self,
        selection: str = "all",
        *,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.log.unlink(missing_ok=True)
        return subprocess.run(
            ["bash", str(LAUNCHER), str(self.plan), selection],
            cwd=ROOT,
            env=environment if environment is not None else self._environment(),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def _events(self) -> list[dict[str, object]]:
        if not self.log.exists():
            return []
        return [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
        ]

    def test_requires_exact_explicit_ack_before_any_cli_or_docker_call(self) -> None:
        for acknowledgement in (None, "wrong-ack"):
            with self.subTest(acknowledgement=acknowledgement):
                result = self._run(
                    "legacy",
                    environment=self._environment(acknowledgement=acknowledgement),
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("requires ROUND_DAGGER_RUNTIME_AB_LAUNCH_ACK", result.stderr)
                self.assertEqual(self._events(), [])

    def test_rejects_host_ipc_and_mismatched_image_identity_before_run(self) -> None:
        environment = self._environment()
        environment["ROUND_DAGGER_ACTOR_DOCKER_IPC"] = "host"
        rejected = self._run("legacy", environment=environment)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("requires private Docker IPC", rejected.stderr)
        self.assertFalse(any(event["event"] == "docker_run" for event in self._events()))

        self.control["inspect_image_sha256"] = "e" * 64
        self._save_control()
        rejected = self._run("legacy")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("image ID differs from immutable plan", rejected.stderr)
        self.assertFalse(any(event["event"] == "docker_run" for event in self._events()))

    def test_input_profile_is_bounded_even_when_fake_authorization_accepts_it(self) -> None:
        for arm in ("legacy", "shadow"):
            self.control["environments"][arm][
                "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE"
            ] = "tolerant"
        self._save_control()
        rejected = self._run("legacy")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("input profile is outside", rejected.stderr)
        self.assertFalse(any(event["event"] == "docker_run" for event in self._events()))

        for arm in ("legacy", "shadow"):
            self.control["environments"][arm][
                "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE"
            ] = "canonical-zero-robot-mask-v1"
        self._save_control()
        accepted = self._run("legacy")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        runs = [event for event in self._events() if event["event"] == "docker_run"]
        self.assertEqual([event["arm"] for event in runs], ["legacy"])

    def test_deterministic_profile_requires_and_forwards_cublas_contract(self) -> None:
        for arm in ("legacy", "shadow"):
            self.control["environments"][arm][
                "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE"
            ] = "canonical-zero-robot-mask-deterministic-cuda-v1"
        self._save_control()
        rejected = self._run("legacy")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("requires CUBLAS_WORKSPACE_CONFIG=:4096:8", rejected.stderr)
        self.assertFalse(any(event["event"] == "docker_run" for event in self._events()))

        for arm in ("legacy", "shadow"):
            self.control["environments"][arm]["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self._save_control()
        accepted = self._run("legacy")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        run = next(event for event in self._events() if event["event"] == "docker_run")
        self.assertEqual(
            run["environment_values"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8"
        )

    def test_repository_binding_mismatch_cannot_run_or_publish_warmup(self) -> None:
        self.control["source_repository_root"] = str(self.root / "other-source")
        self._save_control()
        rejected = self._run("legacy")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("bound to a different repository root", rejected.stderr)
        self.assertFalse(any(event["event"] == "warmup_run" for event in self._events()))
        self.assertFalse(any(event["event"] == "docker_run" for event in self._events()))
        self.assertEqual(
            json.loads(self.state.read_text(encoding="utf-8"))["warmup"], "pending"
        )

    def test_all_uses_only_dynamic_environment_and_runs_immutable_image_in_order(self) -> None:
        result = self._run("all")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self._events()
        warmups = [event for event in events if event["event"] == "warmup_run"]
        runs = [event for event in events if event["event"] == "docker_run"]
        self.assertEqual(len(warmups), 1)
        self.assertEqual([event["arm"] for event in runs], ["legacy", "shadow"])
        self.assertEqual(events[-1]["event"], "compare")
        self.assertLess(events.index(warmups[0]), events.index(runs[0]))
        self.assertLess(events.index(runs[0]), events.index(runs[1]))
        self.assertLess(events.index(runs[1]), len(events) - 1)
        self.assertEqual(warmups[0]["image"], f"sha256:{IMAGE_SHA256}")
        self.assertIn("--ipc=private", warmups[0]["argv"])
        self.assertNotIn("--ipc=host", warmups[0]["argv"])
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", warmups[0]["argv"])
        self.assertIn(
            f"PYTHONPATH={ROOT}:{ROOT}/GR00T-VisualSim2Real", warmups[0]["argv"]
        )
        self.assertIn(
            f"type=bind,src={ROOT},dst={ROOT},readonly", warmups[0]["argv"]
        )
        self.assertEqual(
            warmups[0]["command"],
            [
                "/opt/round-dagger-rtx/bin/python",
                "-m",
                "scripts.round_dagger.warm_image3f_renderer",
                "--endpoint",
                str(self.root / "renderer.txt"),
                "--anchor",
                str(self.root / "anchor.npz"),
                "--output",
                str(self.root / "warmup-evidence.json"),
                "--renderer-sha256",
                "5" * 64,
                "--stream-id",
                "runtime-ab-launcher-warmup",
            ],
        )
        self.assertEqual(
            len([event for event in events if event["event"] == "image_inspect"]),
            1,
        )

        for event in runs:
            arm = str(event["arm"])
            argv = event["argv"]
            self.assertIn("--ipc=private", argv)
            self.assertNotIn("--ipc=host", argv)
            self.assertIn("--shm-size=16g", argv)
            self.assertEqual(event["image"], f"sha256:{IMAGE_SHA256}")
            self.assertEqual(
                event["command"],
                [
                    "/usr/bin/env",
                    "bash",
                    str(ROOT / "scripts/round_dagger/run_actor_validation.sh"),
                ],
            )
            expected = self.control["environments"][arm]
            self.assertEqual(event["environment_names"], sorted(expected))
            self.assertEqual(event["environment_values"], expected)
            self.assertEqual(
                event["fixed_environment"], {"PYTHONDONTWRITEBYTECODE": "1"}
            )
            self.assertIn(
                f"type=bind,src={ROOT},dst={ROOT},readonly", event["argv"]
            )
            self.assertNotIn("ROUND_DAGGER_RUNTIME_AB_LAUNCH_ACK", event["environment_names"])
            self.assertNotIn("ROUND_DAGGER_RUNTIME_AB_HOST_ONLY", event["environment_names"])
            self.assertNotIn("ROUND_DAGGER_RUNTIME_ADAPTER_HOST_ONLY", event["environment_names"])
            self.assertNotIn("WANDB_API_KEY", event["environment_names"])
            self.assertNotIn("RANK", event["environment_names"])
            self.assertEqual(event["environment_values"]["ROUND_DAGGER_SIMULATOR_SEED"], "0")
            self.assertEqual(event["environment_values"]["ROUND_DAGGER_IMAGE3F_SEED"], "42")
            self.assertEqual(
                event["environment_values"][
                    "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE"
                ],
                "live-isaac-v1",
            )
            self.assertEqual(
                event["environment_values"]["IMAGE3F_RUN_ID"],
                f"runtime-ab-{arm}-seed42",
            )

    def test_shadow_rejects_unsealed_legacy_and_partial_evidence(self) -> None:
        rejected = self._run("shadow")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("shadow requires a sealed legacy arm", rejected.stderr)
        self.assertFalse(any(event["event"] == "docker_run" for event in self._events()))

        self._set_states(legacy="sealed", shadow="partial", warmup="validated")
        rejected = self._run("shadow")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("contains immutable partial evidence", rejected.stderr)
        self.assertFalse(any(event["event"] == "docker_run" for event in self._events()))

    def test_zero_exit_without_seal_cannot_advance_to_shadow_or_compare(self) -> None:
        self.control["no_seal_arms"] = ["legacy"]
        self._save_control()
        rejected = self._run("all")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("exited without publishing a sealed cohort", rejected.stderr)
        events = self._events()
        runs = [event for event in events if event["event"] == "docker_run"]
        self.assertEqual([event["arm"] for event in runs], ["legacy"])
        self.assertFalse(any(event["event"] == "compare" for event in events))

    def test_resume_never_duplicates_warmup_and_missing_interarm_warmup_fails(self) -> None:
        self._set_states(legacy="sealed", shadow="empty", warmup="validated")
        resumed = self._run("shadow")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        events = self._events()
        self.assertFalse(any(event["event"] == "warmup_run" for event in events))
        self.assertEqual(
            [event["arm"] for event in events if event["event"] == "docker_run"],
            ["shadow"],
        )
        self.assertFalse(any(event["event"] == "compare" for event in events))

        self._set_states(legacy="sealed", shadow="empty", warmup="pending")
        rejected = self._run("shadow")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("may only be created before an empty legacy arm", rejected.stderr)
        self.assertFalse(any(event["event"] == "warmup_run" for event in self._events()))
        self.assertFalse(any(event["event"] == "docker_run" for event in self._events()))

    def test_warmup_zero_exit_without_evidence_cannot_start_legacy(self) -> None:
        self.control["skip_warmup_evidence"] = True
        self._save_control()
        rejected = self._run("legacy")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("did not publish validated immutable evidence", rejected.stderr)
        events = self._events()
        self.assertEqual(
            len([event for event in events if event["event"] == "warmup_run"]),
            1,
        )
        self.assertFalse(any(event["event"] == "docker_run" for event in events))


class RuntimeABActorValidationInteractionTest(unittest.TestCase):
    def _fixture(self):
        # Import lazily so unittest discovery does not collect the reusable
        # fixture class as a second copy of its entire source test suite.
        from tests.test_round_dagger_actor_validation import ActorValidationDryRunTest

        fixture = ActorValidationDryRunTest(
            methodName="test_one_rank_eval_dry_run_binds_exact_action_contract"
        )
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        return fixture

    def _run_actor(self, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(ACTOR_VALIDATION)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def test_actor_rejects_standalone_shadow_and_partial_ab_metadata(self) -> None:
        fixture = self._fixture()
        standalone = fixture._environment()
        standalone.update(
            {
                "ROUND_DAGGER_RUNTIME_ADAPTER_MODE": "shadow",
                "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK": "metadata-only-v1",
            }
        )
        rejected = self._run_actor(standalone)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("shadow runtime requires an immutable runtime A/B plan", rejected.stderr)

        partial = fixture._environment()
        partial["ROUND_DAGGER_RUNTIME_AB_PLAN"] = str(fixture.root / "plan.json")
        rejected = self._run_actor(partial)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("runtime adapter A/B launch is missing", rejected.stderr)

    def test_actor_preserves_planned_run_id_and_seeds_to_trainer(self) -> None:
        fixture = self._fixture()
        plan = fixture.root / "runtime-ab-plan.json"
        plan.write_bytes(b"runtime-ab-plan")
        arm_contract = fixture.root / "shadow-arm.json"
        arm_contract.write_bytes(b"shadow-arm")
        trace = fixture.root / "trainer-trace.json"
        authorize_log = fixture.root / "authorize.json"

        trainer = fixture.root / "trace-trainer"
        trainer.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import os
                import sys
                with open({str(trace)!r}, "w", encoding="utf-8") as handle:
                    json.dump({{
                        "argv": sys.argv[1:],
                        "environment": {{
                            name: os.environ.get(name)
                            for name in (
                                "IMAGE3F_RUN_ID",
                                "ROUND_DAGGER_IMAGE3F_SEED",
                                "ROUND_DAGGER_SIMULATOR_SEED",
                                "ROUND_DAGGER_RUNTIME_ADAPTER_MODE",
                            )
                        }},
                    }}, handle, sort_keys=True)
                raise SystemExit(23)
                """
            ),
            encoding="utf-8",
        )
        trainer.chmod(0o755)

        fake_bin = fixture.root / "runtime-ab-bin"
        fake_bin.mkdir()
        fake_python = fake_bin / "python3"
        fake_python.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import os
                import sys
                args = sys.argv[1:]
                if args[:3] == ["-m", "round_dagger.runtime_ab", "authorize"]:
                    with open({str(authorize_log)!r}, "w", encoding="utf-8") as handle:
                        json.dump({{
                            "argv": args,
                            "arm": os.environ.get("ROUND_DAGGER_RUNTIME_AB_ARM"),
                            "run_id": os.environ.get("IMAGE3F_RUN_ID"),
                        }}, handle, sort_keys=True)
                    raise SystemExit(0)
                os.execv({sys.executable!r}, [{sys.executable!r}, *args])
                """
            ),
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        environment = fixture._environment(
            purpose="train", train_ack=True, dry_run=False
        )
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "GR00T_PYTHON": str(trainer),
                "IMAGE3F_RUN_ID": "runtime-ab-shadow-seed42",
                "ROUND_DAGGER_IMAGE3F_SEED": "42",
                "ROUND_DAGGER_SIMULATOR_SEED": "0",
                "ROUND_DAGGER_RUNTIME_AB_PLAN": str(plan),
                "ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256": _sha256(plan),
                "ROUND_DAGGER_RUNTIME_AB_ARM": "shadow",
                "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT": str(arm_contract),
                "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT_SHA256": _sha256(
                    arm_contract
                ),
                "ROUND_DAGGER_RUNTIME_AB_EVIDENCE": str(
                    fixture.root / "shadow-comparison.json"
                ),
                "ROUND_DAGGER_RUNTIME_ADAPTER_MODE": "shadow",
                "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK": "metadata-only-v1",
                "ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK": "paired-1x2-v1",
                "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE": "live-isaac-v1",
                "ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE": str(
                    fixture.root / "shadow-runtime.json"
                ),
            }
        )
        result = self._run_actor(environment)
        self.assertEqual(result.returncode, 23, result.stderr)
        authorization = json.loads(authorize_log.read_text(encoding="utf-8"))
        self.assertEqual(authorization["arm"], "shadow")
        self.assertEqual(authorization["run_id"], "runtime-ab-shadow-seed42")
        self.assertIn("--expected-plan-sha256", authorization["argv"])

        observed = json.loads(trace.read_text(encoding="utf-8"))
        self.assertEqual(
            observed["environment"],
            {
                "IMAGE3F_RUN_ID": "runtime-ab-shadow-seed42",
                "ROUND_DAGGER_IMAGE3F_SEED": "42",
                "ROUND_DAGGER_RUNTIME_ADAPTER_MODE": "shadow",
                "ROUND_DAGGER_SIMULATOR_SEED": "0",
            },
        )
        self.assertIn("seed=0", observed["argv"])
        self.assertIn("++algo.config.videogen.seed=42", observed["argv"])


if __name__ == "__main__":
    unittest.main()
