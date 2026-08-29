from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/round_dagger/run_actor_cohort_with_wandb.sh"


class ActorWandbLifecycleTest(unittest.TestCase):
    def test_orphaned_pair_publishes_atomic_numeric_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary).resolve()
            scripts = fixture / "scripts"
            round_scripts = scripts / "round_dagger"
            scaling_scripts = scripts / "scaling"
            round_scripts.mkdir(parents=True)
            scaling_scripts.mkdir()
            shutil.copy2(WRAPPER, round_scripts / WRAPPER.name)
            actor = round_scripts / "run_actor_cohort.sh"
            actor.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            actor.chmod(0o755)
            watcher = scaling_scripts / "wandb_run_watcher.py"
            watcher.write_text(
                "import argparse, pathlib, time\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--run-dir', required=True)\n"
                "p.add_argument('--ready-marker', required=True)\n"
                "p.add_argument('--finish-marker', required=True)\n"
                "p.add_argument('--project'); p.add_argument('--entity')\n"
                "p.add_argument('--expected-email'); p.add_argument('--authorization-scope')\n"
                "p.add_argument('--poll-interval-s')\n"
                "p.add_argument('--max-consecutive-error-scans')\n"
                "a=p.parse_args()\n"
                "ready=pathlib.Path(a.ready_marker); ready.write_text('{}\\n')\n"
                "finish=pathlib.Path(a.run_dir)/a.finish_marker\n"
                "while not finish.exists(): time.sleep(0.01)\n"
                "raise SystemExit(int(finish.read_text().strip()))\n",
                encoding="utf-8",
            )
            run_dir = fixture / "run"
            telemetry = run_dir / "telemetry"
            telemetry.mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (telemetry / "WANDB_PUBLIC_CONFIG.json").write_text(
                "{}\n", encoding="utf-8"
            )
            heartbeat = run_dir / "pair.heartbeat"
            heartbeat.write_text("alive\n", encoding="utf-8")
            old = time.time() - 10
            os.utime(heartbeat, (old, old))
            actor_done = run_dir / "rank1.done"
            finish = run_dir / "pair.finish"
            result = subprocess.run(
                ["bash", str(round_scripts / WRAPPER.name)],
                env={
                    **os.environ,
                    "ROUND_DAGGER_ACTOR_RUN_DIR": str(run_dir),
                    "TRAIN_RUN_DIR": str(run_dir),
                    "ROUND_DAGGER_ACTOR_RANK": "1",
                    "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
                    "ROUND_DAGGER_WANDB_ENABLED": "1",
                    "ROUND_DAGGER_WANDB_SIDECAR_OWNER": "1",
                    "ROUND_DAGGER_WANDB_PYTHON": sys.executable,
                    "ROUND_DAGGER_WANDB_PROJECT": "dagger",
                    "ROUND_DAGGER_WANDB_ENTITY": "nvidia",
                    "ROUND_DAGGER_WANDB_EXPECTED_EMAIL": "ldu@nvidia.com",
                    "ROUND_DAGGER_WANDB_READY_TIMEOUT_SECONDS": "5",
                    "ROUND_DAGGER_WANDB_POLL_SECONDS": "1",
                    "ROUND_DAGGER_WANDB_PAIR_LIFECYCLE": "1",
                    "ROUND_DAGGER_WANDB_PAIR_FINISH_MARKER": finish.name,
                    "ROUND_DAGGER_WANDB_ACTOR_DONE_MARKER": str(actor_done),
                    "ROUND_DAGGER_WANDB_PAIR_HEARTBEAT": str(heartbeat),
                    "ROUND_DAGGER_WANDB_PAIR_ORPHAN_TIMEOUT_SECONDS": "1",
                },
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(finish.read_text(encoding="ascii"), "2\n")
            self.assertFalse(finish.stat().st_mode & stat.S_IWUSR)
            self.assertEqual(actor_done.read_text(encoding="ascii"), "0\n")
            self.assertFalse(actor_done.stat().st_mode & stat.S_IWUSR)
            self.assertIn("supervisor lease expired", result.stderr)
            self.assertIn("lost its pair supervisor lease", result.stderr)


if __name__ == "__main__":
    unittest.main()
