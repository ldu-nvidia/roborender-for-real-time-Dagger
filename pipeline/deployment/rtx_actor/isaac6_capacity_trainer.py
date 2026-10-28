"""Native-only adapter for the explicitly acknowledged Leo capacity probe."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import hashlib

# Same timing as the resident trainer: Kit exists, the scene/buffers do not.
# An inherited/default launch performs no prctl and retains its old semantics.
from native_thp_policy import configure_from_environment, attest as attest_native_thp, DISABLED
NATIVE_THP_CONFIGURATION = configure_from_environment()

from dagger_rtx.durable import atomic_write_json
from dagger_rtx.native_service_trainer import NativeDaggerRTXServiceTrainer
from leo_capacity_probe import run_teacher_capacity, run_teacher_camera_soak


class Isaac6TeacherCapacityTrainer(NativeDaggerRTXServiceTrainer):
    def train(self):
        if not os.environ.get("ROBORENDER_WARM_CAMERA_PLAN"):
            return super().train()
        from recorded_action_camera import ACK, MODE
        recorded = os.environ.get("ROBORENDER_LEO_PROBE_MODE") == MODE
        expected_ack = ACK if recorded else "I_ACKNOWLEDGE_324_NATIVE_CAMERA_SOAK_TEACHER_STEPS"
        if (os.environ.get("ROBORENDER_LEO_PROBE_MODE") not in {"teacher-camera-soak", MODE}
                or os.environ.get("DAGGER_RTX_MODE") != "capture-only"
                or os.environ.get("ROBORENDER_LEO_CAPACITY_STEPS") != "324"
                or os.environ.get("ROBORENDER_LEO_CAPACITY_ACK") != expected_ack
                or recorded != bool(os.environ.get("ROBORENDER_RECORDED_ACTION_REFERENCE"))):
            raise RuntimeError("warm scene comparison is isolated camera-soak only")
        from dagger_rtx.benchmark_barrier import wait_for_benchmark_peer
        from warm_camera_diagnostic import run_native
        wait_for_benchmark_peer("capture-only", os.environ["DAGGER_ACTOR_PREFIX"])
        output = Path(os.environ["DAGGER_RUNTIME_ROOT"]) / "capture-smoke.json"
        def report(row):
            atomic_write_json(output.parent / "capacity-progress.json", row, mode=0o644)
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        return run_native(self, output, report)

    def _capture_only(self, batch, runtime, output):
        thp = attest_native_thp(NATIVE_THP_CONFIGURATION)
        def report(row):
            # These are non-secret diagnostic receipts read by the host UID.
            # Container-root 0600 receipts made the host monitor abort after
            # successful stepping, before its post-exit ownership handoff.
            atomic_write_json(output.parent / "capacity-progress.json", row, mode=0o644)
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

        # Read existing native metadata, not motion payloads. Do not sort or
        # reseed: this probe records the actual native filesystem order.
        files = [str(path) for path in self.env.reset_dataset.motion_files]
        ranges = self.env.reset_dataset.sample_idx_based_on_demo_clip_number
        if not files or len(files) != len(ranges):
            raise RuntimeError("native motion file/usable clip mapping is not one-to-one")
        probe = (run_teacher_camera_soak if os.environ.get("ROBORENDER_LEO_PROBE_MODE")
                 == "teacher-camera-soak" else run_teacher_capacity)
        result = probe(
            batch, self._rollout_lifecycle(),
            steps=int(os.environ["ROBORENDER_LEO_CAPACITY_STEPS"]),
            acknowledgement=os.environ["ROBORENDER_LEO_CAPACITY_ACK"], report=report,
        )
        result["motion_mapping"] = {
            "inventory_size": len(files), "usable_clips": len(ranges),
            "ordered_paths_sha256": hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest(),
            "lane_paths": [files[lane % len(files)] for lane in range(batch.num_envs)],
            "native_rule": "local_env_id % usable_clip_count", "selection_modified": False,
        }
        result["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["cgroup_memory"] = {}
        for field in ("memory.current", "memory.peak", "memory.max", "memory.swap.current"):
            path = Path("/sys/fs/cgroup") / field
            if path.is_file():
                result["cgroup_memory"][field] = path.read_text().strip()
        if thp['policy'] == DISABLED:
            result['native_thp_policy'] = attest_native_thp(thp)
            swap_max = Path('/sys/fs/cgroup/memory.swap.max')
            result['cgroup_memory']['memory.swap.max'] = swap_max.read_text().strip()
            if (result['cgroup_memory'].get('memory.max') != str(21 * (1 << 30))
                    or result['cgroup_memory'].get('memory.swap.current') != '0'
                    or result['cgroup_memory']['memory.swap.max'] != '0'):
                raise RuntimeError('production capacity memory/swap policy changed')
        atomic_write_json(output, result, mode=0o644)
