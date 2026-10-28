"""Bounded beta GPU1-only native qualification for the versioned port.

Capture-only is the default. Explicit cross-host modes permit one vector step
or one 81-step window, never automatic learner submission or production.
This does not qualify success-rate/distribution equivalence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import resolve_native_code

from materialize_isaac6 import materialize
from qualify_graphics import GPU_UUID, HOSTNAME, LABEL, cleanup_owned
from native_thp_policy import INHERITED, DISABLED, ENV as THP_ENV, validate_policy

IMAGE = "sha256:7aa2c5aaf53e55e749c9f63b6430b9554b4ca49928cf35f376a646669da7b509"
REPO = Path("/home/ldu/roborender")
RAVEN = Path("/home/ldu/BC_render/rtx_BC_transfer")
ANCHORS = Path("/home/ldu/roborender_assets/anchors/anchor_pool_2048")
BETA_BENCHMARK_GPUS = {1: GPU_UUID, 2: "GPU-0540f144-9826-ba8f-140d-af4f95a475ce"}


def command(name, output, overlay, code_root, num_envs, image_id=IMAGE, *,
            mode="capture-only", run_id=None, token_file=None, h200_pid=None,
            physical_gpu=1, seed=42, gateway_pids=None, gateway_base=18620,
            memory_gib=40, cpu_count=12, capacity_steps=10, benchmark_barrier=None,
            qualification_config=None, supervisor_config=None, supervisor_status_dir=None,
            compact_host_heap=False, deployment_host=HOSTNAME, warm_camera_plan=None,
            warm_camera_source_sha256=None, recorded_action_reference=None,
            native_thp_policy=INHERITED, storage_paths=None,
            teacher_episode_spec=None, teacher_bc_request=None, teacher_pair_dir=None,
            mixed_inputs=None, mixed_checkpoint=None):
    validate_policy(native_thp_policy)
    if native_thp_policy != INHERITED and (
            mode != "teacher-capacity" or deployment_host != "leo-rtx-6k"
            or memory_gib != 21 or capacity_steps != 10 or not compact_host_heap
            or warm_camera_plan is not None):
        raise ValueError("production memory policy requires compact Leo teacher-capacity B/10-step/21GiB probe")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("an immutable Docker image ID is required")
    if mode not in {"capture-only", "teacher-capacity", "teacher-camera-soak", "recorded-action-camera-soak", "teacher-episodes", "mixed-episodes", "bc-collect", "isolated-one-step", "single-window", "capacity-probe", "multi-window-check", "beta-production"}:
        raise ValueError("only bounded qualification modes are allowed")
    mixed = mode == 'mixed-episodes'
    if mixed != (mixed_inputs is not None and mixed_checkpoint is not None):
        raise ValueError('mixed mode requires both its input manifest and bootstrap checkpoint')
    teacher_episodes = mode in ('teacher-episodes', 'mixed-episodes')
    if teacher_episodes:
        from pipeline.success_rate.teacher_episodes import make_spec, MODE_SCHEMA
        if (deployment_host != "leo-rtx-6k" or physical_gpu not in ((0, 1) if teacher_pair_dir else (1,)) or num_envs != 32
                or memory_gib != 32 or teacher_episode_spec is None or teacher_bc_request is None
                or not compact_host_heap or native_thp_policy != INHERITED):
            raise ValueError("teacher episode smoke requires explicit Leo GPU1 B32/32GiB scope")
        bc_raw = Path(teacher_bc_request).read_bytes()
        episode_raw = Path(teacher_episode_spec).read_bytes()
        mode_based_teacher = mixed or json.loads(episode_raw).get('schema') == MODE_SCHEMA
        if mixed:
            from pipeline.success_rate.mixed_episodes import make_spec as mixed_spec, CHECKPOINT_SHA256
            from pipeline.deployment.rtx_actor.rendered_endpoint_pool import validate_pool_config, SHARDED_ROUTING
            input_raw = Path(mixed_inputs).read_bytes()
            pool = validate_pool_config(json.loads(input_raw)['renderer'])
            if pool['routing'] != SHARDED_ROUTING or hashlib.sha256(Path(mixed_checkpoint).read_bytes()).hexdigest() != CHECKPOINT_SHA256:
                raise ValueError('mixed smoke needs eight disjoint renderer slots and exact bootstrap')
            episode_spec = mixed_spec(json.loads(bc_raw), hashlib.sha256(bc_raw).hexdigest(), seed=seed,
                                      inputs_sha256=hashlib.sha256(input_raw).hexdigest())
        else:
            episode_spec = make_spec(json.loads(bc_raw), hashlib.sha256(bc_raw).hexdigest(), seed=seed,
                                     mode_based=mode_based_teacher)
        if json.loads(episode_raw) != episode_spec:
            raise ValueError("teacher episode specification differs from BC source request")
        if teacher_pair_dir is not None:
            pair_raw = (Path(teacher_pair_dir) / 'plan.json').read_bytes()
            pair = json.loads(pair_raw)
            if (pair.get('schema') != ('roborender.mixed32-pair.v1' if mixed else 'roborender.teacher32-pair.v2' if mode_based_teacher else 'roborender.teacher32-pair.v1')
                    or pair.get('seeds') != ([20261102, 20261103] if mixed else [20261031, 20261101] if mode_based_teacher else [20261029, 20261030])
                    or seed != pair['seeds'][physical_gpu]
                    or pair.get('bc_request_sha256') != hashlib.sha256(bc_raw).hexdigest()
                    or pair.get('expected_episodes') != (128 if mixed else 192 if mode_based_teacher else 128)):
                raise ValueError('invalid paired teacher admission')
            if mode_based_teacher and not re.fullmatch('[0-9a-f]{64}', pair.get('source_sha256','')):
                raise ValueError('teacher mode needs its pinned source identity')
        elif mode_based_teacher:
            raise ValueError('teacher mode requires its explicit paired admission')
    elif teacher_episode_spec is not None or teacher_bc_request is not None:
        raise ValueError("teacher episode inputs require their distinct mode")
    if teacher_pair_dir is not None and not teacher_episodes:
        raise ValueError('paired teacher barrier requires teacher episode mode')
    teacher_capacity = mode in {"teacher-capacity", "teacher-camera-soak", "recorded-action-camera-soak"}
    camera_soak = mode in {"teacher-camera-soak", "recorded-action-camera-soak"}
    recorded = mode == "recorded-action-camera-soak"
    if recorded != (recorded_action_reference is not None) or (recorded and warm_camera_plan is None):
        raise ValueError("recorded actions require their exclusive isolated warm-scene diagnostic")
    if camera_soak and (num_envs != 16 or capacity_steps != 324):
        raise ValueError("camera soak is bounded to B16 and 324 teacher steps")
    if warm_camera_plan is not None:
        if (not camera_soak or benchmark_barrier is None or physical_gpu not in (0, 1)
                or not re.fullmatch(r"[0-9a-f]{64}", warm_camera_source_sha256 or "")):
            raise ValueError("warm camera comparison needs paired isolated B16 soak and source identity")
        from warm_camera_diagnostic import read_plan
        warm_camera_sha = hashlib.sha256(Path(warm_camera_plan).read_bytes()).hexdigest()
        warm_plan = read_plan(warm_camera_plan, warm_camera_sha)
        if recorded:
            from recorded_action_camera import read_reference
            from warm_camera_diagnostic import canonical
            recorded_sha = hashlib.sha256(Path(recorded_action_reference).read_bytes()).hexdigest()
            read_reference(recorded_action_reference, recorded_sha,
                hashlib.sha256(canonical(warm_plan['arms']['0'][-1]['assignment'])).hexdigest())
        evaluation_source = Path(code_root).resolve().parents[3] / "pipeline/evaluation"
        for module in ("beta_small_batch_panel.py", "warm_panel_session.py"):
            if not (evaluation_source / module).is_file():
                raise ValueError("warm resident scene diagnostic lacks evaluation helper source")
    if teacher_capacity and deployment_host != "leo-rtx-6k":
        raise ValueError("teacher-capacity qualification is bound to Leo only")
    if deployment_host == HOSTNAME:
        if physical_gpu not in BETA_BENCHMARK_GPUS:
            raise ValueError("bounded benchmark supports beta GPU1/GPU2 only")
        gpu_uuid = BETA_BENCHMARK_GPUS[physical_gpu]
    elif deployment_host == "leo-rtx-6k":
        from leo_binding import checked_gpu
        gpu_uuid = checked_gpu(physical_gpu, "capture-only" if teacher_capacity or teacher_episodes else mode,
                               campaign=qualification_config is not None)
    else:
        raise ValueError("unknown explicitly bound RTX host")
    maximum_envs = 64 if teacher_capacity else (50 if mode == "bc-collect" else 40)
    if type(num_envs) is not int or not 1 <= num_envs <= maximum_envs:
        raise ValueError("batch exceeds the explicit native mode bound")
    if not 16 <= memory_gib <= 40 or not 4 <= cpu_count <= 12 or capacity_steps not in ({324} if camera_soak else {3, 10}):
        raise ValueError("invalid bounded resource limits")
    gateway_pids = tuple(gateway_pids or ([h200_pid] if h200_pid else []))
    if gateway_pids and (len(gateway_pids) not in {1, 2, 4} or len(set(gateway_pids)) != len(gateway_pids)
                         or any(not isinstance(pid, int) or pid <= 0 for pid in gateway_pids)):
        raise ValueError("invalid attested gateway PID set")
    connected = mode in {"isolated-one-step", "single-window", "capacity-probe", "multi-window-check", "beta-production"}
    run_pattern = r"dagger-beta-[a-z0-9-]{1,64}" if mode == "beta-production" else r"rtx-beta-migration-[a-z0-9-]+"
    if deployment_host == "leo-rtx-6k":
        run_pattern = r"dagger-leo-[a-z0-9-]{1,64}" if mode == "beta-production" else r"rtx-leo-migration-[a-z0-9-]+"
    if connected and (not re.fullmatch(run_pattern, run_id or "")
                      or token_file is None or not gateway_pids):
        raise ValueError("cross-host qualification requires a fresh run, token file and attested PID")
    container_root = "/workspace/rtx_BC_transfer"
    env = {
        "NVIDIA_DRIVER_CAPABILITIES": "compute,graphics,utility",
        "CUDA_VISIBLE_DEVICES": gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "HOME": "/dagger-runtime/home", "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1", "HYDRA_FULL_ERROR": "1",
        "DAGGER_RTX_MODE": "capture-only" if teacher_capacity else mode,
        "DAGGER_RUNTIME_ROOT": "/dagger-runtime",
        "DAGGER_ASSET_MANIFEST": "/isaac6-port/assets.manifest.json",
        "DAGGER_RAVEN_RUNTIME_ROOT": container_root,
        "PYTHONPATH": ":".join(["/isaac6-port", "/workspace/dagger_rtx_server/code/src",
                                  container_root + "/pipeline/GR00T-VisualSim2Real"]),
        "XDG_CACHE_HOME": "/isaac6-cache/xdg", "TORCH_HOME": "/isaac6-cache/torch",
        "WARP_CACHE_PATH": "/isaac6-cache/warp", "CUDA_CACHE_PATH": "/isaac6-cache/cuda",
        "MPLCONFIGDIR": "/isaac6-cache/matplotlib", "HF_HOME": "/isaac6-cache/hf",
        "ROBORENDER_ISAAC6_PORT_MANIFEST": "/isaac6-port/port-manifest.json",
    }
    if compact_host_heap:
        env.update({"ROBORENDER_RTX_COMPACT_HEAP": "1", "MALLOC_ARENA_MAX": "2",
                    "MALLOC_TRIM_THRESHOLD_": "131072"})
    if native_thp_policy == DISABLED:
        env[THP_ENV] = native_thp_policy
    if teacher_capacity:
        from recorded_action_camera import ACK as RECORDED_ACK
        env.update({"ROBORENDER_LEO_CAPACITY_STEPS": str(capacity_steps),
                    "ROBORENDER_LEO_PROBE_MODE": mode,
                    "ROBORENDER_LEO_CAPACITY_ACK": (
                        RECORDED_ACK if recorded else "I_ACKNOWLEDGE_324_NATIVE_CAMERA_SOAK_TEACHER_STEPS" if camera_soak
                        else f"I_ACKNOWLEDGE_{capacity_steps}_NATIVE_TEACHER_VECTOR_STEPS")})
        env["PYTHONPATH"] = "/roborender-source:/capacity-source:" + env["PYTHONPATH"]
    if warm_camera_plan is not None:
        # ResidentTrainingEnvironment inherits the shared panel-reset class;
        # its module imports these helpers even for a teacher-only scene.
        # Match the standing resident launcher's read-only source layout.
        env["PYTHONPATH"] = "/evaluation-source:" + env["PYTHONPATH"]
        env.update({"ROBORENDER_WARM_CAMERA_PLAN": "/run/warm-camera-plan.json",
                    "ROBORENDER_WARM_CAMERA_PLAN_SHA256": warm_camera_sha,
                    "ROBORENDER_WARM_CAMERA_SOURCE_SHA256": warm_camera_source_sha256,
                    "ROBORENDER_WARM_CAMERA_GPU": str(physical_gpu),
                    "ROBORENDER_WARM_CAMERA_BOOTSTRAP_SEED": str(seed)})
        if recorded:
            env.update(ROBORENDER_RECORDED_ACTION_REFERENCE="/run/recorded-action-reference.json",
                       ROBORENDER_RECORDED_ACTION_SHA256=recorded_sha)
    if benchmark_barrier is not None:
        if not re.fullmatch(run_pattern, run_id or ""):
            raise ValueError("benchmark peer gate requires a fresh scoped run")
        env.update({"DAGGER_BENCHMARK_BARRIER_DIR": "/benchmark-barrier",
                    "DAGGER_ACTOR_PREFIX": run_id + f"-g{physical_gpu}"})
    beta_config = None
    if mode == "beta-production":
        import sys
        sys.path.insert(0, str(Path(code_root) / "src"))
        from dagger_rtx.beta_round import validate_config, acknowledgement, gpu_uuids, deployment_host as config_host
        if qualification_config is None:
            raise ValueError("beta production requires its frozen round config")
        beta_config = validate_config(json.loads(Path(qualification_config).read_bytes()))
        if config_host(beta_config) != deployment_host or gpu_uuids(beta_config).get(physical_gpu) != gpu_uuid:
            raise ValueError("campaign schema does not bind this host/GPU")
        lane_rank = list(gpu_uuids(beta_config)).index(physical_gpu)
        if (num_envs != 6 or memory_gib != 21 or not compact_host_heap or cpu_count != 12
                or run_id != beta_config["run_id"] or seed != beta_config["seed"] + lane_rank
                or tuple(beta_config["h200"]["gateway_pids"]) != gateway_pids or gateway_base != 18620):
            raise ValueError("beta production command differs from its qualified topology/config")
    if mode in {"multi-window-check", "beta-production"}:
        if ((num_envs, len(gateway_pids)) not in {(4, 4), (6, 4), (8, 1)}
                or (len(gateway_pids) == 1 and physical_gpu != 1) or qualification_config is None):
            raise ValueError("multi-window runtime check requires its bounded B4/B6 pair or GPU1 B8 single topology")
        if supervisor_config is None or supervisor_status_dir is None:
            raise ValueError("multi-window runtime check requires owned supervised forwards")
        from tunnel_supervisor import read_config
        supervisor, supervisor_sha = read_config(supervisor_config)
        gateways = [(row["local_port"], row["remote_port"], row["expected_ready"]["pid"])
                    for row in supervisor["forwards"] if row["expected_ready"]["role"] == "render_student"]
        if (supervisor["run_id"] != run_id
                or gateways != [(18620 + i, 18620 + i, pid) for i, pid in enumerate(gateway_pids)]
                or not any(row["local_port"] == row["remote_port"] == 18610
                           and row["expected_ready"]["role"] == "coordinator"
                           for row in supervisor["forwards"])):
            raise ValueError("runtime check and supervisor service identities differ")
        supervisor_status_dir = Path(supervisor_status_dir)
        if not supervisor_status_dir.is_absolute() or not supervisor_status_dir.is_dir():
            raise ValueError("supervisor status directory must exist before launching Isaac")
        env.update({"DAGGER_QUALIFICATION_CONFIG": "/run/runtime-check.json",
                    "DAGGER_QUALIFICATION_CONFIG_SHA256": hashlib.sha256(Path(qualification_config).read_bytes()).hexdigest(),
                    "DAGGER_CONNECTION_STATUS": "/run/roborender-connection/status.json",
                    "DAGGER_CONNECTION_CONFIG_SHA256": supervisor_sha})
        if beta_config is not None:
            env.update({"DAGGER_BETA_CONFIG": "/run/runtime-check.json",
                        "DAGGER_BETA_CONFIG_SHA256": hashlib.sha256(Path(qualification_config).read_bytes()).hexdigest()})
        # Check health before paying Isaac's initialization cost. The container
        # reads the directory, not a bind-mounted file inode, so atomic updates
        # to status.json remain visible throughout transport recovery.
        import sys
        sys.path.insert(0, str(Path(code_root) / "src"))
        from dagger_rtx.connection_health import ConnectionHealthGate
        ConnectionHealthGate(supervisor_status_dir / "status.json", run_id=run_id,
                             config_sha256=supervisor_sha, gateway_pids=gateway_pids)()
    if connected:
        env.update({
            "DAGGER_RUN_ID": run_id, "DAGGER_ROUND_ID": "1",
            "DAGGER_ACTOR_PREFIX": run_id + f"-g{physical_gpu}", "DAGGER_WINDOW_INDEX": "0",
            "DAGGER_WIRE_PROTOCOL": "v3", "DAGGER_H200_RPC_WORKERS": str(len(gateway_pids)),
            "DAGGER_H200_TOKEN_FILE": "/run/roborender-h200.token",
            "DAGGER_H200_BASE_URL": f"http://127.0.0.1:{gateway_base}",
            "DAGGER_H200_GATEWAY_URLS": ",".join(f"http://127.0.0.1:{gateway_base+i}" for i in range(len(gateway_pids))),
            "DAGGER_EXPECTED_H200_PIDS": ",".join(map(str, gateway_pids)),
            "DAGGER_PHYSICAL_GPU": str(physical_gpu), "DAGGER_PHYSICAL_GPU_UUID": gpu_uuid,
            "DAGGER_LAUNCH_STARTED_NS": str(time.time_ns()),
            "DAGGER_NATIVE_STEP_ACK": (
                acknowledgement(beta_config) if beta_config is not None
                else "I_ACKNOWLEDGE_THREE_WINDOWS_AND_REAL_LEARNER_UPDATES" if mode == "multi-window-check"
                else "I_ACKNOWLEDGE_ONE_NATIVE_VECTOR_STEP" if mode == "isolated-one-step"
                else ("I_ACKNOWLEDGE_THREE_NATIVE_VECTOR_STEPS" if capacity_steps == 3 else "I_ACKNOWLEDGE_TEN_NATIVE_VECTOR_STEPS")
                if mode == "capacity-probe" else "I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW"
            ),
        })
        if beta_config is not None:
            env["DAGGER_ROUND_ID"] = str(beta_config["round_id"])
        if mode == "capacity-probe":
            env["DAGGER_CAPACITY_STEPS"] = str(capacity_steps)
        if benchmark_barrier is not None:
            env["DAGGER_BENCHMARK_BARRIER_DIR"] = "/benchmark-barrier"
    cmd = ["docker", "run", "--rm", "--name", name, "--label", f"{LABEL}={name}",
           # Outbound access is needed for the original NVIDIA material URLs.
           # Cross-host modes reach only the authenticated loopback SSH forward.
           # No published ports, SSH/API credentials, or automatic learner.
           "--gpus", f"device={gpu_uuid}", "--network", "host" if connected else "bridge", "--shm-size", "8g",
           "--memory", f"{memory_gib}g", "--memory-swap", f"{memory_gib}g", "--cpus", str(cpu_count), "--ulimit", "core=0",
           "--workdir", container_root + "/pipeline/GR00T-VisualSim2Real"]
    if native_thp_policy == DISABLED:
        # Match the production owner's creation-time systemd zero-swap pin.
        # Equal Docker memory/swap flags alone can drift on daemon-reload.
        cmd += ["--annotation", "org.systemd.property.MemorySwapMax=uint64 0"]
    if connected:
        # Replace the inherited interactive Kit-log probe, without rebuilding
        # or relabeling the pinned image. This probe is read-only and must
        # never cause a native process restart or an action replay.
        cmd += ["--health-cmd", "python -m dagger_rtx.container_health",
                "--health-interval", "30s", "--health-timeout", "5s",
                "--health-retries", "3", "--health-start-period", "600s"]
    cache_root = REPO / ("runtime/isaac6-cache" if physical_gpu == 1 else f"runtime/isaac6-cache-gpu{physical_gpu}")
    if deployment_host == "leo-rtx-6k":
        cache_root = REPO / f"runtime/isaac6-cache-leo-gpu{physical_gpu}"
    raven_root, anchors_root = RAVEN, ANCHORS
    if storage_paths is not None:
        if deployment_host != 'leo-rtx-6k' or set(storage_paths) != {'raven', 'anchors', 'cache'}:
            raise ValueError('explicit Leo runtime mount mapping required')
        for path in storage_paths.values():
            if not isinstance(path, Path) or not path.is_absolute() or '..' in path.parts:
                raise ValueError('literal absolute runtime mount path required')
        raven_root, anchors_root, cache_root = (storage_paths[key] for key in ('raven', 'anchors', 'cache'))
    mounts = [(raven_root, container_root, True), (code_root, "/workspace/dagger_rtx_server/code", True),
              (anchors_root, "/assets/anchor_pool_2048", True), (output, "/dagger-runtime", False),
              (overlay, "/isaac6-port", True),
              (cache_root, "/isaac6-cache", False)]
    if teacher_capacity or teacher_episodes:
        # Evaluation classes now live in the canonical success_rate package.
        # Mount the same pinned release; never resolve to a mutable host tree.
        mounts.append((Path(code_root).resolve().parents[3], "/roborender-source", True))
        mounts.append((Path(code_root).resolve().parents[3] / "pipeline/deployment/rtx_actor",
                       "/capacity-source", True))
    if warm_camera_plan is not None:
        mounts.append((warm_camera_plan, "/run/warm-camera-plan.json", True))
        mounts.append((evaluation_source, "/evaluation-source", True))
    if recorded:
        mounts.append((recorded_action_reference, "/run/recorded-action-reference.json", True))
    if benchmark_barrier is not None:
        mounts.append((benchmark_barrier, "/benchmark-barrier", False))
    if connected:
        mounts.append((token_file, "/run/roborender-h200.token", True))
    if mode in {"multi-window-check", "beta-production"}:
        mounts.append((qualification_config, "/run/runtime-check.json", True))
        mounts.append((supervisor_status_dir, "/run/roborender-connection", True))
    manifest = json.loads((overlay / "port-manifest.json").read_text())
    for row in manifest["files"]:
        mounts.append((overlay / row["path"], container_root + "/" + row["path"], True))
    for source, target, readonly in mounts:
        cmd += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    cmd += [image_id, "python", "-u", "-X", "faulthandler", "gr00t/rl/train_agent_trl.py",
            "+exp=loco_manip/wsdpt_image_dagger_3view", f"seed={seed}", f"num_envs={num_envs}",
            "trainer._target_=" + (
                "isaac6_capacity_trainer.Isaac6TeacherCapacityTrainer" if teacher_capacity
                else "dagger_rtx.native_service_trainer.NativeDaggerRTXServiceTrainer" if connected
                else "isaac6_capture_trainer.Isaac6CaptureTrainer"
            ),
            "algo.config.num_steps_per_env=81", "algo.config.num_mini_batches=1",
            "algo.config.num_learning_epochs=0", "algo.config.init_at_random_ep_len=true",
            "++algo.config.raven_exact_raytraced_augmentation=true",
            "algo.config.rollout_with_teacher_num_steps=0", "algo.config.enforce_teacher_rollout=false",
            "algo.config.ratio_teacher_rollout=0.0", "algo.trl.num_total_batches=1",
            "algo.trl.total_episodes=null", "checkpoint=null", "auto_load_latest=false",
            "teacher_actor_path=" + container_root + "/checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt",
            "algo.config.homie_walk_model_path=" + container_root + "/checkpoints/teachers/locomotion/g1_locomotion_walk_controller.pt",
            "algo.config.homie_stand_model_path=" + container_root + "/checkpoints/teachers/locomotion/g1_locomotion_stand_controller.pt",
            "algo.config.videogen.mode=image_dagger", "algo.config.videogen.raytraced_mix_prob=0.5",
            "++algo.config.videogen.seed=0", "++algo.config.videogen.reference_lag=256",
            "algo.config.videogen.anchor_pool_dir=/assets/anchor_pool_2048", "algo.config.videogen.anchor_index=0",
            "simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]",
            "policy_view_count=3", "++env.config.image_dagger_policy_view_count=3",
            "simulator.config.render_results=false", "headless=true", "use_wandb=false",
            "base_dir=/dagger-runtime/logs_rl", "hydra.run.dir=/dagger-runtime/hydra",
            "output_dir=/dagger-runtime/output"]
    if mode == "bc-collect":
        overrides = {
            "+exp": "loco_manip/wsdpt_videogen_collect",
            "trainer._target_": "isaac6_bc_trainer.Isaac6BCQualificationTrainer",
            "algo.trl.num_total_batches": "3",
            "algo.config.videogen.mode": "collect",
        }
        # Online anchor assignment settings do not apply to native BC capture.
        omit = {"algo.config.videogen.raytraced_mix_prob", "++algo.config.videogen.reference_lag",
                "++algo.config.videogen.seed", "algo.config.videogen.anchor_pool_dir",
                "algo.config.videogen.anchor_index", "simulator.config.cameras.policy_views",
                "policy_view_count", "++env.config.image_dagger_policy_view_count",
                "++algo.config.raven_exact_raytraced_augmentation"}
        # BC saves all three conditioning cameras from its original config;
        # online student-observation tiling is irrelevant to teacher collection.
        cmd = [key + "=" + overrides[key] if key in overrides else value
               for value in cmd if (key := value.split("=", 1)[0]) not in omit]
        cmd += ["algo.config.videogen.collect_dir=/dagger-runtime/bc_isaac",
                "algo.config.videogen.dump_dir=null",
                "algo.config.videogen.rollout_action_source=teacher"]
    if teacher_episodes:
        overrides = {
            "+exp": "loco_manip/wsdpt_videogen_collect",
            "trainer._target_": "pipeline.success_rate.teacher_episodes.TeacherEpisodeEvaluator",
            "algo.config.init_at_random_ep_len": "false",
            "simulator._target_": "pipeline.success_rate.teacher_episodes.IsaacSim",
            "env._target_": "pipeline.success_rate.teacher_episodes.Environment",
            "algo.config.videogen.mode": "collect",
            "env.config.reset_from_dataset.use_motion_file_dir": "false",
            "env.config.reset_from_dataset.motion_files": json.dumps(
                [m["path"] for m in episode_spec["motion_inventory"]], separators=(",", ":")),
        }
        omit = {"algo.config.videogen.raytraced_mix_prob", "++algo.config.videogen.reference_lag",
                "++algo.config.videogen.seed", "algo.config.videogen.anchor_pool_dir",
                "algo.config.videogen.anchor_index", "simulator.config.cameras.policy_views",
                "policy_view_count", "++env.config.image_dagger_policy_view_count",
                "++algo.config.raven_exact_raytraced_augmentation"}
        if mixed:
            # Keep 3-view geometry; rendering is exclusively external.
            overrides.update({'+exp': 'loco_manip/wsdpt_image_dagger_3view',
                'trainer._target_': 'pipeline.success_rate.mixed_episodes.MixedEpisodeEvaluator',
                'simulator._target_': 'pipeline.success_rate.mixed_episodes.IsaacSim',
                'env._target_': 'pipeline.success_rate.mixed_episodes.Environment',
                'algo.config.videogen.mode': 'image_dagger',
                '++domain_rand.image_augmentation.enabled': 'false'})
            omit -= {'simulator.config.cameras.policy_views', 'policy_view_count',
                     '++env.config.image_dagger_policy_view_count'}
        seen = set()
        updated = []
        for value in cmd:
            key = value.split("=", 1)[0]
            if key in omit:
                continue
            if key in overrides:
                value = key + "=" + overrides[key]
                seen.add(key)
            if value.startswith("PYTHONPATH="):
                value = value.replace("PYTHONPATH=", "PYTHONPATH=/roborender-source:/capacity-source:", 1)
            updated.append(value)
        cmd = updated + [k + "=" + v for k, v in overrides.items() if k not in seen]
        extra = ["--annotation", "org.systemd.property.MemorySwapMax=uint64 0",
                 "--no-healthcheck", "--mount", f"type=bind,src={teacher_episode_spec},dst=/run/teacher-episodes.json,readonly",
                 "--mount", f"type=bind,src={teacher_bc_request},dst=/run/bc-source-request.json,readonly",
                 "--env", "ROBORENDER_TEACHER_EPISODES_SHA256=" + hashlib.sha256(episode_raw).hexdigest()]
        cmd[cmd.index(image_id):cmd.index(image_id)] = extra
        if mixed:
            cmd[cmd.index(image_id):cmd.index(image_id)] = ['--mount',
                f'type=bind,src={mixed_inputs},dst=/run/mixed-inputs.json,readonly', '--mount',
                f'type=bind,src={mixed_checkpoint},dst=/mixed-assets/bootstrap.pt,readonly']
        if teacher_pair_dir is not None:
            extra = ['--mount', f'type=bind,src={teacher_pair_dir},dst=/teacher-pair,readonly',
                     '--env', 'ROBORENDER_TEACHER_PAIR_SHA256=' + hashlib.sha256(pair_raw).hexdigest(),
                     '--env', 'ROBORENDER_TEACHER_PAIR_RANK=' + str(physical_gpu)]
            cmd[cmd.index(image_id):cmd.index(image_id)] = extra
            if mode_based_teacher:
                cmd[cmd.index('--network') + 1] = 'host'  # Authenticated loopback owner ports only.
                cmd[cmd.index(image_id):cmd.index(image_id)] = [
                    '--env', 'ROBORENDER_TEACHER_MODE_SOURCE_SHA256=' + pair['source_sha256']]
    return cmd


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("owned native qualification stop requested")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--num-envs", type=int, choices=[1, 2], default=2)
    parser.add_argument("--mode", choices=["capture-only", "bc-collect", "isolated-one-step", "single-window"], default="capture-only")
    parser.add_argument("--run-id")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--h200-pid", type=int)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if socket.gethostname() != HOSTNAME or not 60 <= args.timeout <= 1200:
        raise RuntimeError("wrong host or unbounded timeout")
    if args.mode in {"isolated-one-step", "single-window"}:
        if args.token_file is None or not args.token_file.is_file():
            raise RuntimeError("missing scoped H200 bearer token")
        if args.token_file.stat().st_mode & 0o077:
            raise RuntimeError("H200 bearer token must be owner-only")
    try:
        args.code_root = resolve_native_code(args.code_root)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError("qualification uses a separate runtime source snapshot") from error
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        raise RuntimeError("an immutable Docker image ID is required")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image_id]))[0]
    if image["Id"] != args.image_id or image["Config"]["Labels"].get("org.roborender.runtime") != "isaac6-lab3-migration-candidate":
        raise RuntimeError("candidate image identity mismatch")
    for _ in range(3):
        row = subprocess.check_output(["nvidia-smi", "-i", GPU_UUID,
            "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().split(",")
        index, gpu, memory, util = [x.strip() for x in row]
        if index != "1" or gpu != GPU_UUID or int(memory) > 2048 or int(util) > 10:
            raise RuntimeError("GPU1 not idle/identity mismatch")
        time.sleep(1)
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(memory["MemAvailable"].split()[0]) < 46 * 1024 * 1024:
        raise RuntimeError("less than 46GiB host RAM available")
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    name = "roborender-isaac6-native-" + identity.lower()
    output = REPO / "runtime/isaac6-native-qualification" / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "home").mkdir()
    (REPO / "runtime/isaac6-cache").mkdir(parents=True, exist_ok=True)
    overlay = output / "overlay"
    derived = materialize(RAVEN, overlay, args.code_root.parent / "config/assets.manifest.json")
    cmd = command(name, output, overlay, args.code_root, args.num_envs, args.image_id,
                  mode=args.mode, run_id=args.run_id, token_file=args.token_file, h200_pid=args.h200_pid)
    (output / "plan.json").write_text(json.dumps({"command": cmd, "derived_runtime": derived,
        "scope": args.mode, "run_id": args.run_id, "automatic_learning": False}, indent=2) + "\n")
    print(json.dumps({"phase": "native_qualification_started", "mode": args.mode,
                      "output": str(output), "container": name}), flush=True)
    started = time.monotonic()
    exit_code = None
    try:
        with (output / "console.log").open("wb") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                exit_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                cleanup_owned(name)
                exit_code = process.wait(timeout=30)
    finally:
        cleanup_owned(name)
        # The app runs as container root, while the host operator is ldu.
        # Keep private evidence private but readable by its host owner. Only
        # the known qualification receipts are touched (no asset/cache walk).
        receipts = [name for name in ("capture-smoke.json", "capture-error-cameras.json", "native-service-result.json", "bc-collection-smoke.json")
                    if (output / name).is_file()]
        if receipts:
            subprocess.run([
                "docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/chown",
                "--mount", f"type=bind,src={output},dst=/evidence", args.image_id,
                f"{os.getuid()}:{os.getgid()}", *[f"/evidence/{name}" for name in receipts],
            ], check=True, timeout=30)
        result_path = output / ({"capture-only": "capture-smoke.json", "bc-collect": "bc-collection-smoke.json"}
                                .get(args.mode, "native-service-result.json"))
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        passed = exit_code == 0 and result.get("result") == "PASS"
        if args.mode == "capture-only":
            passed = passed and result.get("environment_step_calls") == 0
        elif args.mode == "bc-collect":
            passed = passed and result.get("vector_steps") == 243 and result.get("optimizer_steps") == 0
        else:
            transaction = result.get("transaction", {})
            passed = passed and result.get("mode") == args.mode and result.get("num_envs") == args.num_envs
            if args.mode == "single-window":
                passed = passed and transaction.get("transition_batches") == 81 and transaction.get("physical_transition_rows") == 81 * args.num_envs
            else:
                passed = passed and transaction.get("phase") == "COMPLETE" and len(transaction.get("remote_acknowledgements", [])) == args.num_envs
        report = {"passed": passed, "exit_code": exit_code, "output": str(output),
                  "mode": args.mode, "run_id": args.run_id,
                  "elapsed_seconds": time.monotonic() - started, "production_qualified": False}
        (output / "host-result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
