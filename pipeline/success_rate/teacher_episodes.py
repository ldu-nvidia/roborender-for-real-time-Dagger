"""Collection-bound privileged-teacher evaluation with equal per-lane repeats.

This is a new protocol, not the legacy fixed295 or pooled-episode benchmark.
Native classes are imported only after Isaac startup. No student checkpoint,
renderer, optimizer, collection writer or truncated chunk budget is used.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "roborender.collection-bound-teacher-episodes.v3"
MODE_SCHEMA = "roborender.collection-bound-teacher-episodes.v4"


def require_full_initial_clocks(env):
    """Retain native reset clocks: episode0 and stage counters1 after observation.

    The separately versioned Raven pooled benchmark still randomizes startup
    clocks. Do not apply that startup-shortening rule to this per-lane test.
    Motion/state sampling and ordinary natural resets are unchanged.
    """
    names = ('episode_length_buf', 'time_in_stage_buf', 'actual_time_in_stage_buf')
    values = {name: getattr(env, name).tolist() for name in names}
    expected = dict(episode_length_buf=[0]*env.num_envs,
                    time_in_stage_buf=[1]*env.num_envs,
                    actual_time_in_stage_buf=[1]*env.num_envs)
    if values != expected:
        raise RuntimeError('full-episode evaluation requires native reset clocks (episode0, stages1): '+str(values))
    return dict(schema='roborender.full-episode-initial-clocks.v1',
                num_envs=env.num_envs, clocks=values, randomization_applied=False,
                physics_steps_added=0, state_sampling_modified=False)


def publish_status(path, value):
    """Public run telemetry must remain readable by the host-side observer.

    Container UID0 and the host operator differ; these counters contain no
    credentials. Set permissions on the temporary inode before atomic replace.
    """
    from dagger_rtx.durable import atomic_write_json
    atomic_write_json(Path(path), value, mode=0o644)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def wait_pair_start(output, spec, emit):
    """Optional host-owned barrier; no physics or RNG while a peer initializes."""
    import time
    expected = os.environ.get('ROBORENDER_TEACHER_PAIR_SHA256')
    if expected is None:
        return
    root = Path('/teacher-pair')
    raw = (root/'plan.json').read_bytes()
    rank = int(os.environ['ROBORENDER_TEACHER_PAIR_RANK'])
    plan = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != expected or rank not in (0, 1)
            or plan['seeds'][rank] != spec['seed']
            or plan['bc_request_sha256'] != spec['bc_request_sha256']):
        raise RuntimeError('paired barrier identity mismatch')
    emit('READY_PAIR', rank=rank, pair_sha256=expected)
    deadline = time.monotonic() + 1800
    while not (root/'start.json').exists():
        from pipeline.deployment.rtx_actor.bc_resource_observer import wait_for_observer
        wait_for_observer(output)
        if time.monotonic() > deadline:
            raise RuntimeError('paired startup barrier timed out; no evaluation started')
        time.sleep(.2)
    start = json.loads((root/'start.json').read_text())
    if start != dict(pair_sha256=expected, ranks=[0, 1]):
        raise RuntimeError('paired start receipt mismatch')
    publish_status(output/'evaluation-start.json', dict(time_ns=time.time_ns(),
        rank=rank, pair_sha256=expected))


def make_spec(request, request_sha256, *, seed=20261028, mode_based=False):
    from pipeline.deployment.rtx_actor.isaac6_bc_production import validate_collection_request
    request = validate_collection_request(request)
    motions, inventory = request["motions"], request["motion_inventory"]
    if (type(mode_based) is not bool or len(motions) != 32 or motions != inventory[:32]
            or request.get("global_env_offset", 0) != 0
            or type(seed) is not int or seed < 0 or seed == request["process_seed"]
            or len(request_sha256) != 64
            or any(c not in "0123456789abcdef" for c in request_sha256)):
        raise ValueError("exact BC first32 assignment and a fresh evaluation seed required")
    return dict(schema=MODE_SCHEMA if mode_based else SCHEMA, bc_request_sha256=request_sha256,
                bc_run_id=request["run_id"], motions=deepcopy(motions),
                motion_inventory=deepcopy(inventory), seed=seed, num_envs=32,
                episodes_per_lane=3 if mode_based else 2,
                expected_episodes=96 if mode_based else 64,
                max_vector_steps=3000 if mode_based else 2000,
                policy="privileged_ppo_teacher_plus_homie", renderer_calls=0,
                student_forward_calls=0, optimizer_steps=0,
                initial_clocks="native_reset_episode0_stages1_full_episode_budget",
                stop=("first_three_natural_completions_per_lane_then_physics_idle" if mode_based else
                      "first_two_natural_completions_per_lane_then_physics_idle"),
                extra_fast_lane_episodes="unscored_not_reset_or_paused_at_quota",
                native_replay_allowed=False, compute_release_authorized=False)


def load_spec():
    raw = Path("/run/teacher-episodes.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ["ROBORENDER_TEACHER_EPISODES_SHA256"]:
        raise RuntimeError("teacher evaluation request bytes changed")
    request_raw = Path("/run/bc-source-request.json").read_bytes()
    value = json.loads(raw)
    if value != make_spec(json.loads(request_raw), hashlib.sha256(request_raw).hexdigest(),
                          seed=value["seed"], mode_based=value['schema'] == MODE_SCHEMA):
        raise RuntimeError("teacher evaluation differs from its actual BC source request")
    return value


class TeacherModeBackend:
    """One hash-bound teacher test admission plus warm IDLE commands.

    This backend cannot claim BC/student/DAgger transition readiness. A second
    evaluation needs a new admission; a new command ID is not replay authority.
    The existing owner supplies authentication, serialization and durable IDs.
    """
    schema, mode, route, arm = MODE_SCHEMA, 'TEACHER_TEST', 'evaluate_collection_teacher', 'teacher'

    def __init__(self, spec, *, evaluate, pause, paused, kit_identity, revalidate):
        if spec['schema'] != self.schema:
            raise ValueError('mode runtime requires its versioned three-episode admission')
        self.spec = deepcopy(spec)
        self.evaluate, self.pause, self.paused = evaluate, pause, paused
        self.kit_identity, self.revalidate = kit_identity, revalidate
        self.initial_kit = kit_identity()
        self.used = False

    def validate(self, operation, payload):
        from dagger_rtx.phase_modes import resolve
        if operation != 'run_phase':
            raise ValueError('teacher runtime accepts explicit run_phase commands only')
        mode, route, parameters = resolve(payload)
        if mode == 'IDLE':
            return
        if (mode != self.mode or route != self.route
                or parameters != dict(arm=self.arm, spec_sha256=digest(self.spec))
                or self.used):
            raise ValueError('teacher command differs from unused bound evaluation admission')

    def run(self, operation, payload, report):
        from dagger_rtx.phase_modes import resolve, validate_completion
        self.validate(operation, payload)
        mode, _, _ = resolve(payload)
        if self.revalidate() != self.spec or self.kit_identity() != self.initial_kit:
            raise RuntimeError('teacher admission or Kit identity changed')
        try:
            report('mode_preparing', mode=mode)
            if mode == self.mode:
                self.used = True  # Never replay a failed or partially accepted native run.
                result = self.evaluate()
                if (result.get('spec') != self.spec or result.get('spec_sha256') != digest(self.spec)
                        or result.get('completed') != self.spec['expected_episodes']
                        or result.get('completed_per_lane') != [self.spec['episodes_per_lane']]*32
                        or not self.valid_policy_result(result)):
                    raise RuntimeError('teacher completion differs from admitted full episode/policy evidence')
                receipt = dict(complete=result['complete'], purpose='test', optimizer_steps=result['optimizer_steps'],
                               result=result, spec_sha256=digest(self.spec))
            else:
                receipt = dict(complete=True, optimizer_steps=0)
            validate_completion(mode, receipt)
            self.pause()
            if not self.paused() or self.kit_identity() != self.initial_kit:
                raise RuntimeError('teacher phase did not reach same-Kit physics idle')
            report('mode_idle', mode=mode)
            return dict(receipt, mode=mode, physics_idle=True, kit_identity=self.initial_kit,
                        native_initializations=1, compute_released=False)
        except BaseException:
            try:
                self.pause()
            except Exception:
                pass
            raise

    def phase_capabilities(self):
        return dict(schema='roborender.teacher-phase-capabilities.v1', modes=['IDLE',self.mode],
                    evaluation_admission_unused=not self.used, all_phase_certified=False,
                    compute_release_supported=False)

    def valid_policy_result(self, result):
        return all(type(result.get(k)) is int and result[k] == 0
                   for k in ('optimizer_steps', 'student_forward_calls', 'renderer_calls'))


class PerLaneEpisodes:
    """First K natural completions of EVERY lane; no fast-lane substitution."""
    def __init__(self, count, repeats):
        if type(count) is not int or type(repeats) is not int or count < 1 or repeats < 1:
            raise ValueError("positive integer lane/repeat counts required")
        self.count, self.repeats = count, repeats
        self.counts = [0] * count
        self.goals = [False] * count
        self.starts = [0] * count
        self.outcomes = []
        self.last_step = -1

    @property
    def complete(self):
        return all(n == self.repeats for n in self.counts)

    def update(self, step, goals, dones, timeouts):
        if (self.complete or type(step) is not int or step != self.last_step + 1
                or any(len(v) != self.count for v in (goals, dones, timeouts))
                or any(type(x) is not bool for v in (goals, dones, timeouts) for x in v)
                or any(t and not d for t, d in zip(timeouts, dones))):
            raise ValueError("nonsequential/invalid native vector transition")
        self.last_step = step
        rows = []
        for lane in range(self.count):
            if self.counts[lane] == self.repeats:
                continue
            self.goals[lane] |= goals[lane]
            if dones[lane]:
                row = dict(lane=lane, episode_index=self.counts[lane],
                           terminal_vector_step=step, steps=step + 1 - self.starts[lane],
                           success=self.goals[lane], timeout=timeouts[lane])
                rows.append(row)
                self.outcomes.append(row)
                self.counts[lane] += 1
                self.goals[lane] = False
                self.starts[lane] = step + 1
        return rows

    def summary(self):
        successes = sum(row["success"] for row in self.outcomes)
        completed = len(self.outcomes)
        return dict(complete=self.complete, completed=completed,
                    expected=self.count * self.repeats, successes=successes,
                    failures=completed - successes,
                    success_rate=successes / completed if completed else None,
                    completed_per_lane=list(self.counts), outcomes=deepcopy(self.outcomes))


def _native_classes(spec_loader=None):
    load_spec = spec_loader or globals()['load_spec']
    import time
    import threading
    import torch
    from omegaconf import OmegaConf
    from gr00t.rl.simulator.isaacsim.isaacsim import IsaacSim as BaseSim
    from gr00t.rl.envs.loco_manip.walk_stand_place_grasp_turn_homie import WalkStandPlaceGraspTurnHomie
    from gr00t.rl.trl.trainer.distill_trainer_obj_pred_homie_api import TRLDistillTrainerObjPredHomieAPI
    from gr00t.rl.utils.teleop_reset_lib.motion_loader_wsppt import MotionLoaderWSPPT
    from gr00t.rl.data.tasks.walk_stand_place_grasp_turn_homie.scenario_cfg import isaacsim as scenario
    from dagger_rtx.terminal_goal_accounting import TerminalGoalAccountingMixin
    from pipeline.deployment.rtx_actor.native_motion_assignment import explicit_motion_files_scene_binding
    from pipeline.deployment.rtx_actor.initial_reset_contract import require_completed_initial_reset
    from pipeline.deployment.rtx_actor.native_artifact_io import publish_json
    from pipeline.success_rate.policy_actions import prepare_ppo_teacher, select_policy_actions, RAVEN_PPO

    class IsaacSim(BaseSim):
        def __init__(self, config, *args, **kwargs):
            spec = load_spec()
            with explicit_motion_files_scene_binding(config, spec["motions"], scenario,
                    MotionLoaderWSPPT, motion_inventory=spec["motion_inventory"]) as binding:
                super().__init__(config, *args, **kwargs)
            self.teacher_motion_binding = deepcopy(binding)

    class Environment(TerminalGoalAccountingMixin, WalkStandPlaceGraspTurnHomie):
        pass

    class TeacherEpisodeEvaluator(TRLDistillTrainerObjPredHomieAPI):
        def train(self):
            spec = load_spec()
            if spec['schema'] == MODE_SCHEMA:
                import omni.kit.app
                from dagger_rtx.resident_service import ResidentOwner, serve
                root = Path('/dagger-runtime')
                self.env.simulator.sim.pause()
                backend = TeacherModeBackend(spec, evaluate=self.evaluate,
                    pause=self.env.simulator.sim.pause,
                    paused=lambda: not self.env.simulator.sim.is_playing(),
                    kit_identity=lambda: id(omni.kit.app.get_app()), revalidate=load_spec)
                owner = ResidentOwner(root/'mode-owner', backend,
                    gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                    source_sha256=os.environ['ROBORENDER_TEACHER_MODE_SOURCE_SHA256'])
                publish_status(root/'mode-ready.json', owner.status())
                return serve(owner, port=18640+int(os.environ['ROBORENDER_TEACHER_PAIR_RANK']),
                             token_file=root/'mode-token')
            self.evaluate()
            # Historical one-shot behavior remains explicit; new modes use serve().
            threading.Event().wait()

        def evaluate(self):
            output = Path("/dagger-runtime")
            spec = load_spec()
            original_reset = None
            hook = None
            original_optimizer_step, original_policy_step = self.optimizer.step, self.policy_step
            def emit(phase, **value):
                row = dict(phase=phase, pid=os.getpid(), time_ns=time.time_ns(), **value)
                publish_status(output / "status.json", row)
                print(json.dumps(row, allow_nan=False), flush=True)
            try:
                if (self.env.num_envs != 32 or not isinstance(self.env, Environment)
                        or self.args.num_ppo_epochs != 0 or self.state.global_step != 0):
                    raise RuntimeError("fresh frozen B32 teacher-only evaluator required")
                def forbidden(*args, **kwargs):
                    raise RuntimeError("student forward/optimization forbidden in teacher evaluation")
                self.optimizer.step = forbidden
                self.policy_step = forbidden
                hook = self.policy_model.register_forward_pre_hook(forbidden)
                self._eval_mode()
                self.ref_model.eval()
                self.ref_model.init_rollout()
                proof = prepare_ppo_teacher(self, original_ppo=True)
                publish_json(output / "teacher-parameters.json", proof)
                if list(self.env.reset_dataset.motion_files) != [m["path"] for m in spec["motion_inventory"]]:
                    raise RuntimeError("full reset inventory changed")
                resets = []
                original = self.env._reset_task_states
                original_reset = original
                def observe_reset(lanes, *args, **kwargs):
                    result = original(lanes, *args, **kwargs)
                    ranges = self.env.reset_dataset.sample_idx_based_on_demo_clip_number
                    for lane, sample, index in zip(lanes.tolist(), self.env.sample_idx.tolist(),
                                                   self.env.sample_idx_in_seq.tolist()):
                        if not int(ranges[lane][0]) <= sample <= int(ranges[lane][1]):
                            raise RuntimeError("reset left its bound BC motion condition")
                        resets.append(dict(lane=lane, sample_idx=sample, sample_idx_in_seq=index))
                    return result
                self.env._reset_task_states = observe_reset
                self.env.simulator.sim.play()
                obs = self.env.reset_all()
                reset_proof = require_completed_initial_reset(self.env)
                clocks = require_full_initial_clocks(self.env)
                publish_json(output / "initial-reset.json", dict(reset=reset_proof, clocks=clocks,
                    motion_binding=self.env.simulator.teacher_motion_binding, resets=deepcopy(resets)))
                OmegaConf.save(self.env.config, output / "environment-config.yaml", resolve=True)
                OmegaConf.save(self.config, output / "policy-config.yaml", resolve=True)
                wait_pair_start(output, spec, emit)
                account = PerLaneEpisodes(32, spec['episodes_per_lane'])
                dones = torch.zeros(32, device=self.accelerator.device)
                started = time.monotonic()
                with torch.no_grad(), (output / "transitions.jsonl").open("x") as stream:
                    for step in range(spec["max_vector_steps"]):
                        # External read-only supervisor may hold BEFORE a new step.
                        from pipeline.deployment.rtx_actor.bc_resource_observer import wait_for_observer
                        wait_for_observer(output)
                        obs = {key: value.to(self.accelerator.device) for key, value in obs.items()}
                        actions, _ = select_policy_actions(self, obs, dones, arm="teacher", mode=RAVEN_PPO)
                        obs, reward, dones, infos = self.env.step(actions)
                        goal = infos["grasp_goal_reached"]
                        if not torch.equal(goal, self.env._transition_goal_before_reset):
                            raise RuntimeError("pre-reset goal/export disagreement")
                        rows = account.update(step, goal.bool().tolist(), dones.bool().tolist(),
                                              infos["time_outs"].bool().tolist())
                        stream.write(json.dumps(dict(step=step, completed=rows, resets=resets)) + "\n")
                        resets.clear()
                        if rows or step % 20 == 0:
                            stream.flush()
                            os.fsync(stream.fileno())
                            emit("EVALUATING", vector_steps=step + 1, **account.summary())
                        if account.complete:
                            break
                if not account.complete:
                    raise RuntimeError("safety ceiling reached; incomplete episodes are not failures or successes")
                final_proof = prepare_ppo_teacher(self, original_ppo=True)
                if final_proof["strict_mean_parameters_and_rms_equality"] is not True:
                    raise RuntimeError("teacher changed")
                torch.cuda.synchronize(self.accelerator.device)
                self.env.simulator.sim.pause()
                if self.env.simulator.sim.is_playing():
                    raise RuntimeError('teacher evaluation failed to pause physics')
                result = dict(schema=spec['schema'], spec=spec, spec_sha256=digest(spec),
                    **account.summary(), vector_steps=step + 1, seconds=time.monotonic() - started,
                    teacher_parameter_attestation=final_proof, optimizer_steps=0,
                    student_forward_calls=0, renderer_calls=0, physics_idle=True)
                publish_json(output / "result.json", result)
                emit("COMPLETE_IDLE", **account.summary())
                return result
            except BaseException as exc:
                try:
                    self.env.simulator.sim.pause()
                except Exception:
                    pass
                emit("FAILED", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                if original_reset is not None:
                    self.env._reset_task_states = original_reset
                if hook is not None:
                    hook.remove()
                self.optimizer.step, self.policy_step = original_optimizer_step, original_policy_step

    return IsaacSim, Environment, TeacherEpisodeEvaluator


def __getattr__(name):
    if name not in {"IsaacSim", "Environment", "TeacherEpisodeEvaluator"}:
        raise AttributeError(name)
    classes = _native_classes()
    globals().update(zip(("IsaacSim", "Environment", "TeacherEpisodeEvaluator"), classes))
    return globals()[name]
