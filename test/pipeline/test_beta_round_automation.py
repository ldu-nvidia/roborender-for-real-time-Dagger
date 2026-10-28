"""Control-plane tests: no Isaac, GPU, or fabricated training metrics."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.dagger.beta_campaign import make_first_round_config, next_round_config, validate_actor_completion
from pipeline.dagger.beta_campaign import Campaign
from pipeline.dagger.beta_validation import outcome_payloads, import_comparison, evaluation_id
from pipeline.dagger.production_controller import Controller
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import validate_config, canonical, actor_ids, driver_config
from pipeline.dagger.rtx.code.src.dagger_rtx.qualification_driver import BetaRoundDriver
from pipeline.dagger.rtx.code.src.dagger_rtx.window_runner import RolloutLifecycle
from pipeline.evaluation.beta_small_batch_panel import make_panel, summarize


@pytest.fixture
def cfg(tmp_path):
    panel = tmp_path / "panel.json"
    panel.write_bytes(canonical(make_panel("d" * 64)))
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"test fixture, not a policy")
    return make_first_round_config("dagger-beta-unit-test", seed=610191, checkpoint=ckpt,
        checkpoint_sha=hashlib.sha256(ckpt.read_bytes()).hexdigest(), panel=panel,
        panel_sha=hashlib.sha256(panel.read_bytes()).hexdigest(), source_sha="b" * 64, gateway_pids=[101, 102, 103, 104])


def rows(config):
    parent = config["student"]["initial_checkpoint_sha256"]
    result = []
    for iteration in range(1, 51):
        candidate = hashlib.sha256(canonical([config["round_id"], iteration])).hexdigest()
        result.append({"round_id": config["round_id"], "iteration_id": iteration,
                       "status": "advanced" if iteration < 50 else "validating",
                       "input_checkpoint_sha256": parent, "candidate_checkpoint_sha256": candidate})
        parent = candidate
    return result


def completion(config, iterations):
    return {"actor_ids": actor_ids(config), "owned_actors_stopped": True,
        "actors": [{"physical_gpu": gpu, "run_id": config["run_id"], "round_id": config["round_id"],
                    "phase": "COMPLETE", "uncertain_reason": None, "lifecycle_initializations": 1,
                    "transition_batches": 4050, "physical_transition_rows": 24300,
                    "config_sha256": hashlib.sha256(canonical(config)).hexdigest(),
                    "final_candidate_checkpoint_sha256": iterations[-1]["candidate_checkpoint_sha256"],
                    "windows": [{"index": r["iteration_id"] - 1, "input_checkpoint_sha256": r["input_checkpoint_sha256"],
                                 "output_checkpoint_sha256": r["candidate_checkpoint_sha256"]} for r in iterations]}
                   for gpu in (1, 2)]}


@pytest.mark.parametrize("change", ["gpu0", "b28", "eight", "burst", "replay", "augmentation", "windows", "microbatch", "eval", "beta", "hash"])
def test_reject_contract_drift(cfg, change):
    c = copy.deepcopy(cfg)
    if change == "gpu0": c["rtx_actors"][0]["physical_gpu_index"] = 0
    if change == "b28": c["rtx_actors"][0]["concurrent_environments"] = 28
    if change == "eight": c["h200"]["render_student_worker_count"] = 8
    if change == "burst": c["h200"]["allow_burst"] = True
    if change == "replay": c["replay"]["enabled"] = True
    if change == "augmentation": c["raven_fidelity"]["raytraced_native_augmentation"] = False
    if change == "windows": c["execution"]["window_count"] = 40
    if change == "microbatch": c["student"]["mini_batches_per_iteration"] = 4
    if change == "eval": c["validation"]["completed_episodes"] = 2
    if change == "beta": c["teacher_beta"] = .5
    if change == "hash": c["source_sha256"] = "wrong"
    with pytest.raises((ValueError, RuntimeError)): validate_config(c)


def test_round_lineage_betas_new_seeds_and_logging(cfg):
    original_sha = hashlib.sha256(canonical(cfg)).hexdigest()
    names, ids, seeds = set(), set(), set()
    for rid, beta in enumerate((.75, .5, .25, 0.), 1):
        assert cfg["round_id"] == rid and cfg["teacher_beta"] == beta
        assert len(actor_ids(cfg)) == 12
        names.add(cfg["logging"]["wandb_run_name"])
        ids.add(cfg["logging"]["wandb_run_id"])
        seeds.add(cfg["seed"])
        if rid < 4:
            candidate = hashlib.sha256(str(rid).encode()).hexdigest()
            cfg = next_round_config(cfg, original_sha, candidate, "/checkpoints/final50.pt", [201, 202, 203, 204])
            assert cfg["student"]["initial_checkpoint_sha256"] == candidate
            assert cfg["existing_run_config_sha256"] == original_sha
    assert len(names) == len(ids) == len(seeds) == 4
    with pytest.raises(ValueError): next_round_config(cfg, original_sha, "f"*64, "/p", [1, 2, 3, 4])


def test_smoke_budgets_and_checkpoint_handoffs_do_not_relax_production(cfg, tmp_path):
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import SCHEMA, acknowledgement, SMOKE_ACK
    panel = tmp_path / "smoke-panel.json"
    panel.write_bytes(canonical(make_panel("d" * 64, smoke=True)))
    smoke = make_first_round_config("dagger-beta-smoke-unit-test", seed=610121,
        checkpoint=Path(cfg["student"]["initial_checkpoint"]), checkpoint_sha=cfg["student"]["initial_checkpoint_sha256"],
        panel=panel, panel_sha=hashlib.sha256(panel.read_bytes()).hexdigest(), source_sha="b"*64,
        gateway_pids=[101, 102, 103, 104], smoke=True)
    original_sha = hashlib.sha256(canonical(smoke)).hexdigest()
    for rid, beta in enumerate((.75, .5, .25, 0), 1):
        assert smoke["round_id"] == rid and smoke["teacher_beta"] == beta
        assert driver_config(smoke, original_sha)["windows"] == 3
        assert acknowledgement(smoke) == SMOKE_ACK
        iterations = rows(smoke)[:3]
        iterations[-1]["status"] = "validating"
        receipt = completion(smoke, iterations)
        for actor in receipt["actors"]:
            actor.update(transition_batches=243, physical_transition_rows=1458)
        candidate = validate_actor_completion(smoke, receipt, iterations)
        assert candidate == iterations[-1]["candidate_checkpoint_sha256"]
        from pipeline.dagger.beta_validation import load_panel
        assert load_panel(smoke)["episode_count"] == 10
        assert evaluation_id(smoke).endswith("i0003")
        bad = copy.deepcopy(smoke)
        bad["schema"] = SCHEMA
        with pytest.raises(ValueError): validate_config(bad)
        if rid < 4:
            smoke = next_round_config(smoke, original_sha, candidate, "/checkpoints/smoke-final.pt", [201, 202, 203, 204])


@pytest.mark.parametrize("bad", ["partial", "reset", "checkpoint", "actor", "round", "candidate", "hash"])
def test_incomplete_or_mismatched_actors_cannot_start_evaluation(cfg, bad):
    iterations = rows(cfg)
    receipt = completion(cfg, iterations)
    if bad == "partial": receipt["actors"][0]["transition_batches"] -= 1
    if bad == "reset": receipt["actors"][0]["lifecycle_initializations"] = 2
    if bad == "checkpoint": receipt["actors"][0]["windows"][10]["input_checkpoint_sha256"] = "f"*64
    if bad == "actor": receipt["actor_ids"].pop()
    if bad == "round": receipt["actors"][0]["round_id"] = 2
    if bad == "candidate": iterations[-1]["candidate_checkpoint_sha256"] = None
    if bad == "hash": receipt["actors"][0]["config_sha256"] = "f"*64
    with pytest.raises(RuntimeError): validate_actor_completion(cfg, receipt, iterations)


@pytest.mark.parametrize("mse_mode", [False, True])
def test_production_learner_is_full_b12_no_replay_no_checkpointing(cfg, mse_mode):
    controller = Controller.__new__(Controller)
    controller.config, controller.run_id, controller.round_id = cfg, cfg["run_id"], 1
    controller.actor_ids, controller.wandb_mode = actor_ids(cfg), "online"
    controller.learner = SimpleNamespace(post=Mock())
    controller.mse = SimpleNamespace(protocol={"max_iterations": 1, "interval_iterations": 1}) if mse_mode else None
    iteration = rows(cfg)[0]
    status = {"iterations": [iteration], "barriers": [
        {"round_id": 1, "iteration_id": 1, "actor_id": actor, "chunk_ids": [actor + "-window0"]}
        for actor in controller.actor_ids]}
    controller._submit_learning(iteration, status)
    payload = controller.learner.post.call_args.args[1]
    params = payload["parameters"]
    assert len(payload["train_chunk_ids"]) == 12
    assert params["microbatch_environments"] == 12 and params["num_mini_batches"] == 1
    assert params["raven_exact_full_batch"] and params["activation_checkpointing"] is False
    assert params["deep_validation_samples"] == 0 and "replay_manifest" not in params
    assert params["final_validation_iteration"] == params["validation_interval_iterations"] == (1 if mse_mode else 50)
    assert params.get("metric_protocol") == ("heldout_mse_final_success" if mse_mode else None)
    assert params["wandb_mode"] == "online"


@pytest.mark.parametrize("round_id", [1, 2, 3, 4])
def test_fifty_windows_share_one_native_lifecycle(cfg, tmp_path, round_id):
    while cfg["round_id"] < round_id:
        cfg = next_round_config(cfg, "e"*64, "f"*64, "/p", [1, 2, 3, 4])
    sha = hashlib.sha256(canonical(cfg)).hexdigest()
    normalized = driver_config(cfg, sha)
    all_rows = rows(cfg)
    posts, finished = [], []
    def request(method, path, body=None):
        if method == "POST":
            posts.append(body)
            index = (len(posts) - 1) // 6
            assert body["round_id"] == round_id and body["checkpoint_sha256"] == all_rows[index]["input_checkpoint_sha256"]
            if len(posts) % 6 == 0: finished.append(index)
            return {"ok": True, "expected": 12, "iteration_id": index + 1}
        count = len(finished)
        visible = copy.deepcopy(all_rows[:min(count + 1, 50)])
        if count < 50:
            visible[-1].update(status="collecting", candidate_checkpoint_sha256=None)
        parent = visible[-1]["input_checkpoint_sha256"]
        return {"ok": True, "run": {"config_sha256": normalized["coordinator_config_sha256"],
            "actor_ids": actor_ids(cfg), "current_round": round_id, "beta": cfg["teacher_beta"],
            "rollout_steps": 81, "validation_interval_iterations": 50,
            "status": visible[-1]["status"], "active_checkpoint_sha256": parent},
            "iterations": visible, "barriers": [
                {"round_id": round_id, "iteration_id": idx + 1, "actor_id": actor,
                 "checkpoint_sha256": all_rows[idx]["input_checkpoint_sha256"],
                 "chunk_ids": [f"d{round_id}-{actor}-window-{idx:06d}"]}
                for idx in finished for actor in actor_ids(cfg)]}
    begin, finish = Mock(), Mock()
    runner = SimpleNamespace(num_envs=6, actor_prefix=cfg["run_id"] + "-g1", phase_telemetry=None,
        lifecycle=RolloutLifecycle(begin, finish), run_window=Mock(return_value={"transition_batches": 81, "physical_transition_rows": 486}))
    driver = BetaRoundDriver(runner, request, tmp_path, cfg, sha, 1)
    result = driver.run()
    assert result["phase"] == "COMPLETE" and len(result["windows"]) == 50
    assert result["transition_batches"] == 4050 and result["physical_transition_rows"] == 24300
    begin.assert_called_once()
    finish.assert_called_once()
    assert runner.run_window.call_count == 50 and len(posts) == 300
    assert all(call.kwargs == {"manage_lifecycle": False} for call in runner.run_window.call_args_list)
    with pytest.raises(RuntimeError, match="restart forbidden"):
        BetaRoundDriver(runner, request, tmp_path, cfg, sha, 1)


def comparison(cfg):
    panel = json.loads(Path(cfg["validation"]["scenario_manifest"]).read_bytes())
    result = {"complete": True, "arms": {}}
    for arm, sha in (("bc", cfg["student"]["initial_checkpoint_sha256"]), ("dagger", "c"*64)):
        raw = [{"seed": cohort["seed"], "complete": True, "completed": 5, "optimizer_steps": 0,
            "policy_unchanged": True, "accounting_disagreements": 0,
            "outcomes": [{"lane": lane, "exported_success": lane == 0, "steps": 20, "timeout": True,
                          "episode_reward": float(lane + 1)} for lane in range(5)]} for cohort in panel["cohorts"]]
        result["arms"][arm] = {**summarize(panel, raw), "checkpoint_sha256": sha,
            "runtime_physics_sha256": "e"*64, "policy_unchanged": True,
            "goal_accounting_version": panel["goal_accounting_version"], "paired_initial_states": arm == "dagger"}
    return result


def test_validation_requires_exact_real_complete_outcomes_and_keeps_goal_at_timeout(cfg):
    comp = comparison(cfg)
    first = outcome_payloads(cfg, comp, cfg["student"]["initial_checkpoint_sha256"], "c"*64)
    assert len(first) == len({r["idempotency_key"] for r in first}) == 256
    assert first[0]["goal_reached"] and first[0]["termination"] == "timeout" and first[0]["reward"] == 1.
    assert first == outcome_payloads(cfg, comp, cfg["student"]["initial_checkpoint_sha256"], "c"*64)
    comp["arms"]["dagger"]["outcomes"][0].pop("reward")
    with pytest.raises(ValueError, match="reward is absent"):
        outcome_payloads(cfg, comp, cfg["student"]["initial_checkpoint_sha256"], "c"*64)


@pytest.mark.parametrize("bad", ["partial", "probe", "checkpoint", "steps", "reward"])
def test_bad_success_evidence_is_not_validation(cfg, bad):
    comp = comparison(cfg)
    if bad == "partial": comp["complete"] = False
    if bad == "probe": comp["probe_complete"] = True
    if bad == "checkpoint": comp["arms"]["dagger"]["checkpoint_sha256"] = "f"*64
    if bad == "steps": comp["arms"]["dagger"]["outcomes"][0]["steps"] = 0
    if bad == "reward": comp["arms"]["dagger"]["outcomes"][0]["reward"] = float("nan")
    with pytest.raises(ValueError): outcome_payloads(cfg, comp, cfg["student"]["initial_checkpoint_sha256"], "c"*64)


def test_unbound_template_cannot_launch(cfg):
    cfg["h200"]["gateway_pids"] = []
    validate_config(cfg)
    with pytest.raises(ValueError): validate_config(cfg, runtime=True)
    with pytest.raises(ValueError): driver_config(cfg, "a"*64)


@pytest.mark.parametrize("gpu", [1, 2])
def test_native_beta_plan_preserves_scope_and_science(cfg, tmp_path, gpu):
    import time
    from qualify_native_isaac6 import command
    (tmp_path / "port-manifest.json").write_text('{"files":[]}')
    frozen = tmp_path / "frozen.json"
    frozen.write_bytes(canonical(cfg))
    ssh, known = tmp_path / "ssh", tmp_path / "known"
    ssh.touch(mode=0o600)
    known.touch(mode=0o600)
    forwards = [{"local_port": 18610, "remote_port": 18610, "expected_ready": {"ok": True, "role": "coordinator", "pid": 99}}]
    forwards += [{"local_port": 18620+i, "remote_port": 18620+i,
                  "expected_ready": {"ok": True, "role": "render_student", "pid": pid}}
                 for i, pid in enumerate(cfg["h200"]["gateway_pids"])]
    supervisor = tmp_path / "supervisor.json"
    supervisor.write_bytes(canonical({"schema": "roborender.ssh-supervisor.v1", "run_id": cfg["run_id"],
        "ssh_config": str(ssh), "known_hosts": str(known), "ssh_host": "scoped-host", "lifetime_seconds": 24000, "forwards": forwards}))
    status = tmp_path / "status"
    status.mkdir()
    (status / "status.json").write_bytes(canonical({"schema": "roborender.ssh-supervisor.v1", "run_id": cfg["run_id"],
        "config_sha256": hashlib.sha256(supervisor.read_bytes()).hexdigest(), "state": "ready", "updated_at_ns": time.time_ns(),
        "forwards": [{**r, "healthy": True} for r in forwards]}))
    cmd = command("test-beta", tmp_path / "output", tmp_path, tmp_path / "code", 6,
        mode="beta-production", run_id=cfg["run_id"], token_file=tmp_path / "token", physical_gpu=gpu,
        seed=cfg["seed"]+gpu-1, gateway_pids=cfg["h200"]["gateway_pids"], memory_gib=21, compact_host_heap=True,
        qualification_config=frozen, supervisor_config=supervisor, supervisor_status_dir=status)
    assert cmd[cmd.index("--gpus")+1] == "device=" + cfg["rtx_actors"][gpu-1]["gpu_uuid"]
    assert cmd[cmd.index("--memory")+1] == cmd[cmd.index("--memory-swap")+1] == "21g"
    for flag in ("DAGGER_H200_RPC_WORKERS=4", "DAGGER_ROUND_ID=1", "DAGGER_WIRE_PROTOCOL=v3",
                 "algo.config.init_at_random_ep_len=true", "++algo.config.raven_exact_raytraced_augmentation=true",
                 "simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]",
                 "DAGGER_NATIVE_STEP_ACK=I_ACKNOWLEDGE_BETA_FIFTY_WINDOWS_AND_REAL_LEARNER_UPDATES"):
        assert flag in cmd


def test_existing_parent_registry_provenance_is_not_rewritten(cfg, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical(cfg))
    controller = Controller.__new__(Controller)
    controller.config, controller.config_path = cfg, config_path
    controller.run_id, controller.round_id, controller.actor_ids = cfg["run_id"], 1, actor_ids(cfg)
    digest = cfg["student"]["initial_checkpoint_sha256"]
    controller.registry = SimpleNamespace(get=Mock(return_value={"checkpoint": {"sha256": digest}}), post=Mock())
    controller.coordinator = SimpleNamespace(post=Mock(), get=Mock(return_value={"run": {
        "config_sha256": hashlib.sha256(canonical(cfg)).hexdigest(), "actor_ids": actor_ids(cfg),
        "rollout_steps": 81, "validation_interval_iterations": 50, "current_round": 1, "status": "collecting"}}))
    controller.ensure_registered()
    controller.registry.post.assert_not_called()


def test_run_artifact_parent_is_imported_before_registry_registration(cfg, tmp_path, monkeypatch):
    """Reorganized BC outputs cannot bypass the registry's path allowlist."""
    from pipeline.dagger import production_controller as module
    from pipeline.dagger.services.client import ServiceError
    from test.pipeline.test_dagger_h200_services import running_service
    allowed = tmp_path / "repository" / "checkpoints"
    allowed.mkdir(parents=True)
    monkeypatch.setattr(module, "PROJECT_ROOT", allowed.parent)
    monkeypatch.setenv("DAGGER_CHECKPOINT_SOURCE_ROOT", str(allowed))
    source = Path(cfg["student"]["initial_checkpoint"])
    digest = cfg["student"]["initial_checkpoint_sha256"]
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical(cfg))
    with running_service("registry", tmp_path / "ledger") as registry:
        with pytest.raises(ServiceError, match="checkpoint path must be below"):
            registry.post("/v1/checkpoints/register-local", {"path": str(source), "sha256": digest, "metadata": {}})
        controller = Controller.__new__(Controller)
        controller.config, controller.config_path = cfg, config_path
        controller.run_id, controller.round_id, controller.actor_ids = cfg["run_id"], 1, actor_ids(cfg)
        controller.registry = registry
        controller.coordinator = SimpleNamespace(post=Mock(), get=Mock(return_value={"run": {
            "config_sha256": hashlib.sha256(canonical(cfg)).hexdigest(), "actor_ids": actor_ids(cfg),
            "rollout_steps": 81, "validation_interval_iterations": 50,
            "current_round": 1, "status": "collecting"}}))
        controller.ensure_registered()
        record = registry.get("/v1/checkpoint-records/" + digest)["checkpoint"]
        assert record["metadata"] == {"role": "beta_round_parent", "run_id": cfg["run_id"], "round_id": 1}
        admitted = allowed / "admitted-round-parents" / (digest + ".pt")
        assert admitted.read_bytes() == source.read_bytes()
        controller.ensure_registered()
        assert registry.get("/v1/checkpoint-records/" + digest)["checkpoint"] == record


def test_real_http_outcome_import_complete_retry_and_conflict(cfg, tmp_path, monkeypatch):
    from test.pipeline.test_dagger_h200_services import running_service
    from pipeline.dagger.services.client import ServiceError
    monkeypatch.setenv("DAGGER_CHECKPOINT_SOURCE_ROOT", str(tmp_path))
    checkpoint = tmp_path / "native-candidate.pt"
    checkpoint.write_bytes(b"test checkpoint bytes; no model inference")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    comp = comparison(cfg)
    comp["arms"]["dagger"]["checkpoint_sha256"] = digest
    eid = evaluation_id(cfg)
    with running_service("registry", tmp_path / "ledger") as registry, running_service("evaluation", tmp_path / "ledger") as evaluation:
        registry.post("/v1/checkpoints/register-local", {"path": str(checkpoint), "sha256": digest, "metadata": {"purpose": "test"}})
        record = registry.get("/v1/checkpoint-records/" + digest)["checkpoint"]
        assert record == {"sha256": digest, "size_bytes": checkpoint.stat().st_size, "metadata": {"purpose": "test"}}
        evaluation.post("/v2/evaluations", {"evaluation_id": eid, "run_id": cfg["run_id"], "round_id": 1,
            "purpose": "test", "checkpoint_sha256": digest, "scenario_manifest_sha256": cfg["validation"]["scenario_manifest_sha256"],
            "scenario_ids": [row["case_id"] for row in comp["arms"]["dagger"]["outcomes"]],
            "observation_mode": "raytraced", "metric": "raven_goal_reached_rate"})
        controller = SimpleNamespace(config=cfg, run_id=cfg["run_id"], round_id=1, evaluation=evaluation)
        baseline = cfg["student"]["initial_checkpoint_sha256"]
        report = import_comparison(controller, comp, baseline, digest)
        assert report["completed"] == 256 and report["successes"] == 52 and report["goal_reached_rate"] == 52/256
        assert import_comparison(controller, comp, baseline, digest)["completed"] == 256
        changed = outcome_payloads(cfg, comp, baseline, digest)[0]
        changed["reward"] += 1
        with pytest.raises(ServiceError, match="409"):
            evaluation.post(f"/v2/evaluations/{eid}/outcomes", changed)


@pytest.mark.parametrize("continuation_release", [False, True])
def test_four_round_campaign_resume_release_eval_logging_before_advance(cfg, tmp_path, monkeypatch, continuation_release):
    controllers, events = {}, []
    baseline = cfg["student"]["initial_checkpoint_sha256"]
    class Control:
        def __init__(self, path, token, **kwargs):
            self.config = json.loads(path.read_bytes())
            self.round_id, self.run_id = self.config["round_id"], self.config["run_id"]
            self.rows = rows(self.config)
            checkpoint = tmp_path / f"final-{self.round_id}.pt"
            checkpoint.write_bytes(str(self.round_id).encode())
            candidate = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.rows[-1]["candidate_checkpoint_sha256"] = candidate
            self.result = {"output_checkpoint_path": str(checkpoint), "output_checkpoint_sha256": candidate,
                           "iteration_id": 50, "train_chunks": 12, "train_frames": 972, "replay_manifest_sha256": None,
                           "train_mse": .5}
            self.coordinator = SimpleNamespace(get=lambda p: {"iterations": self.rows}, post=self.advance)
            self.learner = SimpleNamespace(get=lambda p: {"job": {"status": "succeeded", "result": self.result}})
            controllers[self.round_id] = self
        def ensure_registered(self): pass
        def tick(self): return {"action": "wait", "iteration_id": 50, "status": "validating"}
        def advance(self, path, body):
            assert events[-1] == (self.round_id, "logged")
            events.append((self.round_id, "advance"))
            return {"ok": True, "completed_iteration_id": 50, "next_iteration_id": None, "stop_reason": "round_iteration_budget"}
    class Ops:
        def start(self, phase, context):
            self.request = context
            self.phase = phase
            return f"{context['round_id']}-{phase}"
        def poll(self, oid):
            context, phase = self.request, self.phase
            rid, config = context["round_id"], context["config"]
            events.append((rid, phase))
            if phase == "prepare":
                return {"source_sha256": config["source_sha256"], "resource_shape": "custom.gpu.4xh200",
                    "reservation": "cosmos-solution-architect-xcwtffiq", "allow_burst": False,
                    "genuine_backends": True, "owned_control_plane": True, "gateway_pids": [101, 102, 103, 104],
                    "service_state_identity": "c"*64, "token_file": str(tmp_path / "token")}
            if phase == "collect": return completion(config, controllers[rid].rows)
            if phase == "release": return {"owned_actors_stopped": True, "h200_allocation_released": True}
            if phase == "evaluate":
                assert (rid, "release") in events
                comp = comparison(config)
                comp["arms"]["bc"]["checkpoint_sha256"] = baseline
                comp["arms"]["dagger"]["checkpoint_sha256"] = context["final_checkpoint"]["sha256"]
                path = tmp_path / f"comparison-{rid}.json"
                path.write_bytes(canonical(comp))
                return {"comparison_path": str(path), "comparison_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "owned_actors_stopped": True}
            if phase == "control_only":
                return {"h200_gpu_count": 0, "exclusive_ledger_owner": True, "service_state_identity": "c"*64,
                        "token_file": str(tmp_path / "token")}
            raise AssertionError(phase)
    def imported(controller, comp, bc, candidate):
        assert events[-1] == (controller.round_id, "control_only")
        outcome_payloads(controller.config, comp, bc, candidate)
        events.append((controller.round_id, "imported"))
        return {"successes": 52, "goal_reached_rate": 52/256, "checkpoint_sha256": candidate}
    def logged(config, learner, report):
        assert events[-1] == (config["round_id"], "imported")
        events.append((config["round_id"], "logged"))
        return {"online_metrics_verified": True, "run_id": config["logging"]["wandb_run_id"]}
    monkeypatch.setattr("pipeline.dagger.beta_campaign.import_comparison", imported)
    ops = Ops()
    factory = lambda path, token, **kw: controllers.get(json.loads(path.read_bytes())["round_id"]) or Control(path, token, **kw)
    root = tmp_path / "campaign"
    campaign = Campaign(cfg, root, ops, controller_factory=factory, log_validation=logged)
    for index in range(80):
        if continuation_release and campaign.state["round_id"] == 1 and campaign.state["phase"] == "ADVANCE":
            # Software adoption is a round barrier, after evaluation/logging.
            admission = {"schema": "roborender.completed-round-recovery.v1", "run_id": cfg["run_id"],
                "continuation_source_sha256": "e" * 64, "native_restart_allowed": False,
                "completed_training_repeated": False}
            raw = canonical(admission)
            (root / "recovery-admission.json").write_bytes(raw)
            campaign.save(continuation_source_sha256="e" * 64,
                          recovery_admission_sha256=hashlib.sha256(raw).hexdigest())
        if index == 5:  # Resume durable control state, not native simulation.
            campaign = Campaign(cfg, root, ops, controller_factory=factory, log_validation=logged)
        campaign.tick()
        if campaign.state["phase"] == "COMPLETE": break
    assert campaign.state["phase"] == "COMPLETE"
    assert [r["teacher_beta"] for r in campaign.state["completed_rounds"]] == [.75, .5, .25, 0.]
    assert all(events.count((rid, phase)) == 1 for rid in range(1, 5)
               for phase in ("prepare", "collect", "release", "evaluate", "control_only", "imported", "logged", "advance"))
    assert len({r["wandb"]["run_id"] for r in campaign.state["completed_rounds"]}) == 4
    for rid in range(2, 5):
        assert controllers[rid].config["student"]["initial_checkpoint_sha256"] == controllers[rid-1].result["output_checkpoint_sha256"]
        assert controllers[rid].config["source_sha256"] == ("e" if continuation_release else "b") * 64
