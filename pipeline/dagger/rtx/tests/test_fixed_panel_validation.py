from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import threading

import pytest

from dagger_rtx.contracts import canonical_json
from dagger_rtx.fixed_panel_validation import (
    ACTION_POLICY,
    ASSIGNMENT_SCHEMA,
    COHORT_ENVIRONMENTS,
    EXPECTED_EPISODES,
    METRIC,
    OBSERVATION_MODE,
    AuthenticatedEvaluationClient,
    DurableOutcomeJournal,
    EpisodeLatch,
    EpisodeOutcome,
    FixedPanelAssignment,
    FixedPanelValidationError,
    EXPECTED_MOTION_GROUP_NAMES,
    EXPECTED_SCENARIO_MANIFEST_SHA256,
    attest_runtime_reset,
)


GROUPS = EXPECTED_MOTION_GROUP_NAMES
SEEDS = (1103, 2207, 3301, 4409)
CHECKPOINT = "a" * 64
PANEL = EXPECTED_SCENARIO_MANIFEST_SHA256
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def assignment_document() -> dict:
    base = []
    counts = (13, 13, 13, 13, 12)
    for group, count in zip(GROUPS, counts, strict=True):
        for reset in range(count):
            base.append((group, reset, f"{group}__reset_{reset:02d}"))
    episodes = []
    for seed in SEEDS:
        for group, reset, case_id in base:
            episodes.append(
                {
                    "scenario_id": f"seed{seed}.{case_id}",
                    "seed": seed,
                    "case_id": case_id,
                    "motion_group_id": group,
                    "reset_index": reset,
                }
            )
    return {
        "schema": ASSIGNMENT_SCHEMA,
        "evaluation_id": "eval-40",
        "run_id": "run-a",
        "round_id": 1,
        "checkpoint_sha256": CHECKPOINT,
        "scenario_manifest": "/sealed/panel.json",
        "scenario_manifest_sha256": PANEL,
        "observation_mode": OBSERVATION_MODE,
        "action_policy": ACTION_POLICY,
        "metric": METRIC,
        "episode_count": EXPECTED_EPISODES,
        "episodes": episodes,
    }


def write_assignment(tmp_path: Path, value: dict | None = None) -> tuple[Path, str]:
    path = tmp_path / "assignment.json"
    path.write_bytes(canonical_json(value or assignment_document()) + b"\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def load_assignment(tmp_path: Path, value: dict | None = None) -> FixedPanelAssignment:
    path, digest = write_assignment(tmp_path, value)
    return FixedPanelAssignment.load(
        path,
        expected_assignment_sha256=digest,
        expected_evaluation_id="eval-40",
        expected_run_id="run-a",
        expected_round_id=1,
        expected_checkpoint_sha256=CHECKPOINT,
        expected_scenario_manifest_sha256=PANEL,
    )


def test_assignment_builds_four_native_b64_cohorts(tmp_path):
    assignment = load_assignment(tmp_path)
    cohorts = assignment.cohorts()
    assert len(cohorts) == 4
    assert [(row.cohort_index, row.paired_seed, len(row.cases)) for row in cohorts] == [
        (0, 1103, 64),
        (1, 2207, 64),
        (2, 3301, 64),
        (3, 4409, 64),
    ]
    assert [case.motion_group_id for case in cohorts[0].cases[:10]] == list(GROUPS) * 2
    assert cohorts[0].cases[0].reset_index == 0
    assert cohorts[0].cases[5].reset_index == 1
    assert cohorts[0].cases[32].motion_group_id == GROUPS[2]
    assert cohorts[0].cases[32].reset_index == 6
    scenarios = [case.scenario_id for cohort in cohorts for case in cohort.cases]
    assert len(scenarios) == len(set(scenarios)) == 256


def test_runtime_reset_attestation_binds_every_case_to_native_lane(tmp_path):
    cohort = load_assignment(tmp_path).cohorts()[0]
    bounds = [[index * 100, (index + 1) * 100] for index in range(len(GROUPS))]
    sample_indices = [bounds[lane % len(GROUPS)][0] + lane // len(GROUPS) for lane in range(64)]
    result = attest_runtime_reset(
        cohort,
        motion_groups=GROUPS,
        motion_sample_bounds=bounds,
        sample_indices=sample_indices,
        sample_indices_in_sequence=[lane % 8 for lane in range(64)],
        samples_per_motion=8,
    )
    assert result["environment_count"] == COHORT_ENVIRONMENTS == 64
    assert len(result["lanes"]) == 64
    assert result["lanes"][32] == {
        "lane": 32,
        "scenario_id": cohort.cases[32].scenario_id,
        "case_id": cohort.cases[32].case_id,
        "paired_seed": 1103,
        "motion_group_id": GROUPS[2],
        "reset_ordinal": 6,
        "sample_interval": [200, 300],
        "actual_sample_idx": 206,
        "actual_sample_idx_in_seq": 0,
    }
    assert len(result["reset_identity_sha256"]) == 64

    invalid = sample_indices.copy()
    invalid[32] = 199
    with pytest.raises(FixedPanelValidationError, match="outside"):
        attest_runtime_reset(
            cohort,
            motion_groups=GROUPS,
            motion_sample_bounds=bounds,
            sample_indices=invalid,
            sample_indices_in_sequence=[0] * 64,
            samples_per_motion=8,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(episode_count=255),
        lambda value: value.update(observation_mode="generated"),
        lambda value: value["episodes"][0].update(reset_index=1),
        lambda value: value["episodes"][0].update(scenario_id="wrong"),
        lambda value: value["episodes"][0].update(seed=9999),
        lambda value: value["episodes"][0].update(motion_group_id="wrong.npz"),
        lambda value: value["episodes"].pop(),
    ],
)
def test_assignment_divergence_fails_closed(tmp_path, mutate):
    value = assignment_document()
    mutate(value)
    with pytest.raises((FixedPanelValidationError, ValueError)):
        load_assignment(tmp_path, value)


def test_episode_latch_or_latches_and_ignores_post_reset_episode(tmp_path):
    cohort = load_assignment(tmp_path).cohorts()[0]
    latch = EpisodeLatch(cohort)
    batch = len(cohort.cases)
    rewards = [1.0] * batch
    no = [False] * batch
    goal = no.copy()
    goal[0] = True
    assert latch.observe(rewards=rewards, dones=no, grasp_goal_reached=goal) == ()
    dones = no.copy()
    dones[0] = True
    complete = latch.observe(rewards=rewards, dones=dones, grasp_goal_reached=no)
    assert len(complete) == 1
    assert complete[0].case == cohort.cases[0]
    assert complete[0].goal_reached is True
    assert complete[0].reward == 2.0
    assert complete[0].steps == 2
    assert complete[0].termination == "success"

    # Raven auto-resets this lane while waiting for the other first episodes.
    # A later done/success must not create a second panel result.
    assert latch.observe(rewards=rewards, dones=dones, grasp_goal_reached=goal) == ()
    assert latch.completed == 1


def test_episode_latch_timeout_and_failure_are_distinct(tmp_path):
    cohort = load_assignment(tmp_path).cohorts()[0]
    latch = EpisodeLatch(cohort)
    batch = len(cohort.cases)
    dones = [False] * batch
    dones[0] = dones[1] = True
    timeouts = [False] * batch
    timeouts[0] = True
    outcomes = latch.observe(
        rewards=[0.0] * batch,
        dones=dones,
        grasp_goal_reached=[False] * batch,
        timeouts=timeouts,
    )
    assert [item.termination for item in outcomes] == ["timeout", "failure"]


class FakeEvaluationService:
    def __init__(self, assignment: FixedPanelAssignment):
        self.assignment = assignment
        self.requests: dict[str, dict] = {}
        self.calls = 0

    def post_outcome(self, evaluation_id, payload):
        self.calls += 1
        assert evaluation_id == self.assignment.evaluation_id
        key = payload["idempotency_key"]
        if key in self.requests and self.requests[key] != dict(payload):
            raise FixedPanelValidationError("conflict")
        self.requests[key] = dict(payload)
        return {
            "ok": True,
            "evaluation_id": evaluation_id,
            "scenario_id": payload["scenario_id"],
            "status": "complete" if len(self.requests) == 256 else "collecting",
            "completed": len(self.requests),
            "total": 256,
        }

    def get_evaluation(self, evaluation_id):
        successes = sum(int(row["goal_reached"]) for row in self.requests.values())
        completed = len(self.requests)
        return {
            "ok": True,
            "evaluation": {
                "evaluation_id": evaluation_id,
                "run_id": self.assignment.run_id,
                "round_id": self.assignment.round_id,
                "checkpoint_sha256": self.assignment.checkpoint_sha256,
                "scenario_manifest_sha256": self.assignment.scenario_manifest_sha256,
                "observation_mode": OBSERVATION_MODE,
                "purpose": "validation",
                "metric": METRIC,
                "status": "complete" if completed == 256 else "collecting",
                "completed": completed,
                "total": 256,
                "successes": successes,
                "goal_reached_rate": successes / completed if completed else None,
                "wilson_95_interval": [0.0, 1.0] if completed else None,
                "mean_reward": 1.0 if completed else None,
                "mean_steps": 2.0 if completed else None,
                "safety_violations": 0,
            },
        }


def outcome(case, success=False):
    return EpisodeOutcome(
        case=case,
        goal_reached=success,
        reward=1.0,
        steps=2,
        termination="success" if success else "failure",
    )


def test_durable_outcome_is_staged_before_post_and_retry_is_idempotent(tmp_path):
    assignment = load_assignment(tmp_path)
    service = FakeEvaluationService(assignment)
    journal = DurableOutcomeJournal(tmp_path / "ledger", assignment)
    case = assignment.episodes[0]

    journal.stage(outcome(case, True))
    assert service.calls == 0
    first = journal.post_staged(case, service)
    second = journal.submit(outcome(case, True), service)
    assert first == second
    assert service.calls == 1
    assert journal.acked_scenarios() == {case.scenario_id}

    with pytest.raises(FixedPanelValidationError, match="different outcome"):
        journal.stage(outcome(case, False))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response.pop("total"),
        lambda response: response.update(extra="unversioned"),
        lambda response: response.update(completed=0),
        lambda response: response.update(total=255),
        lambda response: response.update(status="complete", completed=1),
        lambda response: response.update(scenario_id="wrong-scenario"),
    ],
)
def test_outcome_acknowledgement_schema_fails_closed(tmp_path, mutate):
    assignment = load_assignment(tmp_path)
    case = assignment.episodes[0]

    class BadService(FakeEvaluationService):
        def post_outcome(self, evaluation_id, payload):
            response = dict(super().post_outcome(evaluation_id, payload))
            mutate(response)
            return response

    journal = DurableOutcomeJournal(tmp_path / "ledger", assignment)
    journal.stage(outcome(case, True))
    with pytest.raises(FixedPanelValidationError, match="acknowledgement"):
        journal.post_staged(case, BadService(assignment))


def test_h200_idempotent_outcome_acknowledgement_is_strictly_accepted(tmp_path):
    assignment = load_assignment(tmp_path)
    case = assignment.episodes[0]

    class IdempotentService(FakeEvaluationService):
        def post_outcome(self, evaluation_id, payload):
            return {
                "ok": True,
                "evaluation_id": evaluation_id,
                "scenario_id": payload["scenario_id"],
                "status": "collecting",
                "idempotent": True,
            }

    journal = DurableOutcomeJournal(tmp_path / "ledger", assignment)
    journal.stage(outcome(case, True))
    response = journal.post_staged(case, IdempotentService(assignment))
    assert response["idempotent"] is True

    class MalformedIdempotentService(FakeEvaluationService):
        def post_outcome(self, evaluation_id, payload):
            return {
                "ok": True,
                "evaluation_id": evaluation_id,
                "scenario_id": payload["scenario_id"],
                "status": "collecting",
                "idempotent": True,
                "extra": "unversioned",
            }

    other = assignment.episodes[1]
    journal.stage(outcome(other, False))
    with pytest.raises(FixedPanelValidationError, match="acknowledgement"):
        journal.post_staged(other, MalformedIdempotentService(assignment))


def test_exact_256_outcomes_finalize_local_and_remote_summary(tmp_path):
    assignment = load_assignment(tmp_path)
    service = FakeEvaluationService(assignment)
    journal = DurableOutcomeJournal(tmp_path / "ledger", assignment)
    for index, case in enumerate(assignment.episodes):
        journal.submit(outcome(case, index % 2 == 0), service)
    path = journal.write_summary(service)
    summary = json.loads(path.read_bytes())
    assert summary["completed_episodes"] == 256
    assert summary["successes"] == 128
    assert summary["goal_reached_rate"] == 0.5
    assert summary["observation_mode"] == "raytraced"
    assert summary["action_policy"] == ACTION_POLICY
    assert summary["views"] == ["head_color", "hand_left", "hand_right"]


def test_authenticated_client_sends_bearer_and_canonical_payload(tmp_path):
    token = tmp_path / "token"
    token.write_text("test-secret\n", encoding="utf-8")
    token.chmod(0o600)
    observed = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            observed["path"] = self.path
            observed["authorization"] = self.headers["Authorization"]
            observed["body"] = self.rfile.read(length)
            payload = json.loads(observed["body"])
            body = canonical_json(
                {
                    "ok": True,
                    "evaluation_id": "eval-40",
                    "scenario_id": payload["scenario_id"],
                    "status": "collecting",
                    "completed": 1,
                    "total": 256,
                }
            )
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = AuthenticatedEvaluationClient(
            f"http://127.0.0.1:{server.server_port}",
            token,
            _allow_test_port=True,
        )
        payload = {"scenario_id": "scenario-a", "idempotency_key": "request-a"}
        response = client.post_outcome("eval-40", payload)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert response["ok"] is True
    assert observed["path"] == "/v2/evaluations/eval-40/outcomes"
    assert observed["authorization"] == "Bearer test-secret"
    assert observed["body"] == canonical_json(payload)


def test_gpu1_only_launcher_is_uuid_bound_and_never_addresses_gpu0():
    launcher = (
        PROJECT_ROOT / "scripts/launch_fixed_panel_validation.sh"
    ).read_text(encoding="utf-8")
    assert "GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9" in launcher
    assert '--gpus "device=$GPU_UUID"' in launcher
    assert 'nvidia-smi -i "$GPU_UUID"' in launcher
    assert "nvidia-smi -i 0" not in launcher
    assert "--gpus device=0" not in launcher
    assert "+num_envs=64" in launcher
    assert "four cohorts" in launcher
    assert "eight sequential 32-environment" not in launcher
    assert "env.config.domain_rand.image_augmentation.enabled=false" in launcher
    assert "2efa9d39c6ace247d6d68b43452f591cd5b22fb2f5e58dcfd3b6c5d1b30df77e" in launcher
    assert "ceb976d2745bdaa99e51ba7141e441dc7a140638ff8e2b0efaf50efd20290054" in launcher


def test_launcher_uses_hydra_append_syntax_for_keys_absent_from_base_eval():
    launcher = (
        PROJECT_ROOT / "scripts/launch_fixed_panel_validation.sh"
    ).read_text(encoding="utf-8")
    for override in (
        "+trainer._target_=dagger_rtx.fixed_panel_raven.FixedPanelEvalTrainer",
        "+env._target_=dagger_rtx.fixed_panel_raven.FixedPanelWalkStandPlaceGraspTurnHomie",
        "+simulator._target_=dagger_rtx.fixed_panel_raven.NormalizedIsaacSim",
        "+algo.config.init_at_random_ep_len=false",
        "+algo.config.use_dagger=false",
        "+env.config.domain_rand.image_augmentation.enabled=false",
        "+env.config.reset_from_dataset.use_motion_file_dir=false",
        '"+env.config.reset_from_dataset.motion_files=$motion_paths"',
    ):
        assert override in launcher


def test_native_adapter_stays_on_stock_action_mean_and_metric_paths():
    source = (
        PROJECT_ROOT / "code/src/dagger_rtx/fixed_panel_raven.py"
    ).read_text(encoding="utf-8")
    assert "result = super().eval(save_dirpath=save_dirpath)" in source
    assert "def load_teacher_actor(self):" in source
    assert "fixed-panel validation forbids a privileged teacher model" in source
    assert 'infos["grasp_goal_reached"]' in source
    assert "journal.stage(outcome)" in source
    assert "journal.post_staged(cohort.cases[lane], client)" in source
    assert "_attest_native_reset" in source
    assert "sample_idx_in_seq" in source
    assert "def _get_sample_idx" not in source
    assert 'if "env_state_dict" in checkpoint:' in source
    assert "rejects checkpoint env_state_dict replay" in source
    assert "_require_action_contract(self, action_state)" in source
    assert "executed action must equal deterministic finite action_mean [64,26]" in source
    launcher = (
        PROJECT_ROOT / "scripts/launch_fixed_panel_validation.sh"
    ).read_text(encoding="utf-8")
    assert "algo.config.use_dagger=false" in launcher
