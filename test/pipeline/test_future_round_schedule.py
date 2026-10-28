from __future__ import annotations
from test.deployment.source_paths import REPO_ROOT

import hashlib
import json
import time
from pathlib import Path

from pipeline.dagger.materialize_next_round import materialize
from pipeline.dagger.services.contracts import file_sha256
from pipeline.dagger.services.store import StateStore


ROOT = REPO_ROOT

def legacy_fixture_config():
    """CPU-only legacy materializer fixture; no ignored deployment dependency."""
    schedule = json.loads((ROOT / "pipeline/dagger/config/future_round_schedule.v1.json").read_text())
    panel = ROOT / "pipeline/dagger/config/fixed_validation_panel.v1.json"
    return dict(
        schema="roborender.cross_gpu_dagger_round.v1", run_id=schedule["coordinator_run_id"],
        round_id=1, teacher_beta=.75, student_action_probability=.25, seed=911223830,
        execution=dict(window_count=50, stop_after_actor_barriers=False,
            wait_for_checkpoint_advance=True, learner_submission_authorized=True, validation_authorized=True),
        rtx_actor=dict(physical_gpu_index=1, gpu_uuid="GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9",
            forbidden_gpu_indices=[0], concurrent_environments=28, rollout_steps_per_iteration=81,
            transitions_per_iteration=28*81),
        student=dict(initial_checkpoint="/cpu-fixture-only/unused-parent.pt", initial_checkpoint_sha256="c"*64),
        validation=dict(scenario_manifest=str(panel), scenario_manifest_sha256=file_sha256(panel),
            completed_episodes=256, interval_iterations=40, minimum_validation_epochs=10,
            maximum_validation_epochs=40, patience=4, minimum_absolute_improvement=.01,
            metric="raven_goal_reached_rate"), logging={})


def test_schedule_has_exact_teacher_probabilities_and_final_validations():
    schedule = json.loads(
        (ROOT / "pipeline/dagger/config/future_round_schedule.v1.json").read_text()
    )
    assert schedule["monitor_interval_seconds"] == 1200
    assert [row["round_id"] for row in schedule["rounds"]] == [2, 3, 4]
    assert [row["teacher_beta"] for row in schedule["rounds"]] == [0.5, 0.25, 0.0]
    assert [row["iterations"] for row in schedule["rounds"]] == [50, 50, 50]
    assert [row["validation_iterations"] for row in schedule["rounds"]] == [
        [50], [50], [50]
    ]
    assert [row["wandb_run_name"].split("-")[2] for row in schedule["rounds"]] == [
        "teacher50", "teacher25", "teacher0"
    ]


def test_materializer_binds_round2_to_final_candidate_not_best_promotion(tmp_path):
    base = tmp_path / "base.json"
    base_config = legacy_fixture_config()
    base_config["validation"]["scenario_manifest"] = str(
        ROOT / "pipeline/dagger/config/fixed_validation_panel.v1.json"
    )
    base.write_text(json.dumps(base_config))
    schedule = json.loads(
        (ROOT / "pipeline/dagger/config/future_round_schedule.v1.json").read_text()
    )
    schedule["coordinator_config_sha256"] = file_sha256(base)
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(schedule))

    state_root = tmp_path / "state"
    store = StateStore(state_root)
    promoted_bytes = b"best-validated-round-one-checkpoint"
    promoted_sha = hashlib.sha256(promoted_bytes).hexdigest()
    final_bytes = b"exact-round-one-iteration-fifty-checkpoint"
    final_sha = hashlib.sha256(final_bytes).hexdigest()
    checkpoint_path = state_root / "checkpoints" / f"{final_sha}.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(final_bytes)
    (state_root / "checkpoints" / f"{promoted_sha}.pt").write_bytes(promoted_bytes)
    now = time.time()
    with store.transaction() as database:
        database.execute(
            "INSERT INTO runs "
            "(run_id,actor_ids_json,rollout_steps,config_sha256,"
            "initial_checkpoint_sha256,active_checkpoint_sha256,current_round,beta,"
            "status,validation_interval_iterations,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                schedule["coordinator_run_id"],
                json.dumps([
                    f"{schedule['coordinator_run_id']}-g1-lane-{lane:03d}"
                    for lane in range(28)
                ]),
                81,
                schedule["coordinator_config_sha256"],
                json.loads(base.read_text())["student"]["initial_checkpoint_sha256"],
                promoted_sha,
                1,
                0.75,
                "round_complete",
                40,
                now,
                now,
            ),
        )
        database.execute(
            "INSERT INTO promotions VALUES (?,?,?,?)",
            (schedule["coordinator_run_id"], 1, promoted_sha, now),
        )
        database.execute(
            "INSERT INTO round_iterations VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                schedule["coordinator_run_id"], 1, 50,
                "a" * 64, final_sha, "final-evaluation", 0.2,
                "complete", now, now,
            ),
        )

    output = tmp_path / "round2.json"
    config = materialize(
        schedule_path=schedule_path,
        base_config_path=base,
        state_root=state_root,
        round_id=2,
        output=output,
    )
    assert config["run_id"] == schedule["coordinator_run_id"]
    assert config["existing_run_config_sha256"] == file_sha256(base)
    assert config["round_id"] == 2
    assert config["teacher_beta"] == 0.5
    assert config["seed"] == 911223831
    assert config["execution"]["window_count"] == 50
    assert config["validation"]["interval_iterations"] == 50
    assert config["student"]["initial_checkpoint_sha256"] == final_sha
    assert config["student"]["initial_checkpoint_sha256"] != promoted_sha
    assert config["rtx_actor"]["runtime_root"].endswith(
        "/dagger-r2-teacher50-seed911223831-b28-50iter"
    )
    assert config["logging"]["wandb_run_id"] == (
        "dagger-r2-teacher50-seed911223831-b28-50iter"
    )
    assert config["logging"]["wandb_run_name"].startswith("dagger-r2-teacher50")
