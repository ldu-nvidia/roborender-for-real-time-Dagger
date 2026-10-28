"""Synthetic CPU tests; these never qualify GPU capacity or send live commands."""
from test.deployment.source_paths import PIPELINE_ROOT
from copy import deepcopy
import fcntl
from pathlib import Path
import sys

import pytest

HERE = PIPELINE_ROOT / "deployment/rtx_actor"
sys.path.insert(0, str(HERE))
from test.deployment.rtx_actor.test_scaled_success_contract import accepted, config_for, partition_result
from pipeline.deployment import scaled_success_dispatch as dispatch
from pipeline.deployment.rtx_actor import scaled_success_contract as scaled


@pytest.fixture
def material(accepted, monkeypatch):
    # Native fixtures import their colocated contracts without package prefixes;
    # mirror only their synthetic panel pins into the package import under test.
    import production_success_contract as fixture_contract
    from pipeline.deployment.rtx_actor import production_success_contract as packaged_contract
    for key in ("PANEL_SHA256", "PANEL_RAW_SHA256", "LEGACY_PANEL_SHA256", "LEGACY_PANEL_RAW_SHA256"):
        monkeypatch.setattr(packaged_contract, key, getattr(fixture_contract, key))
    panel = scaled.build_panel(accepted, batch_size=4, qualification=True)
    config = config_for(accepted, panel)
    owners = [dict(worker_instance_id=str(i + 1) * 32, source_sha256="a" * 64,
        native_pid=1, physical_gpu=i, gpu_uuid=f"GPU-synthetic-{i}") for i in range(2)]
    return accepted, panel, config, owners


def plan_for(material, owner_count=1, arm="bc"):
    _, panel, config, owners = material
    return dispatch.build_plan(panel, config, owners[:owner_count], command_prefix="cpu-scaled-qualification", arm=arm)


def guard(owner, receipt):
    assert receipt["resource_guard"] == {"synthetic_cpu_test_only": True}


class Transport:
    def __init__(self, plan, root):
        self.plan, self.root = plan, root
        self.calls, self.rows, self.active = [], {}, {}
        self.lose_after = self.lose_before = False
        self.ready_changes = {}
        self.outage = False
        self.stage_hook = None

    def get_ready(self, owner):
        if self.outage:
            raise ConnectionError("synthetic transport unavailable")
        key = owner["worker_instance_id"]
        self.calls.append(("ready", key))
        return dict(schema="roborender.isaac-resident.v1", role="isaac_resident", ok=True,
            **{name: owner[name] for name in dispatch.OWNER_KEYS}, native_initializations=1,
            phase="RUNNING" if key in self.active else "IDLE", active_command_id=self.active.get(key),
            **self.ready_changes)

    def get_command(self, owner, cid):
        self.calls.append(("get", owner["worker_instance_id"], cid))
        return deepcopy(self.rows.get(cid))

    def stage_artifacts(self, owner, artifacts):
        self.calls.append(("stage", owner["worker_instance_id"]))
        assert len(artifacts) == 4
        for item in artifacts:
            assert dispatch.digest(item["value"]) == item["raw_sha256"]
        if self.stage_hook:
            self.stage_hook()

    def submit(self, owner, body):
        key, cid = owner["worker_instance_id"], body["command_id"]
        index = next(e["cohort_index"] for e in self.plan["commands"] if e["body"] == body)
        path = self.root / f"cohort-{index:03d}"
        assert dispatch.read(path / "intent.json")["entry"]["body"] == body
        assert list(path.glob("submit-attempt-*.json"))
        assert key not in self.active
        self.calls.append(("post", key, cid, deepcopy(body)))
        if self.lose_before:
            raise TimeoutError("not accepted")
        self.active[key] = cid
        self.rows[cid] = dict(status="RUNNING", payload_sha256=dispatch.digest(body))
        if self.lose_after:
            self.lose_after = False
            raise TimeoutError("accepted but response lost")
        return dict(schema="roborender.isaac-resident.v1", status="accepted", command_id=cid,
            worker_instance_id=key, payload_sha256=dispatch.digest(body))

    def finish(self, material, index):
        original, panel, config, _ = material
        entry = self.plan["commands"][index]
        owner = self.plan["owners"][entry["owner_index"]]
        cid = entry["body"]["command_id"]
        self.rows[cid] = dict(status="COMPLETE", payload_sha256=entry["payload_sha256"], receipt=dict(
            complete=True, source_sha256=owner["source_sha256"], native_pid=1, native_initializations=1,
            optimizer_steps=0, purpose="test", profile="explicit_evaluation",
            resource_guard=dict(synthetic_cpu_test_only=True),
            result=partition_result(original, panel, config, index, self.plan["arm"])))
        self.active.pop(owner["worker_instance_id"], None)


def setup(material, tmp_path, owners=1, **kwargs):
    plan = plan_for(material, owners)
    transport = Transport(plan, tmp_path)
    coordinator = dispatch.Coordinator(plan, tmp_path, transport, resource_validator=guard, **kwargs)
    return coordinator, transport


def posts(transport):
    return [row for row in transport.calls if row[0] == "post"]


@pytest.mark.parametrize('uncertain_owner',[0,1])
def test_opt_in_pair_reconciles_saved_attempt_before_any_fresh_peer(material,tmp_path,uncertain_owner):
    coordinator,transport=setup(material,tmp_path,owners=2,max_submit_attempts=1,stop_peer_on_uncertainty=True)
    entry=next(e for e in coordinator.plan['commands'] if e['owner_index']==uncertain_owner)
    owner=coordinator.plan['owners'][uncertain_owner]
    dispatch.immutable(coordinator._path(entry)/'submit-attempt-00.json',
        dict(plan_sha256=coordinator.plan_sha,body=entry['body'],owner=owner))
    # A real process restart reconstructs the exact saved plan/policy. GET is
    # absent and owner IDLE, but the persisted POST may have been accepted.
    resumed=dispatch.Coordinator(coordinator.plan,tmp_path,transport,resource_validator=guard,
        max_submit_attempts=1,stop_peer_on_uncertainty=True)
    result=resumed.tick()
    assert result['workers']==[dict(owner_index=uncertain_owner,cohort_index=entry['cohort_index'],
        status='SUBMISSION_LIMIT_REQUIRES_RECONCILIATION')]
    assert not posts(transport)
    assert not any(row[0]=='stage' for row in transport.calls)
    assert dispatch.read(tmp_path/'controller-policy.json')['stop_peer_on_uncertainty'] is True
    with pytest.raises(ValueError,match='immutable'):
        dispatch.Coordinator(coordinator.plan,tmp_path,transport,resource_validator=guard,max_submit_attempts=1)


def test_opt_in_pair_stops_at_lost_post_response_without_touching_peer(material,tmp_path):
    coordinator,transport=setup(material,tmp_path,owners=2,max_submit_attempts=1,stop_peer_on_uncertainty=True)
    transport.lose_before=True
    result=coordinator.tick()
    assert [row['status'] for row in result['workers']]==['TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER']
    assert len(posts(transport))==1 and posts(transport)[0][1]==coordinator.plan['owners'][0]['worker_instance_id']


def test_legacy_dispatch_policy_bytes_do_not_opt_in_uncertainty_change(material,tmp_path):
    setup(material,tmp_path,owners=2)
    assert dispatch.read(tmp_path/'controller-policy.json')==dict(max_submit_attempts=3,
        native_replay_allowed=False,compute_release_authorized=False)


@pytest.mark.parametrize("owners", [1, 2])
def test_get_before_post_intent_before_mutation_and_one_cohort_per_owner(material, tmp_path, owners):
    coordinator, transport = setup(material, tmp_path, owners)
    value = coordinator.tick()
    assert len(posts(transport)) == owners
    assert all(row["status"] == "SUBMITTED" for row in value["workers"])
    for post in posts(transport):
        prior = transport.calls[:transport.calls.index(post)]
        assert len([c for c in prior if c[:3] == ("get", post[1], post[2])]) == 2
    coordinator.tick()
    assert len(posts(transport)) == owners
    assert not list(tmp_path.glob("cohort-*/commit.json"))


def test_lost_acceptance_response_reconciles_same_owner_and_id_without_repost(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    transport.lose_after = True
    first = coordinator.tick()
    assert first["workers"][0]["status"] == "TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER"
    assert not (tmp_path / "cohort-000/acceptance.json").exists()
    resumed = dispatch.Coordinator(coordinator.plan, tmp_path, transport, resource_validator=guard)
    assert resumed.tick()["workers"][0]["status"] == "IN_FLIGHT"
    assert len(posts(transport)) == 1
    assert (tmp_path / "cohort-000/acceptance.json").exists()


def test_unaccepted_submission_retries_only_identical_body_and_is_bounded(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    transport.lose_before = True
    for _ in range(3):
        coordinator.tick()
    assert coordinator.tick()["workers"][0]["status"] == "SUBMISSION_LIMIT_REQUIRES_RECONCILIATION"
    assert len(posts(transport)) == 3
    assert all(post == posts(transport)[0] for post in posts(transport))
    assert len(list((tmp_path / "cohort-000").glob("submit-attempt-*.json"))) == 3


def test_stop_before_submission_makes_no_native_mutation(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    dispatch.immutable(tmp_path / "STOP", {"user_requested": True})
    value = coordinator.tick()
    assert value["stop_requested"] and value["workers"][0]["status"] == "NOT_STARTED_STOPPED"
    assert not posts(transport) and not any(c[0] == "stage" for c in transport.calls)


def test_stop_arriving_during_stage_is_checked_again_before_post(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    transport.stage_hook = lambda: dispatch.immutable(tmp_path / "STOP", {"user_requested": True})
    coordinator.tick()
    assert not posts(transport)


def test_stop_commits_accepted_cohort_but_never_starts_next_or_stops_owner(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    coordinator.tick()
    dispatch.immutable(tmp_path / "STOP", {"user_requested": True})
    assert coordinator.tick()["workers"][0]["status"] == "IN_FLIGHT"
    transport.finish(material, 0)
    value = coordinator.tick()
    assert value["committed_cohorts"] == 1 and not value["complete"]
    assert coordinator.tick()["workers"][0]["status"] == "NOT_STARTED_STOPPED"
    assert len(posts(transport)) == 1 and not transport.active


def test_stop_with_uncertain_post_never_retries_or_claims_drained(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    transport.lose_before = True
    coordinator.tick()
    dispatch.immutable(tmp_path / "STOP", {"user_requested": True})
    value = coordinator.tick()
    assert value["workers"][0]["status"] == "STOPPED_UNCERTAIN_ACCEPTANCE"
    assert not value["complete"] and len(posts(transport)) == 1


def test_complete_two_owner_qualification_requires_all_cohorts_and_is_separate_from_test(material, tmp_path):
    coordinator, transport = setup(material, tmp_path, 2)
    coordinator.tick()
    transport.finish(material, 1)
    first = coordinator.tick()
    assert first["committed_cohorts"] == 1 and not first["complete"]
    assert not (tmp_path / "result.json").exists()
    transport.finish(material, 0)
    final = coordinator.tick()
    assert final["complete"] and final["qualification_only"]
    result = dispatch.read(tmp_path / "result.json")
    assert result["completed_episodes"] == 8 and result["qualification_only"]
    assert not result["scaled_success_protocol"]["historical_outcomes_reused"]
    assert len(posts(transport)) == 2


@pytest.mark.parametrize("field,value", [("worker_instance_id", "c" * 32), ("source_sha256", "f" * 64),
    ("gpu_uuid", "GPU-other"), ("native_pid", 2), ("native_initializations", 2)])
def test_changed_owner_fails_before_post(material, tmp_path, field, value):
    coordinator, transport = setup(material, tmp_path)
    original = transport.get_ready
    transport.get_ready = lambda owner: dict(original(owner), **{field: value})
    with pytest.raises(ValueError, match="lifetime"):
        coordinator.tick()
    assert not posts(transport)


def test_previously_accepted_command_disappearance_is_not_replayed(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    coordinator.tick()
    transport.rows.clear(); transport.active.clear()
    with pytest.raises(ValueError, match="disappeared"):
        coordinator.tick()
    assert len(posts(transport)) == 1


def test_native_terminal_fault_is_classified_without_submitting_peer(material, tmp_path):
    coordinator, transport = setup(material, tmp_path, 2)
    entry = coordinator.plan["commands"][0]
    transport.rows[entry["body"]["command_id"]] = dict(status="CRASH_UNCERTAIN", payload_sha256=entry["payload_sha256"])
    assert coordinator.tick()["workers"][0]["status"] == "NATIVE_TERMINAL_FAULT"
    assert not posts(transport)


@pytest.mark.parametrize("fault", ["partial", "checkpoint", "reset", "resource", "wrong_cohort", "teacher"])
def test_full_cohort_semantic_and_resource_validation_precedes_commit(material, tmp_path, fault):
    coordinator, transport = setup(material, tmp_path)
    coordinator.tick(); transport.finish(material, 0)
    row = next(iter(transport.rows.values()))
    result = row["receipt"]["result"]
    if fault == "partial": result["partition_result"]["completed_episodes"] -= 1
    elif fault == "checkpoint": result["checkpoint_sha256"] = "f" * 64
    elif fault == "reset": result["partition_result"]["cohort_receipts"][0]["result"].pop("scaled_initial_state_attestation")
    elif fault == "resource": row["receipt"]["resource_guard"] = {"synthetic_cpu_test_only": False}
    elif fault == "teacher": result["partition_result"]["arm"] = "teacher"
    else:
        original, panel, config, _ = material
        row["receipt"]["result"] = partition_result(original, panel, config, 1)
    with pytest.raises((ValueError, AssertionError)):
        coordinator.tick()
    assert not (tmp_path / "cohort-000/commit.json").exists()


def test_plan_or_commit_tampering_does_not_get_adopted(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    coordinator.tick(); transport.finish(material, 0); coordinator.tick()
    altered = deepcopy(coordinator.plan)
    altered["commands"][0]["owner_index"] = 1
    with pytest.raises(ValueError):
        dispatch.Coordinator(altered, tmp_path, transport, resource_validator=guard)
    commit = tmp_path / "cohort-000/commit.json"
    value = dispatch.read(commit); value["plan_sha256"] = "0" * 64
    commit.chmod(0o644); commit.write_bytes(dispatch.canonical(value) + b"\n")
    with pytest.raises(ValueError, match="immutable"):
        coordinator.tick()


def test_exclusive_controller_lock_prevents_duplicate_dispatch(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    with (tmp_path / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            coordinator.tick()
    assert not posts(transport)


def test_cached_commit_still_rejects_raw_byte_changes(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    coordinator.tick(); transport.finish(material, 0); coordinator.tick()
    path = tmp_path / "cohort-000/commit.json"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw bytes"):
        coordinator.tick()


def test_production_test_plan_requires_qualification_and_never_contains_old_outcomes(material, monkeypatch):
    original, _, _, owners = material
    panel = scaled.build_panel(original, batch_size=16)
    config = config_for(original, panel)
    with pytest.raises(ValueError, match="qualification"):
        dispatch.build_plan(panel, config, owners, command_prefix="cpu-scaled-full295")
    with pytest.raises(ValueError, match="qualification"):
        dispatch.build_plan(panel, config, owners, command_prefix="cpu-scaled-full295", qualification_evidence_sha256="f" * 64)
    # This assertion tests staging/binding only. The actual semantic proof gate
    # above rejects a hash alone; qualification's own CPU suite owns its proof.
    from pipeline.evaluation import scaled_success_qualification as qualification
    proof = {"synthetic_transport_binding_test_only": True}
    verified = []
    monkeypatch.setattr(qualification, "validate_qualification", lambda value, **kwargs: verified.append((value, kwargs)))
    plan = dispatch.build_plan(panel, config, owners, command_prefix="cpu-scaled-full295", qualification_evidence=proof)
    assert len(plan["commands"]) == 19 and len(dispatch.canonical(plan)) < dispatch.LIMIT
    assert [e["owner_index"] for e in plan["commands"]] == [i % 2 for i in range(19)]
    assert not plan["historical_outcomes_reused"]
    assert verified[0][0] == proof
    assert len(plan["shared_artifacts"]) == 3
    for entry in plan["commands"]:
        assert entry["body"]["payload"]["qualification_sha256"] == dispatch.digest(proof)
        assert entry["artifacts"][-1]["value"]["qualification_evidence_sha256"] == dispatch.digest(proof)


def test_duplicate_gpu_owners_and_unbounded_retries_rejected(material, tmp_path):
    _, panel, config, owners = material
    with pytest.raises(ValueError, match="distinct owner"):
        dispatch.build_plan(panel, config, [owners[0], owners[0]], command_prefix="duplicate")
    with pytest.raises(ValueError, match="1..3"):
        setup(material, tmp_path, max_submit_attempts=4)


def test_transport_outage_is_unknown_not_command_absence(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    transport.outage = True
    value = coordinator.tick()
    assert value["workers"][0]["status"] == "TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER"
    assert not posts(transport)


def test_queued_transition_is_accepted_without_duplicate_post(material, tmp_path):
    coordinator, transport = setup(material, tmp_path)
    coordinator.tick()
    cid = coordinator.plan["commands"][0]["body"]["command_id"]
    transport.rows[cid]["status"] = "QUEUED"
    original = transport.get_ready
    transport.get_ready = lambda owner: dict(original(owner), phase="QUEUED")
    assert coordinator.tick()["workers"][0]["status"] == "IN_FLIGHT"
    assert len(posts(transport)) == 1


def test_qualification_probe_forbids_production_proof_claim(material):
    _, panel, config, owners = material
    with pytest.raises(ValueError, match="prior production"):
        dispatch.build_plan(panel, config, owners, command_prefix="cpu-wrong-purpose", qualification_evidence={})


def test_eight_renderer_shards_assign_physical_gpus_not_owner_list_order(material):
    from test.deployment.rtx_actor.test_rendered_endpoint_pool import pool_config
    from pipeline.deployment.rtx_actor.rendered_student_success_contract import make_sharded_config
    _, panel, config, owners = material
    sharded = make_sharded_config(config, panel, config['checkpoint_sha256'], pool_config(8)['endpoints'])
    plan = dispatch.build_plan(panel, sharded, list(reversed(owners)), command_prefix='cpu-eight-shards')
    assert [owner['physical_gpu'] for owner in plan['owners']] == [0, 1]
    assert [plan['owners'][row['owner_index']]['physical_gpu'] for row in plan['commands']] == [0, 1]
    with pytest.raises(ValueError, match='prescribed physical'):
        dispatch.build_plan(panel, sharded, owners[:1], command_prefix='cpu-missing-shard')


def test_actual_tail7_qualification_uses_only_gpu0_and_first_renderer_shard(material):
    from test.deployment.rtx_actor.test_rendered_endpoint_pool import pool_config
    from pipeline.deployment.rtx_actor.rendered_student_success_contract import make_sharded_config
    original, _, template, owners = material
    panel = scaled.build_panel(original, batch_size=16, qualification=True, qualification_variant='tail7')
    config = make_sharded_config(template, panel, template['checkpoint_sha256'], pool_config(8)['endpoints'])
    plan = dispatch.build_plan(panel, config, owners[:1], command_prefix='cpu-tail7-shard')
    assert panel['episode_count'] == 7 and len(plan['commands']) == 1
    assert plan['owners'][plan['commands'][0]['owner_index']]['physical_gpu'] == 0
    with pytest.raises(ValueError, match='prescribed physical'):
        dispatch.build_plan(panel, config, owners[1:], command_prefix='cpu-wrong-tail-shard')
