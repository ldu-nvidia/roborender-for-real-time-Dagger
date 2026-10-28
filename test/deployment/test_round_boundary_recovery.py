"""CPU-only recovery: completed science persists; no native work is replayed."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.dagger.beta_campaign import Campaign, make_first_round_config, next_round_config
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_GPU_UUIDS, canonical
from pipeline.dagger.services.store import SCHEMA as DB_SCHEMA
from pipeline.deployment.beta_operations import CommandOperations, round_runtime_path
from pipeline.deployment.round_boundary_recovery import (
    digest, ledger_snapshot, publish_recovery, validate_binding, validate_boundary, validate_evidence,
)
from pipeline.evaluation.beta_small_batch_panel import make_panel


def insert(db, table, **values):
    keys = list(values)
    db.execute(f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", list(values.values()))


@pytest.fixture
def boundary(tmp_path,request):
    target=getattr(request,"param",3)
    root = tmp_path / "runtime/campaign"
    root.mkdir(parents=True)
    (root / "services").mkdir()
    panel = root / "panel.json"
    panel.write_bytes(canonical(make_panel("d"*64, smoke=True)))
    baseline = root / "baseline.pt"
    baseline.write_bytes(b"CPU-only fake checkpoint; never load as a policy")
    from pipeline.dagger.services.contracts import file_sha256
    first = make_first_round_config("dagger-leo-smoke-boundary-unit", seed=42,
        checkpoint=baseline, checkpoint_sha=file_sha256(baseline), panel=panel,
        panel_sha=file_sha256(panel), source_sha="b"*64, gateway_pids=[101,102,103,104],
        smoke=True, deployment_host="leo-rtx-6k")
    barrier = {"schema": "roborender.isaac-phase-barrier.v1", "workers": [
        {"physical_gpu": g, "gpu_uuid": LEO_GPU_UUIDS[g], "worker_instance_id": str(g+1)*32,
         "source_sha256": "b"*64, "phase": "IDLE"} for g in (0,1)]}
    db = sqlite3.connect(root / "services/state.sqlite3")
    db.executescript(DB_SCHEMA)
    config, completed, configs = first, [], []
    for rid in range(1,target):
        configs.append(deepcopy(config))
        directory = root / f"round-{rid}"
        directory.mkdir()
        (directory / "config.json").write_bytes(canonical(config))
        parent = config["student"]["initial_checkpoint_sha256"]
        for iteration in (1,2,3):
            path = root / f"candidate-{rid}-{iteration}.pt"
            path.write_bytes(f"CPU fixture {rid}/{iteration}".encode())
            candidate = file_sha256(path)
            insert(db, "round_iterations", run_id=first["run_id"], round_id=rid, iteration_id=iteration,
                input_checkpoint_sha256=parent, candidate_checkpoint_sha256=candidate,
                status="complete" if iteration==3 else "advanced", created_at=1, updated_at=1)
            result = {"iteration_id": iteration, "parent_checkpoint_sha256": parent,
                "output_checkpoint_sha256": candidate, "output_checkpoint_path": str(path),
                "train_chunks":12, "train_frames":972, "replay_manifest_sha256":None}
            insert(db, "learner_jobs", job_id=f"r{rid}-i{iteration}", run_id=first["run_id"], round_id=rid,
                input_checkpoint_sha256=parent, chunk_ids_json="[]", status="succeeded", request_json="{}",
                result_json=json.dumps(result), log_path="test", created_at=1, updated_at=1)
            parent = candidate
        eid = f"evaluation-{rid}"
        successes = 1 if rid==1 else 0
        insert(db, "success_evaluations", evaluation_id=eid, run_id=first["run_id"], round_id=rid,
            purpose="validation", checkpoint_sha256=candidate, scenario_manifest_sha256=config["validation"]["scenario_manifest_sha256"],
            scenario_ids_json="[]", observation_mode="raytraced", metric="raven_goal_reached_rate", status="complete",
            created_at=1, updated_at=1)
        for case in range(10):
            insert(db, "success_evaluation_outcomes", evaluation_id=eid, scenario_id=f"case-{case}",
                idempotency_key=f"{eid}-{case}", request_json="{}", goal_reached=int(case<successes), reward=0.,
                steps=1, termination="failure", safety_violation=0, created_at=1)
        comparison = directory / "success-comparison.json"
        comparison.write_bytes(canonical({"test_only": True, "successes": successes}))
        final = {"path": str(path), "sha256": candidate}
        completed.append({"round_id":rid,"teacher_beta":config["teacher_beta"], "final_checkpoint":final,
            "successes":successes,"completed_episodes":10,
            "wandb":{"online_metrics_verified":True,"run_id":config["logging"]["wandb_run_id"]},
            "evaluation":{"ok":True,"phase":"evaluate","round_id":rid,"run_id":first["run_id"],
                "native_phase_barrier":deepcopy(barrier),"owned_actors_stopped":False,
                "comparison_path":str(comparison),"comparison_sha256":file_sha256(comparison)}})
        config = next_round_config(config, digest(first), candidate, path, [101,102,103,104])
    insert(db,"runs",run_id=first["run_id"],actor_ids_json="[]",rollout_steps=81,config_sha256=digest(first),
        initial_checkpoint_sha256=first["student"]["initial_checkpoint_sha256"], active_checkpoint_sha256=candidate,
        current_round=target-1,beta=configs[-1]["teacher_beta"],status="round_complete",created_at=1,updated_at=1)
    db.commit()
    db.close()
    state = {"schema":"roborender.beta-campaign-state.v1","phase":"ABANDONED","run_id":first["run_id"],
        "round_id":target,"config":config,"completed_rounds":completed,"template_sha256":digest(first),
        "first_config_sha256":digest(first),"operation_id":f"r{target}-prepare-old","progress":None,
        "service_binding":{"stale":True},"final_checkpoint":final,"config_path":str(root/f"round-{target-1}/config.json"),
        "abort_receipt":{"ok":True,"phase":"abort","run_id":first["run_id"],"round_id":target,
            "owned_actors_stopped":True,"h200_allocation_released":True,"actor_process_reconciliation_warnings":[]}}
    failure={"phase":"PREPARE","native_restart_allowed":False,"error_type":"RuntimeError"}
    (root/"state.json").write_bytes(canonical(state))
    (root/"failure.json").write_bytes(canonical(failure))
    evidence={"run_id":first["run_id"],"round_id":target,"checked_at":1000.,"allocation_absent":True,
        "management_owners_stopped":True,"next_round_native_commands":0,"native_phase_barrier":barrier,
        "ledger":ledger_snapshot(root,state)}
    return SimpleNamespace(root=root, first=first, state=state, failure=failure, evidence=evidence, configs=configs)


def test_boundary_recovery_preserves_completed_work_and_parent(boundary, monkeypatch):
    b=boundary
    monkeypatch.setattr("pipeline.deployment.round_boundary_recovery.time.time",lambda:1001.)
    original=(b.root/"state.json").read_bytes()
    failure=(b.root/"failure.json").read_bytes()
    database=(b.root/"services/state.sqlite3").read_bytes()
    admission=publish_recovery(b.root,b.first,b.state,b.failure,b.evidence,{},"a"*32)
    state=json.loads((b.root/"state.json").read_bytes())
    assert state["phase"]=="PREPARE" and state["round_id"]==3 and state["operation_id"] is None
    assert state["config"]==b.state["config"] and state["completed_rounds"]==b.state["completed_rounds"]
    assert state["infrastructure_attempts"]=={"3":"a"*32}
    assert "service_binding" not in state and "final_checkpoint" not in state and "config_path" not in state
    assert json.loads((b.root/"recoveries"/("a"*32)/"prior-state.json").read_bytes())==json.loads(original)
    assert (b.root/"failure.json").read_bytes()==failure
    assert (b.root/"services/state.sqlite3").read_bytes()==database
    assert admission["native_restart_allowed"] is False and admission["completed_training_repeated"] is False
    campaign=Campaign(b.first,b.root,SimpleNamespace())
    assert campaign.state==state  # Reattaches control only, never calls a host operation.
    assert campaign.operation_context()["infrastructure_attempts"]=={"3":"a"*32}  # Includes abort/cleanup.
    assert publish_recovery(b.root,b.first,state,b.failure,b.evidence,{},"a"*32)==admission


@pytest.mark.parametrize("boundary",[2,4],indirect=True)
def test_other_completed_round_boundaries_are_supported(boundary,monkeypatch):
    b=boundary;monkeypatch.setattr("pipeline.deployment.round_boundary_recovery.time.time",lambda:1001.)
    publish_recovery(b.root,b.first,b.state,b.failure,b.evidence,{},"a"*32)
    restored=Campaign(b.first,b.root,SimpleNamespace())
    assert restored.state["phase"]=="PREPARE"
    assert restored.state["round_id"]==b.state["round_id"]
    assert restored.state["completed_rounds"]==b.state["completed_rounds"]


@pytest.mark.parametrize("bad",["mid_window","first_round","history","checkpoint","seed","beta","source",
    "wandb","evaluation","cleanup","warning","already_prepared","generation","active","allocation","native_command","stale"])
def test_unsafe_boundaries_fail_closed(boundary,bad):
    b=boundary;s=deepcopy(b.state);f=deepcopy(b.failure);e=deepcopy(b.evidence)
    if bad=="mid_window":f["phase"]="COLLECT_TRAIN"
    if bad=="first_round":s["round_id"]=1
    if bad=="history":s["completed_rounds"].pop(0)
    if bad=="checkpoint":Path(s["config"]["student"]["initial_checkpoint"]).write_bytes(b"corrupt")
    if bad=="seed":s["config"]["seed"]+=1
    if bad=="beta":s["config"]["teacher_beta"]=.5
    if bad=="source":s["config"]["source_sha256"]="e"*64
    if bad=="wandb":s["completed_rounds"][-1]["wandb"]["online_metrics_verified"]=False
    if bad=="evaluation":Path(s["completed_rounds"][-1]["evaluation"]["comparison_path"]).write_bytes(b"changed")
    if bad=="cleanup":s["abort_receipt"]["h200_allocation_released"]=False
    if bad=="warning":s["abort_receipt"]["actor_process_reconciliation_warnings"]=["RuntimeError"]
    if bad=="already_prepared":
        (b.root/"round-3").mkdir();(b.root/"round-3/config.json").write_text("{}")
    if bad=="generation":e["native_phase_barrier"]["workers"][0]["worker_instance_id"]="f"*32
    if bad=="active":e["native_phase_barrier"]["workers"][1]["phase"]="RUNNING"
    if bad=="allocation":e["allocation_absent"]=False
    if bad=="native_command":e["next_round_native_commands"]=1
    if bad=="stale":e["checked_at"]=1.
    with pytest.raises((RuntimeError,ValueError)):
        validate_evidence(b.root,b.first,s,f,e,now=1001.)
    assert json.loads((b.root/"state.json").read_bytes())==b.state


@pytest.mark.parametrize("bad",["iteration","job","evaluation","outcome","new_round"])
def test_ledger_must_be_closed_and_complete(boundary,bad):
    b=boundary
    with sqlite3.connect(b.root/"services/state.sqlite3") as db:
        if bad=="iteration":db.execute("UPDATE round_iterations SET status='collecting' WHERE round_id=2 AND iteration_id=3")
        if bad=="job":db.execute("UPDATE learner_jobs SET status='running' WHERE round_id=2")
        if bad=="evaluation":db.execute("UPDATE success_evaluations SET status='running' WHERE round_id=2")
        if bad=="outcome":db.execute("DELETE FROM success_evaluation_outcomes WHERE evaluation_id='evaluation-2' AND scenario_id='case-0'")
        if bad=="new_round":insert(db,"round_iterations",run_id=b.state["run_id"],round_id=3,iteration_id=1,
            input_checkpoint_sha256="a"*64,status="collecting",created_at=1,updated_at=1)
    with pytest.raises(RuntimeError):ledger_snapshot(b.root,b.state)


def test_admission_tampering_and_science_changes_rejected(boundary,monkeypatch):
    b=boundary;monkeypatch.setattr("pipeline.deployment.round_boundary_recovery.time.time",lambda:1001.)
    publish_recovery(b.root,b.first,b.state,b.failure,b.evidence,{},"a"*32)
    state=json.loads((b.root/"state.json").read_bytes())
    state["config"]["seed"]+=1
    with pytest.raises(RuntimeError,match="scientific"):validate_binding(b.root,state)
    state=json.loads((b.root/"state.json").read_bytes())
    state["completed_rounds"][0]["successes"]=10
    with pytest.raises(RuntimeError,match="history"):validate_binding(b.root,state)
    state=json.loads((b.root/"state.json").read_bytes())
    state["boundary_recovery"]["recovery_id"]="../../escape"
    with pytest.raises(RuntimeError,match="identity"):validate_binding(b.root,state)


def test_pre_switch_crash_retry_is_append_only(boundary,monkeypatch):
    b=boundary
    from pipeline.deployment import round_boundary_recovery as module
    monkeypatch.setattr(module.time,"time",lambda:1001.)
    real=module.atomic
    monkeypatch.setattr(module,"atomic",Mock(side_effect=OSError("test-only control write failure")))
    with pytest.raises(OSError):publish_recovery(b.root,b.first,b.state,b.failure,b.evidence,{},"a"*32)
    assert json.loads((b.root/"state.json").read_bytes())==b.state
    evidence=deepcopy(b.evidence);evidence["checked_at"]=1001.
    monkeypatch.setattr(module,"atomic",real)
    publish_recovery(b.root,b.first,b.state,b.failure,evidence,{},"a"*32)
    assert json.loads((b.root/"state.json").read_bytes())["phase"]=="PREPARE"


def test_resumed_scheduler_prepares_only_unstarted_round_and_pins_new_gateways(boundary,monkeypatch):
    b=boundary;monkeypatch.setattr("pipeline.deployment.round_boundary_recovery.time.time",lambda:1001.)
    publish_recovery(b.root,b.first,b.state,b.failure,b.evidence,{},"a"*32)
    calls=[]
    class Ops:
        def start(self,phase,context):
            calls.append((phase,context))
            assert phase=="prepare" and context["round_id"]==3
            assert context["infrastructure_attempts"]=={"3":"a"*32}
            return "new-prepare"
        def poll(self,oid):
            return {"source_sha256":"b"*64,"resource_shape":"custom.gpu.4xh200",
                "reservation":"cosmos-solution-architect-xcwtffiq","allow_burst":False,
                "genuine_backends":True,"owned_control_plane":True,"gateway_pids":[201,202,203,204],
                "service_state_identity":"c"*64,"token_file":str(b.root/"services.token")}
    no_controller=Mock(side_effect=AssertionError("no training during PREPARE"))
    campaign=Campaign(b.first,b.root,Ops(),controller_factory=no_controller)
    campaign.tick()
    assert campaign.state["phase"]=="REGISTER"
    assert campaign.state["config"]["h200"]["gateway_pids"]==[201,202,203,204]
    assert campaign.state["config"]["student"]==b.state["config"]["student"]
    assert len(calls)==1 and no_controller.call_count==0
    restored=Campaign(b.first,b.root,Ops(),controller_factory=no_controller)
    assert restored.state["completed_rounds"]==b.state["completed_rounds"]
    assert restored.state["phase"]=="REGISTER"


def test_allocation_attempt_is_attested_and_stale_owner_cannot_be_deleted():
    from test.deployment.test_beta_infra import fake_client
    from pipeline.deployment.h200.beta_pod import operate
    client=fake_client();run="dagger-leo-boundary-unit";attempt="a"*32
    operate(client,"create",run,3,["ssh-ed25519 AAAA cpu","ssh-ed25519 AAAA rtx"],infrastructure_attempt=attempt)
    deployment=client.deployment.create.call_args.args[0]
    env={e.name:e.value for e in deployment.spec.envs}
    assert env["ROBORENDER_INFRASTRUCTURE_ATTEMPT"]==attempt
    client.deployment.get.side_effect=None
    deployment.metadata.id_=deployment.metadata.name
    client.deployment.get.return_value=deployment
    assert operate(client,"status",run,3,infrastructure_attempt=attempt)["present"] is True
    with pytest.raises(RuntimeError,match="owner"):
        operate(client,"delete",run,3,infrastructure_attempt="b"*32)
    client.deployment.delete.assert_not_called()


def test_site_forwards_attempt_to_broker_without_changing_ledger_identity():
    from pipeline.deployment.beta_site import Site
    site=Site.__new__(Site)
    site.attempts={"3":"a"*32};site.site={"broker_python":"/usr/bin/python3"}
    site.run="dagger-leo-unit";site.rid=3;site.json=Mock(return_value={"present":False})
    site.allocation("inspect")
    argv=site.json.call_args.args[0]
    assert argv[argv.index("--infrastructure-attempt")+1]=="a"*32
    assert argv[argv.index("--run-id")+1]==site.run


def test_round_attempt_paths_do_not_reuse_release_boot_or_remote_metadata():
    attempts={"3":"a"*32}
    assert round_runtime_path("/run",2,attempts)==Path("/run/round-2")
    assert round_runtime_path("/run",3,attempts)==Path("/run/round-3/recovery-"+"a"*32)
    for bad in ({"3":"../escape"},{"5":"a"*32},{"3":None}):
        with pytest.raises(ValueError):round_runtime_path("/run",3,bad)


def test_new_attempt_has_new_idempotent_operation_identity(tmp_path):
    from pipeline.deployment import beta_operations as module
    from unittest.mock import patch
    entry=tmp_path/"site.py";entry.write_text("# CPU fixture only\n")
    commands={p:{"argv":["/usr/bin/python3",str(entry)],"entrypoint":str(entry),
        "entrypoint_sha256":digest_bytes(entry.read_bytes()),"timeout_seconds":100} for p in module.PHASES}
    ops=CommandOperations({"schema":"roborender.beta-campaign-operations.v1","commands":commands},tmp_path/"operations")
    child=Mock(pid=2_000_000_000)
    with patch.object(module.subprocess,"Popen",return_value=child) as launch:
        old=ops.start("prepare",{"round_id":3,"run_id":"dagger-leo-test"})
        context={"round_id":3,"run_id":"dagger-leo-test","infrastructure_attempts":{"3":"a"*32}}
        new=ops.start("prepare",context)
        assert new!=old and ops.start("prepare",context)==new and launch.call_count==2


def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()
