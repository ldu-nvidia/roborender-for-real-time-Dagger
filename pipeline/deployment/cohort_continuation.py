"""Explicit fresh-namespace continuation after the reconciled update36 failure.

Inspection is read-only. Staging copies only verified completion markers and
creates a new CPU campaign state; execution is a separate CLI action. Historical
native receipts retain their original source identity. This module never
imports a ledger, starts services, restores Isaac, releases an allocation or
dispatches another round. Those admissions precede this controller.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.bc_training.handoff_common import binding, bound, canonical, cli_binding, publish
from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.coverage_campaign import CoverageMSECampaign
from pipeline.dagger.coverage_production import digest, load_plan, require, validate_qualification
from pipeline.dagger.mse_campaign import MSECampaign, online_mse_loggers
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import actor_ids
from pipeline.dagger.services.client import ServiceClient
from pipeline.dagger.services.mse_status import completed_mse_summary
from pipeline.deployment.beta_operations import atomic
from pipeline.deployment.capacity_admission import validate as validate_capacity, validate_production_memory_policy
from pipeline.deployment.cohort_source_transition import build_admission as rebuild_transition
from pipeline.deployment.coverage_operations import CoverageCommandOperations
from pipeline.deployment.native_phase_barrier import native_quiescent

SCHEMA = "roborender.round1-cohort-continuation.v1"
SPEC_SCHEMA = "roborender.round1-cohort-continuation-spec.v1"
PHASES = frozenset({"COLLECT_TRAIN", "RELEASE_TRAINING", "EVALUATE", "IMPORT_VALIDATION",
                    "LOG_VALIDATION", "ADVANCE", "COMPLETE"})
REFS = frozenset({"source_transition", "runtime_rebind", "historical_state", "service_binding",
                  "capacity_admission", "operations", "mse_plan", "site", "resident_pool", "retained",
                  "historical_resident_pool"})


def check_transition(ref):
    value = bound(ref)
    actual = rebuild_transition(
            historical_root=value["historical_source_root"], execution_root=value["execution_source_root"],
            **{key: value[key] for key in ("historical_manifest", "execution_manifest", "historical_config",
                                         "execution_config", "import_receipt")})
    require(actual == value, "source-transition certificate no longer reproduces")
    return value


def _cohorts(plan):
    return [(epoch["coverage_epoch"], cohort) for epoch in plan["rounds"][0]["epochs"]
            for cohort in epoch["cohorts"]]


def _validate_rebind(value, transition_ref, transition):
    require(value.get("schema") == "roborender.dagger-cohort-runtime-rebind.v3"
            and value.get("source_transition") == transition_ref
            and value.get("import_receipt") == transition["import_receipt"]
            and value.get("old_config") == transition["historical_config"]
            and value.get("new_config") == transition["execution_config"]
            and value.get("database_update_complete") is True
            and value.get("completed_updates") == 36 and value.get("next_iteration") == 37
            and value.get("requests_submitted") == 0 and value.get("native_work_started") is False
            and value.get("allocation_retained") is True and value.get("compute_release_requested") is False,
            "actual owner-host source/runtime rebind receipt required")


def _validate_pool(pool, old_state, config, historical_pool):
    from pipeline.deployment.rtx_actor.production_resident_pool import validate_pool
    validate_pool(pool)
    validate_pool(historical_pool)
    require(pool["source_sha256"] == config["source_sha256"], "new pool execution source differs")
    old_ids = {row["worker_instance_id"] for row in
               old_state["service_binding"]["native_phase_barrier"]["workers"]}
    require(not old_ids.intersection(row["worker_instance_id"] for row in pool["workers"]),
            "continuation requires genuinely new native worker generations")
    require(historical_pool["source_sha256"] == old_state["config"]["source_sha256"]
            and {row["worker_instance_id"] for row in historical_pool["workers"]} == old_ids,
            "historical pool differs from the preserved actual native generations")
    def disjoint(left, right):
        left, right = Path(left), Path(right)
        return not left.is_relative_to(right) and not right.is_relative_to(left)
    require(disjoint(pool["jobs_root"], historical_pool["jobs_root"])
            and all(disjoint(new["runtime_root"], old["runtime_root"])
                    and new["container_name"] != old["container_name"]
                    for new, old in zip(pool["workers"], historical_pool["workers"])),
            "fresh remote jobs, native runtime and container namespaces are required")
    return pool


def _validate_service(value, previous, config, pool):
    keys = ("allocation_name", "allocation_owner_run_id", "node_id", "gpu_uuids")
    require(all(value.get(key) == previous.get(key) for key in keys)
            and len(value.get("gpu_uuids", [])) == 4 and value.get("allocation_retained") is True
            and value.get("compute_release_authorized") is False
            and value.get("resource_shape") == "custom.gpu.4xh200"
            and value.get("reservation") == "cosmos-solution-architect-xcwtffiq"
            and value.get("allow_burst") is False and value.get("genuine_backends") is True
            and value.get("owned_control_plane") is True
            and value.get("source_sha256") == config["source_sha256"]
            and value.get("gateway_pids") == config["h200"]["gateway_pids"]
            and native_quiescent(value, source_sha256=config["source_sha256"]),
            "fresh service/native binding changed retained owner or execution identity")
    expected = [{key: row[key] for key in ("physical_gpu", "gpu_uuid", "worker_instance_id")}
                for row in pool["workers"]]
    actual = sorted(value["native_phase_barrier"]["workers"], key=lambda row: row["physical_gpu"])
    require([{key: row[key] for key in expected[0]} for row in actual] == expected,
            "service and fresh native pool generations differ")
    token = Path(value["token_file"])
    require(token.is_file() and not token.is_symlink() and not token.stat().st_mode & 0o077
            and bool(value.get("mse_worker_command")), "owned token and H200 MSE worker required")


def validate_operation_bindings(spec):
    """Every executable phase must use this exact fresh site/pool/allocation."""
    commands = bound(spec["operations"])["commands"]
    pool = bound(spec["resident_pool"])
    remote_pool = None
    for phase in ("collect", "release", "evaluate"):
        argv = commands[phase]["argv"]
        def argument(name):
            require(argv.count(name) == 1, "operation lacks exact " + name)
            return argv[argv.index(name) + 1]
        require(argument("--site") == spec["site"]["path"]
                and argument("--site-sha256") == spec["site"]["raw_sha256"]
                and argument("--retained") == spec["retained"]["path"]
                and argument("--retained-sha256") == spec["retained"]["raw_sha256"]
                and argument("--resident-pool-sha256") == digest(pool)
                and "--management-release" not in argv and "--pool-control" not in argv,
                "operation cannot substitute site/pool/source or retained allocation")
        selected = argument("--resident-pool")
        require(Path(selected).is_absolute() and (remote_pool is None or selected == remote_pool),
                "all phases must use one exact fresh native pool")
        remote_pool = selected
    return remote_pool


def observe_boundary(config, service):
    """Owner RPC only: never open the shared live SQLite from this CPU host."""
    token = Path(service["token_file"]).read_text().strip()
    client = ServiceClient("http://127.0.0.1:18610", token, timeout=30)
    return {"coordinator": client.get("/v1/status/" + config["run_id"]),
            "mse": client.get("/v1/mse-rounds/" + config["run_id"] + "/1")["mse_round"]}


def validate_boundary(observation, config, transition):
    status, mse = observation["coordinator"], observation["mse"]
    rows = status["iterations"]
    require(status["run"].get("config_sha256") == digest(config)
            and status["run"].get("run_id") == config["run_id"]
            and status["run"].get("current_round") == 1
            and status["run"].get("status") == "collecting"
            and status["run"].get("actor_ids") == actor_ids(config)
            and status["run"].get("active_checkpoint_sha256") == transition["parent_checkpoint_sha256"]
            and len(rows) == 37 and [row["iteration_id"] for row in rows] == list(range(1, 38))
            and all(row["round_id"] == 1 for row in rows)
            and all(row["status"] == "advanced" for row in rows[:36])
            and rows[-1]["status"] == "collecting" and rows[-1].get("candidate_checkpoint_sha256") is None
            and mse.get("status") == "active" and mse.get("iteration_id") == 37
            and mse.get("checkpoint_sha256") == transition["parent_checkpoint_sha256"]
            and [row["iteration"] for row in mse["selection"]["history"]] == [0, 20],
            "live owner is not the exact imported empty iteration37 boundary")
    return rows


def inspect(spec_ref, *, observer=observe_boundary):
    spec = bound(spec_ref)
    require(set(spec) == REFS | {"schema", "campaign_root", "historical_round_directory", "wandb_entity",
                                "wandb_key_file", "maximum_round_id", "compute_release_authorized"}
            and spec["schema"] == SPEC_SCHEMA and spec["maximum_round_id"] == 1
            and spec["compute_release_authorized"] is False, "exact round1-only continuation specification required")
    for key in REFS:
        bound(spec[key])
    transition = check_transition(spec["source_transition"])
    old, new = bound(transition["historical_config"]), bound(transition["execution_config"])
    require(Path(transition["execution_config"]["path"]).read_bytes() == canonical(new),
            "execution config must use exact canonical bytes for coordinator registration")
    historical = bound(spec["historical_state"])
    require(historical.get("config") == old and historical.get("phase") == "COLLECT_TRAIN"
            and historical.get("round_id") == 1 and historical.get("completed_rounds") == []
            and historical.get("qualification_completed") is True
            and historical.get("round_logging", {}).get("online_metrics_verified") is True
            and historical["round_logging"]["run_id"] == old["logging"]["wandb_run_id"],
            "historical round1 learner/logging state differs")
    old_plan = load_plan(transition["historical_coverage_plan"])
    validate_qualification(old_plan, historical["native_qualification"], old["student"]["initial_checkpoint_sha256"])
    require(bound(historical["native_qualification_binding"]) == historical["native_qualification"],
            "historical native qualification changed")
    _validate_rebind(bound(spec["runtime_rebind"]), spec["source_transition"], transition)
    capacity = bound(spec["capacity_admission"])
    validate_capacity(capacity, source_sha256=new["source_sha256"], environments_per_actor=16)
    validate_production_memory_policy(capacity)
    pool = _validate_pool(bound(spec["resident_pool"]), historical, new, bound(spec["historical_resident_pool"]))
    service = bound(spec["service_binding"])
    _validate_service(service, historical["service_binding"], new, pool)
    root = Path(spec["campaign_root"]).absolute()
    old_round = Path(spec["historical_round_directory"]).absolute()
    require(old_round == Path(spec["historical_state"]["path"]).parent / "round-1"
            and root != old_round.parent and not root.is_relative_to(old_round.parent)
            and not old_round.parent.is_relative_to(root)
            and root.is_relative_to("/workspace/project/roborender/runtime"),
            "a distinct scoped continuation namespace is required")
    require(Path(service["token_file"]) == root / "services.token", "service token must belong to fresh campaign")
    CoverageCommandOperations(bound(spec["operations"]), root / "operations")
    validate_operation_bindings(spec)
    from pipeline.dagger.mse_campaign import validate_plan
    validate_plan(bound(spec["mse_plan"]), new)
    require(bound(spec["mse_plan"]) == bound(binding(Path(spec["historical_state"]["path"]).parent / "mse-plan.json")),
            "continuation cannot change heldout or stopping plan")
    observation = observer(new, service)
    rows = validate_boundary(observation, new, transition)
    from pipeline.deployment.resident_production_collection import completed_prefix
    markers = []
    parent = old["student"]["initial_checkpoint_sha256"]
    for epoch, cohort in _cohorts(old_plan)[:18]:
        directory = old_round / f"coverage/e{epoch:02d}-c{cohort['cohort_index']:03d}"
        result = completed_prefix(directory, old_plan, 1, epoch, cohort["cohort_index"], parent, rows)
        require(result is not None, "historical completed cohort marker missing")
        saved, record = result
        parent = saved["output_checkpoint_sha256"]
        markers.append({"epoch": epoch, "index": cohort["cohort_index"], "completion": binding(directory / "completion.json")})
    require(parent == transition["parent_checkpoint_sha256"], "completed prefix parent differs from imported checkpoint36")
    return dict(schema=SCHEMA, spec=spec_ref, run_id=new["run_id"], round_id=1,
                campaign_root=str(root), source_transition=spec["source_transition"],
                historical_source_sha256=old["source_sha256"], execution_source_sha256=new["source_sha256"],
                completed_cohorts=18, completed_updates=36, next_iteration=37, historical_markers=markers,
                parent_checkpoint_sha256=parent, preserved_mse_selection=observation["mse"]["selection"],
                qualification_next_updates=[37, 38, 39, 40], maximum_round_id=1,
                automatic_next_round=False, native_replay_allowed=False, compute_release_requested=False)


def validate_admission(ref):
    value = bound(ref)
    require(value.get("schema") == SCHEMA and value.get("completed_updates") == 36
            and value.get("completed_cohorts") == 18 and value.get("next_iteration") == 37
            and value.get("maximum_round_id") == 1 and value.get("automatic_next_round") is False
            and value.get("native_replay_allowed") is False and value.get("compute_release_requested") is False,
            "invalid cohort continuation admission")
    spec = bound(value["spec"])
    for key in REFS:
        bound(spec[key])
    require(spec["source_transition"] == value["source_transition"]
            and spec["campaign_root"] == value["campaign_root"], "continuation specification changed")
    return value, spec


def stage(spec_ref):
    value = inspect(spec_ref)
    root = Path(value["campaign_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root / "state.json").exists() and not (root / "continuation-admission.json").exists(),
                "continuation already staged; never overwrite or reinitialize")
        spec = bound(spec_ref)
        transition = bound(value["source_transition"])
        config = bound(transition["execution_config"])
        old = bound(spec["historical_state"])
        # Only completed markers are copied; their nested references keep the
        # historical bytes/path/source. No failed intent or partial row enters.
        for marker in value["historical_markers"]:
            target = root / "round-1" / f"coverage/e{marker['epoch']:02d}-c{marker['index']:03d}/completion.json"
            require(not target.exists(), "fresh prefix marker already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = Path(marker["completion"]["path"]).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == marker["completion"]["raw_sha256"], "historical marker changed")
            with target.open("xb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            target.chmod(0o444)
        ref = publish(root / "continuation-admission.json", value)
        state = deepcopy(old)
        state.update(config=config, config_path=transition["execution_config"]["path"],
                     template_sha256=digest(config), first_config_sha256=digest(config),
                     service_binding=bound(spec["service_binding"]),
                     mse_worker_command=bound(spec["service_binding"])["mse_worker_command"],
                     operation_id=None, phase="COLLECT_TRAIN", cohort_continuation=ref,
                     collection_deadline=time.time() + 7 * 86400,
                     historical_native_qualification_binding=old["native_qualification_binding"],
                     updated_at=time.time())
        # qualification remains historical, deliberately not relabeled.
        immutable_json(root / "mse-plan.json", bound(spec["mse_plan"]))
        immutable_json(root / "initial-continuation-state.json", state)
        atomic(root / "state.json", state)
        return ref


class HistoricalPrefix:
    """Narrow collector hook: history uses its original plan, new work never does."""
    def __init__(self, ref, config, root):
        self.ref = ref
        self.value, self.spec = validate_admission(ref)
        transition = bound(self.value["source_transition"])
        require(config == bound(transition["execution_config"]) and str(Path(root).absolute()) == self.value["campaign_root"],
                "collector continuation/config/root differs")
        self.old_plan = load_plan(transition["historical_coverage_plan"])
        self.new_plan = load_plan(transition["execution_coverage_plan"])
        self.markers = {(row["epoch"], row["index"]): row["completion"] for row in self.value["historical_markers"]}

    def plan_for(self, directory, epoch, index):
        historical = self.markers.get((epoch, index))
        if historical is None:
            return self.new_plan
        marker = Path(directory) / "completion.json"
        require(marker.is_file() and not marker.is_symlink()
                and hashlib.sha256(marker.read_bytes()).hexdigest() == historical["raw_sha256"]
                and binding(historical["path"]) == historical,
                "historical prefix marker missing, rewritten or relabeled")
        return self.old_plan

    def record_requalification(self, completed, directory, barrier):
        if len(completed) < 20:
            return
        from pipeline.dagger.coverage_production import aggregate_resource_guards, validate_segment
        new = completed[18:20]
        records = [bound(row["segment_result"]) for row in new]
        parent = dict(sha256=self.value["parent_checkpoint_sha256"],
                      optimizer_global_step=bound(bound(self.value["source_transition"])["import_receipt"])["parent_optimizer_global_step"])
        for (epoch, cohort), saved, record in zip(_cohorts(self.new_plan)[18:20], new, records):
            require(saved["input_checkpoint_sha256"] == parent["sha256"] and saved["native_quiescent"] is True,
                    "new qualification cohort parent/quiescence differs")
            parent = validate_segment(self.new_plan, 1, epoch, cohort["cohort_index"], parent, record)
            for ref in saved["learner_receipts"]:
                result = bound(ref)
                require(result.get("optimizer_state_preserved") is True
                        and result.get("train_chunks") == 32 and result.get("train_frames") == 2592,
                        "continuation qualification lacks actual full-batch AdamW receipts")
        require(native_quiescent(barrier, source_sha256=self.value["execution_source_sha256"]),
                "new qualification did not leave actual native owners quiescent")
        resources = aggregate_resource_guards([guard for row in records for guard in row["resource_guards"]], per_gpu=2)
        immutable_json(Path(directory) / "continuation-native-qualification.json", dict(
            schema="roborender.cohort-continuation-joint-qualification.v1", continuation=self.ref,
            source_sha256=self.value["execution_source_sha256"], optimizer_updates=[37, 38, 39, 40],
            segment_receipts=[row["segment_result"] for row in new],
            historical_qualification_preserved=True, accepted_updates_not_repeated=True,
            full_batch_environments=32, completed=True, allocation_retained=True,
            compute_release_requested=False, **resources))


class ContinuationCampaign(CoverageMSECampaign):
    def __init__(self, ref, operations, **kwargs):
        self.admission_ref = ref
        self.admission, self.spec = validate_admission(ref)
        transition = bound(self.admission["source_transition"])
        config = bound(transition["execution_config"])
        self.coverage = load_plan(config["coverage_plan"])
        self.qualification_parent = config["student"]["initial_checkpoint_sha256"]
        # CoverageMSECampaign's ordinary same-source qualification gate is not
        # appropriate for history; validate the old proof against the old plan.
        historical = bound(self.spec["historical_state"])
        validate_qualification(load_plan(transition["historical_coverage_plan"]),
                               historical["native_qualification"], self.qualification_parent)
        MSECampaign.__init__(self, config, self.admission["campaign_root"], operations,
                             plan=bound(self.spec["mse_plan"]), **kwargs)
        self._guard()

    def _guard(self):
        value, spec = validate_admission(self.admission_ref)
        historical = bound(spec["historical_state"])
        config = bound(bound(value["source_transition"])["execution_config"])
        require(self.state["config"] == config and self.state["round_id"] == 1
                and self.state["phase"] in PHASES and self.state["cohort_continuation"] == self.admission_ref
                and self.state["native_qualification"] == historical["native_qualification"]
                and self.state["native_qualification_binding"] == historical["native_qualification_binding"]
                and len(self.state["completed_rounds"]) == (1 if self.state["phase"] == "COMPLETE" else 0),
                "continuation may not alter history/configuration or enter another round")

    def operation_context(self):
        value = super().operation_context()
        value["cohort_continuation"] = self.admission_ref
        return value

    def operation(self, phase):
        self._guard()
        require(phase == {"COLLECT_TRAIN": "collect", "RELEASE_TRAINING": "release", "EVALUATE": "evaluate"}.get(self.state["phase"]),
                "continuation cannot dispatch preparation, abort or another round")
        result = super().operation(phase)
        if result is not None:
            require(result.get("owned_actors_stopped") is not True
                    and native_quiescent(result, source_sha256=self.state["config"]["source_sha256"]),
                    "continuation phase must leave admitted native owners warm")
        return result

    def tick(self):
        self._guard()
        if self.state["phase"] == "ADVANCE":
            self.validate_production_test_evidence()
            context = self.get_controller().mse.context()
            require(context["status"] == "complete"
                    and context["selection"]["best_checkpoint_sha256"] == self.state["final_checkpoint"]["sha256"]
                    and self.state["logging_receipt"].get("online_metrics_verified") is True,
                    "complete corrected295 test and online W&B receipt required")
            row = dict(round_id=1, teacher_beta=.75, selected_checkpoint=self.state["final_checkpoint"],
                       **self.state["mse_completion"], test=self.state["evaluation_report"], wandb=self.state["logging_receipt"])
            immutable_json(self.root / "round1-continuation-completion.json", dict(
                schema="roborender.round1-cohort-continuation-completion.v1", continuation=self.admission_ref,
                completed_round=row, automatic_next_round=False, native_owners_kept_warm=True, compute_released=False))
            self.save("COMPLETE", completed_rounds=[row], progress=dict(action="authorized_round1_complete",
                      automatic_next_round=False, **completed_mse_summary(self.state["evaluation_report"])))
            return dict(self.state)
        return super().tick()


def _live_site(spec, state):
    from pipeline.deployment.retained_campaign_site import RetainedResidentSite
    remote_pool = validate_operation_bindings(spec)
    pool = bound(spec["resident_pool"])
    request = dict(config=state["config"], campaign_root=spec["campaign_root"], run_id=state["run_id"], round_id=1)
    return RetainedResidentSite(bound(spec["site"]), request, remote_pool, digest(pool),
        retained_file=spec["retained"]["path"], retained_sha256=spec["retained"]["raw_sha256"])


def run(ref, *, poll_seconds=20, observer_seconds=7 * 86400):
    value, spec = validate_admission(ref)
    root = Path(value["campaign_root"])
    with (root / "continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            check_transition(spec["source_transition"])
            state = bound(binding(root / "state.json"))
            site = _live_site(spec, state)
            site.verify_sources(); site.verify_retained()
            if state.get("operation_id") is None:
                site.barrier()
            else:
                # Actual existing operations are reconciled by their durable
                # process identity; a lost owner is never relaunched by poll.
                site.pool_action("status")
            begin, log = online_mse_loggers(spec["wandb_entity"], spec["wandb_key_file"])
            campaign = ContinuationCampaign(ref, CoverageCommandOperations(bound(spec["operations"]), root / "operations"),
                                              begin_logging=begin, log_validation=log)
            stop = False
            def signal_stop(*_):
                nonlocal stop
                stop = True
            signal.signal(signal.SIGTERM, signal_stop); signal.signal(signal.SIGINT, signal_stop)
            deadline = time.monotonic() + observer_seconds
            while not stop and time.monotonic() < deadline:
                state = campaign.tick()
                status = dict(phase=state["phase"], continuation=ref, round_id=1, progress=state.get("progress"),
                              checked_unix=time.time(), compute_release_requested=False)
                atomic(root / "continuation-status.json", status)
                print(json.dumps(status, sort_keys=True), flush=True)
                if state["phase"] == "COMPLETE":
                    return
                time.sleep(poll_seconds)
            atomic(root / "continuation-status.json", dict(phase="CPU_OBSERVER_DETACHED", continuation=ref,
                   workload_stop_requested=False, compute_release_requested=False))
        except Exception as error:
            atomic(root / "continuation-status.json", dict(phase="FAILED_CLOSED", continuation=ref,
                   error_type=type(error).__name__, checked_unix=time.time(), native_replay_allowed=False,
                   workload_stop_requested=False, compute_release_requested=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", nargs=2, metavar=("PATH", "RAW_SHA256"))
    parser.add_argument("--admission", nargs=2, metavar=("PATH", "RAW_SHA256"))
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--observer-seconds", type=int, default=7 * 86400)
    args = parser.parse_args()
    if args.admission:
        if args.spec or args.stage or not (args.execute and args.watch):
            parser.error("admitted execution requires only --admission PATH SHA --execute --watch")
        if not 1 <= args.poll_seconds <= 60 or not 60 <= args.observer_seconds <= 7 * 86400:
            parser.error("bounded polling/observer lifetime required")
        run(cli_binding(args.admission), poll_seconds=args.poll_seconds, observer_seconds=args.observer_seconds)
    else:
        if not args.spec or args.execute or args.watch:
            parser.error("inspection/staging requires --spec PATH SHA without execution flags")
        print(json.dumps(stage(cli_binding(args.spec)) if args.stage else inspect(cli_binding(args.spec)), sort_keys=True))


if __name__ == "__main__":
    main()
