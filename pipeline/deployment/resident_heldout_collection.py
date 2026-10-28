"""Host orchestration for one immutable held-out cohort schedule.

Runs against the already owned genuine H200 allocation and resident Leo pool.
No allocation, optimizer submission, checkpoint promotion, or native replay is
performed here. Small identity receipts are copied back; learner arrays remain
with their sole H200 assembler.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.beta_validation import load_panel, panel_seeds
from pipeline.dagger.heldout_collection_plan import load_plan, assignment_for, registration_for
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import canonical, gpu_uuids
from pipeline.dagger.services.client import ServiceClient, ServiceError
from pipeline.dagger.services.contracts import file_sha256
from pipeline.deployment.native_metadata_transfer import fetch_native_json

def training_seeds(config):
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_mse_production
    if is_mse_production(config):
        from pipeline.dagger.coverage_production import round_training_seeds
        return round_training_seeds(config)
    return [config['seed']+i for i,_ in enumerate(config['rtx_actors'])]

def admit_native_cohort(site,receipt,cohort,actor,assignment,name,owner,local,assignments,chunks):
    """Bind actual receipts; native execution can be parallel, admission cannot guess."""
    gpu,cid=cohort['physical_gpu_index'],cohort['cohort_id'];state=receipt['actor_state']
    if (receipt.get('complete') is not True or receipt.get('purpose')!='validation'
        or receipt.get('optimizer_steps')!=0 or state['phase']!='COMPLETE'
        or state['registration_sha256']!=owner['registration_sha256']
        or state['actor_prefix']!=actor['actor_prefix'] or state['transition_batches']!=81
        or state['physical_transition_rows']!=81*len(cohort['lanes'])):
        raise RuntimeError('native held-out completion differs from admitted cohort')
    immutable_json(local/cid/'native-completion.json',receipt)
    worker=next(row for row in site.pool['workers'] if row['physical_gpu']==gpu)
    allowed=Path('/dagger-runtime/phases')/name/'native-assignments'
    for evidence in receipt['episode_assignments']:
        source=Path(evidence['native_assignment_path'])
        if source.parent!=allowed or not source.name.endswith('.json'):
            raise ValueError('native assignment receipt escaped its exact phase output')
    names=[Path(e['native_assignment_path']).name for e in receipt['episode_assignments']]
    payloads=fetch_native_json(site,Path(worker['runtime_root'])/allowed.relative_to('/dagger-runtime'),
        [n for name in names for n in (name,Path(name).with_suffix('.reset.json').name)])
    for evidence in receipt['episode_assignments']:
        source=Path(evidence['native_assignment_path'])
        raw=payloads[source.name]
        if hashlib.sha256(raw).hexdigest()!=evidence['native_assignment_sha256']:
            raise RuntimeError('native assignment evidence changed during transfer')
        destination=local/cid/source.name;site.publish(destination,raw,mode=0o444)
        reset_raw=payloads[source.with_suffix('.reset.json').name];reset=json.loads(reset_raw);lane=reset.get('lane')
        if (type(lane)is not int or lane not in range(len(cohort['lanes']))
            or reset.get('episode_id')!=evidence['episode_id']
            or reset.get('assignment_sha256')!=hashlib.sha256(canonical(assignment)).hexdigest()):
            raise RuntimeError('native reset evidence does not bind the held-out cohort lane')
        reset_destination=destination.with_suffix('.reset.json');site.publish(reset_destination,reset_raw,mode=0o444)
        assignments.append({**evidence,'native_assignment_path':str(destination),
            'actor_id':actor['actor_prefix']+f'-lane-{lane:03d}',
            'native_reset_path':str(reset_destination),'native_reset_sha256':hashlib.sha256(reset_raw).hexdigest()})
    chunks.extend(state['chunk_ids'])


def collect(site):
    site.barrier()
    request = site.request["heldout_collection"]
    if set(request) != {"plan_path", "plan_sha256", "manifest_path", "admission_path"}:
        raise ValueError("exact immutable held-out host-operation request required")
    test_seeds = panel_seeds(load_panel(site.config))
    plan = load_plan(request["plan_path"], request["plan_sha256"], site.config, test_seeds)
    token = site.token.read_text().strip()
    coordinator = ServiceClient("http://127.0.0.1:18610", token, timeout=120)
    registry = ServiceClient("http://127.0.0.1:18612", token, timeout=120)
    parent = site.config["student"]
    try:
        registered = registry.get("/v1/checkpoint-records/" + parent["initial_checkpoint_sha256"])["checkpoint"]
        if registered["sha256"] != parent["initial_checkpoint_sha256"]:
            raise RuntimeError("held-out parent registry identity differs")
    except ServiceError as exc:
        if not str(exc).startswith("404 ") or site.rid != 1:
            raise
        registry.post("/v1/checkpoints/register-local", {"path": parent["initial_checkpoint"],
            "sha256": parent["initial_checkpoint_sha256"],
            "metadata": {"role": "beta_round_parent", "run_id": site.run, "round_id": site.rid}})
    owner = registration_for(plan, site.config, request["plan_sha256"])
    coordinator.post("/v1/validation-collections/register", owner)
    local = site.round / "heldout-population"
    remote = site.remote / "heldout-population"
    immutable_json(local / "registration.json", owner)
    assignments, chunks, pending = [], [], []
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_mse_production
    paired = is_mse_production(site.config)
    # Cohorts may use either worker. Sequential execution makes one exact scene
    # owner explicit and does not change task dynamics; larger parallel batches
    # require their own native qualification rather than assumed capacity.
    for cohort in plan["cohorts"]:
        gpu, cid = cohort["physical_gpu_index"], cohort["cohort_id"]
        assignment = assignment_for(plan, cohort)
        assignment_path = remote / cid / "motion-assignment.json"
        assignment_raw = canonical(assignment)
        assignment_sha = hashlib.sha256(assignment_raw).hexdigest()
        site.publish(assignment_path, assignment_raw, "beta", 0o444)
        in_container = Path("/resident/jobs") / assignment_path.relative_to(site.jobs)
        actor = {"schema": "roborender.dagger-heldout-actor.v1", **owner,
            "actor_prefix": plan["run_id"] + "-" + cid, "num_envs": len(cohort["lanes"]),
            "physical_gpu_index": gpu, "gpu_uuid": gpu_uuids(site.config)[gpu],
            "simulator_seed": cohort["simulator_seed"], "motion_assignment_path": str(in_container),
            "motion_assignment_sha256": assignment_sha, "gateway_pids": site.config["h200"]["gateway_pids"],
            "source_sha256": site.config["source_sha256"]}
        actor_raw = canonical(actor)
        actor_path = remote / cid / "actor.json"
        site.publish(actor_path, actor_raw, "beta", 0o444)
        # Resident processes survive campaigns; an r1/cohort-only name would
        # collide with an earlier completed campaign's immutable phase.
        name = f"mse-heldout-{hashlib.sha256(site.run.encode()).hexdigest()[:12]}-r{site.rid}-{cid}"
        command = {"gpu": gpu, "command_id": plan["run_id"] + "-" + cid,
            "operation": "collect_mse_validation", "payload": {
                "collection_actor_file": str(actor_path.relative_to(site.jobs)),
                "collection_actor_sha256": hashlib.sha256(actor_raw).hexdigest(),
                "motion_assignment_file": str(assignment_path.relative_to(site.jobs)),
                "motion_assignment_sha256": assignment_sha,
                "token_file": str((site.remote / "services.token").relative_to(site.jobs)),
                "output_name": name}}
        pending.append((cohort,actor,assignment,name,command))
        if paired and len(pending)==1 and cohort is not plan['cohorts'][-1]:continue
        commands=[row[4] for row in pending]
        if paired and {row['gpu'] for row in commands}!={0,1}:
            raise RuntimeError('production held-out batch requires one exact cohort per GPU')
        site.pool_action('submit',commands);result=site.pool_action('wait',commands)
        if len(result['results'])!=len(pending):raise RuntimeError('held-out native cohort cardinality differs')
        for context,item in zip(pending,result['results']):
            c,a,assignment,n,_=context
            admit_native_cohort(site,item['receipt'],c,a,assignment,n,owner,local,assignments,chunks)
        pending=[]
        site.barrier()
        print(json.dumps(dict(phase='HELDOUT_NATIVE_COLLECTED',round_id=site.rid,
            completed_chunks=len(chunks),expected_chunks=320 if paired else sum(len(c['lanes']) for c in plan['cohorts']))),flush=True)
    status = coordinator.get("/v1/validation-collections/" + plan["run_id"])["collection"]
    if (status["status"] != "complete" or status["optimizer_authorized"] is not False
            or {row["chunk_id"] for row in status["actors"]} != set(chunks)):
        raise RuntimeError("H200 held-out collection is incomplete")
    collection = {"schema": "roborender.dagger-heldout-collection.v2", "purpose": "validation",
        "source_run_id": plan["run_id"], "round_id": site.rid,
        "collection_plan_path": request["plan_path"], "collection_plan_sha256": request["plan_sha256"],
        **{key: plan[key] for key in ("policy_spec_sha256", "teacher_checkpoint_sha256", "motion_inventory_sha256", "motion_support")},
        "collection_checkpoint_sha256": parent["initial_checkpoint_sha256"], "teacher_beta": site.config["teacher_beta"],
        "training_simulator_seeds": training_seeds(site.config),
        "test_simulator_seeds": sorted(set(test_seeds)), "chunk_ids": sorted(chunks), "episode_assignments": assignments}
    receipt_path = local / "collection.json"
    receipt_sha = immutable_json(receipt_path, collection)
    # The live SQLite WAL has one host owner. Export on that same H200 host;
    # CPU sees only the resulting immutable manifests on shared storage.
    from pipeline.deployment.beta_site import ROOT
    result = site.json([site.site["cpu_python"], "-c",
        "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pipeline.dagger.export_heldout_manifest',run_name='__main__')", ROOT,
        "--ledger", site.root / "services/state.sqlite3", "--collection-receipt", receipt_path,
        "--collection-receipt-sha256", receipt_sha, "--output", request["manifest_path"],
        "--admission-output", request["admission_path"]], "gpu", timeout=120)
    if (result["manifest_sha256"] != file_sha256(Path(request["manifest_path"]))
            or result["admission_sha256"] != file_sha256(Path(request["admission_path"]))
            or result["array_payloads_read"] is not False):
        raise RuntimeError("held-out exporter publication differs")
    return {"purpose": "validation", "optimizer_steps": 0, "collection_complete": True,
            "chunk_count": len(chunks), "transition_rows": 81 * len(chunks), **result, **site.barrier()}
