"""Actor-local episode counters must not cause false collisions or hide leakage."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.export_heldout_manifest import load_collection, export_manifest, validate_export_admission
from pipeline.dagger.heldout_collection_plan import assignment_for
from pipeline.dagger.services.contracts import file_sha256
from pipeline.dagger.services.heldout_mse import canonical_digest, source_episode_identity, load_validation_manifest
from pipeline.dagger.services.mse_protocol import training_provenance
from pipeline.dagger.services.store import StateStore


def fixture(root):
    source = "heldout-actor-scope-fixture"
    plan = {"schema": "roborender.dagger-heldout-plan.v1", "parent_run_id": "dagger-leo-scope-fixture",
        "run_id": source, "round_id": 1, "teacher_checkpoint_sha256": "a"*64,
        "policy_spec_sha256": "b"*64, "motion_inventory_sha256": "c"*64,
        "motion_support": ["motion-one"], "cohorts": []}
    for gpu in (0, 1):
        plan["cohorts"].append({"cohort_id": f"cohort-{gpu}", "physical_gpu_index": gpu,
            "simulator_seed": 100+gpu, "lanes": [{"lane": 0, "motion_id": "motion-one",
                "path": "/cpu-fixture/not-opened.pkl", "sha256": "d"*64}]})
    plan_path = root/"plan.json"
    plan_sha = immutable_json(plan_path, plan)
    collection = {"schema": "roborender.dagger-heldout-collection.v2", "purpose": "validation",
        "source_run_id": source, "round_id": 1, "policy_spec_sha256": "b"*64,
        "teacher_checkpoint_sha256": "a"*64, "collection_checkpoint_sha256": "e"*64,
        "teacher_beta": .75, "motion_inventory_sha256": "c"*64, "motion_support": ["motion-one"],
        "training_simulator_seeds": [200,201], "test_simulator_seeds": [300],
        "chunk_ids": [], "episode_assignments": [], "collection_plan_path": str(plan_path),
        "collection_plan_sha256": plan_sha}
    store = StateStore(root/"store")
    actors = []
    for gpu, cohort in enumerate(plan["cohorts"]):
        assignment = assignment_for(plan, cohort)
        actor = f"{source}-{cohort['cohort_id']}-lane-000"
        actors.append(actor)
        # Both genuine owners would reuse this raw local counter.
        episode = "episode-000-00000000"
        evidence = {"schema":"roborender.native-episode-assignment.v1", "purpose":"validation",
            "run_id":source,"episode_id":episode,"simulator_seed":100+gpu,
            "motion_id":"motion-one","motion_sha256":"d"*64,"observation_source":"generated_controls"}
        path = root/f"native-{gpu}.json"
        immutable_json(path,evidence)
        reset = {"schema":"roborender.native-episode-reset-evidence.v1","episode_id":episode,
            "lane":0,"assignment_sha256":canonical_digest(assignment),"native_episode_reset_index":0}
        reset_path=root/f"reset-{gpu}.json"
        immutable_json(reset_path,reset)
        collection["episode_assignments"].append({key:evidence[key] for key in
            ("episode_id","simulator_seed","motion_id","motion_sha256","observation_source")})
        collection["episode_assignments"][-1].update(actor_id=actor,native_assignment_path=str(path),
            native_assignment_sha256=file_sha256(path),native_reset_path=str(reset_path),native_reset_sha256=file_sha256(reset_path))
        chunk=f"chunk-{gpu}"
        collection["chunk_ids"].append(chunk)
        payload=root/f"chunk-{gpu}.npz"
        payload.write_bytes(f"CPU metadata fixture {gpu}; not arrays".encode())
        meta={"run_id":source,"round_id":1,"actor_id":actor,"rollout_window_id":"window-000000",
            "checkpoint_sha256":"e"*64,"first_step_id":0,"last_step_id":80,
            "teacher_checkpoint_sha256":"a"*64,"sealed_window_sha256":str(gpu+1)*64}
        with store.transaction() as db:
            db.execute("INSERT INTO validation_collection_actors VALUES (?,?,?,?,?)",(source,actor,chunk,"{}",0))
            db.execute("INSERT INTO chunk_reservations VALUES (?,?,?,?,?,?,?,?)",(chunk,file_sha256(payload),payload.stat().st_size,
                json.dumps(meta),"committed",str(payload),0,0))
            db.execute("INSERT INTO chunk_provenance VALUES (?,?,?,?,?,?)",(chunk,file_sha256(payload),json.dumps([episode]),
                "window-000000","a"*64,str(gpu+1)*64))
    owner={"schema":"roborender.dagger-validation-collection.v1","purpose":"validation","run_id":source,
        "round_id":1,"actor_ids":actors,"initial_checkpoint_sha256":"e"*64,"teacher_checkpoint_sha256":"a"*64,
        "policy_spec_sha256":"b"*64,"motion_inventory_sha256":"c"*64,"teacher_beta":.75,"collection_plan_sha256":plan_sha}
    with store.transaction() as db:
        db.execute("INSERT INTO validation_collections VALUES (?,?,?,?,?,?)",(source,canonical_digest(owner),json.dumps(owner),"complete",0,0))
    path=root/"collection.json"
    immutable_json(path,collection)
    return store,path,collection


def test_duplicate_local_ids_are_distinct_and_real_same_scope_leakage_rejected(tmp_path):
    store,path,collection=fixture(tmp_path)
    with store.connection() as db, patch("numpy.load",side_effect=AssertionError("metadata only")):
        manifest,report=export_manifest(db,path,file_sha256(path),tmp_path/"manifest.json")
        provenance=training_provenance(db,collection["source_run_id"],1,collection["chunk_ids"],scoped_episodes=True)
    assert manifest["schema"].endswith(".v2")
    assert len(set(provenance["source_episode_ids"]))==2
    assert manifest["chunks"][0]["raw_source_episode_ids"]==manifest["chunks"][1]["raw_source_episode_ids"]
    report_path=tmp_path/"admission.json"
    immutable_json(report_path,report)
    assert validate_export_admission(report_path,file_sha256(report_path),tmp_path/"manifest.json",report["manifest_sha256"])==collection
    training={"purpose":"train","round_id":1,"chunk_sha256s":["f"*64],"source_episode_ids":[],"source_window_ids":["different-window"]}
    scope=manifest["chunks"][0]["source_scope"]
    training["source_episode_ids"]=[source_episode_identity({**scope,"actor_id":"different-actor"},"episode-000-00000000")]
    load_validation_manifest(tmp_path/"manifest.json",report["manifest_sha256"],training)
    training["source_episode_ids"]=manifest["chunks"][0]["source_episode_ids"]
    with pytest.raises(ValueError,match="overlap"):
        load_validation_manifest(tmp_path/"manifest.json",report["manifest_sha256"],training)


@pytest.mark.parametrize("mutation",["duplicate_scope","wrong_actor","wrong_lane","reused_seed","swapped_actor_chunks"])
def test_scoped_provenance_rejects_ambiguous_or_reused_native_identity(tmp_path,mutation):
    store,path,collection=fixture(tmp_path)
    if mutation=="duplicate_scope": collection["episode_assignments"].append(copy.deepcopy(collection["episode_assignments"][0]))
    if mutation=="wrong_actor": collection["episode_assignments"][0]["actor_id"]=collection["episode_assignments"][1]["actor_id"]
    if mutation=="wrong_lane":
        row=collection["episode_assignments"][0]
        reset_path=tmp_path/"wrong-reset.json"
        reset=json.loads(Path(row["native_reset_path"]).read_bytes());reset["lane"]=1
        immutable_json(reset_path,reset);row.update(native_reset_path=str(reset_path),native_reset_sha256=file_sha256(reset_path))
    if mutation=="reused_seed": collection["training_simulator_seeds"]=[100,201]
    if mutation=="swapped_actor_chunks":
        with store.transaction() as db:
            db.execute("UPDATE validation_collection_actors SET chunk_id=CASE WHEN chunk_id='chunk-0' THEN 'chunk-1' ELSE 'chunk-0' END")
    altered=tmp_path/"altered.json";immutable_json(altered,collection)
    with store.connection() as db,pytest.raises(ValueError):
        export_manifest(db,altered,file_sha256(altered),tmp_path/"bad-manifest.json")
