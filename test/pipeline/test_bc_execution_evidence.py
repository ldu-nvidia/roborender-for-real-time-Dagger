"""CPU-only receipt producer tests; all execution documents are synthetic."""
from test.deployment.source_paths import PIPELINE_ROOT
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline.deployment.rtx_actor.bc_execution_evidence import (
    build_session_receipts, canonical, collector_identity, digest, publish_completed_session,
    RUNTIME_PATHS,
)


def fixture():
    rows = [dict(path=path, sha256="a" * 64) for path in RUNTIME_PATHS]
    source = dict(files=rows, source_sha256=digest(canonical(rows)))
    port = dict(schema="roborender.isaac6.migration.v1", isaac_sim="6.0.0", isaaclab_commit="b" * 40)
    identity = collector_identity(source, port, "sha256:" + "c" * 64)
    request = dict(cohort_id="test-0", motions=[{}], split="test")
    request_sha = digest(canonical(request))
    session = dict(session_id="synthetic-session", requests=[dict(path="/request.json", sha256=request_sha)])
    launch = dict(collector_execution=identity, session_sha256="d" * 64,
        session_canonical_sha256=digest(canonical(session)), source_manifest_raw_sha256="e" * 64,
        port_manifest_raw_sha256="f" * 64, request_identities={request_sha: digest(canonical(request))})
    result = dict(session_id=session["session_id"], status="complete", native_initializations=1,
        scene_initializations=1, failure=None, cohorts=[dict(cohort_id="test-0", request_sha256=request_sha,
        vector_steps=162, output_dir="/dagger-runtime/bc_isaac")])
    cohort = dict(status="complete", request=request, request_sha256=request_sha,
        teacher_only=True, native_payload_writer_unchanged=True, optimizer_steps=0, vector_steps=162)
    return dict(launch=launch, launch_raw_sha256="1" * 64, session=session, result=result,
        result_raw_sha256="2" * 64, host_result=dict(passed=True, exit_code=0, failure=None),
        host_result_raw_sha256="3" * 64,
        cohort_envelopes=[dict(document=cohort, raw_sha256="4" * 64)])


def test_producer_reuses_complete_native_lifecycle_and_preserves_raw_vs_canonical():
    args = fixture()
    session, rows = build_session_receipts(**args)
    assert session["session_sha256"] != session["session_canonical_sha256"]
    assert rows[0]["session_execution_canonical_sha256"] == digest(canonical(session))
    assert rows[0]["actual_num_envs"] == 1 and rows[0]["actual_vector_steps"] == 162


@pytest.mark.parametrize("formatting", ["newline", "pretty"])
def test_native_raw_request_hash_is_not_assumed_canonical(formatting):
    args = fixture()
    doc = args["cohort_envelopes"][0]["document"]
    raw = canonical(doc["request"]) + b"\n" if formatting == "newline" else json.dumps(doc["request"], indent=2).encode()
    raw_sha = digest(raw)
    canonical_sha = digest(canonical(doc["request"]))
    assert raw_sha != canonical_sha
    doc["request_sha256"] = raw_sha
    args["session"]["requests"][0]["sha256"] = raw_sha
    args["launch"]["session_canonical_sha256"] = digest(canonical(args["session"]))
    args["launch"]["request_identities"] = {raw_sha: canonical_sha}
    args["result"]["cohorts"][0]["request_sha256"] = raw_sha
    _, receipts = build_session_receipts(**args)
    assert receipts[0]["request_sha256"] == raw_sha
    assert receipts[0]["request_canonical_sha256"] == canonical_sha


@pytest.mark.parametrize("mutation", ["failure", "reinit", "missing_scene", "request", "writer", "optimizer"])
def test_no_false_receipt_for_partial_or_changed_native_work(mutation):
    args = fixture()
    if mutation == "failure": args["host_result"]["passed"] = False
    elif mutation == "reinit": args["result"]["native_initializations"] = 2
    elif mutation == "missing_scene": args["result"]["scene_initializations"] = 0
    elif mutation == "request": args["cohort_envelopes"][0]["document"]["request"]["split"] = "train"
    elif mutation == "writer": args["cohort_envelopes"][0]["document"]["native_payload_writer_unchanged"] = False
    else: args["cohort_envelopes"][0]["document"]["optimizer_steps"] = 1
    with pytest.raises(ValueError): build_session_receipts(**args)


def test_metadata_only_publication_without_payload_reads(tmp_path, monkeypatch):
    args = fixture()
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(value))
    put(tmp_path / "launch.json", args["launch"])
    put(tmp_path / "bc-session-result.json", args["result"])
    put(tmp_path / "host-result.json", args["host_result"])
    put(tmp_path / "bc_isaac/cohort-manifest.json", args["cohort_envelopes"][0]["document"])
    original = Path.read_bytes
    def read(path):
        assert path.suffix == ".json"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    paths = publish_completed_session(tmp_path, args["session"], args["host_result"], put)
    assert Path(paths["session_execution"]).is_file()
    assert len(paths["cohort_execution_receipts"]) == 1


def test_collector_identity_rejects_drifted_source_and_mutable_image():
    rows = [dict(path=path, sha256="a" * 64) for path in RUNTIME_PATHS]
    source = dict(files=rows, source_sha256=digest(canonical(rows)))
    port = dict(schema="roborender.isaac6.migration.v1", isaac_sim="6.0.0", isaaclab_commit="b" * 40)
    with pytest.raises(ValueError): collector_identity(source, port, "latest")
    source["files"][0]["sha256"] = "c" * 64
    with pytest.raises(ValueError): collector_identity(source, port, "sha256:" + "d" * 64)


def test_modified_session_launcher_import_is_cpu_only():
    expected = PIPELINE_ROOT / "deployment/rtx_actor/run_bc_session.py"
    code = "from pipeline.deployment.rtx_actor import run_bc_session as m; import sys; " \
           f"assert m.__file__ == {str(expected)!r}; " \
           "assert 'torch' not in sys.modules; assert not any(x.startswith('omni') for x in sys.modules); " \
           "assert callable(m.publish_completed_session)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_runtime_contract_ignores_readme_but_rejects_native_guard_or_port_drift():
    rows = [dict(path=path, sha256="a" * 64) for path in RUNTIME_PATHS]
    rows.append(dict(path="README.md", sha256="b" * 64))
    port = dict(schema="roborender.isaac6.migration.v1", isaac_sim="6.0.0", isaaclab_commit="c" * 40)
    overlays = {"native_pre_save_guard": "d" * 64, "host_runtime_guard": "e" * 64}
    def identify(items, extra=overlays, native_port=port):
        return collector_identity(dict(files=items, source_sha256=digest(canonical(items))),
                                  native_port, "sha256:" + "f" * 64, runtime_overlays=extra)
    before = identify(rows)
    rows[-1]["sha256"] = "0" * 64
    docs = identify(rows)
    assert before["source_release_sha256"] != docs["source_release_sha256"]
    assert before["runtime_contract_sha256"] == docs["runtime_contract_sha256"]
    rows[0]["sha256"] = "1" * 64
    assert identify(rows)["runtime_contract_sha256"] != before["runtime_contract_sha256"]
    assert identify(rows, {**overlays, "native_pre_save_guard": "2" * 64})["runtime_contract_sha256"] != identify(rows)["runtime_contract_sha256"]
    assert identify(rows, native_port={**port, "isaac_sim": "different"})["runtime_contract_sha256"] != identify(rows)["runtime_contract_sha256"]
