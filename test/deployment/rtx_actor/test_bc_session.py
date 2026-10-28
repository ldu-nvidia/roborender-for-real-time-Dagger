from test.deployment.source_paths import ACTOR_ROOT
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = ACTOR_ROOT
sys.path.insert(0, str(ROOT))
from isaac6_bc_session import execute_session, request_environment, validate_session
from run_bc_session import build_command, read_requests


def fixtures():
    request = {"schema": "roborender.bc-native-collection-request.v1", "run_id": "test",
        "cohort_id": "cohort-0", "split": "train", "process_seed": 10, "rank": 0,
        "global_env_offset": 0, "motions": [{"motion_id": "m0", "path": "/motions/m0.npz", "sha256": "a"*64}],
        "chunks_per_env": [1], "max_windows": 4, "output_dir": "/dagger-runtime/bc_isaac",
        "sampling_plan_sha256": "b"*64}
    second = copy.deepcopy(request)
    second.update(cohort_id="cohort-1", process_seed=11, output_dir="/dagger-runtime/cohorts/cohort-1/bc_isaac")
    raws = {f"/bc-session-requests/{i}.json": json.dumps(r).encode() for i, r in enumerate([request, second])}
    session = {"schema": "roborender.bc-native-session.v1", "session_id": "bc-session-leo-unit-test",
        "requests": [{"path": p, "sha256": hashlib.sha256(raw).hexdigest()} for p, raw in raws.items()]}
    return session, [request, second], raws


def test_cpu_only_import():
    code = f"import sys; sys.path.insert(0,{str(ROOT)!r}); import isaac6_bc_session; import run_bc_session; assert 'torch' not in sys.modules; assert not any(x.startswith('omni') for x in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_manifest_and_hash():
    session, requests, raw = fixtures()
    assert validate_session(session, read_bytes=raw.__getitem__)[1] == requests
    raw[session["requests"][0]["path"]] += b" "
    with pytest.raises(ValueError, match="bytes changed"):
        validate_session(session, read_bytes=raw.__getitem__)


@pytest.mark.parametrize("field,value", [("cohort_id", "cohort-0"), ("output_dir", "/dagger-runtime/bc_isaac"),
    ("run_id", "different"), ("rank", 1), ("sampling_plan_sha256", "c"*64)])
def test_reject_replay_or_foreign_cohort(field, value):
    session, requests, raw = fixtures()
    requests[1][field] = value
    path = session["requests"][1]["path"]
    raw[path] = json.dumps(requests[1]).encode()
    session["requests"][1]["sha256"] = hashlib.sha256(raw[path]).hexdigest()
    with pytest.raises(ValueError):
        validate_session(session, read_bytes=raw.__getitem__)


def test_phase_environment_restores(monkeypatch):
    monkeypatch.setenv("ROBORENDER_BC_COLLECTION_REQUEST", "prior")
    monkeypatch.delenv("ROBORENDER_BC_COLLECTION_REQUEST_SHA256", raising=False)
    import os
    with request_environment({"path": "new", "sha256": "a"*64}):
        assert os.environ["ROBORENDER_BC_COLLECTION_REQUEST"] == "new"
    assert os.environ["ROBORENDER_BC_COLLECTION_REQUEST"] == "prior"
    assert "ROBORENDER_BC_COLLECTION_REQUEST_SHA256" not in os.environ


def test_unqualified_variable_batch_rejected():
    session, requests, raw = fixtures()
    requests[1]["motions"].append({"motion_id": "m1", "path": "/motions/m1.npz", "sha256": "c" * 64})
    requests[1]["chunks_per_env"].append(1)
    path = session["requests"][1]["path"]
    raw[path] = json.dumps(requests[1]).encode()
    session["requests"][1]["sha256"] = hashlib.sha256(raw[path]).hexdigest()
    with pytest.raises(ValueError, match="fixed environment count"):
        validate_session(session, read_bytes=raw.__getitem__)


def test_complete_lifecycle_exact_order(tmp_path):
    session, requests, _ = fixtures()
    events = []
    def run(request):
        i = requests.index(request)
        events.append(("run", i))
        return {"status": "complete", "request_sha256": session["requests"][i]["sha256"],
            "optimizer_steps": 0, "selected_chunks": 1, "vector_steps": 162}
    path = tmp_path / "result.json"
    execute_session(session, requests, run_cohort=run,
        rebuild_scene=lambda request: events.append(("build", requests.index(request))),
        release_scene=lambda: events.append(("release", None)), kit_identity=lambda: 7,
        report=lambda *a, **k: None, output=path)
    assert events == [("run", 0), ("release", None), ("build", 1), ("run", 1)]
    result = json.loads(path.read_bytes())
    assert result["status"] == "complete"
    assert result["native_initializations"] == 1
    assert len(result["cohorts"]) == 2


def test_partial_cohort_stops_without_retry(tmp_path):
    session, requests, _ = fixtures()
    calls = []
    def run(request):
        calls.append(request)
        raise RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        execute_session(session, requests, run_cohort=run,
            rebuild_scene=lambda request: pytest.fail("must not rebuild after failure"),
            release_scene=lambda: pytest.fail("must not reuse failed native state"), kit_identity=lambda: 7,
            report=lambda *a, **k: None, output=tmp_path / "failed.json")
    assert len(calls) == 1
    result = json.loads((tmp_path / "failed.json").read_bytes())
    assert result["status"] == "failed" and result["cohorts"] == []


def test_kit_replacement_rejected(tmp_path):
    session, requests, _ = fixtures()
    ids = iter([1, 2])
    with pytest.raises(RuntimeError, match="Kit identity"):
        execute_session(session, requests, run_cohort=lambda r: pytest.fail("wrong Kit"),
            rebuild_scene=lambda r: None, release_scene=lambda: None,
            kit_identity=lambda: next(ids), report=lambda *a, **k: None, output=tmp_path / "failed.json")


def test_host_request_mount_cannot_escape(tmp_path):
    session, _, raw = fixtures()
    for path, data in raw.items():
        (tmp_path / Path(path).name).write_bytes(data)
    assert len(read_requests(session, tmp_path)[1]) == 2
    session["requests"][0]["path"] = "/elsewhere/0.json"
    with pytest.raises(ValueError, match="private mount"):
        read_requests(session, tmp_path)


def test_build_uses_one_native_session_no_online_contract(tmp_path):
    session, requests, _ = fixtures()
    image = "sha256:" + "c"*64
    def base(name, runtime, overlay, code, num_envs, **kwargs):
        assert kwargs["mode"] == "bc-collect"
        return ["docker", "run", "--name", name, "--env", "PYTHONPATH=/old", image,
                "python", "+exp=old", "trainer._target_=old", "num_envs=1"]
    name, cmd = build_command(session=session, requests=requests, session_path=tmp_path / "session.json",
        session_sha256="f"*64, requests_dir=tmp_path / "requests", runtime=tmp_path / "runtime",
        overlay=tmp_path / "overlay", source=tmp_path / "source", physical_gpu=0, memory_gib=32,
        base_builder=base, image_id=image)
    assert name == "bc-cohort-leo-unit-test"
    assert "trainer._target_=isaac6_bc_session.Isaac6BCSessionTrainer" in cmd
    assert "ROBORENDER_BC_COLLECTION_REQUEST=/bc-session-requests/0.json" in cmd
    assert "ROBORENDER_BC_SESSION_SHA256=" + "f"*64 in cmd
    assert not any("H200" in item for item in cmd)
