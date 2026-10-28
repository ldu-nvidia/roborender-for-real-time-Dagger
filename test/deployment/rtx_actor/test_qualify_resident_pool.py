"""Static native warm-probe staging gates, not GPU qualification evidence."""
from test.deployment.source_paths import ACTOR_ROOT
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline.deployment.rtx_actor.qualify_resident_pool import validate_spec


def fixture():
    base = "/home/ldu/roborender"
    run = "standing-isaac-leo-mse-b4-unit"
    return {"schema": "roborender.resident-qualification-request.v1", "run_id": run,
        "source_root": base + "/runtime/source-releases/unit/frozen",
        "source_manifest": base + "/runtime/source-releases/unit/source-manifest.json",
        "source_sha256": "a" * 64, "root": base + "/runtime/standing-isaac/" + run,
        "old_round_root": base + "/runtime/old-round", "cold_evaluation_root": base + "/runtime/cold-test",
        "panel_sha256": "b" * 64, "checkpoints": {"bc": "c" * 64, "dagger": "d" * 64},
        "python": base + "/.venv/bin/python"}


def test_dry_run_never_imports_isaac_or_starts_any_process(tmp_path):
    spec = fixture()
    path = tmp_path / "spec.json"
    raw = json.dumps(spec).encode()
    path.write_bytes(raw)
    script = (ACTOR_ROOT / "qualify_resident_pool.py")
    result = subprocess.run([sys.executable, str(script), "--spec", str(path),
        "--spec-sha256", hashlib.sha256(raw).hexdigest()], check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == {"execute": False, "source_sha256": "a" * 64,
        "run_id": spec["run_id"], "gpu_count": 2, "h200_count": 0, "required_host_headroom_gib": 50}


@pytest.mark.parametrize("change", [{"root": "/home/ldu"}, {"run_id": "existing"},
    {"source_root": "/working/mutable"}, {"checkpoints": {"bc": "c" * 64}}, {"panel_sha256": "x"}])
def test_unbound_or_broad_runtime_targets_rejected(change):
    with pytest.raises(ValueError):
        validate_spec({**fixture(), **change})


def test_qualifier_preserves_old_native_probes_and_conservative_host_admission():
    source = (ACTOR_ROOT / "qualify_resident_pool.py").read_text()
    assert "admit_future_caps(current['MemAvailable'] * 1024, samples, target_caps)" in source
    assert "desired+8*1024**2" not in source
    assert "'profiles':['training_b6','evaluation_b5','training_b6']" in source
    assert "'cold_initial_states_equal':True" in source
    assert "'live_collection_learner_validated':False" in source
    assert "spec['run_id']" in source and "20261014-v9" not in source
