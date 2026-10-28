from test.deployment.source_paths import DEPLOYMENT_ROOT
from copy import deepcopy
import pytest
from pipeline.deployment.native_phase_barrier import native_quiescent
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_GPU_UUIDS


def barrier():
    return {"owned_actors_stopped": False, "native_phase_barrier": {
        "schema": "roborender.isaac-phase-barrier.v1", "workers": [
            {"physical_gpu": gpu, "gpu_uuid": uuid, "worker_instance_id": str(gpu+1)*32,
             "source_sha256": "a"*64, "phase": "IDLE"} for gpu, uuid in LEO_GPU_UUIDS.items()]}}


def test_legacy_and_resident_phase_end():
    assert native_quiescent({"owned_actors_stopped": True})
    assert native_quiescent(barrier(), source_sha256="a"*64)
    assert not native_quiescent(barrier(), source_sha256="b"*64)
    assert not native_quiescent({"owned_actors_stopped": False})


@pytest.mark.parametrize("field,value", [("phase", "RUNNING"), ("phase", "QUEUED"),
    ("phase", "CRASH_UNCERTAIN"), ("gpu_uuid", "GPU-wrong"), ("worker_instance_id", ""),
    ("source_sha256", "b"*64), ("physical_gpu", 2)])
def test_active_wrong_or_uncertain_owner_not_a_barrier(field, value):
    receipt = barrier()
    receipt["native_phase_barrier"]["workers"][0][field] = value
    assert not native_quiescent(receipt)


def test_only_one_worker_not_a_barrier():
    receipt = barrier()
    receipt["native_phase_barrier"]["workers"].pop()
    assert not native_quiescent(receipt)


def test_resident_campaign_direct_entrypoint_imports_repository(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    entry = (DEPLOYMENT_ROOT / "resident_campaign_site.py")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(entry), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--resident-pool" in result.stdout
