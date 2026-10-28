"""A completed native phase can leave its attested process resident."""
import re


def native_quiescent(receipt, *, source_sha256=None):
    if receipt.get("owned_actors_stopped") is True:
        return True  # Historical cold launcher completion semantics.
    barrier = receipt.get("native_phase_barrier")
    if not isinstance(barrier, dict) or set(barrier) != {"schema", "workers"} or barrier["schema"] != "roborender.isaac-phase-barrier.v1":
        return False
    rows = barrier["workers"]
    if not isinstance(rows, list) or len(rows) != 2 or {r.get("physical_gpu") for r in rows} != {0, 1}:
        return False
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_GPU_UUIDS
    for row in rows:
        if (set(row) != {"physical_gpu", "gpu_uuid", "worker_instance_id", "source_sha256", "phase"}
                or row.get("gpu_uuid") != LEO_GPU_UUIDS[row["physical_gpu"]] or row.get("phase") != "IDLE"
                or not re.fullmatch(r"[a-f0-9]{32}", row.get("worker_instance_id", ""))
                or not re.fullmatch(r"[a-f0-9]{64}", row.get("source_sha256", ""))):
            return False
    return (len({r["source_sha256"] for r in rows}) == 1
            and (source_sha256 is None or rows[0]["source_sha256"] == source_sha256))
