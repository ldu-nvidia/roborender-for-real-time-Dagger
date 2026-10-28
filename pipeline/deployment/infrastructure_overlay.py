"""Explicit control-only release compatibility; never admits algorithm changes.

The audited provenance/reporting correction below is admitted only as an exact
old/new source pair, not as a blanket exemption for learner or loss code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.deployment.source_snapshot import canonical, verify

SCHEMA = "roborender.control-only-release.v1"
# Historical source-admission path names are evidence, not Python imports.
# Preserve them across test relocation; a source-removing cleanup requires a
# fresh release and must not be admitted as an old control-only hotpatch.
CONTROL_ONLY = frozenset({
    "README.md", "docs/END_TO_END_PIPELINE_CONTRACT.md",
    "pipeline/dagger/beta_campaign.py",
    "pipeline/deployment/beta_operations.py", "pipeline/deployment/beta_site.py",
    "pipeline/deployment/resident_campaign_site.py", "pipeline/deployment/round_boundary_recovery.py",
    "pipeline/deployment/test_round_boundary_recovery.py",
    "pipeline/deployment/infrastructure_overlay.py", "pipeline/deployment/test_infrastructure_overlay.py",
    "pipeline/deployment/recovering_campaign.py", "pipeline/deployment/test_recovering_campaign.py",
    "pipeline/deployment/checkpoint_transfer.py", "pipeline/deployment/test_checkpoint_transfer.py",
    "pipeline/deployment/pending_evaluation_recovery.py", "pipeline/deployment/test_pending_evaluation_recovery.py",
    "pipeline/deployment/h200/beta_pod.py", "pipeline/deployment/h200/beta_pod_entrypoint.sh",
    "pipeline/deployment/test_beta_infra.py", "pipeline/deployment/test_allocation_wait.py",
    "pipeline/deployment/ROUND_BOUNDARY_RECOVERY.md", "pipeline/deployment/STANDING_ISAAC.md",
    "pipeline/deployment/END_TO_END_SMOKE.md",
    "pipeline/tests/test_beta_round_automation.py", "pipeline/tests/test_dagger_metric_contract.py",
    "pipeline/tests/test_dagger_mse_smoke_topology.py", "pipeline/tests/test_dagger_scoped_provenance.py",
    "pipeline/tests/test_dagger_mse_protocol.py", "pipeline/tests/test_control_recovery.py",
    "pipeline/dagger/coverage_queue.py", "pipeline/deployment/retained_campaign_site.py",
    "pipeline/deployment/production_pool_control.py", "pipeline/tests/test_production_pool_control.py",
    "pipeline/deployment/rtx_actor/production_guard_access.py",
    "pipeline/deployment/rtx_actor/recover_production_resident_pool.py",
    "pipeline/tests/test_production_guard_access.py", "pipeline/tests/test_dagger_phase_resource_guard.py",
    "pipeline/tests/test_retained_native_data_route.py",
    "docs/LEDGER_DURABILITY.md", "pipeline/tests/test_ledger_owner.py",
    "pipeline/deployment/retained_service_process.py", "pipeline/tests/test_mode_controller_lifetime.py",
})

REVIEWED_METADATA_DELTAS = {
    # Owner-bound, query-only reader: O_RDWR avoids virtiofs F_GETLK/EINVAL.
    # Only this audited pair is allowed; writer and algorithm changes are not.
    "pipeline/dagger/services/ledger_owner.py": ("0b8935354244bc4572e868f20ebd1ff7aaa2d61ffee6befea90837e5f1aa8a34", "6092451ec8aafec378e411f5c148807d30e13e37874cc01cacf45aaa4dcbc10b"),
    # Exact permission-only writer change; running native source stays frozen.
    "pipeline/dagger/rtx/code/src/dagger_rtx/phase_resource_guard.py": ("c61a678ccd3dd0afae52539e25504e6f5b519966e53c390b6f0fc4a6d5912b43", "57aa945e69c1668bc99d2901810f28d3a8e6d7c4288dc2d219826fdf960139a2"),
    "pipeline/deployment/rtx_actor/production_resident_pool.py": ("47cf84a58ea4df895c217ee02ff560f56812c2a77fa51f206d363be998ee26bd", "3d445dea8ba41c7748b882754fe0f2ab1469f8815f1ec20f226732217d334cac"),
    "pipeline/dagger/services/client.py": ("79689b8de0b9c7d78af1a771b8e0a60d2ceb8fd94e88187037fabccf7b32bcec", "3293847292e428c6a05491aa4262e5c9e099f044fe16f4c99ec2335cfef63f0f"),
    "pipeline/dagger/export_heldout_manifest.py": ("206a581a263765e10a385b34f7898dc5a57e575b08ef826599b00d9638d3aa1a", "22aa1d154f3b9fc4eba6a5a9661e0ccf7544e131871bb22032b229143ea98bb5"),
    "pipeline/dagger/mse_campaign.py": ("6ae9feff4fff9c6ecd4e9d2534b552bce5e1ea3b28977b41e0363d1c5a1fa721", "d855a971708921cb1fa85ed5a9325910603e69a07b2197358b53b3e3d91bd6e1"),
    "pipeline/dagger/production_controller.py": ("ad553832bd09179874fe6b609495896e1552bc809cd1a0a031bf76b6d09ab878", "aa62e942c42b8538dad2866aef1f6fc379455acae7e1b017b7f4182c494c5d6a"),
    "pipeline/dagger/services/heldout_mse.py": ("af55a6710d130e610bc2534b98000ce68315a3c0ffba25a47b0badfcd68ea9fa", "159ef11fc2a588c59ea060d63bb174add04094d0bb92f4ae3358b7c2b26680bd"),
    "pipeline/dagger/services/metric_contract.py": (None, "82a27f41fbdf9d06919039e9055db7744661ff17149da351c65261b5cf463761"),
    "pipeline/dagger/services/mse_protocol.py": ("f0a35fc6357228a046b69d306f800a53bf086ffbb1ee2cd1dde08320f824af78", "d69978ac5120fd125ff35b7f2647957b79261fa63981692a8383c7fc5f8d8522"),
    "pipeline/dagger/services/raven_learner_worker.py": ("87545c7def6729fe50972062980dd7a2e7fc4ace5d9b04c5c9b3f4139d15b456", "027c974f9fc5e8545407e889456477b80ae6b6b248e88f93f0d45ea0ac218ace"),
    "pipeline/deployment/resident_heldout_collection.py": ("ce245660f69f8f50fecb0aa4a05160407a811beb18ba1e80fc674ccc73a075a0", "c2d0d6dee3b7c3af96e9731ff4f4c559d78db9414f085106c203c32aba2b4e66"),
}


def changed_control_paths(native, management):
    for manifest in (native, management):
        if (manifest.get("schema") != "roborender.source-snapshot.v1"
                or manifest.get("file_count") != len(manifest["files"])
                or hashlib.sha256(canonical(manifest["files"])).hexdigest() != manifest["source_sha256"]):
            raise RuntimeError("invalid release manifest")
    old = {r["path"]:r for r in native["files"]}
    new = {r["path"]:r for r in management["files"]}
    if len(old) != len(native["files"]) or len(new) != len(management["files"]):
        raise RuntimeError("duplicate source path")
    changed = sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))
    def admitted(path):
        return path in CONTROL_ONLY or (
            path in REVIEWED_METADATA_DELTAS
            and (old.get(path, {}).get("sha256"), new.get(path, {}).get("sha256"))
                == REVIEWED_METADATA_DELTAS[path]
            and new[path]["executable"] == old.get(path, {}).get("executable", False))
    if not changed or any(not admitted(p) for p in changed) or old.keys()-new.keys():
        raise RuntimeError("release changes native/renderer/learner/scientific code or removes source")
    return changed


def verify_overlay(root, native, path, expected_sha):
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError("control-only release admission changed")
    admission = json.loads(raw)
    if set(admission) != {"schema","native_source_sha256","management_manifest","management_source_sha256","changed_paths"} or admission["schema"] != SCHEMA:
        raise ValueError("explicit control-only release admission required")
    management = json.loads(Path(admission["management_manifest"]).read_bytes())
    changed = changed_control_paths(native, management)
    if (admission["native_source_sha256"] != native["source_sha256"]
            or admission["management_source_sha256"] != management["source_sha256"]
            or admission["changed_paths"] != changed):
        raise RuntimeError("control/native release binding differs")
    verify(Path(root), management)
    return {"management_source_sha256": management["source_sha256"],
            "native_source_sha256": native["source_sha256"], "control_only_release_sha256": expected_sha}
