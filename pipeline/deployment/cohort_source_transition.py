"""Inert admission for the reviewed round1 transport-source transition.

Historical receipts keep their historical source and plan bindings. New native
workers must report the distinct execution source below; this certificate never
relabels them, restores simulator state, updates a ledger, or launches work.
The controller must separately admit the fresh runtime and import continuation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from pipeline.bc_training.handoff_common import bound, canonical, cli_binding, publish
from pipeline.dagger.coverage_production import digest, load_plan
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
    actor_ids, environments_per_actor, is_mse_production, validate_config,
)
from pipeline.deployment.cohort_boundary_recovery import RECEIPT_SCHEMA
from pipeline.deployment.source_snapshot import file_record, safe_relative


SCHEMA = "roborender.dagger-cohort-transport-source-transition.v1"
HISTORICAL_SOURCE_SHA256 = "5f135300ef20ff5e7231c5d9c9c3ed63ecb52def378b4c2356909bbc5880cdbb"
REVIEWED_TRANSPORT_PAIRS = {
    "pipeline/dagger/rtx/code/src/dagger_rtx/connection_health.py": (
        "e93c7ac1ab16581de9614d6fb5f4fd31aba86700f6fba961124300902850e657",
        "ac8a5701d08e3ab5cdbe9aac2fa75e044d8c660b13b63447f466e774ba047973",
    ),
    "pipeline/dagger/rtx/code/src/dagger_rtx/native_service_trainer.py": (
        "45c1ab2374e1abcffc9f8ab825dc37d1d1245076febffc22c75a497370a95175",
        "31a26f9dacfc122ad60b48ffc84b25af5400f86487b26b9b64fa51b0435ad1cc",
    ),
}
# These two orchestration hooks select separately attested historical receipts
# and execution receipts. They do not change native/learner/scientific code;
# exact pairs are required rather than allowing arbitrary control-file edits.
REVIEWED_CONTROL_PAIRS = {
    "pipeline/deployment/resident_production_collection.py": (
        "913044cc3e0e4fce1ff34aa9ce3d11bf2be8a8162327adf30d471058cb519b4c",
        "299ccdfd3d91c6c9285488cf78088bc595a0cdb84f37b7d283e32a72cf1dcc82",
    ),
    "pipeline/deployment/retained_campaign_site.py": (
        "9cd5aae01901832deaaa20b44cca670289ca10d5fa3875621966c24da6d5a215",
        "20e058a477442ae099e03a780e211b34c126a4caa4d2f11a8ae9ac2d117adca9",
    ),
}
# These files are offline recovery tools or tests; none is an alternative native,
# renderer, learner, policy, reset, or resource-guard implementation.
# These original test path strings bind historical source transitions. They are
# not runtime imports and intentionally do not follow current test relocation.
# _source_delta continues to reject source removal rather than waiving a new
# release's identity requirements for cosmetic cleanup.
RECOVERY_PATHS = frozenset({
    "pipeline/deployment/cohort_boundary_recovery.py",
    "pipeline/deployment/cohort_boundary_runtime_rebind.py",
    "pipeline/deployment/cohort_source_transition.py",
    "pipeline/deployment/cohort_continuation.py",
    "pipeline/deployment/test_cohort_boundary_recovery.py",
    "pipeline/deployment/test_cohort_boundary_runtime_rebind.py",
    "pipeline/deployment/test_cohort_source_transition.py",
    "pipeline/deployment/test_cohort_continuation.py",
    "pipeline/dagger/rtx/tests/test_connection_health.py",
    "pipeline/dagger/rtx/tests/test_window_runner.py",
})
HEX64 = re.compile(r"[0-9a-f]{64}")


def require(value, message):
    if not value:
        raise ValueError(message)


def _source_manifest(ref, root):
    """Verify every file and reject undeclared files in an isolated release."""
    root = Path(root).absolute()
    require(root.is_dir() and not root.is_symlink(), "regular isolated source root required")
    value = bound(ref)
    require(isinstance(value, dict)
            and set(value) == {"schema", "source_sha256", "file_count", "files"}
            and value["schema"] == "roborender.source-snapshot.v1",
            "invalid source manifest")
    rows = value["files"]
    require(isinstance(rows, list) and rows and type(value["file_count"]) is int
            and value["file_count"] == len(rows)
            and value["source_sha256"] == hashlib.sha256(canonical(rows)).hexdigest(),
            "source manifest digest/count differs")
    for row in rows:
        require(isinstance(row, dict)
                and set(row) == {"path", "sha256", "bytes", "executable"}
                and isinstance(row["path"], str)
                and isinstance(row["sha256"], str) and HEX64.fullmatch(row["sha256"])
                and type(row["bytes"]) is int and row["bytes"] >= 0
                and type(row["executable"]) is bool,
                "invalid per-file source record")
        safe_relative(row["path"])
        require(file_record(root, row["path"]) == row, "source file bytes/mode differ from manifest")
    names = [row["path"] for row in rows]
    require(names == sorted(set(names)), "source paths must be sorted and unique")
    expected = set(names)
    for path in root.rglob("*"):
        require(not path.is_symlink(), "source tree contains a symlink")
        relative = path.relative_to(root)
        if path.is_file() and "__pycache__" not in relative.parts:
            require(relative.as_posix() in expected, "source tree contains an undeclared file")
    return value, {row["path"]: row for row in rows}


def _source_delta(old, new):
    require(old.keys() <= new.keys(), "source transition cannot remove historical files")
    changed = sorted(path for path in new if old.get(path) != new[path])
    reviewed_pairs = {**REVIEWED_TRANSPORT_PAIRS, **REVIEWED_CONTROL_PAIRS}
    require(set(reviewed_pairs) <= set(changed),
            "all exact reviewed transport/control changes are required")
    for path in changed:
        before, after = old.get(path), new[path]
        if path in reviewed_pairs:
            require(before is not None
                    and (before["sha256"], after["sha256"]) == reviewed_pairs[path]
                    and before["executable"] == after["executable"],
                    "transport/control change is outside the exact reviewed hash pair")
        else:
            documentation = Path(path).suffix == ".md" and (path.startswith("docs/") or path == "README.md")
            require(path in RECOVERY_PATHS or documentation,
                    "source transition changes unreviewed runtime/scientific code")
            require(after["executable"] == (before["executable"] if before else False),
                    "recovery/documentation executable mode changed")
    return changed


def _configs_and_plans(old_ref, new_ref, historical_sha, execution_sha):
    old = validate_config(bound(old_ref), runtime=True)
    new = validate_config(bound(new_ref), runtime=True)
    require(is_mse_production(old) and is_mse_production(new)
            and old["round_id"] == new["round_id"] == 1
            and old["source_sha256"] == historical_sha and new["source_sha256"] == execution_sha
            and environments_per_actor(old) == environments_per_actor(new) == 16
            and len(actor_ids(old)) == len(actor_ids(new)) == 32,
            "source transition is restricted to the exact round1 B16 pair")
    expected = deepcopy(old)
    expected["source_sha256"] = execution_sha
    expected["coverage_plan"] = new["coverage_plan"]
    expected["h200"]["gateway_pids"] = new["h200"]["gateway_pids"]
    require(new == expected,
            "round config may change only source, coverage-plan binding, and gateway PIDs")
    require(old_ref["path"] != new_ref["path"]
            and old["coverage_plan"]["path"] != new["coverage_plan"]["path"],
            "new config and coverage plan require distinct immutable paths")
    old_plan, new_plan = load_plan(old["coverage_plan"]), load_plan(new["coverage_plan"])
    expected_plan = deepcopy(old_plan)
    expected_plan["source_sha256"] = execution_sha
    require(old_plan["source_sha256"] == historical_sha and new_plan == expected_plan
            and old_plan["run_id"] == old["run_id"]
            and old_plan["topology"]["environments_per_actor"] == 16
            and old_plan["topology"]["total_environments"] == 32,
            "coverage plan may change only its source identity")
    return old, new, old_plan, new_plan


def _prefix_receipt(ref, old):
    receipt = bound(ref)
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("import_complete") is True
            and receipt.get("run_id") == old["run_id"] and receipt.get("round_id") == 1
            and receipt.get("source_sha256") == old["source_sha256"]
            and receipt.get("completed_updates") == 36 and receipt.get("completed_cohorts") == 18
            and receipt.get("next_iteration") == 37 and receipt.get("next_cohort_index") == 18,
            "source transition requires the verified 36-update/18-cohort import")
    for key in ("source_database_unchanged", "old_namespace_preserved", "fresh_native_runtime_required",
                "allocation_retained"):
        require(receipt.get(key) is True, "import receipt does not preserve recovery/retention requirements")
    for key in ("native_replay_allowed", "native_work_started", "compute_release_requested"):
        require(receipt.get(key) is False, "import receipt permits replay, native work, or compute release")
    parent = receipt.get("parent_checkpoint_sha256")
    require(isinstance(parent, str) and HEX64.fullmatch(parent)
            and type(receipt.get("parent_optimizer_global_step")) is int
            and receipt["parent_optimizer_global_step"] >= 36
            and receipt.get("parent_learning_rate") == old["student"]["learning_rate"],
            "imported policy/AdamW parent is not bound")
    materialization = receipt.get("checkpoint_materialization", {})
    require(materialization.get("checkpoint_count") == 37
            and materialization.get("all_hashes_verified") is True
            and materialization.get("all_paths_rewritten") is True,
            "import lacks the complete verified checkpoint tree")
    require(receipt.get("mse_history_iterations") == [0, 20],
            "import lacks the preserved fixed-validation selection history")
    for key in ("fixed_heldout_manifest", "fixed_heldout_admission"):
        bound(receipt[key])
    return receipt


def build_admission(*, historical_root, execution_root, historical_manifest, execution_manifest,
                    historical_config, execution_config, import_receipt):
    """Validate immutable artifacts; perform no ledger or runtime mutation."""
    historical_root, execution_root = Path(historical_root).absolute(), Path(execution_root).absolute()
    require(historical_root != execution_root
            and not historical_root.is_relative_to(execution_root)
            and not execution_root.is_relative_to(historical_root),
            "historical and execution source roots must be separate")
    old_manifest, old_files = _source_manifest(historical_manifest, historical_root)
    new_manifest, new_files = _source_manifest(execution_manifest, execution_root)
    historical_sha, execution_sha = old_manifest["source_sha256"], new_manifest["source_sha256"]
    require(historical_sha == HISTORICAL_SOURCE_SHA256 and execution_sha != historical_sha,
            "historical source differs or new native code was relabeled as historical")
    changed = _source_delta(old_files, new_files)
    old, new, old_plan, new_plan = _configs_and_plans(
        historical_config, execution_config, historical_sha, execution_sha,
    )
    receipt = _prefix_receipt(import_receipt, old)
    # Binding the next cohort avoids confusing the global prefix ordinal18
    # with the epoch-local cohort index8.
    cohorts = [(epoch["coverage_epoch"], cohort)
               for epoch in old_plan["rounds"][0]["epochs"] for cohort in epoch["cohorts"]]
    epoch, cohort = cohorts[18]
    require(epoch == 2 and cohort["cohort_index"] == 8 and cohort["start_iteration"] == 36,
            "next cohort differs from the exact iteration37 recovery boundary")
    return {
        "schema": SCHEMA, "run_id": old["run_id"], "round_id": 1,
        "historical_source_sha256": historical_sha, "execution_source_sha256": execution_sha,
        "historical_source_root": str(historical_root), "execution_source_root": str(execution_root),
        "historical_manifest": historical_manifest, "execution_manifest": execution_manifest,
        "historical_config": historical_config, "execution_config": execution_config,
        "historical_config_sha256": digest(old), "execution_config_sha256": digest(new),
        "historical_coverage_plan": old["coverage_plan"], "execution_coverage_plan": new["coverage_plan"],
        "historical_plan_sha256": digest(old_plan), "execution_plan_sha256": digest(new_plan),
        "changed_paths": changed,
        "reviewed_transport_hash_pairs": {path: list(pair) for path, pair in REVIEWED_TRANSPORT_PAIRS.items()},
        "reviewed_control_hash_pairs": {path: list(pair) for path, pair in REVIEWED_CONTROL_PAIRS.items()},
        "import_receipt": import_receipt,
        "completed_updates": 36, "completed_cohorts": 18, "next_iteration": 37,
        "next_coverage_epoch": epoch, "next_cohort_index": cohort["cohort_index"],
        "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"],
        "parent_optimizer_global_step": receipt["parent_optimizer_global_step"],
        "fixed_heldout_manifest": receipt["fixed_heldout_manifest"],
        "fixed_heldout_admission": receipt["fixed_heldout_admission"],
        "mse_history_iterations": receipt["mse_history_iterations"],
        "historical_artifacts_preserved": True, "source_files_verified": True,
        "fresh_native_runtime_required": True, "fresh_native_qualification_required": True,
        "native_replay_allowed": False, "ledger_mutation_performed": False,
        "native_work_started": False, "execution_authorized": False,
        "allocation_retained": True, "compute_release_requested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("historical-root", "execution-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("historical-manifest", "execution-manifest", "historical-config",
                 "execution-config", "import-receipt"):
        parser.add_argument("--" + name, nargs=2, metavar=("PATH", "RAW_SHA256"), required=True)
    parser.add_argument("--output", type=Path, help="publish only this inert admission, without overwriting")
    args = parser.parse_args()
    value = build_admission(
        historical_root=args.historical_root, execution_root=args.execution_root,
        **{name: cli_binding(getattr(args, name)) for name in (
            "historical_manifest", "execution_manifest", "historical_config", "execution_config", "import_receipt")},
    )
    if args.output:
        print(json.dumps({"admission": publish(args.output, value), "execution_authorized": False}, sort_keys=True))
    else:
        print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
