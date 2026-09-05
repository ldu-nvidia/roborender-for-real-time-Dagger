#!/usr/bin/env python3
"""Verify gen-000 BC collection/splits and publish its immutable completion mark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from round_dagger.artifacts import (  # noqa: E402
    ArtifactError,
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.conditioning_split import (  # noqa: E402
    CANONICAL_CAMPAIGN_ROOT,
    ConditioningSplitError,
    EXPECTED_BC_GROUP_COUNTS,
    EXPECTED_BC_NOMINAL_TRANSITIONS,
    EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS,
    EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV,
    PRE_RENDER_AUDIT_SCHEMA,
    SPLITS,
    audit_pre_render_replay_root,
    build_postprocess_provenance,
    generation_lock,
    verify_exact_bc_collection_source,
    verify_split,
)

COMPLETION_SCHEMA = "fast_roborender.round_dagger.bc_collection_complete.v4"
EXPECTED_GROUPS = dict(EXPECTED_BC_GROUP_COUNTS)
EXPECTED_TRANSITIONS = {
    split: count * EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV
    for split, count in EXPECTED_GROUPS.items()
}
_COMPLETION_KEYS = {
    "schema",
    "campaign_id",
    "generation_id",
    "status",
    "collection",
    "split",
    "training_projection_input",
    "postprocess_provenance",
    "rendering_audit",
    "learner_replay_ready",
    "rendering_started",
    "next_stage",
}
_HOST_PHYSICAL_GPU_COUNT = 8
_ADMITTED_PHYSICAL_GPU_INDICES = (0,)
_PHYSICAL_EXECUTION_CONCURRENCY = 1
_VIRTUAL_ACTOR_WORLD_SIZE = 1
_ACTOR_PROCESS_COUNT = 1
_GPUS_PER_ACTOR_PROCESS = 1
_ENVIRONMENTS_PER_ACTOR_PROCESS = 128
_COLLECTION_WAVE_COUNT = 2


def _exact_original_rendering_audit(value: Any) -> dict[str, Any]:
    expected = {
        "schema": PRE_RENDER_AUDIT_SCHEMA,
        "root": "replay",
        "expected_empty_directories": ["replay/manifests", "replay/shards"],
        "observed_inventory": [],
        "observed_entry_count": 0,
        "observed_inventory_sha256": canonical_json_sha256([]),
        "rendering_started": False,
    }
    if value != expected:
        raise ConditioningSplitError(
            "completion marker pre-render audit is not the exact empty replay state"
        )
    return expected


def _load_immutable_completion(path: Path) -> tuple[dict[str, Any], str]:
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ConditioningSplitError(
            f"cannot inspect BC completion marker: {exc}"
        ) from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
        raise ConditioningSplitError(
            "BC completion marker must be an immutable mode-0444 regular file"
        )
    try:
        document = load_strict_json(path)
        digest = sha256_file(path)
    except ArtifactError as exc:
        raise ConditioningSplitError(
            f"cannot load BC completion marker: {exc}"
        ) from exc
    if not isinstance(document, Mapping) or set(document) != _COMPLETION_KEYS:
        raise ConditioningSplitError("BC completion marker fields differ")
    return dict(document), digest


def _completion_document(
    *,
    source_evidence: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    rendering_audit: Mapping[str, Any],
    postprocess_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        _ACTOR_PROCESS_COUNT
        * _ENVIRONMENTS_PER_ACTOR_PROCESS
        * _COLLECTION_WAVE_COUNT
        != EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS
    ):
        raise ConditioningSplitError(
            "BC actor process/wave topology does not produce 256 environment streams"
        )
    split_images = {split: reports[split]["image_count"] for split in SPLITS}
    split_episodes = {split: reports[split]["episode_count"] for split in SPLITS}
    plan_hashes = {reports[split]["verified_plan_sha256"] for split in SPLITS}
    if len(plan_hashes) != 1:
        raise ConditioningSplitError("BC split plans do not share one verified digest")
    plan_sha256 = next(iter(plan_hashes))
    return {
        "schema": COMPLETION_SCHEMA,
        "campaign_id": "dagger-redo-v001",
        "generation_id": "gen-000-bc",
        "status": "complete",
        "collection": {
            "hardware": {
                "host_resource": "8xNVIDIA H200 SM90",
                "host_physical_gpu_count": _HOST_PHYSICAL_GPU_COUNT,
                "admitted_physical_gpu_indices": list(
                    _ADMITTED_PHYSICAL_GPU_INDICES
                ),
                "physical_execution_concurrency": _PHYSICAL_EXECUTION_CONCURRENCY,
            },
            "actor_process_count": _ACTOR_PROCESS_COUNT,
            "gpus_per_actor_process": _GPUS_PER_ACTOR_PROCESS,
            "virtual_actor_world_size": _VIRTUAL_ACTOR_WORLD_SIZE,
            "environments_per_actor_process": _ENVIRONMENTS_PER_ACTOR_PROCESS,
            "collection_wave_count": _COLLECTION_WAVE_COUNT,
            "total_environment_streams": EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS,
            "steps_per_environment": EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV,
            "transitions": EXPECTED_BC_NOMINAL_TRANSITIONS,
            "image_count": source_evidence["image_count"],
            "episode_count": source_evidence["episode_count"],
            "control_hz": 50,
            "render_hz": 5,
            "image_stride": 10,
            "saturation_manifest_path": "saturation_manifest.json",
            "saturation_manifest_sha256": source_evidence["saturation_manifest_sha256"],
            "source_inventory_sha256": source_evidence["source_inventory_sha256"],
        },
        "split": {
            "seed": 42,
            "grouping": "complete_global_environment_stream",
            "pairwise_disjoint": True,
            "exact_coverage": True,
            "plan_path": "../../contracts/bc_collection_split.v1.json",
            "plan_sha256": plan_sha256,
            "groups": EXPECTED_GROUPS,
            "transitions": EXPECTED_TRANSITIONS,
            "image_count": split_images,
            "episode_count": split_episodes,
            "seals": {
                split: {
                    "path": f"../../{split}/conditioning/gen-000-bc/seal.json",
                    "sha256": reports[split]["seal_sha256"],
                }
                for split in SPLITS
            },
        },
        # Keep the historical relative path field byte-identical: the existing
        # completion marker is sealed provenance. Operational consumers use
        # the verified canonical split roots above, not this metadata string.
        "training_projection_input": {
            "split": "train",
            "path": "../../train/conditioning/gen-000-bc",
            "manifest_sha256": reports["train"]["manifest_sha256"],
            "seal_sha256": reports["train"]["seal_sha256"],
            "source_cohorts_allowed": False,
            "validation_test_allowed": False,
        },
        "postprocess_provenance": dict(postprocess_provenance),
        "rendering_audit": dict(rendering_audit),
        "learner_replay_ready": False,
        "rendering_started": False,
        "next_stage": "paused_before_h200_sm90_roborender_engine",
    }


def _finalize_collection_unlocked(
    campaign_root: Path = CANONICAL_CAMPAIGN_ROOT,
    source_root: Path = ROOT,
) -> dict[str, str]:
    campaign = Path(os.path.abspath(campaign_root))
    source = Path(os.path.abspath(source_root))
    if campaign != CANONICAL_CAMPAIGN_ROOT or source != ROOT:
        raise ConditioningSplitError(
            "BC finalization requires canonical campaign/source roots"
        )
    generation = campaign / "BC/collection"
    protocol = source / "configs/round_dagger/native_h200_experiment_protocol.json"
    output = generation / "BC_COLLECTION_COMPLETE.json"

    current_postprocess_provenance = build_postprocess_provenance(source, protocol)
    source_evidence = verify_exact_bc_collection_source(
        campaign,
        protocol_path=protocol,
    )
    if (
        source_evidence["postprocess_provenance"]
        != current_postprocess_provenance
    ):
        raise ConditioningSplitError("deep source audit postprocess provenance differs")

    try:
        os.lstat(output)
        marker_exists = True
    except FileNotFoundError:
        marker_exists = False
    existing: dict[str, Any] | None = None
    existing_digest: str | None = None
    if marker_exists:
        existing, existing_digest = _load_immutable_completion(output)
        # The marker records the exact code that originally produced the
        # already-sealed split.  Keep checking that historical identity against
        # the split plan after relocation; do not reinterpret its source_root
        # metadata as a live checkout path.
        sealed_postprocess_provenance = existing["postprocess_provenance"]
    else:
        sealed_postprocess_provenance = current_postprocess_provenance

    reports = {
        split: verify_split(
            {
                "train": campaign / "BC/train",
                "validation": campaign / "validation/BC",
                "test": campaign / "test/BC",
            }[split],
            consumer="renderer" if split == "train" else split,
            expected_transitions_per_group=EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV,
            expected_source_evidence=source_evidence,
            expected_postprocess_provenance=sealed_postprocess_provenance,
        )
        for split in SPLITS
    }
    if {split: reports[split]["group_count"] for split in SPLITS} != EXPECTED_GROUPS:
        raise ConditioningSplitError("BC split group counts differ from 178/39/39")
    if {
        split: reports[split]["transition_count"] for split in SPLITS
    } != EXPECTED_TRANSITIONS:
        raise ConditioningSplitError("BC split transition counts differ")
    if (
        sum(reports[split]["image_count"] for split in SPLITS)
        != source_evidence["image_count"]
        or sum(reports[split]["episode_count"] for split in SPLITS)
        != source_evidence["episode_count"]
    ):
        raise ConditioningSplitError(
            "BC split image/episode counts differ from collection"
        )
    all_ids = [
        global_id for split in SPLITS for global_id in reports[split]["global_env_ids"]
    ]
    if (
        len(all_ids) != EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS
        or len(set(all_ids)) != EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS
        or sorted(all_ids) != list(range(EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS))
    ):
        raise ConditioningSplitError("BC split coverage is overlapping or incomplete")

    if marker_exists:
        if existing is None or existing_digest is None:  # pragma: no cover
            raise ConditioningSplitError("BC completion marker state is incomplete")
        original_audit = _exact_original_rendering_audit(existing["rendering_audit"])
        expected = _completion_document(
            source_evidence=source_evidence,
            reports=reports,
            rendering_audit=original_audit,
            postprocess_provenance=sealed_postprocess_provenance,
        )
        if existing != expected:
            raise ConditioningSplitError("existing BC completion marker differs")
        return {
            "status": "complete",
            "path": str(output),
            "sha256": existing_digest,
        }

    rendering_audit = audit_pre_render_replay_root(generation)
    _exact_original_rendering_audit(rendering_audit)
    document = _completion_document(
        source_evidence=source_evidence,
        reports=reports,
        rendering_audit=rendering_audit,
        postprocess_provenance=sealed_postprocess_provenance,
    )
    try:
        digest = publish_immutable_json(output, document)
    except ArtifactError as exc:
        raise ConditioningSplitError(
            f"cannot publish BC completion marker: {exc}"
        ) from exc
    published, verified_digest = _load_immutable_completion(output)
    if published != document or verified_digest != digest:
        raise ConditioningSplitError("published BC completion marker differs")
    return {"status": "complete", "path": str(output), "sha256": digest}


def finalize_collection(
    campaign_root: Path = CANONICAL_CAMPAIGN_ROOT,
    source_root: Path = ROOT,
) -> dict[str, str]:
    """Finalize while owning the generation lock for the whole verification."""

    campaign = Path(os.path.abspath(campaign_root))
    source = Path(os.path.abspath(source_root))
    if campaign != CANONICAL_CAMPAIGN_ROOT or source != ROOT:
        raise ConditioningSplitError(
            "BC finalization requires canonical campaign/source roots"
        )
    with generation_lock(campaign / "BC/collection"):
        return _finalize_collection_unlocked(campaign, source)


def main() -> int:
    try:
        report = finalize_collection()
    except (ConditioningSplitError, OSError, ValueError) as exc:
        raise SystemExit(f"BC collection finalization failed: {exc}") from exc
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
