#!/usr/bin/env python3
"""Build one cumulative train-only replay snapshot from sealed H200 waves."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.artifacts import load_strict_json, sha256_file  # noqa: E402
from round_dagger.native_h200_experiment import (  # noqa: E402
    CANONICAL_CAMPAIGN_ID,
    CANONICAL_CAMPAIGN_ROOT,
    GENERATION_IDS,
)
from round_dagger.native_actor_protocol import train_round_relative_root  # noqa: E402
from round_dagger.replay import ReplayError, ReplayLedger, load_snapshot  # noqa: E402


SATURATION_SCHEMA = "fast_roborender.round_dagger.native_actor_saturation.v1"


def _regular_read_only(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ReplayError(f"{label} must be an absolute non-symlink path")
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ReplayError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o222:
        raise ReplayError(f"{label} must be an immutable regular file")
    return path


def _generation_manifest(
    path: Path,
    *,
    campaign: Path,
    macro_round: int,
) -> tuple[list[Path], dict[str, object]]:
    expected_generation = GENERATION_IDS[macro_round]
    source_root = (
        campaign / "BC/collection"
        if macro_round == 0
        else campaign / train_round_relative_root(expected_generation)
    )
    expected = source_root / "saturation_manifest.json"
    supplied = _regular_read_only(path, "saturation manifest")
    if supplied != expected:
        raise ReplayError(
            f"round {macro_round} saturation manifest must be exactly {expected}"
        )
    report = load_strict_json(supplied)
    if not isinstance(report, dict):
        raise ReplayError("saturation manifest must be a JSON object")
    if (
        report.get("schema") != SATURATION_SCHEMA
        or report.get("generation_id") != expected_generation
        or report.get("status") not in {"complete", "budget_capped"}
    ):
        raise ReplayError("saturation manifest identity/status differs")
    if "macro_round" in report and report["macro_round"] != macro_round:
        raise ReplayError("saturation manifest macro-round differs")
    raw = report.get("train_replay_cohorts")
    if not isinstance(raw, list) or not raw:
        raise ReplayError("generation has no whole sealed training waves")
    if len(raw) != len(report.get("waves", ())):
        raise ReplayError("not every reported collection wave has a replay seal")

    seals: list[Path] = []
    transition_total = 0
    for cohort_index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {
            "path",
            "seal_path",
            "seal_sha256",
            "transition_count",
            "purpose",
        }:
            raise ReplayError("saturation replay cohort fields differ")
        cohort = source_root / "replay" / "waves" / f"wave-{cohort_index:03d}"
        seal = cohort / "seal.json"
        if item["path"] != str(cohort) or item["seal_path"] != str(seal):
            raise ReplayError("saturation cohort escaped its generation wave namespace")
        _regular_read_only(seal, "cohort seal")
        if sha256_file(seal) != item["seal_sha256"]:
            raise ReplayError("cohort seal SHA-256 differs from saturation manifest")
        count = item["transition_count"]
        if type(count) is not int or count <= 0 or item["purpose"] != "train":
            raise ReplayError("saturation cohort purpose/count differs")
        transition_total += count
        seals.append(seal)
    if report.get("completed_transitions") != transition_total:
        raise ReplayError("saturation completed-transition total differs")
    return seals, {
        "macro_round": macro_round,
        "generation_id": expected_generation,
        "saturation_manifest_path": str(supplied),
        "saturation_manifest_sha256": sha256_file(supplied),
        "cohort_count": len(seals),
        "transition_count": transition_total,
    }


def build(arguments: argparse.Namespace) -> dict[str, object]:
    campaign = arguments.campaign_root
    if (
        campaign != CANONICAL_CAMPAIGN_ROOT
        or not campaign.is_dir()
        or campaign.is_symlink()
    ):
        raise ReplayError(f"campaign root must be exactly {CANONICAL_CAMPAIGN_ROOT}")
    through = arguments.through_macro_round
    if len(arguments.saturation_manifest) != through + 1:
        raise ReplayError("provide exactly one saturation manifest for every round")
    generation = GENERATION_IDS[through]
    expected_output = (
        campaign
        / train_round_relative_root(generation)
        / "replay"
        / "manifests"
        / f"train-through-round-{through:03d}.json"
    )
    if arguments.output != expected_output:
        raise ReplayError(f"training snapshot output must be exactly {expected_output}")
    if not expected_output.parent.is_dir() or expected_output.parent.is_symlink():
        raise ReplayError("training snapshot output directory is missing or unsafe")

    all_seals: list[Path] = []
    sources: list[dict[str, object]] = []
    for macro_round, path in enumerate(arguments.saturation_manifest):
        seals, source = _generation_manifest(
            path, campaign=campaign, macro_round=macro_round
        )
        all_seals.extend(seals)
        sources.append(source)

    with tempfile.TemporaryDirectory(
        prefix=".native-h200-replay-ledger.", dir=expected_output.parent
    ) as temporary:
        with ReplayLedger(Path(temporary) / "ledger.sqlite3") as ledger:
            accepted = [ledger.ingest_cohort(path) for path in all_seals]
            if any(item["status"] != "ACCEPTED" for item in accepted):
                raise ReplayError("fresh replay ledger did not uniquely accept every cohort")
            snapshot = ledger.build_snapshot(
                CANONICAL_CAMPAIGN_ID,
                through_macro_round=through,
                output=expected_output,
            )
    loaded = load_snapshot(
        expected_output,
        expected_file_sha256=snapshot["file_sha256"],
        artifact_root=campaign,
    )
    return {
        "schema": "fast_roborender.round_dagger.native_h200_train_snapshot_build.v1",
        "campaign_id": CANONICAL_CAMPAIGN_ID,
        "generation_id": generation,
        "through_macro_round": through,
        "path": str(expected_output),
        "file_sha256": loaded["file_sha256"],
        "snapshot_sha256": loaded["snapshot_sha256"],
        "transition_count": loaded["transition_count"],
        "cohort_counts_by_round": loaded["cohort_counts_by_round"],
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=CANONICAL_CAMPAIGN_ROOT)
    parser.add_argument("--through-macro-round", type=int, choices=range(6), required=True)
    parser.add_argument(
        "--saturation-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ReplayError) as error:
        print(f"native H200 replay snapshot failed: {error}", file=sys.stderr)
        raise SystemExit(2)
