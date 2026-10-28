"""Focused tests for Raven-compatible three-view offline render chaining."""

from __future__ import annotations

import hashlib
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from bc_rendering.render_bc_teacher_8gpu import (
    assign_lpt,
    build_render_chains,
    discover_jobs,
    incompatible_existing_outputs,
    raven_stream_tag,
)
from bc_rendering.render_bc_teacher_worker import (
    RAVEN_CHAIN_RENDER_CONTRACT,
    VIEW_SLOTS,
    render_chain,
    validate_completed_chunk,
    validate_lane_plan,
    write_raven_rendered_chunk,
)
from bc_rendering.validate_bc_teacher_render import validate_chunk, expected_chain_contexts, load_expected
from bc_rendering.activate_corrected_render_pool import activate, main as activate_main
from bc_rendering.render_recipes import (
    CURRENT_RECIPE, INDEPENDENT_CONTRACT, LEGACY_RECIPE, LEGACY_CONTRACT, recipe_config,
    collection_provenance,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _planning_job(tag: str, env_id: int, seq: int, *, done: bool = False) -> dict:
    return {
        "stream_tag": tag,
        "env_id": env_id,
        "sequence_id": seq,
        "source_npz": f"/{tag}-{env_id}-{seq}.npz",
        "output_npz": f"/rendered/{tag}-{env_id}-{seq}.npz",
        "global_env_id": env_id,
        "episode_id": seq,
        "frame_count": 81,
        "has_done": done,
    }


class _RecordingPipe:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        refs = kwargs["reference_image_views"]
        self.calls.append(
            {
                "reference_pixels": None
                if refs is None
                else [tuple(np.asarray(image)[0, 0]) for image in refs],
                "depth_view_count": len(kwargs["control_video_views"]),
                "mask_view_count": len(kwargs["control_video_2_views"]),
                "seed": kwargs["seed"],
                "cfg_scale": kwargs["cfg_scale"],
                "steps": kwargs["num_inference_steps"],
            }
        )
        call_number = len(self.calls)
        height = kwargs["height"] // kwargs["num_views"]
        width = kwargs["width"]
        frames = []
        for _ in range(kwargs["num_frames"]):
            stacked = Image.new("RGB", (width, height * 3))
            for view_index in range(3):
                value = call_number * 40 + (view_index + 1) * 10
                stacked.paste(
                    Image.new("RGB", (width, height), (value, value, value)),
                    (0, view_index * height),
                )
            frames.append(stacked)
        return frames


class OfflineReferenceChainingTest(unittest.TestCase):
    def test_activation_cli_defaults_follow_selected_dataset_root(self) -> None:
        root = Path("/unused/custom-bc-pool")
        with (
            patch("sys.argv", ["activate", "--dataset-root", str(root), "--report", "/unused/report.json"]),
            patch("bc_rendering.activate_corrected_render_pool.activate", return_value={}) as mocked,
            patch("bc_rendering.activate_corrected_render_pool.atomic_new_json"),
            patch("builtins.print"),
        ):
            activate_main()
        mocked.assert_called_once_with(
            dataset_root=root,
            source_root=None,
            source_manifest_path=None,
            rendered_root=root / "rendered_independent_v2",
            manifest_path=root / "manifests/independent_v2.json",
            split_dir=root / "splits_independent_v2",
            manifest_only=False,
            recipe=CURRENT_RECIPE, pool_seal=None,
            rgb_validation="full", deep_rgb_sample_count=32, sample_seed=42,
        )

    def test_activation_publishes_only_the_corrected_immutable_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "BC"
            rendered = dataset / "rendered_independent_v2"
            rendered.mkdir(parents=True)
            (dataset / "isaac").mkdir()
            items = []
            for index in range(20):
                items.append({
                    "chunk_id": f"bc_chunk_{index:06d}",
                    "isaac_npz": f"isaac/bc_chunk_{index:06d}.npz",
                    "rendered_npz": f"rendered/bc_chunk_{index:06d}.npz",
                    "source_run_tag": "run-a", "source_batch": 1,
                    "rank": 0, "local_env_id": index,
                })
                np.savez_compressed(
                    rendered / f"bc_chunk_{index:06d}.npz",
                    render_contract=np.asarray(RAVEN_CHAIN_RENDER_CONTRACT),
                    render_chain_id=np.asarray(f"chain-{index}"),
                    render_chain_index=np.asarray(0),
                    render_chain_length=np.asarray(1),
                    render_inherits_previous_reference=np.asarray(False),
                    render_recipe=np.asarray(CURRENT_RECIPE),
                    render_source_stream_id=np.asarray(f"source-stream-{index}"),
                )
            (dataset / "manifest.json").write_text(
                __import__("json").dumps(
                    {"schema": "legacy", "frames_per_chunk": 81, "chunks": items}
                )
            )
            manifest = dataset / "manifests/independent_v2.json"
            splits = dataset / "splits_independent_v2"
            with patch(
                "pipeline.bc_rendering.validate_bc_teacher_render.validate_pool",
                return_value={"verdict": "PASS", "rendered_identities": {
                    p.name: {"sha256": _hash(p), "bytes": p.stat().st_size} for p in rendered.glob("*.npz")}},
            ):
                report = activate(
                    dataset_root=dataset,
                    rendered_root=rendered,
                    manifest_path=manifest,
                    split_dir=splits,
                )
            self.assertEqual(report["verdict"], "PASS")
            self.assertEqual(report["chunk_count"], 20)
            activated = __import__("json").loads(manifest.read_text())
            self.assertEqual(
                activated["chunks"][0]["rendered_npz"],
                "rendered_independent_v2/bc_chunk_000000.npz",
            )
            self.assertEqual(
                activated["chunks"][0]["render_contract"],
                RAVEN_CHAIN_RENDER_CONTRACT,
            )
            self.assertTrue((splits / "train.json").is_file())
            with self.assertRaises(FileExistsError):
                with patch(
                    "pipeline.bc_rendering.validate_bc_teacher_render.validate_pool",
                    return_value={"verdict": "PASS"},
                ):
                    activate(
                        dataset_root=dataset,
                        rendered_root=rendered,
                        manifest_path=manifest,
                        split_dir=splits,
                    )

    def test_stream_tag_matches_raven_filename_not_recovery_run_name(self) -> None:
        self.assertEqual(
            raven_stream_tag(
                {
                    "collection_filename": "chunk_batch1-raven-500_r01_0001.npz",
                    "source_run_tag": "batch1-raven-500-rank1-recovery",
                }
            ),
            "batch1-raven-500",
        )

    def test_groups_consecutive_chunks_and_resets_on_done_or_discontinuity(self) -> None:
        jobs = [
            _planning_job("run-a", 1, 4),
            _planning_job("run-a", 1, 2, done=True),
            _planning_job("run-a", 1, 1),
            _planning_job("run-a", 1, 6),
            _planning_job("run-a", 2, 1),
            _planning_job("run-b", 1, 1),
        ]
        chains = build_render_chains(jobs, recipe=LEGACY_RECIPE)
        identities = sorted([
            (chain["stream_tag"], chain["env_id"], [j["sequence_id"] for j in chain["jobs"]])
            for chain in chains
        ])
        self.assertEqual(
            identities,
            [
                ("run-a", 1, [1, 2]),
                ("run-a", 1, [4]),
                ("run-a", 1, [6]),
                ("run-a", 2, [1]),
                ("run-b", 1, [1]),
            ],
        )
        for chain in chains:
            self.assertEqual(
                [job["chain_index"] for job in chain["jobs"]],
                list(range(len(chain["jobs"]))),
            )
            self.assertFalse(chain["jobs"][0]["inherits_previous_reference"])
            self.assertTrue(
                all(job["inherits_previous_reference"] for job in chain["jobs"][1:])
            )

    def test_chain_is_an_atomic_six_chunk_scheduling_unit(self) -> None:
        chains = build_render_chains(
            [_planning_job("run", 3, seq) for seq in range(8)], max_chain=6,
            recipe=LEGACY_RECIPE,
        )
        self.assertEqual([[j["sequence_id"] for j in c["jobs"]] for c in chains], [
            [0, 1, 2, 3, 4, 5],
            [6, 7],
        ])
        self.assertEqual([c["frame_count"] for c in chains], [486, 162])
        assignments, _ = assign_lpt(chains, 2)
        assigned_ids = [
            chain["chain_id"] for lane in assignments for chain in lane
        ]
        self.assertCountEqual(assigned_ids, [chain["chain_id"] for chain in chains])

    def _source_job(self, root: Path, index: int, chain_length: int) -> dict:
        source = root / f"source-{index}.npz"
        output = root / f"output-{index}.npz"
        arrays = {
            "gt_actions": np.zeros((1, 26), dtype=np.float32),
            "obs_actor_obs": np.zeros((1, 113), dtype=np.float32),
            "dones": np.zeros(1, dtype=np.bool_),
        }
        for _, view in VIEW_SLOTS:
            arrays[f"depth_ctrl__{view}"] = np.zeros((1, 2, 3), dtype=np.uint8)
            arrays[f"rgb_robot__{view}"] = np.zeros((1, 2, 3, 3), dtype=np.uint8)
        np.savez_compressed(source, **arrays)
        return {
            "source_npz": str(source),
            "output_npz": str(output),
            "frame_count": 1,
            "prompt": "robot task",
            "seed": 42,
            "chain_id": "test-chain",
            "chain_index": index,
            "chain_length": chain_length,
            "inherits_previous_reference": index > 0,
            **recipe_config(LEGACY_RECIPE if chain_length > 1 else CURRENT_RECIPE),
            "source_stream_id": "test-source-stream",
        }

    def test_all_three_previous_frames_are_chained_and_resume_is_idempotent(self) -> None:
        config = {
            **recipe_config(LEGACY_RECIPE),
            "height": 2,
            "width": 3,
            "student_height": 108,
            "student_width": 192,
            "cfg_scale": 5.0,
            "num_inference_steps": 50,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [self._source_job(root, index, 2) for index in range(2)]
            chain = {"chain_id": "test-chain", "jobs": jobs}

            first_pipe = _RecordingPipe()
            result = render_chain(first_pipe, chain, config)
            self.assertEqual(result["rendered"], 2)
            self.assertEqual(len(first_pipe.calls), 2)
            self.assertIsNone(first_pipe.calls[0]["reference_pixels"])
            self.assertEqual(
                first_pipe.calls[1]["reference_pixels"],
                [(50, 50, 50), (60, 60, 60), (70, 70, 70)],
            )
            self.assertTrue(all(call["depth_view_count"] == 3 for call in first_pipe.calls))
            self.assertTrue(all(call["mask_view_count"] == 3 for call in first_pipe.calls))
            for job in jobs:
                self.assertTrue(
                    validate_completed_chunk(
                        Path(job["output_npz"]), 1, chain_context=job
                    )
                )
                with np.load(job["output_npz"], allow_pickle=False) as rendered:
                    self.assertEqual(
                        str(rendered["render_contract"].item()),
                        LEGACY_CONTRACT,
                    )

            no_op_pipe = _RecordingPipe()
            result = render_chain(no_op_pipe, chain, config)
            self.assertEqual(result["status"], "skipped_complete")
            self.assertEqual(no_op_pipe.calls, [])

            prefix_hash = _hash(Path(jobs[0]["output_npz"]))
            Path(jobs[1]["output_npz"]).unlink()
            resume_pipe = _RecordingPipe()
            result = render_chain(resume_pipe, chain, config)
            self.assertEqual(result["rendered"], 1)
            self.assertEqual(result["replayed_prefix"], 1)
            self.assertEqual(len(resume_pipe.calls), 2)
            self.assertEqual(_hash(Path(jobs[0]["output_npz"])), prefix_hash)
            self.assertEqual(
                resume_pipe.calls[1]["reference_pixels"],
                [(50, 50, 50), (60, 60, 60), (70, 70, 70)],
            )

    def test_old_independent_render_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "old.npz"
            np.savez_compressed(
                output,
                rgb_gen=np.zeros((1, 108, 192, 3), np.uint8),
                **{
                    f"rgb_gen__{view}": np.zeros((1, 108, 192, 3), np.uint8)
                    for _, view in VIEW_SLOTS
                },
            )
            job = {
                "output_npz": str(output),
                "chain_id": "new-chain",
                "chain_index": 0,
                "chain_length": 1,
                "inherits_previous_reference": False,
                **recipe_config(),
                "source_stream_id": "test-stream",
            }
            self.assertEqual(
                incompatible_existing_outputs(
                    [{"chain_id": "new-chain", "jobs": [job]}]
                ),
                [str(output)],
            )

    def test_final_validator_requires_exact_chain_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            output = root / "output.npz"
            np.savez_compressed(
                source,
                gt_actions=np.zeros((81, 26), np.float32),
                obs_actor_obs=np.zeros((81, 113), np.float32),
                dones=np.zeros(81, np.bool_),
                prompt=np.asarray("task"),
                env_id=np.asarray(2),
                seq=np.asarray(9),
            )
            images = {
                view: [Image.new("RGB", (1, 1)) for _ in range(81)]
                for _, view in VIEW_SLOTS
            }
            context = {
                "chain_id": "chain-final-validation",
                "chain_index": 1,
                "chain_length": 3,
                "inherits_previous_reference": True,
                **recipe_config(LEGACY_RECIPE),
            }
            write_raven_rendered_chunk(
                source, output, images, chain_context=context
            )
            self.assertEqual(
                validate_chunk(source, output, chain_context=context), []
            )
            wrong = {**context, "chain_id": "wrong-chain"}
            self.assertTrue(
                any(
                    "chain provenance mismatch" in error
                    for error in validate_chunk(source, output, chain_context=wrong)
                )
            )

    def test_current_recipe_matches_three_view_launcher_singletons(self) -> None:
        jobs = [_planning_job("run", 2, seq) for seq in range(8)]
        chains = build_render_chains(jobs)
        self.assertEqual([len(c["jobs"]) for c in chains], [1] * 8)
        self.assertTrue(all(c["matches_raven_three_view_launcher"] for c in chains))
        self.assertTrue(all(c["render_contract"] == INDEPENDENT_CONTRACT for c in chains))
        self.assertTrue(all(not c["jobs"][0]["inherits_previous_reference"] for c in chains))
        with self.assertRaisesRegex(ValueError, "explicit legacy"):
            build_render_chains(jobs, max_chain=6)

    def test_legacy_chains_never_join_independent_collector_ranks(self) -> None:
        jobs = [
            {**_planning_job("run", 0, seq), "rank": rank, "global_env_id": rank * 28,
             "source_npz": f"/rank{rank}-seq{seq}.npz"}
            for rank, seq in ((0, 0), (1, 1), (0, 2), (1, 3))
        ]
        chains = build_render_chains(jobs, recipe=LEGACY_RECIPE)
        self.assertEqual(len(chains), 4)
        for chain in chains:
            self.assertEqual(len({job["rank"] for job in chain["jobs"]}), 1)
        self.assertEqual(len({chain["source_stream_id"] for chain in chains}), 2)

    def test_independent_render_calls_never_inherit_previous_chunk(self) -> None:
        config = {**recipe_config(), "height": 2, "width": 3,
                  "student_height": 108, "student_width": 192,
                  "cfg_scale": 5.0, "num_inference_steps": 50}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [self._source_job(root, 0, 1), self._source_job(root, 1, 1)]
            for index, job in enumerate(jobs):
                job.update(chain_index=0, inherits_previous_reference=False,
                           chain_id=f"independent-{index}")
            pipe = _RecordingPipe()
            for job in jobs:
                render_chain(pipe, {"chain_id": job["chain_id"], "jobs": [job]}, config)
            self.assertEqual([call["reference_pixels"] for call in pipe.calls], [None, None])
            self.assertTrue(all((call["seed"], call["cfg_scale"], call["steps"]) == (42, 5.0, 50)
                                for call in pipe.calls))
            # Resuming a complete singleton never needs another chunk or model call.
            again = _RecordingPipe()
            result = render_chain(again, {"chain_id": jobs[1]["chain_id"], "jobs": [jobs[1]]}, config)
            self.assertEqual(result["status"], "skipped_complete")
            self.assertEqual(again.calls, [])
            with np.load(jobs[0]["output_npz"], allow_pickle=False) as output:
                self.assertEqual(output["render_contract"].item(), INDEPENDENT_CONTRACT)

    def test_old_autoregressive_contract_cannot_be_resumed_as_current(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = self._source_job(Path(temporary), 0, 1)
            images = {view: [Image.new("RGB", (1, 1))] for _, view in VIEW_SLOTS}
            old = {**job, **recipe_config(LEGACY_RECIPE)}
            write_raven_rendered_chunk(Path(job["source_npz"]), Path(job["output_npz"]),
                                      images, chain_context=old)
            self.assertEqual(incompatible_existing_outputs([{"jobs": [job]}]), [job["output_npz"]])
            pipe = _RecordingPipe()
            with self.assertRaises(FileExistsError):
                render_chain(pipe, {"chain_id": job["chain_id"], "jobs": [job]}, recipe_config())
            self.assertEqual(pipe.calls, [])

    def test_lane_launch_rejects_legacy_or_ambiguous_contract_before_cuda(self) -> None:
        jobs = [{**_planning_job("run", 1, 0), "seed": 42}]
        chains = build_render_chains(jobs)
        config = {**recipe_config(), "height": 240, "width": 416,
                  "student_height": 108, "student_width": 192, "seed": 42,
                  "num_inference_steps": 50, "cfg_scale": 5.0}
        plan = {"schema": "roborender.bc_teacher_lane_plan.v3", "config": config,
                "chains": chains, "jobs": [job for c in chains for job in c["jobs"]]}
        self.assertEqual(validate_lane_plan(plan)["max_chain_chunks"], 1)
        mutations = [
            lambda p: p.update(schema="roborender.bc_teacher_lane_plan.v2"),
            lambda p: p["config"].update(render_contract=LEGACY_CONTRACT),
            lambda p: p["config"].update(num_inference_steps=5),
            lambda p: p["config"].update(seed=43),
            lambda p: p.update(jobs=[]),
            lambda p: p["chains"][0]["jobs"][0].update(chain_length=2),
        ]
        for mutation in mutations:
            altered = copy.deepcopy(plan)
            mutation(altered)
            with self.assertRaises((ValueError, KeyError)):
                validate_lane_plan(altered)

    def test_changed_source_is_rejected_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = self._source_job(Path(temporary), 0, 1)
            job["source_payload_sha256"] = "0" * 64
            pipe = _RecordingPipe()
            with self.assertRaisesRegex(ValueError, "source payload hash mismatch"):
                render_chain(pipe, {"chain_id": job["chain_id"], "jobs": [job]}, recipe_config())
            self.assertEqual(pipe.calls, [])

    def test_historical_activation_requires_explicit_destinations(self) -> None:
        with patch("sys.argv", ["activate", "--recipe", LEGACY_RECIPE,
                                "--report", "/unused/report.json"]):
            with self.assertRaises(SystemExit):
                activate_main()

    def test_fresh_native_provenance_survives_plan_render_and_validator(self) -> None:
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root, output_root = root / "isaac", root / "rendered"
            source_root.mkdir()
            source = source_root / "chunk_native_0000000.npz"
            np.savez_compressed(source, gt_actions=np.zeros((81, 26), np.float32),
                obs_actor_obs=np.zeros((81, 113), np.float32), dones=np.zeros(81, np.bool_),
                prompt=np.asarray("task"), env_id=np.asarray(0), seq=np.asarray(3))
            provenance = {"source_run_tag": "fresh-train-0001-r001", "rank": 1,
                "local_env_id": 0, "global_env_id": 28, "motion_id": "motion-29",
                "motion_sha256": "a" * 64, "simulator_seed": 3030,
                "episode_id": "fresh-train-0001-r001-e000-p00000001", "split": "train",
                "request_sha256": "b" * 64, "sampling_plan_sha256": "c" * 64,
                "teacher_only": True, "selected_for_split": True, "overcollection": False}
            item = {**provenance, "chunk_id": "bc-0001", "isaac_npz": "isaac/" + source.name,
                "collection_filename": source.name, "source_sha256": _hash(source),
                "source_size_bytes": source.stat().st_size, "sequence_id": 3}
            manifest, seal = root / "manifest.json", root / "SEALED.json"
            manifest.write_text(json.dumps({"status": "sealed", "chunk_count": 1, "chunks": [item]}))
            # A fresh *source* seal truthfully reports no rendered clips yet.
            seal.write_text(json.dumps({"verdict": "PASS", "isaac_chunks": 1, "rendered_chunks": 0}))
            self.assertEqual(load_expected(manifest, seal), [item])
            jobs = discover_jobs(source_root=source_root, output_root=output_root,
                                  pool_manifest=manifest, pool_seal=seal)
            self.assertEqual(jobs[0]["episode_id"], provenance["episode_id"])
            self.assertEqual(jobs[0]["collection_provenance"], provenance)
            context = build_render_chains(jobs)[0]["jobs"][0]
            expected = expected_chain_contexts([item], source_root=source_root, rendered_root=output_root)
            self.assertEqual(expected[source.name]["collection_provenance"], provenance)
            self.assertEqual(expected[source.name]["chain_id"], context["chain_id"])
            images = {view: [Image.new("RGB", (1, 1)) for _ in range(81)] for _, view in VIEW_SLOTS}
            output = Path(context["output_npz"])
            write_raven_rendered_chunk(source, output, images, chain_context=context)
            with np.load(output, allow_pickle=False) as data:
                for key, value in provenance.items():
                    self.assertEqual(data["collection_" + key].item(), value)
            self.assertEqual(validate_chunk(source, output, chain_context=context), [])
            wrong = copy.deepcopy(context)
            wrong["collection_provenance"]["split"] = "validation"
            self.assertFalse(validate_completed_chunk(output, 81, chain_context=wrong))

    def test_partial_or_overcollected_native_provenance_is_not_render_admitted(self) -> None:
        self.assertIsNone(collection_provenance({"source_run_tag": "legacy"}))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            collection_provenance({"motion_id": "unknown"})


if __name__ == "__main__":
    unittest.main()
