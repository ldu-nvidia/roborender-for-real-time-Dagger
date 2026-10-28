from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
import numpy as np
import torch

from pipeline.dagger.replay import (
    build_replay_manifest,
    load_replay_config,
    validate_corrected_bc_chunk,
    validate_replay_manifest,
)
from pipeline.dagger.services.raven_learner_worker import load_batch
from pipeline.dagger.production_controller import Controller, RAVEN_FIDELITY_CONTRACT


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_config(tmp_path: Path, *, split: str = "train") -> dict:
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    chunks = []
    split_items = []
    for index in range(8):
        path = rendered / f"bc_{index}.npz"
        np.savez(
            path,
            render_contract=np.asarray("raven_three_view_autoregressive_v1"),
            render_chain_id=np.asarray(f"chain-{index // 2}"),
            render_chain_index=np.asarray(index % 2, dtype=np.int64),
            render_chain_length=np.asarray(2, dtype=np.int64),
            render_inherits_previous_reference=np.asarray(index % 2 > 0),
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        chunk_id = f"bc_{index}"
        relative = f"rendered/{path.name}"
        chunks.append({
            "chunk_id": chunk_id, "rendered_npz": relative,
            "rendered_sha256": digest,
        })
        split_items.append({"chunk_id": chunk_id, "rendered_npz": relative})
    source = tmp_path / "manifest.json"
    source_sha = write_json(source, {"chunks": chunks})
    train = tmp_path / "splits" / "train.json"
    train_sha = write_json(train, {
        "schema": "roborender.bc_split.v1", "split": split,
        "chunk_count": len(split_items), "items": split_items,
        "source_manifest": "../manifest.json",
        "source_manifest_sha256": source_sha,
    })
    return {
        "schema": "roborender.dagger.replay.v1", "enabled": True,
        "total_chunks_per_update": 4,
        "fractions": {
            "current_dagger": 0.5, "bc": 0.25, "prior_dagger": 0.25,
        },
        "seed": "unit-replay", "bc_train_manifest": str(train),
        "bc_train_manifest_sha256": train_sha,
        "empty_pool_fallback": "bc_then_current",
    }


def barrier(round_id: int, iteration_id: int, actor: int) -> dict:
    return {
        "round_id": round_id, "iteration_id": iteration_id,
        "actor_id": f"actor{actor}",
        "chunk_ids": [f"r{round_id}i{iteration_id}a{actor}"],
    }


def test_absent_or_disabled_replay_preserves_legacy_behavior():
    assert load_replay_config(None) == {"enabled": False}
    assert load_replay_config({"enabled": False}) == {"enabled": False}


def test_selection_is_deterministic_and_first_iteration_falls_back_to_bc(tmp_path):
    config = replay_config(tmp_path)
    barriers = [barrier(1, 1, index) for index in range(4)]
    iterations = [{"round_id": 1, "iteration_id": 1, "status": "ready_for_learning"}]
    arguments = dict(
        config=config, run_id="run", round_id=1, iteration_id=1,
        barriers=barriers, iterations=iterations,
    )
    first = build_replay_manifest(**arguments)
    second = build_replay_manifest(**arguments)
    assert first == second
    assert first["selected_counts"] == {
        "current_dagger": 2, "bc": 2, "prior_dagger": 0,
    }
    assert len(first["samples"]) == 4


def test_prior_pool_uses_only_completed_earlier_iterations(tmp_path):
    config = replay_config(tmp_path)
    barriers = [
        *(barrier(1, 1, index) for index in range(4)),
        *(barrier(1, 2, index) for index in range(4)),
    ]
    iterations = [
        {"round_id": 1, "iteration_id": 1, "status": "advanced"},
        {"round_id": 1, "iteration_id": 2, "status": "ready_for_learning"},
    ]
    result = build_replay_manifest(
        config=config, run_id="run", round_id=1, iteration_id=2,
        barriers=barriers, iterations=iterations,
    )
    assert result["selected_counts"] == {
        "current_dagger": 2, "bc": 1, "prior_dagger": 1,
    }
    prior = [row for row in result["samples"] if row["source"] == "prior_dagger"]
    assert len(prior) == 1 and prior[0]["iteration_id"] == 1


def test_bc_catalog_must_be_training_split(tmp_path):
    config = replay_config(tmp_path, split="validation")
    with pytest.raises(ValueError, match="sealed train split"):
        build_replay_manifest(
            config=config, run_id="run", round_id=1, iteration_id=1,
            barriers=[barrier(1, 1, 0)],
            iterations=[{"round_id": 1, "iteration_id": 1, "status": "ready_for_learning"}],
        )


def test_legacy_unchained_bc_archive_is_rejected(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, gt_actions=np.zeros((81, 26), dtype=np.float32))
    with pytest.raises(ValueError, match="lacks autoregressive chain provenance"):
        validate_corrected_bc_chunk(path)


def test_manifest_validation_binds_current_and_prior_to_durable_barriers(tmp_path):
    config = replay_config(tmp_path)
    barriers = [
        *(barrier(1, 1, index) for index in range(4)),
        *(barrier(1, 2, index) for index in range(4)),
    ]
    iterations = [
        {"round_id": 1, "iteration_id": 1, "status": "advanced"},
        {"round_id": 1, "iteration_id": 2, "status": "ready_for_learning"},
    ]
    manifest = build_replay_manifest(
        config=config, run_id="run", round_id=1, iteration_id=2,
        barriers=barriers, iterations=iterations,
    )
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE round_iterations (
          run_id TEXT, round_id INTEGER, iteration_id INTEGER, status TEXT
        );
        CREATE TABLE iteration_barriers (
          run_id TEXT, round_id INTEGER, iteration_id INTEGER,
          actor_id TEXT, chunk_ids_json TEXT
        );
        CREATE TABLE replay_bc_attestations (
          path TEXT PRIMARY KEY, size_bytes INTEGER, mtime_ns INTEGER,
          sha256 TEXT, checked_at REAL
        );
    """)
    for row in iterations:
        db.execute(
            "INSERT INTO round_iterations VALUES (?,?,?,?)",
            ("run", row["round_id"], row["iteration_id"], row["status"]),
        )
    for row in barriers:
        db.execute(
            "INSERT INTO iteration_barriers VALUES (?,?,?,?,?)",
            ("run", row["round_id"], row["iteration_id"], row["actor_id"],
             json.dumps(row["chunk_ids"])),
        )
    current = {
        row["chunk_ids"][0] for row in barriers if row["iteration_id"] == 2
    }
    dagger, bc = validate_replay_manifest(
        manifest, db=db, run_id="run", round_id=1, iteration_id=2,
        admitted_current=current, bc_root=tmp_path / "rendered",
    )
    assert len(dagger) == 3
    assert len(bc) == 1

    changed = json.loads(json.dumps(manifest))
    prior = next(row for row in changed["samples"] if row["source"] == "prior_dagger")
    prior["iteration_id"] = 2
    body = dict(changed)
    body.pop("sha256")
    changed["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="not from the past"):
        validate_replay_manifest(
            changed, db=db, run_id="run", round_id=1, iteration_id=2,
            admitted_current=current, bc_root=tmp_path / "rendered",
        )


def test_offline_uint8_and_online_float_policy_rgb_share_float01_contract(tmp_path):
    arrays = {
        "obs_actor_obs": np.zeros((81, 113), dtype=np.float32),
        "gt_actions": np.zeros((81, 26), dtype=np.float32),
    }
    for view in ("head_color", "hand_left", "hand_right"):
        arrays[f"rgb_gen__{view}"] = np.full(
            (81, 108, 192, 3), 255, dtype=np.uint8
        )
    offline = tmp_path / "offline.npz"
    np.savez(offline, **arrays)
    spec = {
        "input_contract": {
            "views": ["head_color", "hand_left", "hand_right"],
            "rgb_mean": [0.0, 0.0, 0.0], "rgb_std": [1.0, 1.0, 1.0],
        },
        "output_contract": {"policy_action_dim": 11},
    }
    vision, actor, target = load_batch([offline], torch.device("cpu"), spec)
    assert vision.dtype == torch.float32
    assert float(vision.min()) == 1.0 and float(vision.max()) == 1.0
    assert actor.shape == (1, 81, 113) and target.shape == (1, 81, 11)


def test_controller_submits_exact_manifest_and_selected_dagger_ids(tmp_path):
    class Learner:
        def __init__(self):
            self.payload = None

        def post(self, path, payload):
            assert path == "/v1/jobs"
            self.payload = payload
            return {"ok": True}

    controller = Controller.__new__(Controller)
    controller.run_id = "run"
    controller.round_id = 1
    controller.wandb_mode = "disabled"
    controller.actor_ids = [f"actor{index}" for index in range(4)]
    controller.learner = Learner()
    controller.config = {
        "execution": {"window_count": 40}, "seed": 42, "teacher_beta": 0.75,
        "rtx_actor": {"gpu_uuid": "gpu1"},
        "student": {
            "learning_rate": 2e-4, "max_gradient_norm": 1.0,
            "mini_batches_per_iteration": 1,
        },
        "h200": {"render_student_worker_count": 8},
        "validation": {"interval_iterations": 40},
        "logging": {"wandb_project": "dagger", "wandb_run_name": "run"},
        "raven_fidelity": dict(RAVEN_FIDELITY_CONTRACT),
        "replay": replay_config(tmp_path),
    }
    barriers = [barrier(1, 1, index) for index in range(4)]
    controller._submit_learning(
        {"iteration_id": 1, "input_checkpoint_sha256": "a" * 64},
        {
            "barriers": barriers,
            "iterations": [
                {"round_id": 1, "iteration_id": 1, "status": "ready_for_learning"}
            ],
        },
    )
    payload = controller.learner.payload
    manifest = payload["parameters"]["replay_manifest"]
    assert manifest["selected_counts"] == {
        "current_dagger": 2, "bc": 2, "prior_dagger": 0,
    }
    assert payload["train_chunk_ids"] == [
        row["chunk_id"] for row in manifest["samples"] if row["source"] != "bc"
    ]
