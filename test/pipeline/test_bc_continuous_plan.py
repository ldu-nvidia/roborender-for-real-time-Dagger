import json

import pytest

from pipeline.bc_collection import continuous_plan as plan


def motions(count=34):
    return [{"motion_id": f"m{i}", "path": f"/motions/m{i}.npz", "sha256": f"{i + 1:064x}"}
            for i in range(count)]


def make(**kwargs):
    return plan.build_plan(motions(), run_id="new-continuous", max_windows=8,
                           output_dir="/dagger-runtime/bc_isaac", **kwargs)


def test_one_scene_full_inventory_and_fixed_prefix():
    value = make()
    request = plan.build_request(value)
    assert value["scene_count"] == 1
    assert value["num_envs"] == 32
    assert value["process_seed"] == 0
    assert value["vector_steps"] == 648
    assert request["motions"] == motions()[:32]
    assert request["motion_inventory"] == motions()
    assert request["sampling_plan_sha256"] == plan.digest(value)
    assert "chunks_per_env" not in request
    assert request["split"] == "train"
    assert value["production_admitted"] is False
    from pipeline.deployment.rtx_actor.isaac6_bc_production import validate_collection_request
    assert validate_collection_request(request) == request


def test_resolved_order_is_not_sorted_or_shuffled():
    rows = list(reversed(motions()))
    value = plan.build_plan(rows, run_id="fresh", max_windows=1, output_dir="/run/raw", num_envs=2)
    assert value["motions"] == rows[:2]
    assert plan.build_request(value)["motion_inventory"] == rows


@pytest.mark.parametrize("budget", [0, -1, 50001, True, 1.2, None])
def test_explicit_budget_bounds(budget):
    with pytest.raises(ValueError):
        plan.build_plan(motions(), run_id="fresh", max_windows=budget, output_dir="/run/raw")


def test_budget_is_required():
    with pytest.raises(TypeError):
        plan.build_plan(motions(), run_id="fresh", output_dir="/run/raw")


def test_repeated_mapping_not_silently_introduced():
    with pytest.raises(ValueError, match="repeated modulo"):
        plan.build_plan(motions(2), run_id="fresh", max_windows=1, output_dir="/run/raw")


def test_full_loader_inventory_limit():
    with pytest.raises(ValueError, match="10000"):
        plan.build_plan(motions(10001), run_id="fresh", max_windows=1, output_dir="/run/raw")


@pytest.mark.parametrize("field,value", [
    ("scene_count", 2), ("split", "validation"), ("selection_uses_outcomes", True),
    ("optimizer_steps", 1), ("production_admitted", True),
    ("episode_reset_rule", "reset_every_window"), ("conditioning_views", ["head_color"]),
])
def test_protocol_mutations_rejected(field, value):
    candidate = make()
    candidate[field] = value
    with pytest.raises(ValueError):
        plan.validate_plan(candidate)


def test_duplicate_and_unsafe_motion_descriptors():
    rows = motions(2)
    for bad in ([rows[0], rows[0]], [{**rows[0], "path": "/motions/../m0.npz"}, rows[1]],
                [{**rows[0], "extra": True}, rows[1]]):
        with pytest.raises(ValueError):
            plan.build_plan(bad, run_id="fresh", max_windows=1, output_dir="/run/raw", num_envs=2)


def test_directory_resolution_preserves_actual_enumeration_without_loading_tensors(tmp_path, monkeypatch):
    for name in ("a.npz", "z.npz", "ignored.txt"):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(plan.os, "listdir", lambda directory: ["z.npz", "ignored.txt", "a.npz"])
    resolved = plan.resolve_inventory(tmp_path, "/native/motions")
    assert [r["motion_id"] for r in resolved] == ["z", "a"]
    assert resolved[0]["path"] == "/native/motions/z.npz"
    assert resolved[0]["sha256"] == plan.file_sha256(tmp_path / "z.npz")


def test_directory_resolution_rejects_symlink_motion(tmp_path):
    (tmp_path / "target").write_bytes(b"motion")
    (tmp_path / "alias.npz").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="without aliases"):
        plan.resolve_inventory(tmp_path, "/native/motions")


def test_directory_cli_requires_explicit_native_mapping(tmp_path):
    with pytest.raises(SystemExit):
        plan.main(["--motion-dir", str(tmp_path), "--run-id", "fresh", "--max-windows", "1",
                   "--output-dir", "/run/raw", "--plan-out", str(tmp_path / "p.json"),
                   "--request-out", str(tmp_path / "r.json")])


def test_staging_never_writes_active_selector_or_overwrites(tmp_path):
    source = tmp_path / "motions.json"
    source.write_text(json.dumps(motions()))
    args = ["--eligible-motions", str(source), "--run-id", "fresh", "--max-windows", "4",
            "--output-dir", "/dagger-runtime/bc_isaac", "--plan-out", str(tmp_path / "plan.json"),
            "--request-out", str(tmp_path / "request.json")]
    assert plan.main(args) == 0
    from pipeline.deployment.rtx_actor.isaac6_bc_production import validate_collection_request
    staged_request = json.loads((tmp_path / "request.json").read_bytes())
    assert validate_collection_request(staged_request) == staged_request
    assert set(p.name for p in tmp_path.iterdir()) == {"motions.json", "plan.json", "request.json"}
    with pytest.raises(FileExistsError):
        plan.main(args)
