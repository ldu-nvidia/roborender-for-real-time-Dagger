"""Tiny files only; these tests never hash or claim real model evidence."""
import json
from pathlib import Path

import pytest

from pipeline.bc_training.render_asset_identity import hash_regular, produce, sha
from pipeline.bc_training.production_protocol import verify_render_asset_identity


def tiny_registry(tmp_path):
    for name in ("base/weights", "offline-adapter", "online-adapter", "engine", "source.py", "prompt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"raw")
    artifact = lambda name, path: dict(id=name, relative_path=path, bytes=3, sha256=sha(b"raw"))
    registry = dict(schema="roborender.renderer_registry.v1", artifacts=[
        dict(id="wan21_fun_control_1p3b_foundation", relative_root="base",
             files=[dict(path="weights", bytes=3, sha256=sha(b"raw"))]),
        artifact("roborender_offline_bc_video_teacher_50step_lora_step05294", "offline-adapter"),
        artifact("roborender_online_dagger_image3f_5step_lora_step05000", "online-adapter"),
        artifact("roborender_online_dagger_image3f_sm90_bf16_3view", "engine")])
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    return path


def test_streaming_producer_is_checker_compatible_with_optional_online_receipts(tmp_path):
    registry = tiny_registry(tmp_path)
    bindings = produce(registry_path=registry, renderer_root=tmp_path,
        source_files={"source.py": tmp_path / "source.py"}, prompt_path=tmp_path / "prompt",
        output_dir=tmp_path / "proof", include_online_assets=True)
    assert verify_render_asset_identity(bindings) == bindings["render_asset_identity"]
    receipt = json.loads(Path(bindings["render_asset_identity"]["path"]).read_bytes())
    assert len(receipt["additional_online_assets"]) == 2
    assert receipt["no_cuda_or_model_load"] is True
    with pytest.raises(FileExistsError):
        produce(registry_path=registry, renderer_root=tmp_path,
            source_files={"source.py": tmp_path / "source.py"}, prompt_path=tmp_path / "prompt",
            output_dir=tmp_path / "proof")


def test_streaming_hash_rejects_wrong_registry_and_mutation(tmp_path):
    path = tmp_path / "small"
    path.write_bytes(b"abcdef")
    assert hash_regular(path, expected_sha256=sha(b"abcdef"), expected_bytes=6, block_bytes=2)["bytes"] == 6
    with pytest.raises(ValueError, match="size differs"): hash_regular(path, expected_bytes=5)
    with pytest.raises(ValueError, match="SHA256 differs"): hash_regular(path, expected_sha256="0" * 64)


def test_unchanged_big_assets_reuse_prior_bound_identity_without_reread(tmp_path, monkeypatch):
    import pipeline.bc_training.render_asset_identity as module
    registry = tiny_registry(tmp_path)
    kwargs = dict(registry_path=registry, renderer_root=tmp_path, source_files={"source.py": tmp_path / "source.py"},
                  prompt_path=tmp_path / "prompt", include_online_assets=True)
    first = produce(**kwargs, output_dir=tmp_path / "first")
    original = module.hash_regular
    def metadata_only(path, **options):
        assert Path(path).name in {"source.py", "prompt"}
        return original(path, **options)
    monkeypatch.setattr(module, "hash_regular", metadata_only)
    second = produce(**kwargs, output_dir=tmp_path / "second",
        reuse_receipt=first["render_asset_identity"]["path"], reuse_receipt_sha256=first["render_asset_identity"]["raw_sha256"])
    assert verify_render_asset_identity(second)
