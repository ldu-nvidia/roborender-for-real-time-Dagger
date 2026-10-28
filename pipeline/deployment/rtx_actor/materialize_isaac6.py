"""Generate a hash-bound, separately mounted Raven compatibility overlay.

Inputs are immutable. Unknown source versions, overlapping edits and reuse of an
output directory fail closed. The original Raven trainer/teacher bytes remain
unchanged; the output manifest names the simulator port explicitly.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

BASE_COMMIT = "94d79851d9c471b1210be81c75116d9bea835a4d"
SOURCE_PREFIX = "pipeline/GR00T-VisualSim2Real/"
PINS = {
    "gr00t/rl/envs/legged_base_task/legged_robot_base.py": "78d0a736125d09c9ebe650e871e87338bdcb463941dca949d4a347edb70dde5c",
    "gr00t/rl/envs/base_task/loco_manip_base.py": "b2fb387386e2ab6f72860467859d8eba440c9198b3fb869034e25096b2c1900e",
    "gr00t/rl/train_agent_trl.py": "85005db0cab61eea52a01d2d6f4860e868703935bfa78c8576458ccc7882b565",
    "gr00t/rl/simulator/isaacsim/isaacsim.py": "1d03ebca26f8c5f64d65d2146e79bbf8ceab6af2b83dfb65afa690b784b1073f",
    "gr00t/rl/simulator/isaacsim/events.py": "52e87b21b190440ec2a822edd44338f13ef5e657617a965132a07e07d41b9433",
    "gr00t/rl/simulator/isaacsim/utils.py": "4628ed5a0841144d7b3a5e71382b671a3fb7cbe01225a0a23a8390b71eb61d75",
    "gr00t/rl/isaac_utils/playground/utils/math_utils.py": "303ad124dc0075536fd552dfda30ef5a61625158c03e267d63911bc420687967",
    "gr00t/rl/envs/loco_manip/walk_stand_grasp.py": "d985d3e74414b6ceabefd917bdaa3b36c67ff9e141882436f931815f16bd3be9",
    "gr00t/rl/envs/base_task/staged_task_base.py": "c67503aad0a9ebf9c66c6dcc629f5ea0872b58451fe3d3a098b83faf1d949e26",
    "gr00t/rl/isaac_utils/playground/env_rand/table_grid.py": "480576c511c74614509e7301d1498a58c8d1206e3197608818c0e73b736b2ed6",
    "gr00t/rl/isaac_utils/playground/env_rand/domelight.py": "401e78cef85abacfc2d4838928a905787ce9b687d8c1daeefbe9adec48077e38",
    "gr00t/rl/data/tasks/walk_stand_place_grasp_turn_homie/scenario_cfg/isaacsim.py": "1a1531fcedd29c4ef6f89b514d59794084afb8fabc0a02589adb61d0ef78c473",
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"expected exactly one occurrence: {before!r}")
    return text.replace(before, after, 1)


def replace_ast(text, replacements):
    """Replace exact AST source ranges while retaining all untouched source."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node, replacement in replacements:
        # AST column offsets are UTF-8 bytes, not Unicode code points.
        start_col = len(lines[node.lineno - 1].encode()[:node.col_offset].decode())
        end_col = len(lines[node.end_lineno - 1].encode()[:node.end_col_offset].decode())
        edits.append((offsets[node.lineno - 1] + start_col,
                      offsets[node.end_lineno - 1] + end_col, replacement))
    edits.sort()
    if any(left[1] > right[0] for left, right in zip(edits, edits[1:])):
        raise ValueError("overlapping compatibility edits")
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


def boundary_edits(text, *, simulator):
    edits = []
    for node in ast.walk(ast.parse(text)):
        if (simulator and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance" and len(node.args) == 2
                and isinstance(node.args[1], ast.Name)
                and node.args[1].id in {"Articulation", "RigidObject", "DeformableObject"}):
            edits.append((node.args[1], f"rr6.asset_base({node.args[1].id!r})"))
        if isinstance(node, ast.Attribute):
            source = ast.get_source_segment(text, node.value)
            if node.attr == "root_physx_view":
                edits.append((node, f"rr6.physx_view({source})"))
            if simulator and node.attr == "data":
                edits.append((node, f"rr6.data({source})"))
            if simulator and node.attr in {"_external_force_b", "_external_torque_b"}:
                component = "force" if node.attr == "_external_force_b" else "torque"
                edits.append((node, f"rr6.external_wrench({source}, {component!r})"))
        if simulator and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            function = node.func
            if function.attr in {"write_root_state_to_sim", "set_world_poses", "visualize"}:
                # All such calls in this exact simulator file target Lab assets,
                # cameras or markers, not the simulator's public wrapper methods.
                target = ast.get_source_segment(text, function.value)
                args = [ast.get_source_segment(text, item) for item in node.args]
                args += [f"{item.arg}={ast.get_source_segment(text, item.value)}"
                         for item in node.keywords]
                renamed = {"write_root_state_to_sim": "write_root_state",
                           "set_world_poses": "set_world_poses", "visualize": "visualize"}
                edits.append((node, f"rr6.{renamed[function.attr]}({', '.join([target] + args)})"))
            if ast.unparse(function) == "TiledCameraCfg.OffsetCfg":
                for keyword in node.keywords:
                    if keyword.arg == "rot":
                        expr = ast.get_source_segment(text, keyword.value)
                        edits.append((keyword.value, f"rr6.xyzw({expr})"))
    text = replace_ast(text, edits)
    # Import after __future__ when it exists, before normal imports otherwise.
    if "from __future__ import annotations\n" in text:
        return replace_once(text, "from __future__ import annotations\n",
                            "from __future__ import annotations\nimport roborender_isaac6 as rr6\n")
    return "import roborender_isaac6 as rr6\n" + text


def transform(relative, text):
    if relative in {"gr00t/rl/envs/base_task/loco_manip_base.py",
                    "gr00t/rl/envs/legged_base_task/legged_robot_base.py"}:
        # The active WalkStandPlaceGraspTurnHomie hierarchy inherits from
        # StagedTaskBase -> LeggedRobotBase, not LocoManipBase. Both independent
        # implementations require the same post-write bookkeeping correction.
        methods = [n for n in ast.walk(ast.parse(text))
                   if isinstance(n, ast.FunctionDef) and n.name == "reset_all"]
        if len(methods) != 1:
            raise ValueError("expected exactly one pinned reset_all")
        method = methods[0]
        tail = method.body[-1]
        if not isinstance(tail, ast.Return) or ast.unparse(tail.value) != "self.obs_buf_dict":
            raise ValueError("reset_all completion boundary differs")
        first = method.body[0]
        text = replace_ast(text, [
            (first, "from initial_reset_contract import begin_initial_reset, complete_initial_reset\n"
                    "        begin_initial_reset(self)\n        " + ast.get_source_segment(text, first)),
            (tail, "complete_initial_reset(self)\n        return self.obs_buf_dict"),
        ])
    elif relative.endswith("/train_agent_trl.py"):
        start = text.index("        # Copy headless rendering kit file")
        end = text.index("        app_launcher = AppLauncher(args_cli)", start)
        text = text[:start] + (
            "        # Versioned migration: use Lab3's native Kit, never overlay Kit5.1.\n"
            "        if args_cli.enable_cameras and args_cli.headless:\n"
            "            args_cli.experience = '/workspace/isaaclab/apps/isaaclab.python.headless.rendering.kit'\n\n"
        ) + text[end:]
    elif relative == "gr00t/rl/simulator/isaacsim/isaacsim.py":
        text = boundary_edits(text, simulator=True)
        text = replace_once(text, "from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext",
                            "from isaaclab.sim import SimulationCfg, SimulationContext\n    from isaaclab_physx.physics import PhysxCfg\n    from isaaclab_physx.renderers import IsaacRtxRendererCfg")
        text = replace_once(text, "physx=PhysxCfg(", "physics=PhysxCfg(")
        text = replace_once(text, "if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:",
                            "if self.sim.has_gui:")
        if text.count("self.sim.has_gui() or self.sim.has_rtx_sensors()") != 2:
            raise ValueError("rendering guard source mismatch")
        text = text.replace("self.sim.has_gui() or self.sim.has_rtx_sensors()", "self.sim.is_rendering")
        for expression in ("obj_cfg.init_state.rot", "robot_articulation_config.init_state.rot"):
            text = replace_once(text, f"torch.tensor({expression}, device=self.sim_device)",
                                f"rr6.wxyz(torch.tensor({expression}, device=self.sim_device))")
        text = replace_once(text, "from isaacsim.core.utils.semantics import add_update_semantics",
                            "from roborender_isaac6 import add_update_semantics")
        text = replace_once(text, "translation=wpos, orientation=wrot,",
                            "translation=wpos, orientation=rr6.xyzw(wrot),")
        # Lab3 owns annotator options on the renderer, not the deprecated
        # CameraCfg field. Pin an independent renderer cfg so camera copying
        # cannot silently return colorized RGBA labels instead of integer IDs.
        text = replace_once(text, "colorize_semantic_segmentation=False,",
                            "renderer_cfg=IsaacRtxRendererCfg(colorize_semantic_segmentation=False),")
        text = replace_once(
            text, "                self.sim.reset()",
            "                self.sim.reset()\n"
            "                for _camera in getattr(self, 'conditioning_cameras', {}).values():\n"
            "                    rr6.require_integer_semantic_ids(_camera)",
        )
        # Lab2 expanded env_.*/name and assigned assets in environment order.
        # Lab3 instead expands a wildcard in the asset basename. Explicit paths
        # retain Raven's one already-selected asset per environment (including
        # its control-camera replica scenes), without sampling or renaming.
        text = replace_once(
            text,
            "obj_cfg.spawn.assets_cfg = obj_cfg.spawn.assets_cfg[: self.scene.cfg.num_envs]",
            "obj_cfg.spawn.assets_cfg = obj_cfg.spawn.assets_cfg[: self.scene.cfg.num_envs]\n"
            "                    obj_cfg.spawn.spawn_paths = [\n"
            "                        f'/World/envs/env_{i}/{name}'\n"
            "                        for i in range(self.scene.cfg.num_envs)\n"
            "                    ]",
        )
    elif relative.endswith("/events.py"):
        text = boundary_edits(text, simulator=False)
    elif relative.endswith("/staged_task_base.py"):
        text = boundary_edits(text, simulator=True)
    elif relative.endswith("/utils.py") or relative.endswith("/utils/math_utils.py"):
        text = replace_once(text, "from isaaclab.utils.math import", "from roborender_isaac6 import")
    elif relative.endswith("/walk_stand_grasp.py"):
        text = replace_once(text, "from isaaclab.utils.math import", "from roborender_isaac6 import")
    elif relative.endswith("/table_grid.py") or relative.endswith("/domelight.py"):
        text = replace_once(text, "import isaacsim.core.utils.prims as prim_utils",
                            "import roborender_isaac6 as prim_utils")
        if relative.endswith("/table_grid.py"):
            text = replace_once(
                text,
                "    if orientation is None:\n        orientation = (1.0, 0.0, 0.0, 0.0)",
                "    # The Lab3 spawner supplies XYZW; Raven Rotation consumes WXYZ.\n"
                "    if orientation is None:\n        orientation = (1.0, 0.0, 0.0, 0.0)\n"
                "    else:\n        orientation = prim_utils.wxyz(orientation)",
            )
        else:
            text = replace_once(text, "translation=translation, orientation=orientation)",
                                "translation=translation, orientation=prim_utils.wxyz(orientation))")
    elif relative.endswith("/scenario_cfg/isaacsim.py"):
        text = replace_once(text, "rot=[1.0, 0.0, 0.0, 0.0]", "rot=[0.0, 0.0, 0.0, 1.0]")
    else:
        raise ValueError(f"unknown port target: {relative}")
    ast.parse(text, filename=relative)
    return text


def materialize(source_root, output, assets_manifest):
    sources = {}
    for relative, expected in PINS.items():
        payload = (source_root / SOURCE_PREFIX / relative).read_bytes()
        if sha(payload) != expected:
            raise ValueError(f"base Raven source hash mismatch: {relative}")
        sources[relative] = payload
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for relative, payload in sources.items():
        converted = transform(relative, payload.decode()).encode()
        target = output / SOURCE_PREFIX / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(converted)
        entries.append({"path": SOURCE_PREFIX + relative, "source_sha256": sha(payload),
                        "derived_sha256": sha(converted), "bytes": len(converted)})
    helper = Path(__file__).with_name("roborender_isaac6.py")
    shutil.copy2(helper, output / helper.name)
    reset_helper = Path(__file__).with_name("initial_reset_contract.py")
    shutil.copy2(reset_helper, output / reset_helper.name)
    capture = Path(__file__).with_name("isaac6_capture_trainer.py")
    shutil.copy2(capture, output / capture.name)
    bc_trainer = Path(__file__).with_name("isaac6_bc_trainer.py")
    shutil.copy2(bc_trainer, output / bc_trainer.name)
    manifest = json.loads(assets_manifest.read_text())
    by_path = {entry["path"]: entry for entry in entries}
    for artifact in manifest["artifacts"]:
        relative = str(Path(artifact["path"]).relative_to(manifest["raven_root"]))
        if relative in by_path:
            entry = by_path[relative]
            artifact["source_sha256"] = artifact["sha256"]
            artifact["sha256"], artifact["bytes"] = entry["derived_sha256"], entry["bytes"]
    derived = {"schema": "roborender.isaac6.migration.v1", "base_raven_commit": BASE_COMMIT,
               "isaac_sim": "6.0.0", "isaaclab_commit": "28a37cecdd433c22d9eabd6a5954add9f13a8951",
               "production_qualified": False, "helper_sha256": sha(helper.read_bytes()),
               "capture_trainer_sha256": sha(capture.read_bytes()), "files": entries}
    derived["bc_qualification_trainer_sha256"] = sha(bc_trainer.read_bytes())
    derived["initial_reset_contract"] = "roborender.initial-reset-completion.v1"
    derived["initial_reset_helper_sha256"] = sha(reset_helper.read_bytes())
    manifest["derived_runtime"] = derived
    (output / "port-manifest.json").write_text(json.dumps(derived, indent=2) + "\n")
    (output / "assets.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_root, args.output, args.assets_manifest), indent=2))


if __name__ == "__main__":
    main()
