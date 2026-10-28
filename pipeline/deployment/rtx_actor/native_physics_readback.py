"""Bounded read-only native physics evidence; importing this is stdlib-only.

No simulation updates, setters, Kit operations, owner creation or compute APIs.
Getters below exist in the preserved PhysX tensor API; unsupported versions are
reported explicitly. A diagnostic buffer is never labelled a PhysX readback.
"""
from __future__ import annotations

import math

MAX_VALUES = 16384
MAX_JOINTS = 128


def _plain(value, limit=MAX_VALUES):
    shape = getattr(value, "shape", None)
    if shape is not None:
        n = 1
        for dim in shape:
            n *= int(dim)
        if n > limit:
            raise ValueError("readback array exceeds bound")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "torch"):
        value = value.torch.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    remaining = [limit]
    def visit(item):
        if isinstance(item, (list, tuple)):
            if len(item) > limit:
                raise ValueError("readback sequence exceeds bound")
            return [visit(x) for x in item]
        remaining[0] -= 1
        if remaining[0] < 0:
            raise ValueError("readback value count exceeds bound")
        if item is None or type(item) is bool:
            return item
        if isinstance(item, (int, float)) and math.isfinite(item):
            return item
        raise ValueError("readback contains nonfinite or unsupported value")
    return visit(value)


def _read(target, name, *, getter=False):
    if target is None:
        return {"status": "unsupported", "reason": "view_unavailable"}
    try:
        value = getattr(target, name)
    except AttributeError:
        return {"status": "unsupported", "reason": "getter_unavailable" if getter else "field_unavailable"}
    try:
        if getter:
            if not callable(value):
                return {"status": "unsupported", "reason": "getter_not_callable"}
            value = value()
            if value is None:
                return {"status": "unsupported", "reason": "getter_returned_none"}
        return {"status": "ok", "value": _plain(value)}
    except (AttributeError, NotImplementedError):
        return {"status": "unsupported", "reason": "getter_not_implemented"}
    except Exception as exc:
        # Do not print native exception messages: they can include paths/tokens.
        return {"status": "unavailable", "error_type": type(exc).__name__}


def _view(asset):
    for name in ("root_view", "root_physx_view"):
        try:
            result = getattr(asset, name)
        except AttributeError:
            continue
        if result is not None:
            return result
    return None


def _scope(env):
    if type(env.num_envs) is not int or not 1 <= env.num_envs <= 2:
        raise ValueError("readback is qualified for B1/B2 diagnostics only")
    return env.simulator, env.simulator._robot


def _known_getters(view, names):
    return {name: _read(view, name, getter=True) for name in names}


def drive_readback(env):
    """Actual backend targets and forces; never claims user targets are applied."""
    sim, robot = _scope(env)
    view = _view(robot)
    return {"schema": "roborender.native-drive-readback.v1", "num_envs": env.num_envs,
        "native_joint_order": list(robot.joint_names),
        "backend": _known_getters(view, ("get_dof_position_targets", "get_dof_velocity_targets",
            "get_dof_actuation_forces", "get_dof_projected_joint_forces")),
        "user_position_target": _read(robot.data, "joint_pos_target"),
        "staging_position_target": _read(robot, "_joint_pos_target_sim"),
        "force_semantics": {
            "get_dof_actuation_forces": "backend actuation-force field; not certified total implicit drive torque",
            "get_dof_projected_joint_forces": "incoming joint wrench projected onto DOF motion direction; not isolated motor torque"},
        "read_only": True}


def _names(value):
    if not isinstance(value, (list, tuple)) or len(value) > MAX_JOINTS:
        raise ValueError("joint name count exceeds bound")
    if any(not isinstance(x, str) or len(x) > 192 for x in value):
        raise ValueError("invalid joint name")
    return list(value)


def _usd_readback(sim, count):
    """Read exact existing prim attributes; no stage traversal or schema writes."""
    stage = getattr(getattr(sim, "sim", None), "stage", None)
    if stage is None:
        return {"status": "unsupported", "reason": "stage_unavailable"}
    fields = ("physxArticulation:solverPositionIterationCount", "physxArticulation:solverVelocityIterationCount",
        "physxArticulation:enabledSelfCollisions", "physics:collisionEnabled", "physics:approximation",
        "physxCollision:contactOffset", "physxCollision:restOffset", "physics:mass", "physics:density",
        "physics:diagonalInertia", "physics:centerOfMass", "physxRigidBody:disableGravity")
    rows = []
    suffixes = ("Robot", "Robot/pelvis", "front_object/bottle", "hold_object/bottle", "tray_object/tray")
    for lane in range(count):
        for suffix in suffixes:
            path = f"/World/envs/env_{lane}/{suffix}"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                rows.append({"lane": lane, "path": path, "status": "unsupported", "reason": "prim_path_unavailable"})
                continue
            attrs = {}
            for name in fields:
                attr = prim.GetAttribute(name)
                if not attr:
                    attrs[name] = {"status": "unsupported", "reason": "attribute_unavailable"}
                    continue
                try:
                    value = attr.Get()
                    attrs[name] = {"status": "ok", "value": _plain(value),
                        "authored": bool(attr.HasAuthoredValueOpinion())}
                except Exception as exc:
                    attrs[name] = {"status": "unavailable", "error_type": type(exc).__name__}
            material = {"status": "unsupported", "reason": "usd_material_api_unavailable"}
            try:
                from pxr import UsdShade
                bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                if bound:
                    mp = bound.GetPrim()
                    properties = {}
                    for name in ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution"):
                        attr = mp.GetAttribute(name)
                        properties[name] = _plain(attr.Get()) if attr else None
                    material = {"status": "ok", "path": str(bound.GetPath()), "purpose_requested": "physics",
                        "attributes": properties, "effective_friction_inferred": False,
                        "fallback": "unresolved; USD binding may fall back to all-purpose/default; use backend material getter"}
                else:
                    material = {"status": "unsupported", "reason": "no_bound_material", "effective_friction_inferred": False}
            except Exception as exc:
                material = {"status": "unavailable", "error_type": type(exc).__name__}
            rows.append({"lane": lane, "path": path, "status": "ok", "attributes": attrs, "physics_material": material})
    return {"status": "ok", "rows": rows, "coverage": "fixed paths only; not an exhaustive collider traversal"}


def physics_readback(env):
    """One bounded two-lane configuration readback; no stepping/asset mutation."""
    sim, robot = _scope(env)
    native_names = _names(robot.joint_names)
    names = _names(sim.dof_names)
    ids = _plain(sim.dof_ids, MAX_JOINTS)
    if (len(names) != len(ids) or any(type(i) is not int or not 0 <= i < len(native_names) for i in ids)):
        raise ValueError("invalid native-to-config joint map")
    view = _view(robot)
    getters = ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_max_forces", "get_dof_max_velocities",
        "get_dof_armatures", "get_dof_friction_properties", "get_masses", "get_inertias",
        "get_material_properties", "get_contact_offsets", "get_rest_offsets")
    actuators = []
    for name, actuator in robot.actuators.items():
        if len(actuators) >= 8 or not isinstance(name, str) or len(name) > 192:
            raise ValueError("actuator count/name exceeds bound")
        indices = getattr(actuator, "joint_indices", None)
        if indices is None or isinstance(indices, slice):
            if isinstance(indices, slice) and indices != slice(None):
                raise ValueError("partial actuator slice is unsupported")
            indices = list(range(len(native_names)))
        actuators.append({"name": name, "class": type(actuator).__module__ + "." + type(actuator).__name__,
            "is_implicit_model": _read(actuator, "is_implicit_model"), "joint_names": _names(actuator.joint_names),
            "joint_indices": _plain(indices, MAX_JOINTS),
            "configured": {field: _read(actuator, field) for field in
                ("stiffness", "damping", "effort_limit_sim", "velocity_limit_sim", "armature", "friction")}})
    tasks = {}
    for name in ("front_object", "hold_object", "tray_object"):
        key = getattr(env.config.task, name)
        task = sim._task.get(key)
        tasks[name] = _known_getters(_view(task), ("get_masses", "get_inertias", "get_material_properties",
            "get_contact_offsets", "get_rest_offsets"))
    return {"schema": "roborender.native-physics-readback.v1", "num_envs": env.num_envs,
        "joint_mapping": {"native_joint_names": native_names, "simulator_joint_names": names,
            "simulator_native_joint_indices": ids, "mapping_names_match": [native_names[i] for i in ids] == names,
            "right_finger_simulator_indices": _read(env, "_right_hand_dof_idx")},
        "has_implicit_actuators": _read(robot, "_has_implicit_actuators"),
        "has_newton_actuators": _read(robot, "_has_newton_actuators"),
        "actuators": actuators, "robot_backend": _known_getters(view, getters), "tasks_backend": tasks,
        "material_columns": ["static_friction", "dynamic_friction", "restitution"],
        "shape_to_body_mapping": "unsupported; material/offset arrays remain in native shape order",
        "usd": _usd_readback(sim, env.num_envs), "read_only": True}
