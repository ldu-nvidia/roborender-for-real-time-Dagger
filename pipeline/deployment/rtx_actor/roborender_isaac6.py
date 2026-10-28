"""Explicit Raven (WXYZ/Torch) -> IsaacLab 3 (XYZW/ProxyArray) boundary.

Migration candidate only. Never import this from the certified Isaac 5 runtime.
No policy, action, task, timing, or reward logic is changed here.
"""
from __future__ import annotations

import torch


def asset_base(name):
    # Lab3 factories return a backend implementation derived from Base*, not
    # an instance of the public factory class used to construct it.
    from isaaclab.assets.articulation.base_articulation import BaseArticulation
    from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
    from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
    return {"Articulation": BaseArticulation, "RigidObject": BaseRigidObject,
            "DeformableObject": BaseDeformableObject}[name]


def as_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        return {key: as_torch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [as_torch(item) for item in value]
    if type(value).__module__.startswith("isaaclab") and hasattr(value, "torch"):
        return value.torch
    if type(value).__module__.startswith("warp"):
        import warp as wp
        return wp.to_torch(value)
    return value


def _reorder(value, order):
    if value is None:
        return None
    value = as_torch(value)
    if isinstance(value, tuple):
        return tuple(value[index] for index in order)
    if isinstance(value, list):
        return [value[index] for index in order]
    if value.shape[-1] != len(order):
        raise ValueError(f"invalid quaternion shape: {value.shape}")
    return value[..., order]


def xyzw(value):
    """Convert Raven WXYZ to Lab3 XYZW without mutating its input."""
    return _reorder(value, [1, 2, 3, 0])


def wxyz(value):
    """Convert Lab3 XYZW to Raven WXYZ without mutating its input."""
    return _reorder(value, [3, 0, 1, 2])


def root_state(value, *, to_lab):
    value = as_torch(value)
    if value.shape[-1] != 13:
        raise ValueError(f"invalid root state shape: {value.shape}")
    result = value.clone()
    result[..., 3:7] = xyzw(value[..., 3:7]) if to_lab else wxyz(value[..., 3:7])
    return result


class _LegacyData:
    def __init__(self, native_data):
        self._native_data = native_data

    def __getattr__(self, name):
        value = as_torch(getattr(self._native_data, name))
        if name in {"root_state_w", "default_root_state"}:
            return root_state(value, to_lab=False)
        if name in {"body_quat_w", "target_quat_source", "root_quat_w"}:
            return wxyz(value)
        return value


def data(asset):
    return _LegacyData(asset.data)


def write_root_state(asset, state, env_ids):
    return asset.write_root_state_to_sim(root_state(state, to_lab=True), env_ids)


def set_world_poses(camera, positions, orientations, **kwargs):
    return camera.set_world_poses(positions, xyzw(orientations), **kwargs)


def visualize(markers, positions, orientations=None, scales=None):
    return markers.visualize(positions, xyzw(orientations), scales)


def external_wrench(asset, component):
    if component not in {"force", "torque"}:
        raise ValueError(component)
    return as_torch(getattr(asset.permanent_wrench_composer, f"out_{component}_b"))


class _PhysxView:
    """Convert only Raven's direct PhysX accesses; do not patch Lab internals."""
    def __init__(self, asset):
        self.view = asset.root_view

    def get_coms(self):
        # PhysX's low-level COM quaternion convention is XYZW in both releases.
        return as_torch(self.view.get_coms())

    def set_coms(self, values, indices):
        import warp as wp
        return self.view.set_coms(wp.from_torch(values.contiguous()),
                                  wp.from_torch(indices.to(dtype=torch.int32).contiguous()))

    def get_disable_gravities(self):
        return as_torch(self.view.get_disable_gravities())

    def set_disable_gravities(self, values, indices):
        import warp as wp
        return self.view.set_disable_gravities(
            wp.from_torch(values.contiguous()),
            wp.from_torch(indices.to(dtype=torch.int32).contiguous()))


def physx_view(asset):
    return _PhysxView(asset)


def quat_from_euler_xyz(*args, **kwargs):
    from isaaclab.utils.math import quat_from_euler_xyz as native
    return wxyz(native(*args, **kwargs))


def quat_from_matrix(*args, **kwargs):
    from isaaclab.utils.math import quat_from_matrix as native
    return wxyz(native(*args, **kwargs))


def euler_xyz_from_quat(quat, *args, **kwargs):
    from isaaclab.utils.math import euler_xyz_from_quat as native
    return native(xyzw(quat), *args, **kwargs)


def axis_angle_from_quat(quat, *args, **kwargs):
    from isaaclab.utils.math import axis_angle_from_quat as native
    return native(xyzw(quat), *args, **kwargs)


def wrap_to_pi(*args, **kwargs):
    from isaaclab.utils.math import wrap_to_pi as native
    return native(*args, **kwargs)


def add_update_semantics(prim, label):
    from isaacsim.core.experimental.utils.semantics import add_labels
    add_labels(prim, labels=[label], taxonomy="class")


def require_integer_semantic_ids(camera, *, synthetic_data=None):
    """Restore Raven's raw-ID contract after Replicator attaches its nodes.

    Lab3 passes mapping='{}' to Replicator 1.13.25. Its attach path treats
    len(keys)*4 == len(values) (0 == 0) as a color mapping and overrides
    colorize=False. Explicitly restore both nodes on this camera's own render
    product, before Raven consumes any segmentation. No labels are fabricated
    and no color-to-class heuristic is used.
    """
    if camera._renderer.cfg.colorize_semantic_segmentation is not False:
        raise RuntimeError("Raven conditioning requires integer semantic IDs")
    render_data = camera._render_data
    if "semantic_segmentation" not in render_data.annotators or not render_data.render_product_paths:
        raise RuntimeError("semantic annotator is not initialized")
    if synthetic_data is None:
        from omni.syntheticdata import SyntheticData
        synthetic_data = SyntheticData.Get()
    for product_path in render_data.render_product_paths:
        if not isinstance(product_path, str) or not product_path.startswith("/"):
            raise RuntimeError("invalid semantic render-product path")
        for name in ("SemanticSegmentationPostRender", "semantic_segmentation"):
            synthetic_data.set_node_attributes(
                name, {"inputs:colorize": False}, product_path
            )


def get_prim_at_path(*args, **kwargs):
    from isaaclab.sim.utils import get_prim_at_path as native
    return native(*args, **kwargs)


def create_prim(*args, **kwargs):
    from isaaclab.sim.utils import create_prim as native
    if kwargs.get("orientation") is not None:
        kwargs["orientation"] = xyzw(kwargs["orientation"])
    return native(*args, **kwargs)
