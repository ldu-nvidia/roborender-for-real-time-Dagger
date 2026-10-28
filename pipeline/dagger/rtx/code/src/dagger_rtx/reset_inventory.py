"""Versioned full Raven reset inventory with paired local scene lanes.

Only the IDs passed to Raven's motion sampler are translated. Physics/reset
callbacks still receive local lane IDs; sampling/resampling remain native.
"""
from pathlib import Path
import hashlib
from .contracts import canonical_json

CONFIG_FIELD = 'reset_inventory_protocol'
PROTOCOL = 'roborender.raven-full-reset-inventory.v1'


def binding(motions, lane_offset):
    return dict(protocol=PROTOCOL, motion_inventory_sha256=hashlib.sha256(canonical_json(motions)).hexdigest(),
                reset_lane_offset=lane_offset, reset_motion_count=len(motions))


def validate_binding(value, *, lane_offset):
    if (not isinstance(value, dict) or set(value) != {'protocol','motion_inventory_sha256','reset_lane_offset','reset_motion_count'}
            or value['protocol'] != PROTOCOL or type(value['reset_lane_offset']) is not int
            or value['reset_lane_offset'] != lane_offset or type(value['reset_motion_count']) is not int
            or value['reset_motion_count'] != 295):
        raise ValueError('exact full295 native reset inventory binding required')
    from .contracts import require_sha256
    require_sha256(value['motion_inventory_sha256'], 'full reset inventory')
    return value


def descriptors(inventory, motion_root):
    """Preserve the boot-verified inventory order, including unassigned motions."""
    root = Path(motion_root)
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('explicit native reset motion root required')
    return [dict(motion_id=r['motion_id'], path=str(root/r['filename']), sha256=r['sha256'])
            for r in inventory['motions']]


class FullResetInventoryMixin:
    def _get_sample_idx(self, env_ids, processed_ratio=None, reset_from_teleop_start=False):
        offset = getattr(self.config.reset_from_dataset, 'roborender_reset_lane_offset', None)
        if offset is not None:
            if (type(offset) is not int or offset < 0
                    or len(self.reset_dataset.sample_idx_based_on_demo_clip_number)
                    != len(self.reset_dataset.motion_files)
                    or offset + self.num_envs > len(self.reset_dataset.motion_files)):
                raise ValueError('full reset inventory lost an eligible motion or lane mapping')
            env_ids = env_ids + offset
        return super()._get_sample_idx(env_ids, processed_ratio=processed_ratio,
                                      reset_from_teleop_start=reset_from_teleop_start)
