import numpy as np
import pytest

from test.pipeline.audit_bc_smoke_contents import inspect_array


def test_nonfinite_raw_depth_reported_not_changed():
    depth = np.ones((81, 2, 2), dtype=np.float16)
    depth[0, 0, 0] = np.inf
    stats = inspect_array('depth_raw__head_color', depth)
    assert stats['nonfinite'] == 1 and stats['finite_min'] == 1
    assert np.isinf(depth[0, 0, 0])


@pytest.mark.parametrize('name', ['gt_actions', 'obs_actor_obs'])
def test_bad_supervision_rejected(name):
    data = np.zeros((81, 26), dtype=np.float32)
    data[3, 2] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        inspect_array(name, data)


def test_bad_quantization_and_frames_rejected():
    with pytest.raises(ValueError, match='14-bit'):
        inspect_array('depth_rawq__head_color', np.full((81, 2, 2), 16384, dtype=np.uint16))
    with pytest.raises(ValueError, match='81 rows'):
        inspect_array('rgb_robot__hand_left', np.zeros((80, 2, 2, 3), dtype=np.uint8))


def test_serialized_dones_are_not_actual_terminal_flags():
    with pytest.raises(ValueError, match='dones convention'):
        inspect_array('dones', np.ones(81, dtype=bool))
    assert inspect_array('dones', np.zeros(81, dtype=bool))['nonzero'] == 0
