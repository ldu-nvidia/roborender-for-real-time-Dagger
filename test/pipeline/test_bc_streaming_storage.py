from types import SimpleNamespace
import numpy as np
import pytest

from pipeline.bc_collection import streaming_storage as disk


def free(monkeypatch, amount):
    monkeypatch.setattr(disk.os, 'statvfs', lambda path: SimpleNamespace(f_frsize=1, f_bavail=amount))


def test_policy_keeps_512mib_ceiling_and64gib_reserve():
    value = disk.policy(32)
    assert value['window_admission_bytes'] == 81 * disk.GIB
    assert value['whole_campaign_fit_guaranteed'] is False
    assert value['shortage_is_success'] is False


def test_window_headroom_does_not_use_compressed_mean(tmp_path, monkeypatch):
    value = disk.StreamingDiskGuard(tmp_path, 32)
    free(monkeypatch, 81 * disk.GIB)
    value.admit_window()
    free(monkeypatch, 81 * disk.GIB - 1)
    with pytest.raises(RuntimeError, match='next full window'):
        value.admit_window()
    assert value.windows_admitted == 1


def test_prewrite_size_bound_and_actual_incompressible_zip(tmp_path, monkeypatch):
    value = disk.StreamingDiskGuard(tmp_path, 32)
    free(monkeypatch, 82 * disk.GIB)
    arrays = dict(rgb=np.random.default_rng(73).integers(0, 256, (81, 32, 32, 3), dtype=np.uint8),
                  prompt=np.array('teacher'), gt_actions=np.zeros((81, 26), dtype=np.float32))
    target = tmp_path / 'chunk.npz'
    bound = value.admit_archive(target, arrays)
    np.savez_compressed(target, **arrays)
    value.verify_archive(target, bound)
    assert target.stat().st_size < bound and value.emissions_verified == 1
    with pytest.raises(ValueError, match='fresh'):
        value.admit_archive(target, arrays)


def test_production_camera_schema_fits_ceiling_without_assuming_compression():
    values = {}
    for view in ('head_color', 'hand_left', 'hand_right'):
        for name, shape, dtype in (
            ('depth_rawq', (81,480,832), 'uint16'), ('depth_raw', (81,240,416), 'float16'),
            ('depth_ctrl', (81,240,416), 'uint8'), ('rgb_robot', (81,240,416,3), 'uint8')):
            values[f'{name}__{view}'] = np.broadcast_to(np.array(0, dtype=dtype), shape)
    values['gt_actions'] = np.zeros((81,26), dtype='float32')
    # Wider than current observations: size bound is checked again on actual data.
    values['obs'] = np.zeros((81,2048), dtype='float32')
    assert 320 << 20 < disk.archive_size_bound(values) < 512 << 20


def test_shortage_before_write_and_output_replacement_fail_closed(tmp_path, monkeypatch):
    root = tmp_path / 'output'
    root.mkdir()
    value = disk.StreamingDiskGuard(root, 32)
    free(monkeypatch, 65 * disk.GIB)
    with pytest.raises(RuntimeError, match='before archive'):
        value.admit_archive(root / 'x.npz', dict(x=np.zeros(1)))
    root.rename(tmp_path / 'preserved')
    root.mkdir()
    free(monkeypatch, 100 * disk.GIB)
    with pytest.raises(RuntimeError, match='identity changed'):
        value.admit_window()


@pytest.mark.parametrize('values', [dict(x=np.array([object()])), {'../x':np.zeros(1)},
    dict(x=np.broadcast_to(np.array(0, dtype='uint8'), (512 << 20,)))])
def test_object_path_and_oversized_payloads_rejected(values):
    with pytest.raises((RuntimeError, ValueError)):
        disk.archive_size_bound(values)
