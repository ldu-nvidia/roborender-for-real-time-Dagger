"""The full-resolution replay is CPU-only and cannot claim native capacity."""
import inspect
import pytest
from test.deployment.rtx_actor import bc_bounded_captured_input as probe


def test_missing_cpu_scope_rejected_before_input_reads(monkeypatch, tmp_path):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with pytest.raises(ValueError, match='CPU-only'):
        probe.compare(tmp_path, tmp_path / 'missing', tmp_path / 'out',
                      legacy_order_proof=tmp_path / 'missing-proof')


def test_reuses_full_provenance_checks_and_labels_new_writer():
    source = inspect.getsource(probe.compare)
    assert 'return original.compare(runtime, reference_path, output,' in source
    assert 'storage.write_window(self)' in source
    assert 'native_payload_writer_unchanged=False' in source
    assert 'b32_capacity_certified=False' in source
    assert 'cuda_transfer_tested=False' in source
    assert 'rng_unchanged=True' in source


def test_actual_comparator_with_bounded_writer_preserves_all_outputs(tmp_path, monkeypatch):
    from test.pipeline.test_bc_native_same_input import recorded_fixture, REFERENCE
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    runtime = recorded_fixture(tmp_path)
    result = probe.compare(runtime, REFERENCE, tmp_path / 'bounded-output',
                           legacy_order_proof=None)
    assert result['verdict'] == 'PASS'
    assert result['schema'] == 'roborender.bc-bounded-captured-input.v1'
    assert result['chunks'] > 0 and result['windows'] == 4
    assert result['rng_unchanged'] is True
    assert result['native_payload_writer_unchanged'] is False
    assert result['b32_capacity_certified'] is False
