"""Consolidation keeps one implementation, lazy imports and forwarding patches."""
import importlib
from pathlib import Path

from pipeline import success_rate


def test_explicit_modules_and_no_duplicate_compatibility_implementations():
    root = Path(success_rate.__file__).parent
    assert {path.stem for path in root.glob('*.py')} == {
        '__init__', '__main__', 'contracts', 'dispatch', 'runtime', 'resources',
        'episode_metrics', 'evaluation_loop', 'policy_actions', 'student_observations',
        'teacher_episodes', 'mixed_episodes'}
    for old, (target, prefix, names) in success_rate._COMPATIBILITY.items():
        alias = importlib.import_module('pipeline.success_rate.' + old)
        module = importlib.import_module('pipeline.success_rate.' + target)
        for name in names.split():
            assert getattr(alias, name) is getattr(module, prefix + name)


def test_legacy_monkeypatch_reaches_actual_function_globals(monkeypatch):
    from pipeline.success_rate import scaled_success_resources as old
    from pipeline.success_rate import resources
    monkeypatch.setattr(old, 'SCHEMA', 'test-only')
    assert resources.SCHEMA == 'test-only'
    assert old.summarize.__globals__['SCHEMA'] == 'test-only'
    from pipeline.success_rate import scaled_success_qualification as gate
    monkeypatch.setattr(gate, 'SCHEMA', 'other-test-only')
    assert resources.qualification_SCHEMA == 'other-test-only'
    assert gate.validate_qualification.__globals__['qualification_SCHEMA'] == 'other-test-only'
