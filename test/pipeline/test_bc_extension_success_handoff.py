"""Extension-schema dispatch only; synthetic metadata is not native evidence."""
from test.deployment.source_paths import REPO_ROOT
import os
from pathlib import Path

import pytest

from pipeline.bc_training import plan_success_test as success, training_extension
from pipeline.bc_training.handoff_common import binding, bound, publish


def fixture(tmp_path, monkeypatch, *, opt_in=True):
    root = tmp_path / 'run'; checkpoints = root / 'validation_checkpoints'
    checkpoints.mkdir(parents=True)
    best = checkpoints / 'epoch_0001_step_000160.pt'; best.write_bytes(b'cpu-fixture-not-a-policy')
    os.link(best, root / 'best_validation.pt')
    panel = binding(REPO_ROOT /
        'runtime/production-audits/20261015-raven-faithful/plan-only-native295-b2-20261015.success-panel.json')
    from pipeline.deployment.rtx_actor.production_success_contract import build_rendered_panel
    from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config
    config=rendering_config(build_rendered_panel(bound(panel)))
    rendering=publish(tmp_path/'rendering-dependencies.json',config)
    train = publish(tmp_path / 'train.json', {'split': 'train'})
    val = publish(tmp_path / 'validation.json', {'split': 'validation'})
    report = dict(schema=training_extension.ADMISSION_SCHEMA, purpose='legacy_training_extension',
        production_admitted=False, training_extension_admitted=True,
        split_files={'train': train, 'validation': val}, closed_loop_test=dict(panel=panel,
            episode_count=295, motion_bindings={str(i): 'hash' for i in range(295)}))
    admission = publish(tmp_path / 'admission.json', report)
    state = publish(root / 'validation_state.json', dict(status='complete', stop_reason='max_epochs',
        selected_checkpoint=str(root / 'best_validation.pt'), best_epoch=1, best_mse=0.2,
        train_split_sha256=train['raw_sha256'], validation_split_sha256=val['raw_sha256'],
        history=[dict(epoch=0, mse=0.9, checkpoint=str(checkpoints / 'epoch_0000.pt')),
                 dict(epoch=1, mse=0.2, checkpoint=str(best))]))
    config = publish(root / 'run_config.json', dict(output=str(root), data_admission_json=admission['path'],
        train_split=train['path'], validation_split=val['path'], allow_training_extension=opt_in,
        run_name='prod-extension-test', bootstrap='/bootstrap.pt', policy_spec='/spec.json', dataset_root=str(tmp_path)))
    called = []
    def validate(*args, **kwargs):
        called.append(kwargs)
        if not kwargs['allow_training_extension']: raise ValueError('explicit extension opt-in required')
        return report
    monkeypatch.setattr(training_extension, 'validate_training_admission', validate)
    return dict(training_state=state, training_config=config, data_admission=admission,
        resident_source_sha256='a' * 64, resident_gpu_uuid='GPU-test', resident_port=18700,
        resident_source_root='/frozen', resident_token_file='/private/token', resident_jobs_root='/jobs',
        resident_assets_root='/assets', panel_relative_path='panels/test.json',
        command_id='bc-success-extension-test',rendered_observation_config=rendering), called


def test_extension_uses_exact_opt_in_and_raw_split_hashes(tmp_path, monkeypatch):
    args, called = fixture(tmp_path, monkeypatch)
    result = success.plan(**args, output=tmp_path / 'planned.json')
    plan = bound(result['plan'])
    assert called[0]['allow_training_extension'] is True
    assert plan['training_extension_completed'] is True
    assert plan['production_training_completed'] is False
    assert plan['selected_epoch'] == 1 and plan['test_case_count'] == 295
    assert plan['success_test_completed'] is False and plan['executed'] is False
    assert plan['command']['payload']['teacher_action_mode'] == 'distill'
    assert plan['command']['payload']['diagnostic_initial_robot_refresh'] == 'student_clear_already_applied_once_v1'


def test_extension_cannot_bypass_explicit_training_opt_in(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch, opt_in=False)
    with pytest.raises(ValueError, match='opt-in'):
        success.plan(**args, output=tmp_path / 'planned.json')
    assert not (tmp_path / 'planned.json').exists()
