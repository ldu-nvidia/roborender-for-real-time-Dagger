"""No GPU calls: default launches cannot silently select historical validation."""
import sys

import pytest

from pipeline.deployment.training_launch_policy import require_training_launch


def test_no_validation_protocol_remains_allowed():
    require_training_launch(validation_enabled=False)


def test_legacy_validation_requires_explicit_opt_in():
    with pytest.raises(ValueError, match='MSE validation is disabled'):
        require_training_launch(validation_enabled=True)
    require_training_launch(validation_enabled=True, allow_legacy_validation=True)


@pytest.mark.parametrize('value', ['false', 'true', None, 0, 1])
def test_strings_and_integer_flags_cannot_bypass_policy(value):
    with pytest.raises(ValueError, match='explicit boolean'):
        require_training_launch(validation_enabled=value)
    with pytest.raises(ValueError, match='explicit boolean'):
        require_training_launch(validation_enabled=True, allow_legacy_validation=value)


def test_old_mode_cli_refuses_execution_before_reading_settings(tmp_path, monkeypatch):
    from pipeline.deployment import run_mode
    output=tmp_path/'not-created'
    monkeypatch.setattr(sys,'argv',['run_mode','--mode','smoke','--settings',
        str(tmp_path/'missing-settings.json'),'--output',str(output),'--execute'])
    with pytest.raises(ValueError, match='MSE validation is disabled'):
        run_mode.main()
    assert not output.exists()


@pytest.mark.parametrize('schema_name,extra_args,rejection', [
    ('SCALED_FRESH_SCHEMA', [], 'MSE validation is disabled'),
    ('FIXED100_QUEUE_SCHEMA', [], 'historical fixed100 cohort execution requires'),
    ('FIXED100_QUEUE_SCHEMA', ['--allow-legacy-cohort-lifecycles'], None),
    ('CONTINUOUS_FIXED100_QUEUE_SCHEMA', [], None),
    ('CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA', [], None),
])
def test_queue_gate_precedes_controller_and_preserves_fixed100(
        tmp_path, monkeypatch, capsys, schema_name, extra_args, rejection):
    from pipeline.dagger import coverage_queue
    config=tmp_path/'queue.json';config.write_text('{}')
    output=tmp_path/'queue-root'
    monkeypatch.setattr(coverage_queue,'binding',lambda path: {'raw_sha256':'a'*64})
    monkeypatch.setattr(coverage_queue,'validate_queue',lambda value: {
        'schema': getattr(coverage_queue, schema_name),
        'root':str(output)})
    monkeypatch.setattr(sys,'argv',['coverage_queue','--queue-config',str(config),
        '--queue-sha256','a'*64,'--execute',*extra_args])
    if rejection is None:
        # It passes the no-validation policy, but still needs the independent
        # --watch/execution guards; this is not a GPU admission.
        with pytest.raises(SystemExit) as exc:
            coverage_queue.main()
        assert exc.value.code == 2
        assert '--execute requires --watch' in capsys.readouterr().err
    else:
        with pytest.raises(ValueError, match=rejection):
            coverage_queue.main()
        assert not output.exists()
    assert not (output/'queue.lock').exists()


def test_direct_mse_cli_requires_legacy_opt_in_before_operations(tmp_path, monkeypatch):
    from pipeline.dagger import mse_campaign
    config=tmp_path/'config.json';config.write_text('{}')
    plan=tmp_path/'plan.json';plan.write_text('{}')
    output=tmp_path/'campaign-root'
    monkeypatch.setattr(mse_campaign,'validate_config',lambda value:value)
    monkeypatch.setattr(mse_campaign,'validate_plan',lambda value,config:value)
    monkeypatch.setattr(mse_campaign,'is_fixed100',lambda config:False)
    monkeypatch.setattr(sys,'argv',['mse_campaign','--first-round-config',str(config),
        '--mse-plan',str(plan),'--root',str(output),'--execute'])
    with pytest.raises(ValueError, match='MSE validation is disabled'):
        mse_campaign.main()
    assert not output.exists()
