"""CPU-only final-epoch BC selection for the new paired rendered295 handoff.

Revalidates the trainer's actual seal and immutable union/base provenance. This
does not unpickle a checkpoint, execute a test, qualify capacity, or authorize
training. The strict learner separately verifies the policy and AdamW payload.
Historical MSE-selected v1 metadata is not converted into this schema.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from pipeline.bc_training.handoff_common import binding, bound, cli_binding, publish

SCHEMA = 'roborender.bc-production-selected.v2'


def build_selection(*, training_state, training_config, data_admission, test_panel,
                    rendered_observation_config):
    from pipeline.bc_training.plan_success_test import (
        _fixed_final_selection, checkpoint_identity, validate_union_admission)
    from pipeline.deployment.rendered_success_dispatch import configuration
    from pipeline.deployment.rtx_actor import production_success_contract as original
    from pipeline.deployment.rtx_actor import scaled_success_contract as scaled
    from pipeline.deployment.rtx_actor.rendered_endpoint_pool import SHARDED_ROUTING

    state, config = bound(training_state), bound(training_config)
    report = validate_union_admission(state, config, data_admission)
    if (config.get('student_lifecycle') != 'raven_offline_replay_v1'
            or config.get('experimental_shuffle', False) is not False
            or state.get('training_batch_contract', {}).get('training_chunk_count') != report['chunk_count']):
        raise ValueError('new production selection requires the full sorted Raven-replay union')
    root = Path(config['output']).resolve(strict=True)
    checkpoint, selected = _fixed_final_selection(state, config, training_state=training_state,
        training_config=training_config, data_admission=data_admission, root=root)
    name = config.get('run_name') or root.name
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', name):
        raise ValueError('bounded actual BC run name required')
    policy_spec = binding(config['policy_spec'])
    specification = bound(policy_spec)
    bootstrap = checkpoint_identity(config['bootstrap'])
    if (state.get('policy_spec_sha256') != policy_spec['raw_sha256']
            or state.get('bootstrap_sha256') != bootstrap['sha256']
            or specification.get('checkpoint_sha256') != bootstrap['sha256']):
        raise ValueError('completed BC bootstrap/policy specification identity differs')
    protected = report['protected_historical_test']
    source = bound(protected['panel'])
    if source.get('observation_mode') == 'raytraced_3view':
        source = original.build_rendered_panel(source)
    else:
        original.require_panel(source, raw_sha256=protected['panel']['raw_sha256'])
    panel = scaled.require_panel(bound(test_panel))
    if (protected.get('episode_count') != 295 or panel['scaled_protocol']['purpose'] != 'test'
            or panel['scaled_protocol']['batch_size'] != 16 or panel['episode_count'] != 295
            or panel['scaled_protocol']['source_panel'] != source):
        raise ValueError('new BC selection requires scaled B16 full295 from the protected original test')
    rendering = configuration(rendered_observation_config, panel, checkpoint['sha256'])
    if (rendering['renderer'].get('routing') != SHARDED_ROUTING
            or len(rendering['renderer'].get('endpoints', [])) != 8):
        raise ValueError('new BC testing requires the two disjoint four-H200 renderer shards')
    return dict(schema=SCHEMA, run_name=name, selection='last_completed_epoch',
        selected_epoch=selected['selected_epoch'], selected_global_step=state['global_step'],
        selected_training_mse=selected['selected_training_mse'], selected_validation_mse=None,
        validation_enabled=False, training_protocol=selected['training_protocol'],
        optimizer_state_preserved=True, rng_state_preserved=True,
        training_state=deepcopy(training_state), training_config=deepcopy(training_config),
        data_admission=deepcopy(data_admission), policy_spec=policy_spec,
        bootstrap_sha256=bootstrap['sha256'], checkpoint=checkpoint,
        training_batch_contract=deepcopy(state['training_batch_contract']),
        test_panel=deepcopy(test_panel), source_data_admission_test_panel=deepcopy(protected['panel']),
        rendered_observation_config=deepcopy(rendered_observation_config),
        expected_test_episodes=295, success_test_executed=False,
        production_training_completed=True, production_execution_authorized=False,
        checkpoint_payload_validation='deferred_to_strict_learner_admission')


def validate_selection(value):
    """Re-derive every field; metadata flags alone cannot promote a checkpoint."""
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise ValueError('versioned final-epoch production BC selection required')
    keys = ('training_state', 'training_config', 'data_admission', 'test_panel',
            'rendered_observation_config')
    if not all(key in value for key in keys):
        raise ValueError('complete bound final-epoch selection inputs required')
    expected = build_selection(**{key: value[key] for key in keys})
    if value != expected:
        raise ValueError('BC selection differs from its actual sealed final epoch and admitted union')
    return deepcopy(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-state', 'training-config', 'data-admission', 'test-panel',
                 'rendered-observation-config'):
        parser.add_argument('--' + name, nargs=2, required=True, metavar=('PATH', 'RAW_SHA256'))
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop('output')
    value = build_selection(**{key: cli_binding(pair) for key, pair in args.items()})
    print(json.dumps(dict(selection=publish(output, value), executed=False)))


if __name__ == '__main__':
    main()
