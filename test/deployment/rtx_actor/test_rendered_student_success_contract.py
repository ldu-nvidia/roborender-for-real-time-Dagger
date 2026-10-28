"""CPU-only rendered student protocol regressions and reusable receipt fixtures."""
from test.deployment.source_paths import ACTOR_ROOT
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(ACTOR_ROOT))
import rendered_student_success_contract as contract
import production_success_contract as production
import smoke_success_contract as smoke
import initial_robot_refresh_diagnostic as refresh
from explicit_motion_panel import canonical, summarize, validate_panel
from rendered_training_support_contract import ADAPTER_SHA256
from test.deployment.rtx_actor.test_explicit_motion_panel import result as original_result


def rendering_config(panel, checkpoint_sha256='b' * 64):
    template = dict(renderer=dict(host='127.0.0.1', port=18550, identity=dict(
        renderer='Image3F', adapter_sha256=ADAPTER_SHA256, engine_sha256='e' * 64,
        steps=5, cfg=1., teacache_threshold=.3, seed=42, server_rng_isolated=True)),
        anchors=dict(seed=1442, files=[dict(path=f'/assets/anchor_{i:05d}.npz',
            sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(2048)]))
    return contract.bind_observation_config(template, panel, checkpoint_sha256)


def cohort_result(panel, index=0, arm='bc', config=None, successes=None, *, _config_validated=False):
    config = rendering_config(panel) if config is None else config
    spec = panel['cohorts'][index]
    count = len(spec['case_ids'])
    successes = [True] * count if successes is None else successes
    row = original_result(spec, successes=successes)
    row.update(arm=arm, completed=count, expected=count,
        original_raven_metric_completed=count, observation_mode=contract.MODE,
        checkpoint_sha256=config['checkpoint_sha256'], vector_steps=123)
    mode = (refresh.STUDENT_CLEAR_ONCE if panel['panel_id'] == production.PANEL_ID else refresh.SMOKE_STUDENT_CLEAR_ONCE)
    env = SimpleNamespace(num_envs=count, need_to_refresh_envs=torch.ones(count, dtype=torch.bool))
    row.update({refresh.FIELD: mode, 'initial_robot_refresh_control': refresh.control_contract(mode),
        'initial_robot_refresh_attestation': refresh.clear_already_applied_initial_refresh(
            env, mode=mode, arm=arm, teacher_action_mode='distill')})
    def views(kind):
        return {view: dict(shape=[240, 416] if kind == 'depth' else [240, 416, 3],
            dtype='uint8', sha256='d' * 64) for view in contract.VIEWS}
    row['rendered_observation_receipt'] = dict(
        schema='roborender.rendered-success-observation-receipt.v1', observation_mode=contract.MODE,
        renderer_identity=deepcopy(config['renderer']['identity']), view_order=list(contract.VIEWS),
        policy_view_shape=[108, 192, 3], control_view_shape=[240, 416],
        vector_steps_rendered=123, render_requests_completed=count * 123,
        raytraced_fallbacks=0, failed=False,
        anchor_seed=(config['anchors']['seed'] + index) % (2 ** 32),
        anchor_policy='fixed_frame0_per_episode', anchor_selections=[dict(lane=lane,
            selected_at_vector_step=0, file_sha256=config['anchors']['files'][lane]['sha256'],
            frame_index=0, prompt_sha256='f' * 64) for lane in range(count)],
        normalizations_per_vector_step=1, controls_from_current_simulator_state=True,
        simulator_steps_added=0, optimizer_steps=0, offline_BC_teacher50_equivalent=False,
        elapsed_seconds=2., trace_sha256='d' * 64,
        bounded_first_inputs=[dict(step=step, lanes=[dict(lane=lane,
            inputs={kind: views(kind) for kind in ('depth', 'mask', 'previous')},
            generated=views('generated')) for lane in range(count)]) for step in range(2)],
        evaluation_binding=(contract._observation_binding(panel, config, index) if _config_validated
            else contract.observation_binding(panel, config, index)))
    return row


def complete_result(panel, arm='bc', config=None):
    config = rendering_config(panel) if config is None else config
    config = contract.validate_config(config, panel)
    rows = [cohort_result(panel, index, arm, config, _config_validated=True) for index in range(len(panel['cohorts']))]
    value = summarize(panel, rows)
    value['rendered_student_success_protocol'] = contract.protocol_identity(panel, config)
    if panel['panel_id'] == production.PANEL_ID:
        value['production_success_protocol'] = production.protocol_identity(panel)
    else:
        value['smoke_success_protocol'] = smoke.protocol_identity(panel)
    return value


@pytest.fixture
def accepted(monkeypatch):
    # Reuse the shared synthetic fixed295 fixture, not any runtime assets.
    from test.deployment.rtx_actor.test_production_success_contract import accepted as fixture
    return fixture.__wrapped__(monkeypatch)


def test_exact_case_assignments_are_unchanged_and_legacy_execution_rejected(accepted):
    legacy = contract.raytraced_shape(accepted)
    assert production.build_rendered_panel(legacy) == accepted
    assert legacy['cohorts'] == accepted['cohorts']
    assert {key for key in accepted if legacy[key] != accepted[key]} == {'panel_id', 'observation_mode'}
    assert production.require_legacy_panel(legacy) == legacy
    with pytest.raises(ValueError):
        production.evaluation_options(legacy)


@pytest.mark.parametrize('fault', ['checkpoint', 'panel', 'renderer', 'anchors', 'host', 'seed', 'extra'])
def test_config_is_explicit_and_hash_bound(accepted, fault):
    value = rendering_config(accepted)
    if fault == 'checkpoint': value['checkpoint_sha256'] = 'not-a-sha'
    if fault == 'panel': value['panel_sha256'] = '0' * 64
    if fault == 'renderer': value['renderer']['identity']['steps'] = 50
    if fault == 'anchors': value['anchors']['files'].pop()
    if fault == 'host': value['renderer']['host'] = '0.0.0.0'
    if fault == 'seed': value['anchors']['seed'] = True
    if fault == 'extra': value['paired_baseline_file'] = '/not-required'
    with pytest.raises(ValueError): contract.validate_config(value, accepted)


@pytest.mark.parametrize('index', [0, 147])
def test_each_native_cohort_including_last_single_lane_has_bound_rendering(accepted, index):
    config = rendering_config(accepted)
    row = cohort_result(accepted, index, config=config)
    contract.validate_cohort_observations(row, count=len(accepted['cohorts'][index]['case_ids']),
        config=config, panel=accepted, cohort_index=index)
    subset = deepcopy(accepted)
    subset['cohorts'] = [subset['cohorts'][index]]
    subset['episode_count'] = len(subset['cohorts'][0]['case_ids'])
    assert validate_panel(subset) == subset
    assert summarize(subset, [row])['completed_episodes'] == subset['episode_count']


@pytest.mark.parametrize('fault', ['fallback', 'count', 'normalization', 'mode', 'binding', 'anchor', 'views', 'initial'])
def test_missing_or_changed_observations_fail_closed(accepted, fault):
    config = rendering_config(accepted)
    row = cohort_result(accepted, config=config)
    receipt = row['rendered_observation_receipt']
    if fault == 'fallback': receipt['raytraced_fallbacks'] = 1
    if fault == 'count': receipt['render_requests_completed'] -= 1
    if fault == 'normalization': receipt['normalizations_per_vector_step'] = 2
    if fault == 'mode': receipt['observation_mode'] = 'raytraced_3view'
    if fault == 'binding': receipt['evaluation_binding']['checkpoint_sha256'] = 'a' * 64
    if fault == 'anchor': receipt['anchor_selections'][0]['file_sha256'] = 'a' * 64
    if fault == 'views': receipt['bounded_first_inputs'][0]['lanes'][0]['generated'].pop('hand_left')
    if fault == 'initial': receipt['anchor_selections'][0]['selected_at_vector_step'] = 1
    with pytest.raises(ValueError): contract.validate_cohort_observations(row, count=2,
        panel=accepted, config=config, cohort_index=0)


def test_full_and_smoke_results_require_rendering_protocol(accepted):
    for panel in (accepted, smoke.make_panel(accepted)):
        config = rendering_config(panel)
        value = complete_result(panel, config=config)
        assert contract.validate_rendered_result(panel, value, 'b' * 64, config) == value
        value.pop('rendered_student_success_protocol')
        with pytest.raises(ValueError): contract.validate_rendered_result(panel, value, 'b' * 64, config)


def test_configuration_read_checks_raw_hash_and_checkpoint(accepted, tmp_path):
    config = rendering_config(accepted)
    path = tmp_path / 'config.json'
    path.write_bytes(canonical(config) + b'\n')
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert contract.read_inputs(path, raw_sha, accepted, checkpoint_sha256='b' * 64) == config
    with pytest.raises(ValueError): contract.read_inputs(path, 'a' * 64, accepted)
    with pytest.raises(ValueError): contract.read_inputs(path, raw_sha, accepted, checkpoint_sha256='c' * 64)
