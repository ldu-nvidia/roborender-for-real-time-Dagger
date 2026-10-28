"""CPU protocol checks only: no native/student/renderer qualification claim."""
from copy import deepcopy
import json

import pytest

from pipeline.success_rate.contracts import (
    INITIAL_STUDENT_SHA256, collection_initial_reference, collection_student_plan,
    validate_collection_initial_reference, validate_collection_student_plan,
)
from pipeline.success_rate.teacher_episodes import PerLaneEpisodes
from test.pipeline.test_continuous_bc_launcher import request


def reference():
    return collection_initial_reference(json.dumps(request(), indent=2).encode())


def plan(ref=None, renderer='online_image3f5'):
    return collection_student_plan(ref or reference(), checkpoint_sha256=INITIAL_STUDENT_SHA256,
                                    renderer=renderer, seeds=[20261102, 20261103])


def test_reference_preserves_original_bytes_order_inventory_not_fake_states():
    raw = json.dumps(request(), indent=2).encode()
    value = collection_initial_reference(raw)
    assert value['source_request_utf8'].encode() == raw
    assert value['num_conditions'] == 32
    assert [v['motion'] for v in value['lane_assignments']] == request()['motions']
    assert value['full_reset_inventory_count'] == len(request()['motion_inventory'])
    assert value['exact_historical_state_available'] is False
    assert validate_collection_initial_reference(value) == value


@pytest.mark.parametrize('field', ['lane_assignments', 'source_request_sha256', 'full_reset_inventory_sha256',
                                  'collection_seed', 'exact_historical_state_available'])
def test_reference_tampering_rejected(field):
    value = reference()
    value[field] = 'changed'
    with pytest.raises(ValueError):
        validate_collection_initial_reference(value)


def test_smoke_has_192_equal_outcomes_no_teacher_fallback_or_learning():
    ref = reference(); value = plan(ref)
    assert validate_collection_student_plan(value, ref) == value
    assert value['expected_total'] == 192 and value['episodes_per_lane'] == 3
    assert value['distinct_motion_conditions'] == 32 and value['concurrent_environments'] == 64
    assert [a['h200_slots'] for a in value['actors']] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert not value['ready'] and value['execution_admission'] is None
    assert value['optimizer_steps'] == value['teacher_action_probability'] == 0
    assert not value['raytraced_fallback_allowed'] and not value['chunk_count_reset_allowed']


@pytest.mark.parametrize('field,value', [('ready', True), ('expected_total', 295),
    ('teacher_takeover', True), ('generated_only', False), ('optimizer_steps', 1),
    ('chunk_count_reset_allowed', True), ('future_state_conditioning_allowed', True)])
def test_cannot_relax_student_invariants(field, value):
    ref = reference(); altered = plan(ref); altered[field] = value
    with pytest.raises(ValueError):
        validate_collection_student_plan(altered, ref)


def test_renderer_roles_remain_distinct_and_offline_is_not_live_admission():
    online, offline = plan(), plan(renderer='offline_bc_teacher50')
    assert online['renderer']['causal_adapter_implemented']
    assert not offline['renderer']['causal_adapter_implemented'] and not offline['ready']
    assert online['renderer']['slot_order'] != offline['renderer']['slot_order']
    assert online['renderer']['adapter_sha256'] != offline['renderer']['adapter_sha256']
    with pytest.raises(ValueError, match='explicit named renderer'):
        plan(renderer='renderer1')


@pytest.mark.parametrize('seeds', [[0, 1], [1, 1], [True, 2], [1], [1, 2**31]])
def test_fresh_independent_reset_seeds(seeds):
    with pytest.raises(ValueError):
        collection_student_plan(reference(), checkpoint_sha256=INITIAL_STUDENT_SHA256,
                                renderer='online_image3f5', seeds=seeds)


def test_trained_or_retired_parent_is_not_untrained_student():
    with pytest.raises(ValueError, match='initial student'):
        collection_student_plan(reference(), checkpoint_sha256='f'*64,
                                renderer='online_image3f5', seeds=[1, 2])


def test_three_repeats_every_lane_no_fast_lane_substitution():
    accounts = [PerLaneEpisodes(32, 3), PerLaneEpisodes(32, 3)]
    for account in accounts:
        for step in range(9):
            account.update(step, [True]*31+[False], [True]*31+[step in (2, 5, 8)], [False]*32)
    assert sum(a.summary()['completed'] for a in accounts) == 192
    assert sum(a.summary()['successes'] for a in accounts) == 186
    assert all(a.summary()['completed_per_lane'] == [3]*32 for a in accounts)


def test_catalog_cli_is_immutable_and_never_starts_work(tmp_path, capsys):
    from pipeline.success_rate.__main__ import main
    source, target = tmp_path/'request.json', tmp_path/'reference.json'
    source.write_text(json.dumps(request()))
    args = ['preserve-bc-conditions', '--request', str(source), '--output', str(target)]
    assert main(args) == main(args) == 0
    assert json.loads(target.read_text()) == collection_initial_reference(source.read_bytes())
    changed = request(); changed['process_seed'] = 1
    source.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='immutable'):
        main(args)
