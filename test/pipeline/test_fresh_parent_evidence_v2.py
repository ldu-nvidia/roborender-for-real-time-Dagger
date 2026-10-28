"""Run-local read-only W&B reconciliation; fixtures do not certify GPU work."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[2]/'runtime/raven-dagger-alignment-20261026-7pYLxI/fresh_parent_evidence_v2.py'
if not PATH.is_file():
    pytest.skip('optional run-local helper absent', allow_module_level=True)
spec = importlib.util.spec_from_file_location('fresh_parent_v2_test', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rows = [dict(update=i, train_mse=.8/i, gradient_norm_before_clip=.06,
                 update_seconds=1., bn_counter_delta=82*i) for i in (1, 2)]
    result = dict(wandb=dict(id='test'), updates=deepcopy(rows), result='PASS', optimizer_updates=2,
        chunks_per_update=32, policy_forward_population=2592, production_promoted=False,
        initial_state_exact_saved_checkpoint=True, policy_reload_exact=True,
        optimizer_reload_exact=True, rng_restore_exact=True)
    def scan_history(*, page_size):
        assert page_size == 50  # A keys parameter is intentionally not accepted.
        return iter(rows)
    run = SimpleNamespace(id='test', url='fixture-not-live', state='finished',
        summary={key: result[key] for key in m.SUMMARY}, scan_history=scan_history)
    api = SimpleNamespace(run=lambda path: run)
    return result, rows, run, api


def test_full_scan_matches_exact_logged_results_without_key_filter():
    result, rows, run, api = fixture()
    assert m.online_readback(result, api)['history'] == rows


@pytest.mark.parametrize('change', [
    lambda rows, run: rows.clear(), lambda rows, run: rows.pop(),
    lambda rows, run: rows.append(dict(rows[0])), lambda rows, run: rows.reverse(),
    lambda rows, run: rows[0].pop('train_mse'),
    lambda rows, run: rows[0].update(train_mse=.7),
    lambda rows, run: rows[1].update(bn_counter_delta=82),
    lambda rows, run: setattr(run, 'state', 'running'),
    lambda rows, run: setattr(run, 'id', 'other'),
    lambda rows, run: run.summary.pop('result'),
    lambda rows, run: run.summary.update(production_promoted=True),
])
def test_missing_mismatched_or_wrong_run_history_fails_closed(change):
    result, rows, run, api = fixture()
    change(rows, run)
    with pytest.raises(ValueError):
        m.online_readback(result, api)
