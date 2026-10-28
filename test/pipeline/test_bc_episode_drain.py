import pytest

from pipeline.bc_collection.episode_drain import EpisodeDrain, validate_completion, is_bounded_qualification


def current(*ages):
    return [dict(episode_id=f'e{i}', episode_step_index=age) for i, age in enumerate(ages)]


def test_functional_qualification_has_distinct_bounded_request_and_launch():
    import hashlib
    from pipeline.bc_collection.continuous_plan import build_plan, build_request
    from pipeline.bc_collection.run_continuous import validate_launch, storage_budget
    from pipeline.bc_collection.storage_guard import STORAGE_SCHEMA, GIB
    from pipeline.deployment.rtx_actor.bounded_bc_storage import PROTOCOL
    inventory = [dict(motion_id=f'm{i}', path=f'/motions/m{i}.npz',
                      sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(32)]
    options = dict(run_id='drain-qualification', max_windows=16, num_envs=32,
                   output_dir='/dagger-runtime/bc_isaac', collection_storage_protocol=PROTOCOL)
    request = build_request(build_plan(inventory, **options, target_total_chunks=1))
    assert is_bounded_qualification(request)
    validate_launch(request, qualification=True, expected_num_envs=32,
                    physical_gpu=0, memory_gib=43, timeout=3600)
    with pytest.raises(ValueError, match='qualification-only'):
        validate_launch(request, qualification=False, expected_num_envs=32,
                        physical_gpu=0, memory_gib=43, timeout=3600)
    for target in (65, 10000):
        with pytest.raises(ValueError):
            build_plan(inventory, **options, target_total_chunks=target)
    # The existing conservative whole-probe storage budget is NOT reduced.
    policy = dict(schema=STORAGE_SCHEMA, root='/data1', reserve_bytes=64 * GIB,
                  bytes_per_emission_budget=512 << 20,
                  session_metadata_budget_bytes=GIB, poll_seconds=1)
    assert storage_budget(policy, request)['required_available_bytes'] == 321 * GIB


def test_target_is_total_and_checked_at_durable_window_boundary():
    value = EpisodeDrain(2, 10)
    value.accept_window(1, 9, current(81, 12))
    assert value.targets is None
    assert value.finish_transition(81, ['e0', 'e1'], [False, False]) == [True, True]
    value.accept_window(2, 11, current(162, 93))
    assert value.trigger['committed_chunks'] == 11
    assert value.trigger['vector_steps'] == 162
    assert not value.complete


def test_fast_lanes_stop_recording_slow_lanes_continue_without_quota():
    value = EpisodeDrain(2, 10)
    value.accept_window(1, 10, current(81, 81))
    assert value.finish_transition(81, ['e0', 'e1'], [True, False]) == [True, True]
    for step in range(82, 1000):  # More than the old nine-chunk cutoff.
        assert value.finish_transition(step, ['new-e0', 'e1'], [False, False]) == [False, True]
    assert not value.complete
    assert value.finish_transition(1000, ['new-e0', 'e1'], [False, True]) == [False, True]
    assert value.complete
    assert value.receipt()['recorded_drain_rows'] == [1, 920]
    assert value.receipt()['unrecorded_after_drain_rows'] == [919, 0]


def test_just_reset_zero_row_episode_is_not_started_for_drain():
    value = EpisodeDrain(2, 1)
    value.accept_window(1, 1, current(0, 0))
    assert value.complete and value.targets == [None, None]


def test_complete_remaining_window_does_not_record_new_episodes():
    value = EpisodeDrain(2, 1)
    value.accept_window(1, 1, current(0, 81))
    assert value.finish_transition(81, ['new-e0', 'e1'], [False, True]) == [False, True]
    for step in range(82, 162):
        assert value.finish_transition(step, ['new-e0', 'new-e1'], [False, False]) == [False, False]
    value.accept_window(2, 1, current(80, 80))
    assert value.complete and value.trigger['window'] == 1
    assert value.receipt()['unrecorded_after_drain_rows'] == [81, 80]


def test_invalid_vector_batch_is_atomic_and_replay_rejected():
    value = EpisodeDrain(2, 1)
    value.accept_window(1, 1, current(81, 81))
    before = value.receipt()
    with pytest.raises(ValueError, match='without its native terminal'):
        value.finish_transition(81, ['e0', 'wrong'], [True, False])
    assert value.receipt() == before
    value.finish_transition(81, ['e0', 'e1'], [True, False])
    with pytest.raises(ValueError, match='replay or gap'):
        value.finish_transition(81, ['e0', 'e1'], [True, False])


@pytest.mark.parametrize('target', [0, -1, True, 1.5])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError):
        EpisodeDrain(32, target)


def test_no_rearm_and_receipt_is_not_mutable_state():
    value = EpisodeDrain(1, 1)
    value.accept_window(1, 1, current(81))
    value.finish_transition(81, ['e0'], [True])
    value.accept_window(2, 2, [dict(episode_id='new', episode_step_index=80)])
    receipt = value.receipt()
    receipt['triggered']['episode_ids'][0] = 'tampered'
    assert value.targets == ['e0'] and value.complete


def test_real_writer_ledger_and_durable_journal_drain_same_episode_rows(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    from pathlib import Path
    from pipeline.bc_collection.continuous_plan import build_plan, build_request
    from pipeline.deployment.rtx_actor import bounded_bc_storage as writer
    from pipeline.deployment.rtx_actor import isaac6_bc_production as native
    from test.pipeline.test_bounded_bc_storage import owner, populate
    from test.pipeline.test_isaac6_bc_continuous import reset

    value = owner(tmp_path / 'drained', 2)
    from pipeline.bc_collection.streaming_storage import StreamingDiskGuard, GIB
    value._bc_disk_guard = StreamingDiskGuard(value.vg_collect_dir, 2)
    monkeypatch.setattr(value._bc_disk_guard, '_free', lambda: 100 * GIB)
    inventory = [dict(motion_id=f'm{i}', path=f'/motions/m{i}.npz',
                      sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(2)]
    plan = build_plan(inventory, run_id='drain-test', max_windows=4, num_envs=2,
        output_dir=value.vg_collect_dir, collection_storage_protocol=writer.PROTOCOL,
        target_total_chunks=2)
    ledger = native.ChunkProvenanceLedger(build_request(plan))
    value._bc_ledger = ledger
    journal = native.ContinuousCollectionJournal(ledger)
    for lane in range(2):
        reset(ledger, lane)
    try:
        for window in range(4):
            value._bc_disk_guard.admit_window()
            populate(value, window, ['head_color', 'hand_left', 'hand_right'])
            value.storage.dones.zero_()
            for t in range(81):
                step = window * 81 + t
                flags = [step in (129, 259), step == 299]
                value.storage.dones[t, :, 0] = __import__('torch').tensor(flags)
                ledger.begin_transition([r['episode_step_index'] + 123 for r in ledger.current])
                for lane, done in enumerate(flags):
                    if done:
                        reset(ledger, lane)
                ledger.finish_transition(flags)
            before = value._vg_saved_chunks
            with np.errstate(invalid='ignore'):
                writer.write_window(value)
            pending = ledger.pending()
            assert value._vg_saved_chunks - before == len(pending)
            for index, expected in enumerate(pending, before):
                path = Path(value.vg_collect_dir) / f'chunk_{value._vg_run_tag}_{index:07d}.npz'
                with np.load(path, allow_pickle=False) as data:
                    rows = [row['vector_step_index'] for row in expected['rows']]
                    np.testing.assert_array_equal(data['gt_actions'][:, 0], rows)
                    assert int(data['seq']) == expected['seq']
                    assert rows[-1] < (130 if expected['local_env_id'] == 0 else 300)
                record = ledger.describe(expected, path)
                sidecar = path.with_suffix('.provenance.json')
                native.publish_json(sidecar, record)
                journal.append(record, sidecar)
                ledger.acknowledge(expected)
            receipt = journal.accept_window(value._vg_saved_chunks)
            assert receipt['episode_drain']['triggered']['window'] == 1
            assert value._ep_seq_base == ledger.sequences
            assert [len(buf.get('gt', [])) for buf in value._ep_buf] == [len(buf) for buf in ledger.buffers]
        journal.require_completed(native_saved_chunks=value._vg_saved_chunks,
            native_tail_rows=[0, 0], initial_reset_all_calls=1)
        assert ledger.emitted_counts == [1, 3]
        assert ledger.discarded_tail_rows == [49, 57]
        assert ledger.unrecorded_rows == [194, 24]
        assert ledger.drain.complete
        assert value._bc_disk_guard.windows_admitted == 4
        assert value._bc_disk_guard.emissions_verified == 4
        validate_completion(ledger.drain.receipt(), num_envs=2, target_chunks=2,
            windows=4, chunks=4, unrecorded_rows=[194, 24], tail_rows=[0, 0])
        invalid = ledger.drain.receipt()
        invalid['terminal_vector_steps'][0] += 1
        with pytest.raises(ValueError, match='exact natural terminal'):
            validate_completion(invalid, num_envs=2, target_chunks=2, windows=4,
                chunks=4, unrecorded_rows=[194, 24], tail_rows=[0, 0])
    finally:
        journal.close()
