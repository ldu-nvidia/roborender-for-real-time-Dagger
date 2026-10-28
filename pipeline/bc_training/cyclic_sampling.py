"""Raven's persistent sorted-file cursor, independent of DataLoader prefetch.

A reporting epoch is ceil(N/B) complete updates, not an exactly-once dataset
pass when N % B != 0. Only accepted optimizer updates advance the saved cursor.
"""
from __future__ import annotations

SCHEMA = 'roborender.bc-sampling.v2'
CURSOR_SCHEMA = 'roborender.bc-cyclic-cursor.v1'

def updates_per_epoch(count, batch):
    if type(count) is not int or count < 1 or type(batch) is not int or batch not in (16, 32):
        raise ValueError('cyclic BC requires a nonempty pool and full B16/B32 batches')
    return (count + batch - 1) // batch

def cursor_state(count, batch, accepted_updates):
    updates_per_epoch(count, batch)
    if type(accepted_updates) is not int or accepted_updates < 0:
        raise ValueError('accepted optimizer update count must be nonnegative')
    consumed = accepted_updates * batch
    return dict(schema=CURSOR_SCHEMA, accepted_updates=accepted_updates,
        chunks_consumed=consumed, next_index=consumed % count,
        completed_pool_passes=consumed // count, chunk_count=count, batch_size=batch)

def validate_cursor(value, count, batch, accepted_updates):
    if value != cursor_state(count, batch, accepted_updates):
        raise ValueError('saved BC cyclic cursor differs from committed optimizer updates')

def validate_checkpoint_sampling(sampling, batch_contract, cursor, accepted_bc_updates):
    """Validate immutable BC sampling provenance, also in DAgger descendants.

    DAgger updates do not consume the offline BC file cursor. Descendants must
    therefore supply the saved BC step, not their current optimizer step.
    """
    import re
    count, batch = batch_contract.get('training_chunk_count'), batch_contract.get('chunks_per_update')
    updates = updates_per_epoch(count, batch)
    expected = dict(schema=SCHEMA, mode='raven_sorted_cyclic', chunk_count=count,
        batch_size=batch, updates_per_reporting_epoch=updates,
        reporting_epoch='ceil_N_over_B_full_updates_cursor_never_reset',
        cursor_commit='accepted_optimizer_updates_only', partial_batch='wrap_sorted_pool')
    if (not isinstance(sampling, dict) or set(sampling) != set(expected) | {'ordered_paths_sha256'}
            or any(sampling.get(k) != v for k, v in expected.items())
            or not isinstance(sampling.get('ordered_paths_sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', sampling['ordered_paths_sha256'])
            or batch_contract.get('updates_per_epoch') != updates
            or batch_contract.get('drop_last') is not False
            or batch_contract.get('partial_batch') != 'wrap_sorted_pool'
            or batch_contract.get('epoch_is_exact_dataset_pass') is not (count % batch == 0)):
        raise ValueError('BC cyclic sampling/batch provenance differs')
    validate_cursor(cursor, count, batch, accepted_bc_updates)

class CyclicBatchSampler:
    """An immutable iteration snapshot; prefetch never commits training progress."""
    def __init__(self, count, batch):
        self.count, self.batch = count, batch
        self.updates = updates_per_epoch(count, batch)
        self.start = 0

    def start_after(self, accepted_updates):
        self.start = cursor_state(self.count, self.batch, accepted_updates)['chunks_consumed']

    def __len__(self):
        return self.updates

    def __iter__(self):
        start = self.start
        for update in range(self.updates):
            yield [(start + update * self.batch + lane) % self.count for lane in range(self.batch)]
