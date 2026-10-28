"""Versioned end-of-campaign recording boundary, not an Isaac reset policy.

This user-requested extension is NOT Raven's fixed-window stopping rule.
The target is checked only after a durable complete-window save. Finish the
episodes with already recorded rows at that boundary; never record a new
episode afterwards. Isaac still advances its full vector and performs its
ordinary native resets, including on lanes whose recording has finished.
No simulator state can be restored from this accounting object.
"""
from copy import deepcopy

PROTOCOL = 'roborender.bc-total-chunks-natural-drain.v1'
QUALIFICATION_MAX_WINDOWS = 16


def is_bounded_qualification(request):
    """Admit only a small functional probe, never the production campaign.

    Separate from full-buffer capacity: a drained population can stop recording
    early and therefore cannot prove sustained full-population capacity.
    """
    drain = request.get('episode_drain')
    windows = request.get('max_windows')
    width = len(request.get('motions', []))
    return (isinstance(drain, dict) and drain.get('protocol') == PROTOCOL
            and type(drain.get('target_chunks')) is int
            and 1 <= drain['target_chunks'] <= 2 * width
            and type(windows) is int and 1 <= windows <= QUALIFICATION_MAX_WINDOWS)


def validate_completion(receipt, *, num_envs, target_chunks, windows, chunks, unrecorded_rows, tail_rows):
    """Independent receipt accounting, not simulator/capacity certification."""
    def integer(value, minimum=0):
        return type(value) is int and value >= minimum
    if (not isinstance(receipt, dict) or receipt.get('protocol') != PROTOCOL
            or receipt.get('num_envs') != num_envs or receipt.get('target_chunks') != target_chunks
            or receipt.get('complete') is not True or receipt.get('accepted_windows') != windows
            or receipt.get('committed_chunks') != chunks or not integer(chunks, target_chunks)
            or receipt.get('state_restoration_supported') is not False
            or receipt.get('native_reset_policy_changed') is not False
            or tail_rows != [0] * num_envs):
        raise ValueError('missing completed target/natural-drain receipt')
    trigger = receipt.get('triggered')
    if (not isinstance(trigger, dict) or not integer(trigger.get('window'), 1)
            or trigger['window'] > windows or trigger.get('vector_steps') != trigger['window'] * 81
            or not integer(trigger.get('committed_chunks'), target_chunks)
            or trigger['committed_chunks'] > chunks):
        raise ValueError('invalid durable total-target trigger')
    names = ('terminal_vector_steps', 'recorded_drain_rows', 'unrecorded_after_drain_rows')
    arrays = [trigger.get('episode_ids')] + [receipt.get(name) for name in names]
    if any(not isinstance(array, list) or len(array) != num_envs for array in arrays):
        raise ValueError('missing full-population drain accounting')
    if receipt['unrecorded_after_drain_rows'] != unrecorded_rows:
        raise ValueError('native and drain excluded-row accounting differ')
    start, end = trigger['vector_steps'], windows * 81
    for episode, terminal, recorded, excluded in zip(*arrays):
        if (not integer(recorded) or not integer(excluded) or recorded + excluded != end - start):
            raise ValueError('drain rows do not cover actual remaining vector windows')
        if episode is None:
            if terminal is not None or recorded:
                raise ValueError('unstarted recorded episode cannot produce drain rows')
        elif (not isinstance(episode, str) or not episode or not integer(terminal, start)
              or terminal >= end or recorded != terminal - start + 1):
            raise ValueError('recorded episode lacks exact natural terminal step')
    return receipt


class EpisodeDrain:
    def __init__(self, num_envs, target_chunks):
        if type(num_envs) is not int or num_envs < 1 or type(target_chunks) is not int or target_chunks < 1:
            raise ValueError('positive integer lane count and total chunk target required')
        self.num_envs, self.target_chunks = num_envs, target_chunks
        self.targets = None
        self.pending = None
        self.trigger = None
        self.last_window = self.last_chunks = 0
        self.recorded_drain_rows = [0] * num_envs
        self.excluded_rows = [0] * num_envs
        self.terminal_steps = [None] * num_envs
        self.last_transition = None

    @property
    def complete(self):
        return self.pending is not None and not any(self.pending)

    def accept_window(self, window, chunks, current_episodes):
        """Called only AFTER writer/provenance fsync and window acceptance.

        A zero-age current episode has no recorded row, so need not be drained.
        `episode_step_index` is our actual transition count, not randomized
        native episode clocks. It must not be substituted by episode_length_buf.
        """
        if (type(window) is not int or window != self.last_window + 1
                or type(chunks) is not int or chunks < self.last_chunks
                or len(current_episodes) != self.num_envs):
            raise ValueError('monotonic consecutive accepted-window accounting required')
        for row in current_episodes:
            if (not isinstance(row, dict) or not isinstance(row.get('episode_id'), str)
                    or not row['episode_id'] or type(row.get('episode_step_index')) is not int
                    or row['episode_step_index'] < 0):
                raise ValueError('actual episode identity and recorded age required')
        self.last_window, self.last_chunks = window, chunks
        if self.targets is None and chunks >= self.target_chunks:
            self.targets = [row['episode_id'] if row['episode_step_index'] > 0 else None
                            for row in current_episodes]
            self.pending = [value is not None for value in self.targets]
            self.trigger = dict(window=window, vector_steps=window * 81,
                                committed_chunks=chunks, episode_ids=list(self.targets))
            self.last_transition = window * 81 - 1

    def finish_transition(self, vector_step, episode_ids, dones):
        """Return per-lane recording mask for the just-completed native step.

        Call once with PRE-step episode identities and actual native outcomes.
        Terminal rows remain selected; their ordinary short tails are discarded
        by the writer. A new episode's rows are excluded, never padded or joined.
        """
        if (type(vector_step) is not int or vector_step < 0
                or len(episode_ids) != self.num_envs or len(dones) != self.num_envs
                or any(type(done) is not bool for done in dones)
                or any(not isinstance(value, str) or not value for value in episode_ids)):
            raise ValueError('one complete actual native vector outcome required')
        if self.targets is None:
            return [True] * self.num_envs
        if vector_step != self.last_transition + 1:
            raise ValueError('drain transition replay or gap')
        # Validate the entire batch before mutating any lane.
        for lane, episode in enumerate(episode_ids):
            if self.pending[lane] and episode != self.targets[lane]:
                raise ValueError('recorded episode changed without its native terminal row')
            if (not self.pending[lane] and self.targets[lane] is not None
                    and episode == self.targets[lane]):
                raise ValueError('terminated recorded episode identity reused')
        selected = list(self.pending)
        for lane, done in enumerate(dones):
            if selected[lane]:
                self.recorded_drain_rows[lane] += 1
                if done:
                    self.pending[lane] = False
                    self.terminal_steps[lane] = vector_step
            else:
                self.excluded_rows[lane] += 1
        self.last_transition = vector_step
        return selected

    def receipt(self):
        return deepcopy(dict(protocol=PROTOCOL, target_chunks=self.target_chunks,
            num_envs=self.num_envs, triggered=self.trigger, complete=self.complete,
            accepted_windows=self.last_window, committed_chunks=self.last_chunks,
            terminal_vector_steps=self.terminal_steps,
            recorded_drain_rows=self.recorded_drain_rows,
            unrecorded_after_drain_rows=self.excluded_rows,
            state_restoration_supported=False, native_reset_policy_changed=False))
