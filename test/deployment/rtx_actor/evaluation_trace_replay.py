"""CPU-only replay of recorded first-episode scoring, never native qualification.

The raw goals/dones are inputs; completed_outcomes and final scores are only
comparison targets. Raven's actual pinned callbacks and completion/reset block
are executed. Physics, teacher/student inference, rendering, reset correctness,
velocity metrics and Raven's default pooled protocol are NOT re-qualified.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from pipeline.deployment import scaled_teacher_dispatch as dispatch
from pipeline.deployment.rtx_actor import scaled_teacher_benchmark as teacher
from pipeline.deployment.rtx_actor.success_diagnostic_core import FirstEpisodeAccounting

PINS = {
    'envs/loco_manip/walk_stand_grasp.py':
        'd985d3e74414b6ceabefd917bdaa3b36c67ff9e141882436f931815f16bd3be9',
    'trl/trainer/distill_trainer_obj_pred_homie_api.py':
        '3da5a2557ae18a8ba53c4da14d8841b80a609b851b326a21534ee2c5001b7584',
}
METRIC_METHODS = {
    'init_eval_metrics_tracking', 'update_eval_metrics_per_step',
    'process_eval_episode_completions', 'reset_eval_episode_tracking',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'physical evidence file required: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def _pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    def invalid(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def read(path):
    file_identity(path)
    return parse(Path(path).read_bytes())


def load_oracle(reference_root):
    """Extract unmodified original source, not a reimplementation of its logic."""
    trees = {}
    identities = {}
    for relative, expected in PINS.items():
        path = Path(reference_root) / relative
        identity = file_identity(path)
        require(identity['sha256'] == expected, f'pinned Raven source changed: {relative}')
        trees[relative] = ast.parse(path.read_bytes())
        identities[relative] = identity
    env_tree, trainer_tree = trees.values()
    env_cls = next(n for n in env_tree.body if isinstance(n, ast.ClassDef) and n.name == 'WalkStandGrasp')
    methods = [deepcopy(n) for n in env_cls.body
               if isinstance(n, ast.FunctionDef) and n.name in METRIC_METHODS]
    require({n.name for n in methods} == METRIC_METHODS, 'original metric methods absent')
    cls = ast.ClassDef(name='OriginalMetrics', bases=[], keywords=[], body=methods, decorator_list=[])
    trainer_cls = next(n for n in trainer_tree.body if isinstance(n, ast.ClassDef)
                       and n.name == 'TRLDistillTrainerObjPredHomieAPI')
    evaluate = next(n for n in trainer_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'eval')
    loop = next(n for n in ast.walk(evaluate) if isinstance(n, ast.While)
                and ast.unparse(n.test) == 'not terminate_rollout()')
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.AugAssign)
                 and ast.unparse(n.target) == 'self.cur_episode_length')
    block = deepcopy(loop.body[start:start + 4])
    require(len(block) == 4 and isinstance(block[-1], ast.If)
            and ast.unparse(block[-1].test) == 'len(new_ids) > 0', 'original scoring block changed')
    shell = ast.parse('def original_step(self, dones, infos):\n    eval_num_envs_episodes = True\n    pbar = self.pbar\n').body[0]
    shell.body.extend(block)
    namespace = dict(torch=torch, logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls, shell], type_ignores=[])),
                 '<hash-pinned-Raven-scoring>', 'exec'), namespace)
    return namespace['OriginalMetrics'], namespace['original_step'], identities


def replay_events(events, count, oracle):
    """One recorded cohort; no summarized successes are used as replay inputs."""
    metric_class, original_step, _ = oracle
    env = metric_class()
    env.num_envs = count
    env.get_object_class_names = lambda: None
    env.init_eval_metrics_tracking('cpu')
    original = SimpleNamespace(env=env, cur_episode_length=torch.zeros(count),
        env_episode_completed=torch.zeros(count, dtype=torch.bool),
        config=SimpleNamespace(eval=SimpleNamespace(save_videos=False)),
        pbar=SimpleNamespace(update=lambda _count: None))
    current = FirstEpisodeAccounting(count)
    completion_order = []
    steps = 0
    for event in events:
        require(not current.result()['complete'], 'extra event after all first episodes completed')
        require(type(event.get('step')) is int and event['step'] == steps, 'nonsequential trace step')
        require(event.get('arm') == 'teacher', 'trace behavior arm changed')
        lanes, exported = event.get('lanes'), event.get('exported_goals')
        require(isinstance(lanes, list) and len(lanes) == count and isinstance(exported, list)
                and len(exported) == count, 'trace lane cardinality mismatch')
        require(all(type(v) is bool for v in exported), 'raw exported goals must be bools')
        native, done, timeout, rewards = [], [], [], []
        for row in lanes:
            require(isinstance(row, dict), 'raw lane object required')
            goal, flag, timed = (row.get(k) for k in (
                'current_last_stage_completed_task_buf', 'reset_buf', 'time_out_buf'))
            reward = row.get('episode_reward')
            require(type(goal) is bool and type(timed) is bool and type(flag) is int
                    and flag in (0, 1), 'raw goal/done/timeout field invalid')
            require(type(reward) in (int, float) and math.isfinite(reward), 'recorded reward invalid')
            require(float(torch.tensor(reward, dtype=torch.float32)) == reward,
                    'recorded cumulative reward is not exact native float32')
            native.append(goal); done.append(flag); timeout.append(timed); rewards.append(reward)
        require(native == exported, 'corrected native/exported raw goals disagree')
        # Captured cumulative rewards are passed through, not reconstructed from
        # success summaries or fabricated per-step rewards.
        original.cur_reward_sum = torch.tensor(rewards, dtype=torch.float32)
        original_step(original, torch.tensor(done), {'grasp_goal_reached': torch.tensor(exported)})
        completed = current.update(steps, native, exported, done, timeout, lanes)
        require(completed == event.get('completed_outcomes'), 'recorded completion differs from raw events')
        completion_order.extend(completed)
        require(env.eval_metrics['goal_reached_buffer'] == [r['exported_success'] for r in completion_order],
                'original/current first-episode successes differ')
        require(env.eval_metrics['reward_buffer'] == [r['episode_reward'] for r in completion_order],
                'original/current first-episode rewards differ')
        require(env.eval_metrics['length_buffer'] == [r['steps'] for r in completion_order],
                'original/current first-episode lengths differ')
        steps += 1
    result = current.result()
    require(result['complete'], 'truncated trace: missing first-episode outcomes')
    return dict(vector_steps=steps, accounting=result,
        original_successes=sum(env.eval_metrics['goal_reached_buffer']),
        original_completed=len(env.eval_metrics['goal_reached_buffer']),
        original_completion_lane_order=[r['lane'] for r in completion_order])


def validate_commit(plan, entry, commit):
    """Read-only lineage validation, deliberately not a resource re-qualification."""
    owner = plan['owners'][entry['owner_index']]
    row = commit['native_command_receipt']
    receipt = row['receipt']
    expected = dict(schema=dispatch.COMMIT_SCHEMA, plan_sha256=dispatch.common.digest(plan),
        cohort_index=entry['cohort_index'], owner=owner, command_id=entry['body']['command_id'],
        payload_sha256=entry['payload_sha256'], native_command_receipt=row, result=receipt['result'])
    require(commit == expected, 'commit plan/owner/command/result binding differs')
    require(row.get('status') == 'COMPLETE' and row.get('payload_sha256') == entry['payload_sha256'],
            'completed command payload binding differs')
    for key in ('source_sha256', 'worker_instance_id', 'native_pid'):
        require(receipt.get(key) == owner[key], f'historical native {key} differs')
    require(receipt.get('command_id') == entry['body']['command_id'] and receipt.get('complete') is True,
            'historical native command completion differs')
    value = teacher.validate_result(plan['panel'], commit['result'])
    require(value['benchmark'] == entry['artifacts'][0]['value'], 'historical benchmark changed')
    return value


def replay_campaign(dispatch_root, trace_root, reference_root):
    dispatch_root, trace_root = Path(dispatch_root), Path(trace_root)
    plan_path = dispatch_root / 'plan.json'
    plan = dispatch.validate_plan(read(plan_path))
    oracle = load_oracle(reference_root)
    entries = plan['commands']
    expected_paths = set()
    cohort_rows, validated, case_rows = [], [], []
    for entry in entries:
        index = entry['cohort_index']
        cid = entry['body']['command_id']
        base = trace_root / f"gpu{entry['owner_index']}" / 'phases' / cid / f'cohort-{index:03d}'
        trace_path, raw_path = base / 'steps.jsonl', base / 'result.json'
        wrapper_path = base.parent / 'result.json'
        expected_paths.update((trace_path, raw_path, wrapper_path))
        commit_path = dispatch_root / f'cohort-{index:03d}' / 'commit.json'
        value = validate_commit(plan, entry, read(commit_path))
        require(read(wrapper_path) == value, 'native phase wrapper differs from committed result')
        raw = read(raw_path)
        require(all(key in value['cohort_result'] and value['cohort_result'][key] == item
                    for key, item in raw.items()), 'raw native result differs from committed result')
        spec = plan['panel']['cohorts'][index]
        count = len(spec['case_ids'])
        with trace_path.open() as handle:
            replay = replay_events((parse(line) for line in handle), count, oracle)
        require(replay['vector_steps'] == raw.get('vector_steps'), 'raw trace/result vector steps differ')
        require(all(raw.get(key) == item for key, item in replay['accounting'].items()),
                'raw events/result accounting differs')
        require(replay['original_completed'] == raw.get('original_raven_metric_completed')
                and replay['original_successes'] == raw.get('original_raven_metric_successes'),
                'raw events/original native metric summary differs')
        cases = [dict(case_id=case, cohort_index=index, **outcome)
                 for case, outcome in zip(spec['case_ids'], replay['accounting']['outcomes'])]
        case_rows.extend(cases)
        cohort_rows.append(dict(cohort_index=index, command_id=cid, count=count,
            successes=replay['original_successes'], vector_steps=replay['vector_steps'],
            trace=file_identity(trace_path), raw_result=file_identity(raw_path),
            native_wrapper=file_identity(wrapper_path),
            commit=file_identity(commit_path),
            historical_numeric_prefix_sha256=raw.get('numeric_trace_sha256'),
            historical_numeric_prefix_steps=raw.get('numeric_trace_steps_recorded'),
            original_completion_lane_order=replay['original_completion_lane_order']))
        validated.append(value)
    observed = set(trace_root.rglob('steps.jsonl')) | set(trace_root.rglob('result.json'))
    require(observed == expected_paths, 'unexpected or omitted cohort trace/result files')
    require(len(case_rows) == 295 and len({r['case_id'] for r in case_rows}) == 295,
            'full295 exact case membership differs')
    final_path = dispatch_root / 'result.json'
    final = teacher.aggregate_results(plan['panel'], validated)
    require(read(final_path) == final, 'historical full295 aggregate differs')
    successes = sum(r['exported_success'] for r in case_rows)
    require(successes == final['successes'], 'trace-derived full295 success count differs')
    from pipeline.deployment.rtx_actor import success_diagnostic_core
    return dict(schema='roborender.historical-first-episode-scoring-replay.v1', verdict='PASS',
        scope='historical_recorded_outcome_scoring_only', completed=len(case_rows), successes=successes,
        failures=len(case_rows)-successes, success_rate=successes/len(case_rows),
        native_source_sha256=final['source_sha256'], panel_sha256=final['panel_sha256'],
        plan=file_identity(plan_path), historical_aggregate=file_identity(final_path),
        oracle_sources=oracle[2], current_accounting_source=file_identity(success_diagnostic_core.__file__),
        comparator=file_identity(__file__), cohorts=cohort_rows, outcomes=case_rows,
        full_trace_hashes_contemporaneously_sealed=False,
        full_trace_hash_status='freshly_observed_historical_files',
        numeric_prefix_bytes_replayed=False, resource_requalified=False,
        protocol='matched_first_episode_per_lane_not_Raven_default_pooled',
        reward_scope='recorded_float32_cumulative_reward_not_independent_reward_recomputation',
        velocity_metrics_compared=False, independent_native_trajectory_tested=False,
        reset_correctness_retested=False, policy_tested=False, renderer_tested=False,
        stage4_independent_live_Raven_evaluation_passed=False, compute_release_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch-root', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'fresh output path required')
    result = replay_campaign(args.dispatch_root, args.trace_root, args.reference_root)
    with args.output.open('x') as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'scope', 'completed', 'successes', 'failures')}))


if __name__ == '__main__':
    main()
