"""Summarize small timing/receipt sidecars; never scan rollout payloads."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def percentile(values, p):
    values = sorted(values)
    position = (len(values)-1)*p
    lo, hi = int(position), min(int(position)+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def analyze(root: Path):
    pair = json.loads((root/'pair-result.json').read_text())
    if not pair['passed']:
        return {'run_id': pair['run_id'], 'qualified': False, 'failure': pair.get('failure'),
                'pair_result': pair}
    actors, starts, ends, window_starts, window_ends = [], [], [], [], []
    for gpu in (1, 2):
        actor_root = root/f'gpu{gpu}'
        native = json.loads((actor_root/'native-service-result.json').read_text())
        telemetry_path = next((p for p in (actor_root/'capacity-telemetry.jsonl', actor_root/'run-telemetry.jsonl') if p.exists()))
        rows = jsonl(telemetry_path)
        steps = [r for r in rows if r.get('event') == 'vector_step_complete']
        expected = native.get('capacity_steps') or 81
        if len(steps) != expected or len(steps) < 3:
            raise RuntimeError('incomplete measured steps')
        indices = [r['step_keys'][0]['rollout_step_index'] for r in steps]
        if indices != list(range(expected)):
            raise RuntimeError('duplicate/missing/out-of-order vector steps')
        batch = native['num_envs']
        for row in steps:
            if len(row['step_keys']) != batch or any(row[k] != batch for k in (
                'unique_prepare_receipts','unique_capture_digests','unique_selection_receipts',
                'unique_commit_receipts','unique_remote_commit_receipts')):
                raise RuntimeError('receipt cardinality mismatch')
        # Exclude cold step zero; full-window step 80 includes seal and is
        # reported separately rather than counted as an ordinary warm step.
        warm = steps[1:-1] if expected == 81 else steps[1:]
        times = [r['timings_ns']['vector_step_total']/1e9 for r in warm]
        start = warm[0]['time_ns']/1e9 - times[0]
        end = warm[-1]['time_ns']/1e9
        starts.append(start)
        ends.append(end)
        window_start = steps[0]['time_ns']/1e9 - steps[0]['timings_ns']['vector_step_total']/1e9
        window_end = steps[-1]['time_ns']/1e9
        window_starts.append(window_start)
        window_ends.append(window_end)
        http = [r for r in rows if r.get('event') in {'http_request','http_batch_request_v2','http_batch_request_v3'}]
        errors = sum(r.get('outcome') != 'success' for r in http)
        retries = sum(r.get('attempt', 1) > 1 for r in http)
        phases = {
            'capture_teacher': lambda t:t.get('native_capture_batch',0),
            'local_prepare_excluding_capture': lambda t:t['local_prepare_barrier']-t.get('native_capture_batch',0),
            'prepare_encode_transfer_render_student_select': lambda t:t['h200_prepare_render_student_select_barrier'],
            'native_env_step': lambda t:t.get('native_env_step_batch',0),
            'local_commit_excluding_env_step': lambda t:t['local_commit_vector_barrier']-t.get('native_env_step_batch',0),
            'remote_commit': lambda t:t['h200_commit_barrier'],
        }
        phase_means = {name:statistics.mean(fn(r['timings_ns'])/1e9 for r in warm) for name,fn in phases.items()}
        phase_means['residual'] = statistics.mean(times)-sum(phase_means.values())
        actors.append({'physical_gpu':gpu,'num_envs':batch,'steps':len(steps),
            'all_committed_rows':len(steps)*batch,
            'collection_window_seconds':window_end-window_start,
            'final_vector_step_seconds':steps[-1]['timings_ns']['vector_step_total']/1e9,
            'warm_steps':len(warm),'warm_rows':len(warm)*batch,'cold_step_seconds':steps[0]['timings_ns']['vector_step_total']/1e9,
            'mean_warm_step_seconds':statistics.mean(times),'p95_warm_step_seconds':percentile(times,.95),
            'phase_mean_seconds':phase_means,
            'phase_share_percent':{k:100*v/statistics.mean(times) for k,v in phase_means.items()},
            'http_errors':errors,'http_retries':retries,
            'request_bytes':sum(r.get('request_body_bytes',0) for r in http),
            'generated_rows':sum(r['observation_source_counts']['generated_controls'] for r in steps),
            'raytraced_rows':sum(r['observation_source_counts']['raytraced'] for r in steps),
            'teacher_rows':sum(r['selected_source_counts']['teacher'] for r in steps),
            'student_rows':sum(r['selected_source_counts']['student'] for r in steps)})
    extent = max(ends)-min(starts)
    window_extent = max(window_ends)-min(window_starts)
    overlap = max(0,min(ends)-max(starts))
    resources = jsonl(root/'resources.jsonl')
    return {'run_id':pair['run_id'], 'qualified':all(a['http_errors']==a['http_retries']==0 for a in actors),
        'actors':actors,'aggregate_warm_rows':sum(a['warm_rows'] for a in actors),
        'aggregate_warm_seconds':extent,'actor_overlap_seconds':overlap,
        'aggregate_committed_transitions_per_second':sum(a['warm_rows'] for a in actors)/extent,
        'full_collection_wall_seconds_including_cold_step_and_seal':window_extent,
        'all_committed_rows':sum(a['all_committed_rows'] for a in actors),
        'full_collection_transitions_per_second':sum(a['all_committed_rows'] for a in actors)/window_extent,
        'min_host_available_gib':min(r['mem_available_kib'] for r in resources)/1024**2,
        'host_swap_growth_mib':(resources[0]['swap_free_kib']-min(r['swap_free_kib'] for r in resources))/1024,
        'all_elapsed_including_cold_initialization_seconds':pair['elapsed_seconds'],
        'not_a_success_rate_or_production_certification':True}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,help='Optional generated JSON summary beside the evidence')
    args=parser.parse_args()
    summary=json.dumps(analyze(args.root),indent=2)+'\n'
    if args.output:
        if args.output.resolve().parent != args.root.resolve() or args.output.suffix != '.json':
            parser.error('output must be a JSON file directly inside the evidence directory')
        args.output.write_text(summary)
    print(summary,end='')
