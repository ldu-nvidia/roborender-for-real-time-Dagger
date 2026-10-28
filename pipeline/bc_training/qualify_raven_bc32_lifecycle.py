"""Two genuine full B32x81 BC updates from the immutable initial student.

CPU-only admission by default. No production epoch/selection, validation pass,
resume, GPU allocation, renderer change, automatic retry, or resource release.
The balanced merged-pool diagnostic sample is NOT the production sorted cursor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.bc_training import qualify_raven_bc_lifecycle as base
from pipeline.bc_training import train_bc_h200 as bc

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'roborender.bc32-lifecycle-qualification.v1'
BOOTSTRAP_SHA = '720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a'
SPEC_SHA = '3e903c55218ca2aae4c970fab6aeb820de0a35d8965a34262b8206fd62f74dee'
BATCH = 32


def source_identity():
    return dict(base.source_identity(), **{
        name:bc.sha256_file(ROOT/name) for name in (
            'pipeline/bc_training/qualify_raven_bc32_lifecycle.py','pipeline/bc_training/fresh_smoke_inputs.py',
            'pipeline/bc_rendering/continuous_source.py','pipeline/bc_rendering/render_recipes.py',
            'pipeline/bc_rendering/render_bc_teacher_8gpu.py','pipeline/bc_rendering/validate_bc_teacher_render.py')})


def selected_rows(index, paths, root, origins):
    indexed = {str((root / row['rendered_npz']).resolve()): row for row in index['items']}
    pools = {'train': [], 'validation': []}
    for path in sorted(paths, key=lambda p: str(p.relative_to(root.resolve()))):
        row = indexed[str(path)]; role = origins[row['chunk_id']]
        if role not in pools:
            raise ValueError('merged chunk has an unrecognized original role')
        pools[role].append(dict(path=str(path), chunk_id=row['chunk_id'], original_role=role,
                                size_bytes=path.stat().st_size))
    if any(len(pool) < 32 for pool in pools.values()):
        raise ValueError('diagnostic requires32 original-train and32 former-validation chunks')
    result = []
    for update in range(2):
        for role in ('train', 'validation'):
            result.extend(pools[role][update * 16:(update + 1) * 16])
    if len({r['chunk_id'] for r in result}) != 64 or len({r['path'] for r in result}) != 64:
        raise ValueError('diagnostic chunks must be unique')
    return result


def plan(args):
    from pipeline.bc_training import fresh_smoke_inputs
    if fresh_smoke_inputs.requested(args):
        return fresh_smoke_inputs.plan(args,source_identity=source_identity())
    admitted = base.plan(args)
    if admitted['bootstrap_sha256'] != BOOTSTRAP_SHA or admitted['policy_spec_sha256'] != SPEC_SHA:
        raise ValueError('B32 qualification requires the exact saved canonical initial student checkpoint/spec')
    root = Path(admitted['dataset_root'])
    report = base.validate_training_admission(Path(admitted['data_admission']), Path(admitted['train_split']),
        Path(admitted['retired_validation_index']), root)
    index, paths = bc.load_split(Path(admitted['train_split']), 'train', root)
    rows = selected_rows(index, paths, root, report['origin_roles'])
    return dict(admitted, schema=SCHEMA, chunks_per_update=BATCH, policy_forward_population=BATCH * 81,
        full_epoch_updates=90, source_identity=source_identity(), selected_chunks=rows,
        selection='diagnostic_each_update_16_original_train_plus16_former_validation_sorted_role_prefixes',
        selected_origin_counts=dict(Counter(r['original_role'] for r in rows)),
        production_sorted_cursor_exercised=False, heldout_validation_examples=0,
        initial_policy='strict_full_state_load_from_saved_canonical_initial_student_checkpoint',
        batch_schema=bc.RAVEN_REPLAY_B32_BATCHING_CONTRACT,
        memory_requirements=dict(minimum_free_device_gib=120, device_headroom_floor_gib=16,
            minimum_host_available_gib=16, minimum_cgroup_headroom_gib=8,
            allocator_cap='min(112GiB,actual_free_before-16GiB,0.8*actual_device_total)',
            no_retry=True, no_microbatch_fallback=True, no_renderer_eviction=True),
        initial_checkpoint_is_bc116=False, random_constructor_weights_retained=False)


def memory_admission(identity):
    free, total = identity['gpu_free_bytes_before'], identity['gpu_total_bytes']
    if free < 120 << 30 or identity['host_available_bytes_before'] < 16 << 30:
        raise RuntimeError('B32 requires120 GiB free device and16 GiB host memory before construction')
    if identity['cgroup_headroom_bytes_before'] is not None and identity['cgroup_headroom_bytes_before'] < 8 << 30:
        raise RuntimeError('B32 requires8 GiB cgroup headroom')
    cap = min(112 << 30, free - (16 << 30), int(.8 * total))
    return dict(allocator_cap_bytes=cap, allocator_fraction=cap / total,
        actual_device_headroom_floor_bytes=16 << 30, minimum_free_before_bytes=120 << 30,
        cap_formula='min(112GiB,actual_free_before-16GiB,0.8*actual_device_total)',
        prior_b16_allocated_bytes=38444574208, prior_b16_reserved_bytes=59919826944,
        linear_b32_allocated_estimate_bytes=2 * 38444574208, linear_b32_reserved_estimate_bytes=2 * 59919826944,
        estimate_is_not_capacity_proof=True, retry_allowed=False, microbatch_fallback_allowed=False,
        renderer_eviction_allowed=False, compute_release_authorized=False)


def verify_initial_policy(policy, bootstrap):
    """All parameters AND buffers must equal the saved student, before update1."""
    if bc.sha256_file(bootstrap) != BOOTSTRAP_SHA:
        raise ValueError('initial saved checkpoint bytes changed')
    saved = torch.load(bootstrap, map_location='cpu', weights_only=False)['policy_state_dict']
    actual = policy.state_dict()
    base.assert_equal(saved, actual)
    digest = hashlib.sha256()
    for name, tensor in sorted(actual.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(initial_state_exact_saved_checkpoint=True, initial_state_tensor_count=len(actual),
                initial_state_sha256=digest.hexdigest(), initial_checkpoint_sha256=BOOTSTRAP_SHA)


def execute(args, admitted):
    output = base.output_path(args)
    if not math.isfinite(args.min_free_gpu_gib) or args.min_free_gpu_gib < 120:
        raise ValueError('B32 smoke requires at least120 GiB free-memory admission')
    device_identity = base.cuda_identity(args.min_free_gpu_gib)
    memory_policy = memory_admission(device_identity)
    torch.cuda.set_per_process_memory_fraction(memory_policy['allocator_fraction'], 0)
    output.mkdir(parents=True, exist_ok=False)
    run = None; started = time.monotonic()
    try:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda:0'); generator = torch.Generator().manual_seed(args.seed)
        paths = [Path(row['path']) for row in admitted['selected_chunks']]
        payload_hashes = {str(path): bc.sha256_file(path) for path in paths}
        bc.atomic_json(output / 'plan.json', dict(admitted, device=device_identity,
            memory_policy=memory_policy, selected_payload_sha256=payload_hashes))
        if args.wandb_mode == 'online':
            import wandb
            run = wandb.init(project=args.wandb_project, name=output.name, group='bc32-lifecycle-qualification',
                mode='online', resume='never', dir=str(output), config=admitted)
        policy, spec = bc.build_policy(Path(admitted['policy_spec']), Path(admitted['bootstrap']), device)
        initial = verify_initial_policy(policy, Path(admitted['bootstrap']))
        bc.atomic_json(output / 'initial-student-proof.json', initial)
        print(json.dumps(dict(phase='initial_saved_student_verified', **initial)), flush=True)
        bc.configure_raven_offline_student(policy)
        pools = bc.install_index_safe_max_pool2d(policy)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.)
        dataset = bc.BCChunks(paths, int(spec['input_contract']['actor_obs_dim']), 11, include_object_target=True)
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0, generator=generator, drop_last=False)
        mean = torch.tensor(spec['input_contract']['rgb_mean'], device=device).view(1, 1, 1, 1, 3)
        std = torch.tensor(spec['input_contract']['rgb_std'], device=device).view(1, 1, 1, 1, 3)
        torch.cuda.reset_peak_memory_stats()
        memory_observations = []
        def memory_check(phase, update):
            free, total = torch.cuda.mem_get_info(0)
            row = dict(phase=phase, update=update, free_bytes=free, total_bytes=total,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            memory_observations.append(row)
            bc.atomic_json(output / 'memory-observations.json', dict(memory_policy=memory_policy, observations=memory_observations))
            if free < memory_policy['actual_device_headroom_floor_bytes'] or row['peak_reserved_bytes'] > memory_policy['allocator_cap_bytes']:
                raise RuntimeError('B32 device headroom/capped-allocator guard failed; no retry or batch substitution')
        def progress(row):
            memory_check('after_update', row['update'])
            if run is not None: run.log(row, step=row['update'])
        metrics = base.run_updates(policy, optimizer, loader, device, mean, std, chunk_batch_size=BATCH,
            pre_update=lambda update: memory_check('before_update', update), progress=progress)
        contract = dict(bc.student_lifecycle_batch_fields(bc.REPLAY_STUDENT_LIFECYCLE, BATCH),
            chunks_per_update=BATCH, frames_per_chunk=81, policy_forward_population=2592,
            microbatching=False, gradient_accumulation=False, training_chunk_count=64, updates_per_epoch=None)
        checkpoint = output / 'diagnostic-checkpoint.pt'
        bc.save_checkpoint(checkpoint, policy, optimizer, 0, 2, None, None, dict(admitted,
            policy_spec=spec, selection='diagnostic_only_not_completed_epoch', qualification_optimizer_steps=2,
            training_batch_contract=contract, training_protocol=dict(schema=SCHEMA, optimizer_updates=2),
            initial_student_proof=initial), generator)
        reloaded = base.reload_checkpoint(checkpoint, policy, optimizer, generator,
            Path(admitted['policy_spec']), Path(admitted['bootstrap']))
        if plan(args) != admitted or source_identity() != admitted['source_identity'] or any(
                bc.sha256_file(Path(path)) != digest for path, digest in payload_hashes.items()):
            raise RuntimeError('admitted source/data/bootstrap changed during B32 smoke')
        result = dict(admitted, **metrics, **reloaded, **initial, result='PASS', device=device_identity,
            memory_policy=memory_policy, memory_observations=memory_observations,
            training_batch_contract=contract, index_safe_max_pools=pools, checkpoint=str(checkpoint),
            checkpoint_sha256=bc.sha256_file(checkpoint), selected_payload_sha256=payload_hashes,
            gpu_free_bytes_after=torch.cuda.mem_get_info(0)[0], gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), total_seconds=time.monotonic() - started)
        if run is not None:
            result['wandb'] = dict(id=run.id, url=run.url)
            run.summary.update(dict(result='PASS', optimizer_updates=2, chunks_per_update=BATCH,
                policy_forward_population=2592, production_promoted=False, **initial, **reloaded))
            run.finish(); run = None
        bc.atomic_json(output / 'smoke-result.json', result)
        print(json.dumps(dict(result='PASS', output=str(output), chunks_per_update=BATCH, **reloaded)), flush=True)
        return result
    except BaseException as error:
        bc.atomic_json(output / 'failure.json', dict(schema=SCHEMA, result='FAIL', error_type=type(error).__name__,
            error=str(error), compute_release_authorized=False, elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        if run is not None: run.finish(exit_code=1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--output', type=Path)
    parser.add_argument('--asset-root', type=Path, default=Path(os.environ.get('ROBORENDER_ASSET_ROOT', ROOT)))
    parser.add_argument('--registry', type=Path, default=ROOT / 'configs/storage_paths.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fresh-continuous-pool',type=Path)
    parser.add_argument('--fresh-render-plan',type=Path)
    parser.add_argument('--fresh-rendered-root',type=Path)
    parser.add_argument('--min-free-gpu-gib', type=float, default=120.)
    parser.add_argument('--wandb-mode', choices=('disabled', 'online'), default='disabled')
    parser.add_argument('--wandb-project', default='dagger')
    args = parser.parse_args(argv)
    if not args.asset_root.is_absolute() or not args.registry.is_absolute(): parser.error('absolute asset root and registry required')
    return args


def main(argv=None):
    args = parse_args(argv); admitted = plan(args)
    if args.execute: execute(args, admitted)
    else: print(json.dumps(admitted, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
