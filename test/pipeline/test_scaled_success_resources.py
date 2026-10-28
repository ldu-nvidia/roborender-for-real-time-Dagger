from copy import deepcopy

import pytest

from pipeline.evaluation.scaled_success_resources import GIB, SCHEMA, summarize


def fixture():
    native = [dict(owner_id="isaac0", gpu_uuid="rtx0", source_sha256="a"*64)]
    renderers = [dict(owner_id="renderer0", gpu_uuid="h200-0", source_sha256="b"*64)]
    samples = []
    for t in (100., 105., 110.):
        samples.append(dict(schema=SCHEMA, at_unix=t, host_available_bytes=24*GIB,
            native=[dict(**native[0], alive=True, vram_used_bytes=20*GIB, vram_total_bytes=96*GIB,
                anon_bytes=10*GIB, shmem_bytes=GIB, kernel_bytes=GIB, memory_limit_bytes=21*GIB,
                swap_bytes=0, swap_limit_bytes=0, oom_events=0, oom_kill_events=0,
                limit_hit_events=0, thp_policy="disable-v1")],
            renderers=[dict(**renderers[0], alive=True, vram_used_bytes=25*GIB, vram_total_bytes=141*GIB)]))
    for sample in samples:
        sample['native'][0]['safety_sample'] = dict(process_id=17, cgroup_device=32,
            cgroup_inode=123, cgroup_path='/sys/fs/cgroup/test-fixture',
            host_available_bytes=24*GIB, memory_current_bytes=18*GIB,
            memory_limit_bytes=21*GIB, swap_bytes=0, swap_limit_bytes=0,
            inactive_file_bytes=6*GIB, working_set_bytes=12*GIB,
            events=dict(max=0, oom=0, oom_kill=0), some_avg10=0., full_avg10=0.)
    return samples, dict(native_owners=native, renderer_owners=renderers, started_unix=101., ended_unix=109.)


def test_measured_resources_never_certify_pipeline_alone():
    samples, options = fixture()
    result = summarize(samples, **options)
    assert result["resources_passed"]
    assert not result["production_ready"]
    assert result["maximum_native_private_bytes"] == {"isaac0": 12*GIB}
    assert not result["compute_release_authorized"]


@pytest.mark.parametrize("field,value", [
    ("swap_bytes", 1), ("swap_limit_bytes", -1), ("oom_events", 1),
    ("oom_kill_events", 1), ("limit_hit_events", 1), ("thp_policy", "inherited"),
    ("memory_limit_bytes", 32*GIB), ("anon_bytes", 15*GIB),
    ("vram_used_bytes", 72*GIB), ("alive", False), ("source_sha256", "c"*64),
    ("gpu_uuid", "other"), ("sampling_error", "unavailable"), ("kernel_bytes", None),
])
def test_reject_bad_native_sample(field, value):
    samples, options = fixture()
    samples[1]["native"][0][field] = value
    with pytest.raises(ValueError):
        summarize(samples, **options)


@pytest.mark.parametrize("mutate", [
    lambda s: s[1].update(host_available_bytes=11*GIB),
    lambda s: s[1].update(native=[]),
    lambda s: s[1].update(renderers=[]),
    lambda s: s[1].update(at_unix=float("nan")),
    lambda s: s[1].update(at_unix=100.),
    lambda s: s[2].update(at_unix=120.),
    lambda s: s[1]["renderers"][0].update(vram_used_bytes=120*GIB),
    lambda s: s[1]["renderers"][0].update(alive=False),
])
def test_reject_gaps_and_missing_renderer_capacity(mutate):
    samples, options = fixture()
    mutate(samples)
    with pytest.raises(ValueError):
        summarize(samples, **options)


def test_distinct_ports_are_not_distinct_gpu_capacity():
    samples, options = fixture()
    options["renderer_owners"].append(dict(options["renderer_owners"][0], owner_id="renderer1"))
    with pytest.raises(ValueError, match="duplicate GPU"):
        summarize(samples, **options)


@pytest.mark.parametrize("key,value", [("started_unix", 99.), ("ended_unix", 111.)])
def test_require_whole_probe_bracket(key, value):
    samples, options = fixture()
    options[key] = value
    with pytest.raises(ValueError):
        summarize(samples, **options)


def test_two_rtx_eight_real_h200_owners_not_eight_ports():
    samples, options = fixture()
    native = [dict(options['native_owners'][0], owner_id=f'isaac{i}', gpu_uuid=f'rtx{i}') for i in range(2)]
    renderers = [dict(options['renderer_owners'][0], owner_id=f'renderer{i}', gpu_uuid=f'h200-{i}') for i in range(8)]
    options.update(native_owners=native, renderer_owners=renderers)
    for sample in samples:
        sample['native'] = [dict(sample['native'][0], **owner) for owner in native]
        sample['renderers'] = [dict(sample['renderers'][0], **owner) for owner in renderers]
    value = summarize(samples, **options)
    assert len(value['renderer_owners']) == 8 and len(value['native_owners']) == 2
    assert not value['production_ready']
    options['renderer_owners'].append(dict(renderers[0], owner_id='renderer8', gpu_uuid='h200-8'))
    with pytest.raises(ValueError, match='topology'):
        summarize(samples, **options)


@pytest.mark.parametrize('kind', ['native', 'renderers', None])
def test_malformed_sample_rejected_cleanly(kind):
    samples, options = fixture()
    if kind is None:
        samples[1] = None
    else:
        samples[1][kind][0] = None
    with pytest.raises(ValueError):
        summarize(samples, **options)


def test_cache_reclaim_counter_not_fabricated_zero_or_confused_with_oom():
    samples, options = fixture()
    for i, sample in enumerate(samples):
        sample['native'][0]['limit_hit_events'] = 100+i
        sample['native'][0]['safety_sample']['events']['max'] = 100+i
    value = summarize(samples, **options)
    assert value['initial_limit_events']['isaac0'] == 100
    assert value['final_limit_events']['isaac0'] == 102
    assert value['interval_limit_events']['isaac0'] == 2


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(safety_sample=None),
    lambda r: r['safety_sample'].update(full_avg10=21),
    lambda r: r['safety_sample'].update(some_avg10=41),
    lambda r: r['safety_sample'].update(working_set_bytes=21*GIB),
    lambda r: r['safety_sample'].update(cgroup_inode=124),
    lambda r: r['safety_sample']['events'].update(max=5),
])
def test_actual_v3_memory_readings_are_required_and_cross_checked(mutate):
    samples, options = fixture()
    mutate(samples[1]['native'][0])
    with pytest.raises(ValueError):
        summarize(samples, **options)


def test_sustained_pressure_never_passes_even_with_private_memory_headroom():
    samples, options = fixture()
    samples = [deepcopy(samples[0]) for _ in range(8)]
    for i, sample in enumerate(samples):
        sample['at_unix'] = 100.+5*i
        sample['native'][0]['safety_sample']['full_avg10'] = 6.
    options['ended_unix'] = 134.
    with pytest.raises(ValueError):
        summarize(samples, **options)
