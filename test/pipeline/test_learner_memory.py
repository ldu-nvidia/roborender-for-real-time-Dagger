"""CPU-only tests: admission arithmetic is not a GPU capacity result."""
import pytest

from pipeline.dagger.services import learner_memory as m


def test_cap_accounts_for_other_gpu_clients_and_keeps_more_than_quarter_free():
    p = m.admission(free_bytes=120*m.GIB, total_bytes=140*m.GIB, reserved_bytes=0)
    assert p['allocator_cap_bytes'] == 83*m.GIB
    assert p['external_before_bytes'] + p['allocator_cap_bytes'] == 103*m.GIB
    assert p['estimate_is_capacity_proof'] is False
    assert p['compute_release_authorized'] is False


def test_existing_allocator_reservation_is_not_double_counted_as_external():
    p = m.admission(free_bytes=118*m.GIB, total_bytes=140*m.GIB, reserved_bytes=2*m.GIB)
    assert p['allocator_cap_bytes'] == 83*m.GIB


@pytest.mark.parametrize('free,total,reserved', [(True,140,0),(141,140,0),(0,140,0),(120,140,21)])
def test_bad_actual_counters_rejected(free,total,reserved):
    with pytest.raises(ValueError):m.admission(free_bytes=free,total_bytes=total,reserved_bytes=reserved)


def test_inadequate_budget_does_not_authorize_retry_or_smaller_batch():
    with pytest.raises(RuntimeError):
        m.admission(free_bytes=100*m.GIB,total_bytes=140*m.GIB,reserved_bytes=0)


class Cuda:
    free=120*m.GIB
    reserved=0
    peak=0
    def device_count(self):return 1
    def mem_get_info(self,device):assert device==0;return self.free,140*m.GIB
    def memory_reserved(self,device):return self.reserved
    def memory_allocated(self,device):return self.reserved
    def max_memory_reserved(self,device):return self.peak
    def max_memory_allocated(self,device):return self.peak
    def set_per_process_memory_fraction(self,fraction,device):self.fraction=fraction
    def reset_peak_memory_stats(self,device):self.reset=True


def test_begin_caps_before_any_learner_allocation_and_actual_observation_is_recorded():
    c=Cuda();ctx=m.begin(c)
    assert c.fraction==83/140 and c.reset
    c.reserved=c.peak=75*m.GIB;c.free=45*m.GIB
    assert m.observe(ctx,c,'after_update')['free_bytes']==45*m.GIB
    assert len(ctx['observations'])==1


@pytest.mark.parametrize('change',[
    lambda c:setattr(c,'free',35*m.GIB),
    lambda c:setattr(c,'peak',84*m.GIB),
    lambda c:setattr(c,'reserved',84*m.GIB),
])
def test_observed_headroom_failure_fails_closed(change):
    c=Cuda();ctx=m.begin(c);change(c)
    with pytest.raises(RuntimeError):m.observe(ctx,c,'before_update')
    assert ctx['observations']==[]


def test_multi_device_or_legacy_context():
    c=Cuda();c.device_count=lambda:8
    with pytest.raises(RuntimeError):m.begin(c)
    assert m.observe(None,c,'legacy') is None
