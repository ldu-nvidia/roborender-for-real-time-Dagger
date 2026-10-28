import pytest

from dagger_rtx.gateway_topology import bounded_gateway_ports, require_gateway_count


@pytest.mark.parametrize("mode", ["isolated-one-step", "single-window", "capacity-probe"])
def test_one_gateway_requires_explicit_bounded_v3_mode(mode):
    assert bounded_gateway_ports("v3", mode, 1, 18620) == (18620,)
    with pytest.raises(RuntimeError):
        require_gateway_count("v4", mode, 1)


@pytest.mark.parametrize("protocol", ["v3", "v4"])
def test_production_remains_eight_gateways(protocol):
    for count in (1, 2, 4, 7):
        with pytest.raises(RuntimeError):
            require_gateway_count(protocol, "production", count)
    assert bounded_gateway_ports(protocol, "production", 8, 18620) == tuple(range(18620, 18628))


def test_unrecognized_mode_does_not_allow_single_gateway():
    with pytest.raises(RuntimeError):
        require_gateway_count("v3", "unbounded", 1)


@pytest.mark.parametrize("workers,base", [(2, 18620), (2, 18622), (4, 18620)])
def test_bounded_capacity_pool_does_not_relax_production(workers, base):
    assert bounded_gateway_ports("v3", "capacity-probe", workers, base) == tuple(range(base, base + workers))
    with pytest.raises(RuntimeError):
        require_gateway_count("v3", "production", workers)
    with pytest.raises(RuntimeError):
        bounded_gateway_ports("v3", "capacity-probe", workers, 65535)
