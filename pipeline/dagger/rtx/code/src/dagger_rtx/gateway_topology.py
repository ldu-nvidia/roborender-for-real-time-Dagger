"""Resource-bounded qualification without relaxing production topology."""
from __future__ import annotations


def require_gateway_count(protocol: str, mode: str, workers: int) -> None:
    if mode == "beta-production":
        if protocol != "v3" or workers != 4:
            raise RuntimeError("beta production requires exactly four v3 gateways")
        return
    if mode == "multi-window-check":
        if protocol != "v3" or workers not in {1, 4}:
            raise RuntimeError("multi-window check requires one or four v3 gateways, bound by its frozen config")
        return
    if protocol not in {"v3", "v4"}:
        return
    bounded_pool = (
        protocol == "v3" and workers in {1, 2, 4}
        and mode in {"isolated-one-step", "single-window", "capacity-probe"}
    )
    if workers != 8 and not bounded_pool:
        raise RuntimeError(
            f"{protocol} production requires eight gateways; 1, 2 or 4 v3 gateways are "
            "allowed only for an explicitly bounded native qualification"
        )


def bounded_gateway_ports(protocol: str, mode: str, workers: int, base_port: int) -> tuple[int, ...]:
    require_gateway_count(protocol, mode, workers)
    if workers != 8:
        if not 1024 <= base_port <= 65535 - workers + 1:
            raise RuntimeError("invalid bounded gateway port")
        return tuple(range(base_port, base_port + workers))
    return tuple(range(18620, 18628))
