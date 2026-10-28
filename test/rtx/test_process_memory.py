import pytest

from dagger_rtx.process_memory import compact_host_heap


def test_default_never_trims_or_reads_runtime(monkeypatch):
    monkeypatch.delenv("ROBORENDER_RTX_COMPACT_HEAP", raising=False)
    assert compact_host_heap("test") is None


def test_opt_in_requires_startup_allocator_limit(monkeypatch):
    monkeypatch.setenv("ROBORENDER_RTX_COMPACT_HEAP", "1")
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    with pytest.raises(RuntimeError, match="before process startup"):
        compact_host_heap("test")


def test_opt_in_trims_only_cpu_heap_and_reports_boundary(monkeypatch, capsys):
    import json
    from dagger_rtx import process_memory

    monkeypatch.setenv("ROBORENDER_RTX_COMPACT_HEAP", "1")
    monkeypatch.setenv("MALLOC_ARENA_MAX", "2")
    readings = iter(["VmRSS:\t2048 kB\n", "VmRSS:\t1024 kB\n"])
    monkeypatch.setattr(process_memory.Path, "read_text", lambda self: next(readings))
    calls = []
    monkeypatch.setattr(process_memory.gc, "collect", lambda: calls.append("gc"))

    class Trim:
        def __call__(self, pad):
            calls.append(("malloc_trim", pad))
            return 1

    class Libc:
        malloc_trim = Trim()

    monkeypatch.setattr(process_memory.ctypes, "CDLL", lambda library: Libc())
    report = compact_host_heap("native_setup_complete")
    assert calls == ["gc", ("malloc_trim", 0)]
    assert report["boundary"] == "native_setup_complete"
    assert report["rss_kib_before"] == 2048
    assert report["rss_kib_after"] == 1024
    assert report["allocator_released_pages"] is True
    assert json.loads(capsys.readouterr().out) == report
