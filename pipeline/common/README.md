# Common contracts

The active Raven stages share only canonical path and provenance conventions;
their scientific logic remains in the owning phase and hash-pinned GR00T files.
`../round_dagger/` is the archived custom replay/split implementation and must
not be imported by new Raven-equivalent stage code. The production runtime
does not place the parent `pipeline/` directory on `PYTHONPATH`; only the clean
exact GR00T checkout is importable by the Raven launchers.
