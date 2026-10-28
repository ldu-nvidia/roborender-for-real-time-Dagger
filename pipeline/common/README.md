# Common contracts

The active Raven stages share only canonical path and provenance conventions;
their scientific logic remains in the owning phase and hash-pinned GR00T files.
The superseded custom replay/split implementation has been removed; new code
must use `../dagger/` and its `rtx/` component. Do not create a parallel DAgger
package or copy host-specific source into this directory.

Shared helpers may configure a runtime or resolve the canonical project root,
but they must not own renderer, learner, simulator, or dataset behavior. The
Raven launchers import the explicitly pinned GR00T checkout; distributed
service code imports the canonical repository package.
