# Launchers

Canonical H200 launchers live inside their owning phase directories:
`../bc_collection`, `../bc_rendering`, `../bc_training`, `../dagger`, and
`../evaluation`. This directory is documentation-only. New launchers must be
thin, contain no algorithmic logic, and bind only canonical `data/`,
`renderer/`, `checkpoints/`, and `runtime/` paths.

Foreign GPU engines, RTX hostnames, transfer staging, alternate project roots,
and archived custom-protocol controllers are forbidden in active launchers.
