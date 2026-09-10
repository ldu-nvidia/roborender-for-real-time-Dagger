# Launchers

Canonical launchers live inside their owning phase directories:
`../bc_collection`, `../bc_rendering`, `../bc_training`, `../dagger`,
`../dagger/rtx`, and `../evaluation`. This directory is documentation-only.
New launchers must be thin, contain no algorithmic logic, and resolve source
relative to the common Git checkout.

Foreign GPU engines, embedded credentials, transfer staging, alternate project
roots, and archived controllers are forbidden. Deployment hostnames, asset
roots, and SSH proxy details must be explicit environment/config bindings, not
copied source trees.
