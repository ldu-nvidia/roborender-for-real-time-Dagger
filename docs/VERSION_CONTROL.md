# Source versioning and releases

The canonical online repository is
https://github.com/ldu-nvidia/roborender-for-real-time-Dagger.
The user selected direct commits to `main` on October 28, 2026; a review branch
is not required. Do not force-push or overwrite remote history.

## One history, two working checkouts

- Cosmos CPU access: `/workspace/project/roborender`.
- Leo RTX: `/data1/workspaces/roborender/source`.
- Both use the same repository and commit history, not independently maintained
  RTX and H200 products. Hardware-specific modules remain in that source tree.
- Reconcile dirty checkouts using per-file hashes and source-only backups before
  copying changes. Never synchronize `.git`, blindly overwrite local changes,
  or delete a workspace to make it match another host.

## Commit boundary

Version production code, first-party tests under `test/`, configuration templates,
dependency specifications, container build recipes, documentation, and reviewed
upstream patches with their provenance. Review existing vendored code and its
redistribution permissions before declaring a public release.

Keep datasets, motion/model payloads, checkpoints, built containers/engines,
Python environments, credentials, logs, mutable run receipts and source archives
out of Git. Reference artifacts by registry identifier and checksum. `.gitignore`
does not remove already tracked files: inspect the staged tree as well.

Before committing: review the exact staged diff and renames, scan for credentials
and large/binary artifacts, run the applicable CPU regression suite and shell
syntax checks, and run `git diff --cached --check`. Do not suppress failing tests
to label a release certified. Report skips and native-test limits explicitly.

## Push and deployment

1. Authenticate to this exact remote without placing a token in a command,
   source file, or chat. Verify SSH host keys against GitHub's published keys;
   never disable host verification.
2. Fetch remote refs and reconcile `main` before a normal fast-forward push.
   If access is unavailable, local commits may be prepared, but do not claim a
   push or assume the remote branch is empty.
3. Bring both editable checkouts to the same committed source and verify both
   Git HEAD and per-file source digests. A matching HEAD with dirty files is not
   source equivalence.
4. Tag a release only with evidence appropriate to that label. Record the Git
   commit, source manifest, container digest, renderer/checkpoint identities and
   test evidence. A source checkpoint is not production/native certification.

Running Isaac and H200 workers execute immutable releases. A Git commit, pull,
push or source synchronization must not hotpatch them or restart workloads.
Deploy changes only through an admitted safe phase boundary. Retain all compute
unless the user explicitly authorizes release of the particular resource.
