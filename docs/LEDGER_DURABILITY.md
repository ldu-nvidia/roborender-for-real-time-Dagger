# Live ledger ownership and recovery

Every service and SQL reader sharing a run ledger must execute on its recorded
owner host and boot lifetime. A shared filesystem path does **not** make SQLite
WAL safe to read from another machine. CPU orchestration uses authenticated
service endpoints; diagnostics execute on the owner through pinned SSH.

New ledgers use `DELETE` rollback journals, `synchronous=FULL`, one transaction
per commit batch, and a private `ledger-owner.json` binding. Service constructors
serialize initialization. Application readers reject foreign owners before
opening SQLite. This cannot prevent an arbitrary external `sqlite3` script:
never inspect a live shared ledger directly from another host.

Query-only readers use an existing-file `mode=rw` descriptor plus SQLite
`query_only=ON`; writes remain forbidden and missing databases are never created.
The retained virtiofs mount rejects `F_GETLK/F_WRLCK` on `O_RDONLY` descriptors
with `EINVAL` (`SQLITE_IOERR_CHECKRESERVEDLOCK`). Do not work around this by
disabling locks or marking a changing database immutable.

An existing unowned or WAL ledger is rejected, not silently converted. After a
failure, stop only the exact run-owned workloads, retain compute, preserve the
closed database and sidecars with hashes, and reconcile native receipts against
the owning ledger. A clean SQLite integrity check is not proof that every
acknowledged transition exists. Never replay an uncertain native window or
promote incomplete training data. A new host/lifetime requires an explicitly
admitted offline recovery into a fresh namespace; changing the owner marker is
not recovery.

Paired native observers surface either peer's failure and cancel only the other
read-only observer. They do not kill, restart, submit, or step native work.
Reconcile both native owners before any subsequent lifecycle.

## October 18 incident

`10-17-prod-recovery-v12` stopped after two successful optimizer updates when
eight acknowledged step-50 commits were absent from the durable ledger. Native
step-51 was rejected; the incomplete cohort was excluded. A diagnostic helper
had opened the live WAL from the CPU access host, violating the same-host rule.
The actual gateway SQLite 3.50.4 also predates the upstream WAL-reset race fix.
Neither fact alone establishes the exact cause; do not claim a proven root cause
or successful production qualification from the repair's unit tests.

Authoritative SQLite constraints and upstream race details:
https://www.sqlite.org/wal.html (sections "Overview" and "WAL-Reset Bug").

Tests: `test/pipeline/test_ledger_owner.py` exercises owner rejection,
rollback, and independent writer processes with acknowledged-commit readback;
`test/pipeline/test_production_peer_failure.py` exercises bounded peer-failure
reporting without native mutation. New-source joint runtime qualification is
still required before production admission.

Control-only fixes must be wired into every host-operation argv using
`--management-release` and its exact hash; a separate audit note is not an
executable source admission. Keep native source frozen, verify the complete
management manifest, and admit only reviewed old/new runtime hashes. After a
source check fails before dispatch, preserve the failed operation and reconcile
zero native commands/rows/learner jobs before assigning a new control-operation
identity. Never use this path for a dispatched or uncertain simulator window.
