"""Shared control/SSH deadlines; no simulator or GPU dependencies."""

SSH_STARTUP_SECONDS = 45.0
SSH_RECOVERY_SECONDS = 75.0
CONTROL_REQUEST_TIMEOUT_SECONDS = 15.0
CONTROL_RECOVERY_SECONDS = 120.0

# A control read must not abandon the owner while its pinned tunnel is still
# allowed to reconnect. These are transport deadlines, never replay authority.
assert CONTROL_RECOVERY_SECONDS > SSH_RECOVERY_SECONDS + CONTROL_REQUEST_TIMEOUT_SECONDS
