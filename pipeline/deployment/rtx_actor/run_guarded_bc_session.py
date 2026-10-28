"""Opt-in fixed-width B1..B6 collection using the qualified shared RAM guards.

The ordinary historical run_bc_session remains untouched. This entrypoint
requires matching actual-B capacity and fresh immutable source/request pins.
It never resumes a simulator or an incomplete native cohort.
"""
from run_guarded_bc_capacity import main


if __name__ == '__main__':
    main(qualification_only=False)
