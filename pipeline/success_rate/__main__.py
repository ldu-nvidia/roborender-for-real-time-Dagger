"""Safe CPU entry point for planning and validating frozen-policy evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

from . import dispatch as policies
from .scaled_success_dispatch import read, immutable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser('preserve-bc-conditions', help='Preserve ordered BC32 conditions; CPU only')
    catalog.add_argument('--request', required=True)
    catalog.add_argument('--output', required=True)
    student = commands.add_parser('plan-collection-student', help='Plan the untrained B32x2 three-repeat test; does not launch')
    student.add_argument('--reference', required=True)
    student.add_argument('--checkpoint', required=True)
    student.add_argument('--renderer', required=True, choices=('online_image3f5', 'offline_bc_teacher50'))
    student.add_argument('--seeds', nargs=2, required=True, type=int)
    student.add_argument('--output', required=True)
    plan = commands.add_parser("plan", help="Select student or privileged teacher; plan only, never submit")
    plan.add_argument("--policy", choices=("student", "teacher"), default="student")
    plan.add_argument("--protocol", required=True, choices=(policies.POOLED_PROTOCOL, policies.FIXED_PROTOCOL),
        help="Explicit legacy topology; never silently downgrade a requested B32-per-RTX run")
    for name in ("panel", "owners", "output", "command-prefix"):
        plan.add_argument("--" + name, required=True)
    plan.add_argument("--arm", choices=("bc", "dagger"))
    plan.add_argument("--config", help="Student-only hash-bound rendered observation configuration")
    plan.add_argument("--qualification", help="Student fixed295 current-source qualification proof")
    plan.add_argument("--shadow-checkpoint-sha256",
        help="Teacher-only inert student checkpoint required by the current native harness; not teacher weights")
    plan.add_argument("--qualification-only", action="store_true")
    check = commands.add_parser("validate-plan", help="Validate a plan without contacting owners")
    check.add_argument("--plan", required=True)
    aggregate = commands.add_parser("aggregate", help="Validate all planned results and calculate rate")
    aggregate.add_argument("--plan", required=True)
    aggregate.add_argument("--results", nargs="+", required=True, metavar="COHORT_RESULT")
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == 'preserve-bc-conditions':
        from .contracts import collection_initial_reference
        path = Path(args.request)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20:
            raise ValueError('bounded regular original BC request required')
        value = collection_initial_reference(path.read_bytes())
        immutable(args.output, value)
        print(json.dumps(dict(output=str(args.output), source_request_sha256=value['source_request_sha256'],
                              num_conditions=value['num_conditions'], exact_historical_state_available=False)))
        return 0
    elif args.command == 'plan-collection-student':
        from .contracts import collection_student_plan
        digest = hashlib.sha256()
        with Path(args.checkpoint).open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        value = collection_student_plan(read(args.reference), checkpoint_sha256=digest.hexdigest(),
                                         renderer=args.renderer, seeds=args.seeds)
        immutable(args.output, value)
    elif args.command == "plan":
        value = policies.build_plan(read(args.panel), read(args.owners), policy=args.policy,
            protocol=args.protocol, config=read(args.config) if args.config else None,
            qualification_evidence=read(args.qualification) if args.qualification else None,
            shadow_checkpoint_sha256=args.shadow_checkpoint_sha256,
            command_prefix=args.command_prefix, qualification_only=args.qualification_only, arm=args.arm)
        immutable(args.output, value)
    else:
        value = policies.validate_plan(read(args.plan))
        if args.command == "aggregate":
            value = policies.aggregate(value, [read(path) for path in args.results])
            immutable(args.output, value)
    print(json.dumps(value, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
