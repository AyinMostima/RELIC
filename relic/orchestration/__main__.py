"""
Orchestration CLI entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cli import run_compile, run_validate, _write_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RELIC orchestration CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate Graph IR")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--strict", action="store_true", default=False)
    validate_parser.add_argument("--no-topology-audit", action="store_true", default=False)

    compile_parser = subparsers.add_parser("compile", help="Compile Graph IR")
    compile_parser.add_argument("path", type=Path)
    compile_parser.add_argument("--strict", action="store_true", default=False)
    compile_parser.add_argument("--auto-adapt", action="store_true", default=False)
    compile_parser.add_argument("--no-topology-audit", action="store_true", default=False)
    compile_parser.add_argument("--run-bundle", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "validate":
        result = run_validate(
            args.path,
            strict=args.strict,
            include_topology=not args.no_topology_audit,
        )
    else:
        result = run_compile(
            args.path,
            auto_adapt=args.auto_adapt,
            strict=args.strict,
            include_topology=not args.no_topology_audit,
            run_bundle_dir=args.run_bundle,
        )
    _write_output(result)


if __name__ == "__main__":
    main()
