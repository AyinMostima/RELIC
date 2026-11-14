"""
CLI utilities for RELIC extension workflows.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from .discovery import build_extension_info, discover_extensions
from relic.core.simulator.dependency_ops import _require_execution_group

from .protocol import MANIFEST_FILENAME, ManifestError
from .validator import validate_extension

logger = logging.getLogger(__name__)

_BRIDGE_CHOICES = (
    "biological_process",
    "physical_field",
    "transport_mechanism",
    "scale_coupling",
)

_BRIDGE_TEMPLATE_MAP = {
    "biological_process": "biological_process",
    "physical_field": "physical_field",
    "transport_mechanism": "transport_mechanism",
    "scale_coupling": "scale_coupling",
}

_BRIDGE_CLASS_MAP = {
    "biological_process": "BiologicalProcessFunctor",
    "physical_field": "PhysicalFieldFunctor",
    "transport_mechanism": "TransportMechanismFunctor",
    "scale_coupling": "ScaleCouplingFunctor",
}

_CATEGORY_MAP = {
    "biological_process": "biological_process",
    "physical_field": "physical_field",
    "transport_mechanism": "transport_mechanism",
    "scale_coupling": "scale_coupling",
}

def _to_extension_key(name: str) -> str:
    cleaned = name.strip().replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in cleaned if ch.isalnum() or ch == "_").lower().strip("_")


def _to_class_name(name: str) -> str:
    key = _to_extension_key(name)
    if key.endswith("_functor"):
        key = key[: -len("_functor")]
    parts = [part for part in key.split("_") if part]
    if not parts:
        return "CustomExtensionFunctor"
    return "".join(part.capitalize() for part in parts) + "Functor"


def _render_template(template_text: str, replacements: Dict[str, str]) -> str:
    result = template_text
    for key, value in replacements.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _template_root() -> Path:
    return Path(__file__).resolve().parent / "templates"


def _scaffold_extension(args: argparse.Namespace) -> bool:
    ext_name = args.name.strip()
    if not ext_name:
        print("[ERROR] Extension name cannot be empty.")
        return False

    bridge = args.bridge
    # The execution group is stated on the command line and judged by the
    # simulator's execution-group predicate; it is never implied by the bridge.
    try:
        execution_group = _require_execution_group(
            args.execution_group.strip(),
            source="relic.extensions.cli scaffold --execution-group",
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return False
    template_dir = _template_root() / _BRIDGE_TEMPLATE_MAP[bridge]
    if not template_dir.exists():
        print(f"[ERROR] Missing template directory: {template_dir}")
        return False

    output_dir = Path(args.output).resolve() if args.output else (Path.cwd() / ext_name).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[ERROR] Output directory is not empty: {output_dir}")
        return False
    output_dir.mkdir(parents=True, exist_ok=True)

    extension_key = _to_extension_key(ext_name)
    class_name = _to_class_name(ext_name)
    output_var = f"{extension_key}_flux" if extension_key else "custom_flux"
    test_name = f"test_{extension_key or 'extension'}"

    replacements = {
        "EXTENSION_NAME": ext_name,
        "EXTENSION_KEY": extension_key,
        "FUNCTOR_CLASS": class_name,
        "BRIDGE_CLASS": _BRIDGE_CLASS_MAP[bridge],
        "CATEGORY": _CATEGORY_MAP[bridge],
        "EXECUTION_GROUP": execution_group,
        "OUTPUT_VARIABLE": output_var,
        "TEST_NAME": test_name,
    }

    template_to_output = {
        "functor.py.template": "functor.py",
        MANIFEST_FILENAME: MANIFEST_FILENAME,
        "parameters.yaml": "parameters.yaml",
        "test_functor.py.template": "test_functor.py",
    }

    for template_name, output_name in template_to_output.items():
        template_path = template_dir / template_name
        if not template_path.exists():
            print(f"[ERROR] Missing template file: {template_path}")
            return False
        content = template_path.read_text(encoding="utf-8-sig")
        rendered = _render_template(content, replacements)
        _write_file(output_dir / output_name, rendered)

    print(f"[OK] Extension scaffold created: {output_dir}")
    print("[OK] Generated files:")
    for output_name in template_to_output.values():
        print(f"  - {output_name}")
    return True


def _list_extensions(args: argparse.Namespace) -> bool:
    discovered = discover_extensions()
    source_filter = args.source
    filtered = [
        info for info in discovered
        if source_filter == "all" or info.source == source_filter
    ]

    if not filtered:
        print("[OK] No extensions discovered.")
        return True

    print(f"[OK] Discovered extensions: {len(filtered)}")
    for info in filtered:
        print(f"- {info.name} | source={info.source} | root={info.root_path}")
        for declaration in info.manifest.functors:
            print(
                "    functor=%s bridge=%s category=%s group=%s"
                % (
                    declaration.class_path,
                    declaration.bridge,
                    declaration.category,
                    declaration.execution_group,
                )
            )
    return True


def _validate_extension_path(args: argparse.Namespace) -> bool:
    target = Path(args.path).resolve()
    try:
        info = build_extension_info(target, source="directory")
    except (ManifestError, OSError) as exc:
        # CLI boundary: parse_manifest reports a missing or unreadable file
        # (OSError) and invalid YAML/schema (ManifestError); the command
        # prints the error and main() exits non-zero.
        print(f"[ERROR] Failed to load extension manifest: {exc}")
        return False

    result = validate_extension(info)
    print(f"[OK] Extension: {info.name}")
    print(f"[OK] Valid: {result.valid}")
    if result.warnings:
        print("[OK] Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.errors:
        print("[ERROR] Errors:")
        for error in result.errors:
            print(f"  - {error}")
    return result.valid


def _extension_info(args: argparse.Namespace) -> bool:
    discovered = discover_extensions()
    match = next((info for info in discovered if info.name == args.name), None)
    if match is None:
        print(f"[ERROR] Extension not found: {args.name}")
        return False

    payload: Dict[str, Any] = {
        "name": match.name,
        "root_path": str(match.root_path),
        "source": match.source,
        "manifest": match.manifest.model_dump(by_alias=True),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return True


def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Attach `ext` subcommands to a parser."""

    subparsers = parser.add_subparsers(dest="ext_command", help="Extension commands")

    scaffold = subparsers.add_parser(
        "scaffold",
        help="Generate an extension template project.",
    )
    scaffold.add_argument("name", help="Extension name")
    scaffold.add_argument(
        "--bridge",
        choices=_BRIDGE_CHOICES,
        required=True,
        help="Bridge type (abstract contract) of the generated functor.",
    )
    scaffold.add_argument(
        "--execution-group",
        required=True,
        help="Execution group of the generated functor (fast, medium or slow).",
    )
    scaffold.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for generated extension files.",
    )

    list_cmd = subparsers.add_parser("list", help="List discovered extensions.")
    list_cmd.add_argument(
        "--source",
        choices=("directory", "entry_point", "all"),
        default="all",
        help="Filter extensions by discovery source.",
    )

    validate_cmd = subparsers.add_parser(
        "validate",
        help="Validate one extension directory or manifest file.",
    )
    validate_cmd.add_argument("path", help="Extension directory or relic_extension.yaml path")

    info_cmd = subparsers.add_parser("info", help="Show details for one extension.")
    info_cmd.add_argument("name", help="Extension name")


def execute(args: argparse.Namespace, config: Optional[Dict[str, Any]] = None) -> bool:
    """Execute extension CLI command."""

    _ = config
    command = getattr(args, "ext_command", None)
    if not command:
        print("[ERROR] Missing ext subcommand.")
        return False

    if command == "scaffold":
        return _scaffold_extension(args)
    if command == "list":
        return _list_extensions(args)
    if command == "validate":
        return _validate_extension_path(args)
    if command == "info":
        return _extension_info(args)

    print(f"[ERROR] Unknown ext subcommand: {command}")
    return False


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m relic.extensions.cli",
        description="RELIC extension tools",
    )
    setup_parser(parser)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "ext_command", None):
        parser.print_help()
        return 1
    return 0 if execute(args, config={}) else 1


if __name__ == "__main__":
    sys.exit(main())
