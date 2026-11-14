"""Parameter operations used by CLI and integrations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from . import get_global_manager, load
from .category_aliases import normalize_category

logger = logging.getLogger(__name__)


def parse_value(value_str: str) -> Any:
    """Parse string value to appropriate Python type."""
    value_str = value_str.strip()

    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    try:
        if "." not in value_str and "e" not in value_str.lower():
            return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    return value_str


def export_parameters(namespace: Optional[str] = None, format_type: str = "yaml", output_file: Optional[Path] = None) -> int:
    """Export parameters to file or stdout."""
    manager = get_global_manager()
    output_path = Path(output_file) if output_file else None
    output_text = manager.export(
        namespace=namespace,
        format=format_type,
        output_path=output_path,
    )
    if output_file:
        logger.info("Parameters exported to: %s", output_file)
        return 0
    if output_text:
        logger.info("%s", output_text)
    return 0


def diff_parameters(file1: Path, file2: Path, ignore_metadata: bool = True) -> int:
    """Compare two parameter files."""
    path1 = Path(file1)
    path2 = Path(file2)
    if not path1.exists():
        logger.error("File not found: %s", path1)
        return 1
    if not path2.exists():
        logger.error("File not found: %s", path2)
        return 1

    data1 = load(path1)
    data2 = load(path2)
    params1 = data1.get("parameters", data1) if isinstance(data1, dict) else data1
    params2 = data2.get("parameters", data2) if isinstance(data2, dict) else data2

    if ignore_metadata:
        params1 = params1 or {}
        params2 = params2 or {}

    all_keys = set(params1.keys()) | set(params2.keys())
    differences = []
    for key in sorted(all_keys):
        val1 = params1.get(key, "<MISSING>")
        val2 = params2.get(key, "<MISSING>")
        if val1 != val2:
            differences.append((key, val1, val2))

    if not differences:
        logger.info("[OK] No parameter differences found")
        return 0

    logger.info("Found %d differences", len(differences))
    for key, val1, val2 in differences:
        logger.info("Parameter: %s", key)
        logger.info("  File 1: %s", val1)
        logger.info("  File 2: %s", val2)
    return 1


def list_parameters(namespace: Optional[str] = None, pattern: Optional[str] = None, category: Optional[Any] = None):
    """List parameters using the global manager.

    A category string is normalized by the canonical category owner
    (parameters.category_aliases); an unknown category raises ValueError.
    Manager errors propagate.
    """
    manager = get_global_manager()
    if isinstance(category, str):
        category = normalize_category(category)
    return manager.list(namespace=namespace, pattern=pattern, category=category)


def get_parameter(key: str, default: Any = None, namespace: Optional[str] = None):
    """Get parameter value using the global manager; manager errors propagate."""
    manager = get_global_manager()
    return manager.get(key, default=default, namespace=namespace)


def set_parameter(key: str, value: Any, units: Optional[str] = None, validate: bool = True, namespace: Optional[str] = None) -> None:
    """Set parameter value using the global manager; a rejected value raises."""
    manager = get_global_manager()
    manager.set_value(key, value, units=units, validate=validate)


def validate_parameters(strict: bool = False, namespace: Optional[str] = None, fix: bool = False):
    """Validate parameters using the global manager.

    Returns the validation findings (empty when every parameter is valid). An
    error raised while validating propagates; it is not reported as a finding.
    """
    manager = get_global_manager()

    if namespace:
        keys = manager.list(namespace=namespace)
    else:
        keys = list(manager.parameters.keys())

    errors = manager.validate_parameters(keys)
    return [error["error"] for error in errors] if errors else []


def execute_cli(args) -> bool:
    """Execute parameter CLI command using the unified manager.

    Returns False for a missing subcommand or a failed command outcome. An
    error raised while running the command propagates to the process-level
    CLI boundary (relic.interfaces.cli.main.main), which reports it and exits
    with 1.
    """
    if not hasattr(args, "params_command") or not args.params_command:
        logger.error("params command requires a subcommand")
        logger.error("Available commands: list, get, set, validate, export, diff, audit")
        return False

    preload_all = getattr(args, "params_command", None) in {
        "list",
        "validate",
        "export",
        "diff",
        "audit",
    }
    manager = get_global_manager(preload_all=preload_all, warmup_cache=False)
    return _execute_with_manager(args, manager)


def _execute_with_manager(args, manager) -> bool:
    """Execute command using unified parameter manager.

    Errors propagate to the process-level CLI boundary
    (relic.interfaces.cli.main.main), which reports them and exits with 1.
    """
    if args.params_command == "list":
        category = getattr(args, "category", None)
        if isinstance(category, str):
            # Canonical category owner; an unknown category raises ValueError.
            category = normalize_category(category)
        keys = manager.list(
            namespace=getattr(args, "namespace", None),
            pattern=getattr(args, "pattern", None),
            category=category,
        )
        results = []
        for key in keys:
            spec = manager.get_parameter(key)
            value = spec.value if spec is not None else manager.get(key)
            results.append(
                {
                    "key": key,
                    "value": value,
                    "units": getattr(spec, "units", None),
                    "description": getattr(spec, "description", None),
                }
            )
        _print_parameter_list(results, getattr(args, "details", False))
        return True

    if args.params_command == "get":
        try:
            value = manager.get(
                args.key,
                default=getattr(args, "default", None),
                namespace=getattr(args, "namespace", None),
                required=getattr(args, "default", None) is None,
            )
        except KeyError:
            # required=True raises KeyError for an undefined key without a
            # --default: the command fails (exit code 1), nothing is substituted.
            logger.error("Parameter '%s' not found", args.key)
            return False
        logger.info("%s = %s", args.key, value)
        return True

    if args.params_command == "set":
        key = args.key
        namespace = getattr(args, "namespace", None)
        if namespace and not key.startswith(namespace):
            key = f"{namespace}.{key}"
        parsed_value = parse_value(args.value)
        manager.set_value(
            key,
            parsed_value,
            units=getattr(args, "units", None),
            validate=not getattr(args, "no_validate", False),
        )
        logger.info("Set %s = %s", key, parsed_value)
        return True

    if args.params_command == "validate":
        namespace = getattr(args, "namespace", None)
        keys = manager.list(namespace=namespace)
        errors = manager.validate_parameters(keys)
        if errors:
            logger.error("Found %d validation issues", len(errors))
            for issue in errors:
                logger.error("  - %s", issue.get("error", issue))
            return False
        logger.info("All parameters validated successfully")
        return True

    if args.params_command == "export":
        return (
            export_parameters(
                namespace=getattr(args, "namespace", None),
                format_type=getattr(args, "format", "yaml"),
                output_file=getattr(args, "output", None),
            )
            == 0
        )

    if args.params_command == "diff":
        return (
            diff_parameters(
                file1=args.file1,
                file2=args.file2,
                ignore_metadata=getattr(args, "ignore_metadata", False),
            )
            == 0
        )

    if args.params_command == "audit":
        report = manager.run_registry_audit(
            expected_params=getattr(args, "expected_params", None),
            namespace_filter=getattr(args, "namespace", None),
            include_unused=getattr(args, "include_unused", False),
        )

        output = manager.export_audit_report(
            report,
            format=getattr(args, "format", "table"),
            output_path=getattr(args, "output", None),
        )

        if output:
            logger.info("%s", output)
        else:
            logger.info("Audit report saved to %s", args.output)

        missing_requests = manager.get_missing_requests()
        if missing_requests and not getattr(args, "output", None):
            logger.info("Top missing parameter requests:")
            for key, count in sorted(
                missing_requests.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]:
                logger.info("  %s: %d requests", key, count)

        if getattr(args, "clear_missing", False):
            manager.clear_missing_requests()
            logger.info("Cleared missing requests tracking")

        return not report.has_failures()

    logger.error("Unknown command: %s", args.params_command)
    return False


def _print_parameter_list(results, show_details: bool = False) -> None:
    """Print parameter list results."""
    if not results:
        logger.info("No parameters found")
        return

    if show_details:
        for param in results:
            logger.info("%s: %s", param.get("key", "unknown"), param.get("value", "N/A"))
            if "units" in param:
                logger.info("  Units: %s", param["units"])
            if "description" in param:
                logger.info("  Description: %s", param["description"])
    else:
        for param in results:
            key = param.get("key", "unknown")
            value = param.get("value", "N/A")
            units = f" ({param['units']})" if param.get("units") else ""
            logger.info("%s: %s%s", key, value, units)
