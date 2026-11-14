"""Guardrail test: scientific modules must use the unified parameter import entrypoint."""

from pathlib import Path


def test_scientific_modules_do_not_import_parameter_manager_module_directly() -> None:
    root = Path("relic") / "scientific"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from relic.parameters.manager import get_global_manager" in text:
            offenders.append(str(path))

    assert offenders == [], (
        "Found forbidden imports of 'relic.parameters.manager.get_global_manager' in scientific modules: "
        + ", ".join(offenders)
    )

