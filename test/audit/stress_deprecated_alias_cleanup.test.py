"""Audit gate to ensure deprecated stress functor aliases are removed."""

from __future__ import annotations

from relic.scientific.biosphere import stress


def test_stress_module_has_no_deprecated_functor_aliases() -> None:
    deprecated = [
        "WaterStressFunctor",
        "ThermalStressFunctor",
        "NutrientStressFunctor",
        "OzoneStressFunctor",
    ]
    leaked = [name for name in deprecated if hasattr(stress, name)]
    assert not leaked, "Deprecated stress aliases must be removed: " + ", ".join(leaked)


def test_stress_exports_only_canonical_class_names() -> None:
    exports = getattr(stress, "__all__", [])
    for alias in (
        "WaterStressFunctor",
        "ThermalStressFunctor",
        "NutrientStressFunctor",
        "OzoneStressFunctor",
    ):
        assert alias not in exports, f"{alias} should not be exported in stress.__all__"
