
from relic.coupling.capability_registry import (
    CouplingCapabilityRegistry,
    CouplingErrorCode,
    create_not_implemented_error,
    _resolve_capability_snapshot,
)


class _FakeParameterManager:
    """Lightweight stand-in to drive capability overrides without touching the real database."""

    def __init__(self, mapping):
        self.mapping = mapping

    def get(self, key, units=None, required=False, **kwargs):
        return self.mapping.get(key)


def test_error_codes_are_unique_and_well_formed():
    codes = [code.value for code in CouplingErrorCode]
    assert len(codes) == len(set(codes)), "Error codes must be unique"
    assert all(code.startswith("COUPLING_E") for code in codes)
    assert all(code[10:].isdigit() for code in codes)


def test_capability_status_and_error_codes():
    status = CouplingCapabilityRegistry.get_implementation_status()
    assert "regular_grid_bilinear" in status["implemented"]
    assert "irregular_grid_bilinear" in status["implemented"]

    error_code = CouplingCapabilityRegistry.get_error_code("conservative_regridding_irregular")
    assert error_code == CouplingErrorCode.IRREGULAR_GRID_NOT_SUPPORTED.value


def test_create_not_implemented_error_includes_code_and_feature():
    err = create_not_implemented_error("conservative_regridding_irregular")
    message = str(err)
    assert "COUPLING_E1001" in message
    assert "irregular grids" in message.lower()
    assert "Alternatives" in message


def test_capability_overrides_with_parameter_manager():
    fake_manager = _FakeParameterManager(
        {
            "coupling.capabilities.irregular_grid_bilinear.enabled": True,
            "coupling.capabilities.regular_grid_bilinear.enabled": False,
        }
    )

    effective = _resolve_capability_snapshot(fake_manager)

    assert effective["irregular_grid_bilinear"].status == "implemented"
    assert effective["regular_grid_bilinear"].status == "not_implemented"
    assert CouplingCapabilityRegistry.CAPABILITIES[
        "regular_grid_bilinear"
    ].status == "implemented"


def test_capability_declarations_have_references_and_limitations():
    for name, decl in CouplingCapabilityRegistry.CAPABILITIES.items():
        assert decl.feature, f"{name} should declare a feature description"
        assert decl.status in {"implemented", "partial", "not_implemented"}
        if decl.error_code:
            assert isinstance(decl.error_code, CouplingErrorCode)
        # At least one of limitations/alternatives/references should be present for non-implemented features
        if decl.status != "implemented":
            assert decl.limitations or decl.alternatives or decl.references


def test_categorical_alternative_names_the_basis_rule_not_a_fallback():
    """Categorical fields are regridded by nearest neighbour by their basis; nothing falls back."""
    from relic.coupling import capability_registry

    declaration = capability_registry.CouplingCapabilityRegistry.get_capability("conservative_regridding_irregular")
    assert not any("fallback" in text.lower() for text in declaration.alternatives)
    assert any("categorical" in text and "nearest" in text for text in declaration.alternatives)


def test_capability_override_lookup_errors_and_invalid_values_raise():
    """A failing override lookup or a non-bool override raises; only an absent override is skipped."""
    import pytest

    from relic.coupling.capability_registry import CouplingCapabilityRegistry

    class _Failing:
        def get(self, key, **kwargs):
            raise ValueError(f"invalid override {key}")

    with pytest.raises(ValueError, match="invalid override"):
        _resolve_capability_snapshot(_Failing())

    class _NotBool:
        def get(self, key, **kwargs):
            return "yes"

    with pytest.raises((TypeError, ValueError), match="bool"):
        _resolve_capability_snapshot(_NotBool())

    class _Absent:
        def get(self, key, **kwargs):
            assert kwargs.get("required") is False
            return None

    effective = _resolve_capability_snapshot(_Absent())
    assert {name: item.status for name, item in effective.items()} == {
        name: item.status for name, item in CouplingCapabilityRegistry.CAPABILITIES.items()
    }
