"""Tests for cross-dimension unit conversion authorization rules."""

from pathlib import Path
from typing import Dict, Any, Optional, List

import pytest
import torch
import yaml

import relic.data.standards.dimensionality as dimensionality_module
from relic.data.standards.unit_converter import UnitConverter, ConversionContext
from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace
from relic.utils.physical_constants import get_physical_constant


def _rules_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "relic" / "parameters" / "schemas" / "cross_dimension_rules.yaml"


def _load_rules() -> List[Dict[str, Any]]:
    rules_path = _rules_path()
    data = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    rules = data.get("rules", [])
    if not isinstance(rules, list):
        return []
    return rules


def _get_context_value(pm, field: str, units: str) -> Optional[float]:
    conv_ns = resolve_namespace(ParameterNamespace.CONVERSION_CONTEXT)

    if field == "soil_depth":
        return pm.get("soil_depth", namespace=conv_ns, units=units, expected_type=float)
    if field == "temperature":
        return get_physical_constant(
            "relic.physical.constants.temperature.reference_temperature"
        )
    if field == "pressure":
        return get_physical_constant(
            "relic.physical.constants.pressure.reference_pressure"
        )
    if field == "water_density":
        return get_physical_constant(
            "relic.physical.constants.water.density_liquid"
        )
    return None


def _build_context(required: Dict[str, str]) -> ConversionContext:
    pm = get_global_manager()
    context = ConversionContext(allow_defaults=False)

    for field, units in required.items():
        value = _get_context_value(pm, field, units)
        assert value is not None, f"Missing configured context value for {field}"
        setattr(context, field, float(value))

    return context


def _rule_variable_cases():
    cases = []
    for rule in _load_rules():
        variables = rule.get("variables") or []
        assert variables, f"Rule {rule.get('name')} must declare variables"
        cases.extend((rule, variable_name) for variable_name in variables)
    return cases


@pytest.mark.parametrize(("rule", "variable_name"), _rule_variable_cases())
def test_cross_dimension_rules_require_context(
    rule: Dict[str, Any],
    variable_name: str,
) -> None:
    converter = UnitConverter()
    source_units = rule.get("lhs")
    target_units = rule.get("rhs")

    assert converter.can_convert(
        source_units,
        target_units,
        variable_name=variable_name,
    ), f"Rule {rule.get('name')} declares unsupported variable {variable_name}"

    required = rule.get("require_context") or {}
    context = _build_context(required)

    result = converter.convert(1.0, variable_name, source_units, target_units, context)
    assert result is not None

    if rule.get("forbid_if_missing_context", False) and required:
        missing_context = ConversionContext(allow_defaults=False)
        required_fields = list(required.keys())
        for field in required_fields[1:]:
            setattr(missing_context, field, getattr(context, field))
        with pytest.raises(ValueError):
            converter.convert(1.0, variable_name, source_units, target_units, missing_context)


def test_carbon_flux_day1_to_si_conversion() -> None:
    converter = UnitConverter()
    context = ConversionContext()
    result = converter.convert(
        1.0,
        "carbon_flux",
        "g C m-2 day-1",
        "kg C m-2 s-1",
        context,
    )
    expected = 1.0 / (1000.0 * 86400.0)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("source_units", "target_units", "expected"),
    [
        ("K", "C", 0.0),
        ("K", "degC", 0.0),
        ("C", "K", 273.15),
        ("degC", "K", 273.15),
    ],
)
def test_temperature_affine_pairs_allowed_for_generic(
    source_units: str,
    target_units: str,
    expected: float,
) -> None:
    converter = UnitConverter()
    result = converter.convert(
        273.15 if source_units == "K" else 0.0,
        "generic",
        source_units,
        target_units,
        ConversionContext(),
    )
    assert float(result) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("source_units", "target_units"),
    [
        ("dimensionless", "MPa"),
        ("MPa", "dimensionless"),
        ("day-1", "dimensionless"),
        ("dimensionless", "day-1"),
    ],
)
def test_semantically_invalid_identity_conversion_is_not_supported(
    source_units: str,
    target_units: str,
) -> None:
    converter = UnitConverter()

    assert converter.can_convert(
        source_units,
        target_units,
        variable_name="generic",
    ) is False
    with pytest.raises(ValueError):
        converter.convert(
            1.0,
            "generic",
            source_units,
            target_units,
            ConversionContext(),
        )


def test_coulomb_case_sensitive_semantics_preserved() -> None:
    converter = UnitConverter()
    assert converter.get_dimension("C") == "electric_charge"
    assert converter.get_dimension("c") == "temperature"


def test_get_dimension_delegates_to_canonical_dimensionality_owner(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dimensionality_module,
        "dimension_category_from_unit",
        lambda unit: "canonical_test_dimension",
    )

    assert UnitConverter().get_dimension("custom_unit") == "canonical_test_dimension"


@pytest.mark.parametrize(
    ("units", "expected_dimension"),
    [
        ("s day-1", "dimensionless"),
        ("g C m-2", "mass_areal_density"),
        ("kg N m-2", "mass_areal_density"),
        ("umol photons m-2 s-1", "amount_flux"),
        ("mol mol-1", "dimensionless"),
        ("degC_day", "thermal_time"),
    ],
)
def test_get_dimension_uses_canonical_composite_taxonomy(
    units: str,
    expected_dimension: str,
) -> None:
    assert UnitConverter().get_dimension(units) == expected_dimension


@pytest.mark.parametrize(
    ("units", "expected_dimension"),
    [
        ("mol m-2 s-1", "amount_flux"),
        ("umol m-2 s-1", "amount_flux"),
        ("kg m-2 s-1", "mass_flux"),
        ("g N m-2 s-1", "mass_flux"),
    ],
)
def test_flux_dimension_distinguishes_amount_from_mass(
    units: str,
    expected_dimension: str,
) -> None:
    assert UnitConverter().get_dimension(units) == expected_dimension


def test_missing_cross_dimension_schema_fails_fast(monkeypatch) -> None:
    monkeypatch.setattr(
        UnitConverter,
        "_find_repo_root",
        staticmethod(lambda start_path: None),
    )

    with pytest.raises(RuntimeError, match="repository root"):
        UnitConverter()._load_cross_dimension_rules()


def test_malformed_cross_dimension_schema_fails_fast(monkeypatch) -> None:
    monkeypatch.setattr(yaml, "safe_load", lambda payload: {"rules": {}})

    with pytest.raises(ValueError, match="rules must be a list"):
        UnitConverter()._load_cross_dimension_rules()


def test_conductance_conversion_is_bidirectional_elementwise_and_differentiable() -> None:
    converter = UnitConverter()
    conductance_mol = torch.tensor(
        [0.1, 0.2], dtype=torch.float64, requires_grad=True
    )
    temperature = torch.tensor(
        [280.0, 310.0], dtype=torch.float64, requires_grad=True
    )
    pressure = torch.tensor(
        [80000.0, 105000.0], dtype=torch.float64, requires_grad=True
    )
    context = ConversionContext(
        temperature=temperature,
        pressure=pressure,
        allow_defaults=False,
    )

    conductance_ms = converter.convert(
        conductance_mol,
        "boundary_layer_conductance",
        "mol m-2 s-1",
        "m s-1",
        context,
    )
    restored = converter.convert(
        conductance_ms,
        "stomatal_conductance",
        "m s-1",
        "mol m-2 s-1",
        context,
    )

    gas_constant = get_physical_constant(
        "relic.physical.constants.universal.gas_constant"
    )
    expected = conductance_mol * gas_constant * temperature / pressure
    assert converter.can_convert(
        "m s-1",
        "mol m-2 s-1",
        variable_name="stomatal_conductance",
    )
    for variable_name in (
        "stomatal_conductance",
        "canopy_conductance",
        "boundary_layer_conductance",
    ):
        assert converter.can_convert(
            "mol m-2 s-1",
            "m s-1",
            variable_name=variable_name,
        )
        assert converter.can_convert(
            "m s-1",
            "mol m-2 s-1",
            variable_name=variable_name,
        )
        assert converter._normalize_variable_name(variable_name) == variable_name
    assert conductance_ms.dtype == torch.float64
    assert torch.allclose(conductance_ms, expected, rtol=1e-12, atol=0.0)
    assert torch.allclose(restored, conductance_mol, rtol=1e-12, atol=0.0)

    gradients = torch.autograd.grad(
        conductance_ms.sum(),
        (conductance_mol, temperature, pressure),
    )
    assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients)
    assert all(torch.any(gradient != 0) for gradient in gradients)


def test_conversion_context_preserves_state_tensors_and_spatial_variation() -> None:
    converter = UnitConverter()
    air_temperature = torch.tensor(
        [285.0, 305.0], dtype=torch.float64, requires_grad=True
    )
    air_pressure = torch.tensor(
        [90000.0, 101325.0], dtype=torch.float64, requires_grad=True
    )

    context = converter.get_conversion_context(
        {
            "air_temperature": air_temperature,
            "atmospheric_pressure": air_pressure,
        }
    )
    converted = converter.convert(
        torch.full((2,), 0.15, dtype=torch.float64),
        "stomatal_conductance",
        "mol m-2 s-1",
        "m s-1",
        context,
    )

    assert context.temperature is air_temperature
    assert context.pressure is air_pressure
    assert converted.shape == air_temperature.shape
    assert converted[0] != converted[1]

    converted.sum().backward()
    assert air_temperature.grad is not None
    assert air_pressure.grad is not None


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (0.0, 101325.0),
        (-1.0, 101325.0),
        (float("nan"), 101325.0),
        (float("inf"), 101325.0),
        (298.15, 0.0),
        (298.15, -1.0),
        (298.15, float("nan")),
        (298.15, float("inf")),
    ],
)
def test_conductance_conversion_rejects_invalid_thermodynamic_context(
    temperature: float,
    pressure: float,
) -> None:
    converter = UnitConverter()
    context = ConversionContext(
        temperature=temperature,
        pressure=pressure,
        allow_defaults=False,
    )

    with pytest.raises(ValueError, match="temperature|pressure"):
        converter.convert(
            torch.tensor(0.1),
            "stomatal_conductance",
            "mol m-2 s-1",
            "m s-1",
            context,
        )


@pytest.mark.parametrize("conductance", [float("nan"), float("inf"), -0.1])
def test_conductance_conversion_rejects_invalid_conductance(
    conductance: float,
) -> None:
    converter = UnitConverter()
    context = ConversionContext(
        temperature=298.15,
        pressure=101325.0,
        allow_defaults=False,
    )

    with pytest.raises(ValueError, match="conductance"):
        converter.convert(
            torch.tensor(conductance),
            "stomatal_conductance",
            "mol m-2 s-1",
            "m s-1",
            context,
        )


def test_conductance_conversion_coerces_context_to_data_device() -> None:
    if not torch.cuda.is_available():
        return

    converter = UnitConverter()
    conductance = torch.tensor(
        [0.1, 0.2], device="cuda", dtype=torch.float64, requires_grad=True
    )
    temperature = torch.tensor(
        [290.0, 300.0], dtype=torch.float32, requires_grad=True
    )
    pressure = torch.tensor(
        [95000.0, 100000.0], dtype=torch.float32, requires_grad=True
    )
    context = ConversionContext(
        temperature=temperature,
        pressure=pressure,
        allow_defaults=False,
    )

    converted = converter.convert(
        conductance,
        "stomatal_conductance",
        "mol m-2 s-1",
        "m s-1",
        context,
    )

    assert converted.device.type == "cuda"
    assert converted.dtype == torch.float64
    converted.sum().backward()
    assert conductance.grad is not None
    assert temperature.grad is not None
    assert pressure.grad is not None
