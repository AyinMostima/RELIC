from __future__ import annotations

import pytest
import torch

from relic.validation.framework.models import (
    ValidationCode,
    ValidationContext,
    ValidationLevel,
)
from relic.validation.framework.rules.coupling_contracts import (
    validate_biosphere_atmosphere_coupling,
    validate_interface_type_consistency,
    validate_soil_vegetation_coupling,
    validate_stress_factor_bounds,
)


def _messages(results) -> str:
    return " ".join(result.message for result in results)


def _has_error(results) -> bool:
    return any(
        result.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
        for result in results
    )


def _has_success(results) -> bool:
    return any("success" in result.tags for result in results)


# References:
#   - Best et al. (2015), J. Hydrometeorol. 16:1425-1442.
#   - Foken (2008), Micrometeorology.
# Mathematical expressions:
#   ET = T + E_soil.
#   R_n = H + LE + G.
# Parameter meanings:
#   ET, T, E_soil: water fluxes [kg m-2 s-1].
#   R_n, H, LE, G: surface energy fluxes [W m-2].
def _biosphere_context(**output_overrides) -> ValidationContext:
    outputs = {
        "evapotranspiration_flux": 1.0,
        "transpiration_rate": 0.7,
        "soil_evaporation_flux": 0.3,
        "sensible_heat_flux": 150.0,
        "latent_heat_flux": 300.0,
        "co2_flux": -1.0e-6,
    }
    outputs.update(output_overrides)
    return ValidationContext(
        metadata={
            "biosphere_atmosphere_coupling": {
                "input_state": {
                    "net_radiation": 500.0,
                    "soil_heat_flux": 50.0,
                    "air_temperature": 298.15,
                    "vapor_pressure_deficit": 1.0,
                    "atmospheric_pressure": 101325.0,
                    "wind_speed": 2.0,
                },
                "output_fluxes": outputs,
            }
        }
    )


def test_biosphere_rule_accepts_closed_water_and_energy_ledgers() -> None:
    results = validate_biosphere_atmosphere_coupling(_biosphere_context())

    assert not _has_error(results)
    assert any("water_balance" in result.tags for result in results)
    assert any("energy_balance" in result.tags for result in results)


def test_biosphere_rule_fails_closed_without_soil_evaporation() -> None:
    context = _biosphere_context()
    del context.metadata["biosphere_atmosphere_coupling"]["output_fluxes"][
        "soil_evaporation_flux"
    ]

    results = validate_biosphere_atmosphere_coupling(context)

    assert _has_error(results)
    assert "not evaluable" in _messages(results)
    assert "soil_evaporation_flux" in _messages(results)
    assert not any(
        "water_balance" in result.tags and "success" in result.tags
        for result in results
    )


def test_biosphere_rule_reports_energy_without_soil_heat_flux_as_not_evaluable() -> None:
    """G is a budget term; G ~ 0 (FAO-56 Eq. 42, day-long steps) must be supplied, not assumed."""
    context = _biosphere_context()
    del context.metadata["biosphere_atmosphere_coupling"]["input_state"]["soil_heat_flux"]

    results = validate_biosphere_atmosphere_coupling(context)

    energy = [result for result in results if "energy_balance" in result.tags]
    assert len(energy) == 1
    assert energy[0].passed is False
    assert "not evaluable" in energy[0].message
    assert "soil_heat_flux" in energy[0].message
    assert energy[0].details[0].value is None


def test_biosphere_rule_rejects_unclosed_energy_ledger() -> None:
    results = validate_biosphere_atmosphere_coupling(
        _biosphere_context(sensible_heat_flux=100.0)
    )

    assert _has_error(results)
    assert "energy" in _messages(results).lower()
    assert not any(
        "energy_balance" in result.tags and "success" in result.tags
        for result in results
    )


# References:
#   - Feddes et al. (1978), Simulation of Field Water Use.
# Mathematical expression:
#   T_eff = sum_i(u_i * Delta z_i).
# Parameter meanings:
#   u_i: root uptake density [kg m-3 s-1].
#   Delta z_i: soil layer thickness [m].
#   T_eff: effective transpiration [kg m-2 s-1].
def _soil_context(input_state, output_fluxes) -> ValidationContext:
    return ValidationContext(
        metadata={
            "soil_vegetation_coupling": {
                "input_state": input_state,
                "output_fluxes": output_fluxes,
            }
        }
    )


def test_soil_rule_fails_closed_when_water_contract_is_missing() -> None:
    results = validate_soil_vegetation_coupling(
        _soil_context({}, {"root_fraction_profile": [0.4, 0.6]})
    )

    assert _has_error(results)
    assert "not evaluable" in _messages(results)
    for name in ("soil_depth_layers", "root_uptake_profile", "effective_transpiration"):
        assert name in _messages(results)
    assert not _has_success(results)


def test_soil_rule_rejects_incompatible_layer_shapes_without_success() -> None:
    results = validate_soil_vegetation_coupling(
        _soil_context(
            {"soil_depth_layers": [0.5, 1.0]},
            {
                "root_uptake_profile": [1.0],
                "effective_transpiration": 1.0,
            },
        )
    )

    assert _has_error(results)
    assert "shape" in _messages(results).lower()
    assert not _has_success(results)


def test_soil_rule_rejects_nonfinite_root_fractions() -> None:
    results = validate_soil_vegetation_coupling(
        _soil_context(
            {"soil_depth_layers": [0.5, 1.0]},
            {
                "root_uptake_profile": [1.0, 1.0],
                "effective_transpiration": 1.0,
                "root_fraction_profile": [float("nan"), 1.0],
            },
        )
    )

    assert _has_error(results)
    assert "finite" in _messages(results).lower()
    assert not _has_success(results)


def test_soil_rule_accepts_closed_water_and_root_fraction_ledgers() -> None:
    results = validate_soil_vegetation_coupling(
        _soil_context(
            {"soil_depth_layers": [0.5, 1.0]},
            {
                "root_uptake_profile": [1.0, 1.0],
                "effective_transpiration": 1.0,
                "root_fraction_profile": [0.4, 0.6],
            },
        )
    )

    assert not _has_error(results)
    assert _has_success(results)


# References:
#   - Jackson et al. (1996), Oecologia 108:389-411 (root fractions of a profile).
#   - Higham (2002), Accuracy and Stability of Numerical Algorithms, section 4.2.
# Mathematical expression:
#   s = sum_i f_i; normalized iff |s - 1| <= gamma_K(u) * (sum_i |f_i| + 1), K = 2 (n + 1).
# Parameter meanings:
#   f_i: root fraction of layer i [1]; n: layer count; u: unit roundoff of the dtype of f.
# Units: fractions, sums and the bound dimensionless [1].
# Domain/constraints: a normalized profile differs from 1 by rounding only; a truncated
#   profile (here a 0.5 % deficit) is not normalized, whatever its size.
# Sign convention: f_i >= 0.
def _root_fraction_results(root_fractions):
    return [
        result
        for result in validate_soil_vegetation_coupling(
            _soil_context(
                {"soil_depth_layers": [0.5, 1.0]},
                {
                    "root_uptake_profile": [1.0, 1.0],
                    "effective_transpiration": 1.0,
                    "root_fraction_profile": root_fractions,
                },
            )
        )
        if "root_fraction" in result.tags
    ]


def test_soil_rule_reports_a_root_profile_that_is_not_normalized() -> None:
    reported = _root_fraction_results([0.4, 0.595])

    assert len(reported) == 1
    assert reported[0].code == ValidationCode.COUPLING_CONTRACT_VIOLATION
    assert reported[0].details[0].value == pytest.approx(0.995)


def test_soil_rule_accepts_a_profile_normalized_in_float32() -> None:
    generator = torch.Generator().manual_seed(3)
    weights = torch.rand(37, generator=generator, dtype=torch.float32)
    fractions = weights / weights.sum()

    assert _root_fraction_results(fractions) == []
# Mathematical expression:
#   0 <= f_stress <= 1.
# Parameter meanings:
#   f_stress: dimensionless environmental limitation factor [-].
@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([]),
        -0.01,
        1.01,
    ],
)
def test_stress_rule_rejects_nonfinite_empty_and_unbounded_values(value) -> None:
    results = validate_stress_factor_bounds(
        ValidationContext(metadata={"stress_factors": {"water_stress_factor": value}})
    )

    assert _has_error(results)
    assert not any(
        result.code == ValidationCode.STRESS_FACTOR_OK for result in results
    )


def test_stress_rule_accepts_finite_closed_interval() -> None:
    results = validate_stress_factor_bounds(
        ValidationContext(
            metadata={
                "stress_factors": {
                    "water_stress_factor": [0.0, 0.5, 1.0],
                    "temperature_stress_factor": 0.75,
                }
            }
        )
    )

    assert not _has_error(results)
    assert any(result.code == ValidationCode.STRESS_FACTOR_OK for result in results)


@pytest.mark.parametrize(
    "source_spec,target_spec,expected_text",
    [
        (
            {"dtype": "float32", "shape": [2], "units": "kg"},
            {"dtype": "float32", "shape": [3], "units": "kg"},
            "Shape mismatch",
        ),
        (
            {"dtype": "float32", "shape": [2], "units": "kg"},
            {"dtype": "float32", "shape": [2], "units": "K"},
            "inconsistent units",
        ),
    ],
)
def test_interface_rule_does_not_report_success_for_contract_mismatch(
    source_spec,
    target_spec,
    expected_text: str,
) -> None:
    results = validate_interface_type_consistency(
        ValidationContext(
            metadata={
                "coupling_interfaces": {
                    "test_interface": {
                        "source_variables": {"shared": source_spec},
                        "target_variables": {"shared": target_spec},
                    }
                }
            }
        )
    )

    assert expected_text in _messages(results)
    assert not _has_success(results)


def test_interface_rule_marks_interface_without_shared_variables_unchecked() -> None:
    results = validate_interface_type_consistency(
        ValidationContext(
            metadata={
                "coupling_interfaces": {
                    "test_interface": {
                        "source_variables": {"source_only": {"dtype": "float32"}},
                        "target_variables": {"target_only": {"dtype": "float32"}},
                    }
                }
            }
        )
    )

    assert any(result.code == ValidationCode.MISSING_REQUIRED_DATA for result in results)
    assert not _has_success(results)


def test_interface_rule_accepts_matching_shared_contract() -> None:
    spec = {"dtype": "float32", "shape": [2], "units": "kg"}
    results = validate_interface_type_consistency(
        ValidationContext(
            metadata={
                "coupling_interfaces": {
                    "test_interface": {
                        "source_variables": {"shared": dict(spec)},
                        "target_variables": {"shared": dict(spec)},
                    }
                }
            }
        )
    )

    assert not _has_error(results)
    assert any(result.code == ValidationCode.INTERFACE_TYPE_OK for result in results)


@pytest.mark.parametrize("field", ["dtype", "shape", "units"])
def test_interface_rule_requires_explicit_declarations(field) -> None:
    """A shared variable without a declared dtype, shape or units is not compared by default."""
    declared = {"dtype": "float32", "shape": [2], "units": "kg"}
    incomplete = {key: value for key, value in declared.items() if key != field}
    results = validate_interface_type_consistency(
        ValidationContext(
            metadata={
                "coupling_interfaces": {
                    "test_interface": {
                        "source_variables": {"shared": dict(declared)},
                        "target_variables": {"shared": incomplete},
                    }
                }
            }
        )
    )

    missing = [
        result for result in results if result.code == ValidationCode.MISSING_REQUIRED_DATA
    ]
    assert len(missing) == 1
    assert missing[0].level == ValidationLevel.ERROR
    assert field in missing[0].message and "shared" in missing[0].message
    assert not _has_success(results)


def test_biosphere_co2_check_has_no_tolerance() -> None:
    """The CO2 check tests finiteness only; it reports no tolerance and no residual."""
    results = validate_biosphere_atmosphere_coupling(_biosphere_context(co2_flux=float("nan")))

    (co2,) = [result for result in results if "co2_flux_finite" in result.tags]
    assert co2.passed is False
    assert co2.details[0].threshold is None
    assert co2.details[0].value is None


@pytest.mark.parametrize(
    ("rule", "label"),
    [
        (validate_biosphere_atmosphere_coupling, "biosphere_atmosphere_coupling"),
        (validate_soil_vegetation_coupling, "soil_vegetation_coupling"),
    ],
)
def test_absent_coupling_data_is_not_applicable(rule, label) -> None:
    (result,) = rule(ValidationContext(metadata={}))

    assert result.code == ValidationCode.MISSING_REQUIRED_DATA
    assert "not applicable (domain absent)" in result.message
    assert label in result.message
    assert result.details[0].value == "not applicable (domain absent)"


# Boundary handlers of the coupling-contract rules (RULES 2, 2026-09-26): a data failure after
# a verdict leaves one explicit failed result; a context-schema fault (AttributeError) reaches
# the validated_rule boundary unrelabelled.
def test_soil_rule_failure_leaves_no_partial_verdict(monkeypatch) -> None:
    from relic.validation.framework.rules import coupling_contracts

    def failing_bound(dtype, count):
        raise RuntimeError("rounding bound failed")

    monkeypatch.setattr(coupling_contracts, "reduction_rounding_bound", failing_bound)
    results = validate_soil_vegetation_coupling(
        _soil_context(
            {"soil_depth_layers": [0.5, 1.0]},
            {
                "root_uptake_profile": [1.0, 1.0],
                "effective_transpiration": 1.0,
                "root_fraction_profile": [0.4, 0.6],
            },
        )
    )

    assert [result.code for result in results] == [ValidationCode.GENERIC_VALIDATION_FAILED]
    assert results[0].passed is False
    assert "rounding bound failed" in results[0].message


def test_stress_rule_does_not_relabel_a_context_schema_fault(monkeypatch) -> None:
    from relic.validation.framework.rules import coupling_contracts

    def schema_fault(stress_factors):
        raise AttributeError("stress factor entry has no mapping interface")

    monkeypatch.setattr(coupling_contracts, "check_stress_factor_bounds_impl", schema_fault)
    results = validate_stress_factor_bounds(
        ValidationContext(metadata={"stress_factors": {"water_stress_factor": 0.5}})
    )

    assert [result.code for result in results] == [ValidationCode.GENERIC_VALIDATION_FAILED]
    assert results[0].passed is False
    assert results[0].message.startswith(
        "Validation rule 'stress_factor_bounds_validation' execution failed"
    )
