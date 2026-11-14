#!/usr/bin/env python3
"""
L1 tests for radiation conservation audit functor.

Coverage:
- Metadata sanity (category, execution_phase, conservation_laws)
- Basic conservation check with balanced fluxes
- Functor forward integration storing diagnostics
"""

import pytest
import torch

from relic.scientific.atmosphere.radiation_energy_conservation import (
    RadiationEnergyConservationValidator,
    ConservationViolationType,
)
from relic.core.functor import ProcessCategory, ExecutionPhase
from relic.utils.physical_constants import get_physical_constant


def test_metadata_category_and_phase():
    """Validator should register as physical_field functor in validation phase with energy conservation."""
    validator = RadiationEnergyConservationValidator()
    metadata = validator.metadata
    assert metadata.category == ProcessCategory.PHYSICAL_FIELD
    assert metadata.execution_phase == ExecutionPhase.VALIDATION
    assert "energy" in metadata.conservation_laws


def test_validate_energy_conservation_balanced_flux():
    """Balanced shortwave fluxes should yield minor violation with near-zero relative error."""
    validator = RadiationEnergyConservationValidator()
    fluxes = {
        "shortwave_down": torch.tensor(100.0),
        "shortwave_net": torch.tensor(100.0),
    }
    report = validator.validate_energy_conservation(radiation_fluxes=fluxes)
    assert report.violation_type == ConservationViolationType.MINOR
    assert report.relative_error <= validator.relative_minor + 1e-9


def test_forward_stores_last_report_and_diagnostics():
    """Forward path should run audit and expose diagnostic output."""
    validator = RadiationEnergyConservationValidator()
    state = {
        "shortwave_down": torch.tensor(50.0),
        "shortwave_net": torch.tensor(50.0),
    }
    updates = validator.forward(state=state, time=0.0, dt=60.0)
    assert "diagnostics.energy_conservation_report" in updates
    assert validator.last_report is not None
    assert "energy_conservation" in validator.diagnostics
    diag = validator.diagnostics["energy_conservation"]
    assert "relative_error" in diag
    assert diag["violation_type"] in {v.value for v in ConservationViolationType}


def test_forward_reads_declared_inputs_from_the_state_only():
    """The state is the one input channel: an environment keyword is not read, and the
    temperature the audit checks is the declared optional input air_temperature."""
    validator = RadiationEnergyConservationValidator()
    assert "air_temperature" in validator.metadata.input_variables
    assert "air_temperature" in validator.optional_input_variables
    state = {
        "shortwave_down": torch.tensor(120.0),
        "shortwave_net": torch.tensor(100.0),
    }
    # The governed call rejects an environment keyword outright.
    hot = {"air_temperature": torch.tensor(500.0)}
    with pytest.raises(TypeError, match="environment"):
        validator(dict(state), time=0.0, dt=60.0, environment=hot)
    # Without the temperature in the state the audit has no temperature to flag.
    validator.forward(state=dict(state), time=0.0, dt=60.0)
    assert validator.last_report.temperature_valid
    # The same temperature in the state is read and flagged.
    validator.forward(state={**state, **hot}, time=0.0, dt=60.0)
    assert not validator.last_report.temperature_valid


def test_managerless_overrides(monkeypatch):
    """Validator should run from full overrides without touching ParameterManager lookups."""
    overrides = {
        "minor_tolerance": 0.5,
        "moderate_tolerance": 1.0,
        "severe_tolerance": 2.0,
        "relative_minor": 0.01,
        "relative_moderate": 0.05,
        "relative_severe": 0.1,
        "min_albedo": 0.05,
        "max_albedo": 0.99,
        "min_emissivity": 0.8,
        "max_emissivity": 1.0,
        "min_temperature": 200.0,
        "max_temperature": 350.0,
        "canopy_conservation_tolerance": 0.05,
        "enable_canopy_layer_check": True,
    }
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(
            RadiationEnergyConservationValidator,
            "get_parameter_value",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unexpected ParameterManager lookup")
            ),
        )
        validator = RadiationEnergyConservationValidator(
            config={"parameters": overrides}
        )
        report = validator.validate_energy_conservation(
            radiation_fluxes={
                "shortwave_down": torch.tensor(100.0),
                "shortwave_net": torch.tensor(100.0),
            }
        )
        assert report.violation_type == ConservationViolationType.MINOR
        assert validator.minor_tolerance == overrides["minor_tolerance"]


def _full_radiation_fluxes():
    """Independent surface ledger: 500 + 300 = 300 + 100 + 400."""
    return {
        key: torch.tensor(value, dtype=torch.float64)
        for key, value in {
            "shortwave_down": 500.0,
            "longwave_down": 300.0,
            "shortwave_net": 400.0,
            "longwave_net": -100.0,
            "net_radiation": 300.0,
            "shortwave_reflected": 100.0,
            "longwave_up": 400.0,
        }.items()
    }


@pytest.mark.parametrize("include_total_net", [True, False])
def test_full_radiative_ledger_counts_each_energy_transfer_once(include_total_net):
    # References: Campbell and Norman (1998), surface radiation balance.
    # Mathematical expression: SW_down + LW_down = R_net + SW_reflected + LW_up.
    # Parameter meanings: R_net = SW_net + LW_net is the signed absorbed flux.
    # Units: every term is W m-2; relative_error is dimensionless.
    # Domain/constraints: the net total and its two components are redundant observations.
    # Sign convention: downwelling and emitted magnitudes are positive; net may be negative.
    validator = RadiationEnergyConservationValidator()
    fluxes = _full_radiation_fluxes()
    if not include_total_net:
        del fluxes["net_radiation"]
    output = validator(fluxes, time=0.0, dt=60.0)
    report = validator.last_report
    assert report.total_energy_in == pytest.approx(800.0)
    assert report.total_energy_out == pytest.approx(800.0)
    assert report.energy_balance_error == pytest.approx(0.0, abs=1e-12)
    assert report.relative_error == pytest.approx(0.0, abs=1e-12)
    assert report.diagnostics["radiation_balance_scope"] == "full_radiative_flux"
    assert output["diagnostics.energy_conservation_report"].item() == 0.0
    assert not report.violations


def test_redundant_net_observation_cannot_hide_an_inconsistent_component():
    validator = RadiationEnergyConservationValidator()
    fluxes = _full_radiation_fluxes()
    fluxes["shortwave_net"] = torch.tensor(350.0, dtype=torch.float64)
    report = validator.validate_energy_conservation(fluxes)
    assert report.diagnostics["net_component_residual"] == pytest.approx(50.0)
    assert report.violations


def test_nighttime_negative_net_radiation_closes_without_clipping():
    validator = RadiationEnergyConservationValidator()
    fluxes = {key: torch.tensor(value) for key, value in {
        "shortwave_down": 0.0, "shortwave_net": 0.0,
        "shortwave_reflected": 0.0, "longwave_down": 200.0,
        "longwave_up": 350.0, "longwave_net": -150.0,
        "net_radiation": -150.0,
    }.items()}
    report = validator.validate_energy_conservation(fluxes)
    assert report.total_energy_in == report.total_energy_out == 200.0
    assert report.relative_error == 0.0
    assert not report.violations


def test_consumer_albedo_in_radiation_state_supplies_reflected_port():
    validator = RadiationEnergyConservationValidator()
    fluxes = _full_radiation_fluxes()
    del fluxes["shortwave_reflected"]
    report = validator.validate_energy_conservation(
        fluxes, radiation_state={"surface_albedo": torch.tensor(0.2, dtype=torch.float64)}
    )
    assert report.total_energy_in == pytest.approx(800.0)
    assert report.total_energy_out == pytest.approx(800.0)
    assert report.relative_error == pytest.approx(0.0, abs=1e-12)
    assert report.shortwave_balance["sw_reflected"] == pytest.approx(100.0)
    assert report.diagnostics["reflected_flux_source"] == "surface_albedo"


def test_shortwave_only_contract_discloses_limited_coverage():
    validator = RadiationEnergyConservationValidator()
    report = validator.validate_energy_conservation(
        {"shortwave_down": torch.tensor(100.0), "shortwave_net": torch.tensor(100.0)}
    )
    assert report.relative_error == 0.0
    assert report.diagnostics["radiation_balance_scope"] == "shortwave_supplied_components"
    assert any("shortwave" in message and "limited" in message for message in report.warnings)


@pytest.mark.parametrize("alias", ["incoming_shortwave", "solar_radiation"])
def test_flux_aliases_are_single_observations_and_conflicts_are_rejected(alias):
    validator = RadiationEnergyConservationValidator()
    fluxes = _full_radiation_fluxes()
    canonical = "shortwave_down"
    fluxes[alias] = fluxes[canonical].clone()
    report = validator.validate_energy_conservation(fluxes)
    assert report.relative_error == 0.0
    fluxes[alias] = fluxes[alias] + 1.0
    with pytest.raises(ValueError, match="Conflicting.*" + canonical):
        validator.validate_energy_conservation(fluxes)


def test_surface_emission_is_not_an_alias_for_total_outgoing_longwave():
    """An emitted-only descriptor must not impersonate the measured total flux."""
    validator = RadiationEnergyConservationValidator()
    fluxes = _full_radiation_fluxes()
    fluxes["surface_emission"] = fluxes["longwave_up"].clone()
    with pytest.raises(ValueError, match="surface_emission.*emitted.*total"):
        validator.validate_energy_conservation(fluxes)
    assert validator.total_checks == 0
    assert not validator.conservation_history


@pytest.mark.parametrize("emissivity", [0.0, 0.97, 1.0])
def test_property_based_longwave_audit_includes_reflection_and_reports_actual_upward(emissivity):
    """Property inference verifies the greybody identity and discloses partial coverage."""
    validator = RadiationEnergyConservationValidator()
    incident, temperature = 300.0, 295.15
    sigma = get_physical_constant('stefan_boltzmann_constant')
    # References: Noah-MP Technical Note, Eqs. 3.6.330 and 3.6.416-417;
    #   Nicodemus (1965), doi:10.1364/AO.4.000767.
    # Mathematical expression: emitted=eps*sigma*Ts^4; reflected=(1-eps)*Ldown;
    #   Lup=emitted+reflected; Lnet=eps*Ldown-emitted.
    # Parameter meanings: eps surface LW emissivity/absorptivity; sigma SSOT
    #   Stefan-Boltzmann constant; Ts surface temperature; Ldown incident LW.
    # Units: all fluxes [W m-2]; Ts [K]; sigma [W m-2 K-4]; eps [1].
    # Domain/constraints: reciprocal opaque grey surface, finite scalars,
    #   eps in [0,1]; only local LW inference is qualified, the missing total
    #   upward observation does not certify a full measured global ledger.
    #   Existing model plausibility limits, including eps screening, stay active.
    # Sign convention: incident/net inward-positive; emitted/reflected/Lup outward-positive.
    emitted = emissivity * sigma * temperature ** 4
    total_up = emitted + (1.0 - emissivity) * incident
    net = emissivity * incident - emitted
    report = validator.validate_energy_conservation(
        {"longwave_down": torch.tensor(incident, dtype=torch.float64),
         "longwave_net": torch.tensor(net, dtype=torch.float64)},
        radiation_state={"surface_emissivity": torch.tensor(emissivity, dtype=torch.float64),
                         "surface_temperature": torch.tensor(temperature, dtype=torch.float64)},
    )
    assert report.longwave_balance["lw_error"] == pytest.approx(0.0, abs=1e-12)
    assert report.longwave_balance["lw_up"] == pytest.approx(total_up, abs=1e-12)
    assert report.longwave_balance["lw_net"] == pytest.approx(net, abs=1e-12)
    assert report.diagnostics["radiation_balance_scope"] == "partial_radiative_flux"
    assert any("limited" in message for message in report.warnings)


def test_explicit_total_upward_observation_is_not_replaced_by_property_inference():
    """A complete measured ledger retains its supplied total upward LW."""
    validator = RadiationEnergyConservationValidator()
    report = validator.validate_energy_conservation(
        _full_radiation_fluxes(),
        radiation_state={"surface_emissivity": torch.tensor(0.97, dtype=torch.float64),
                         "surface_temperature": torch.tensor(285.15, dtype=torch.float64)},
    )
    assert report.longwave_balance["lw_up"] == 400.0
    assert report.longwave_balance["lw_error"] == 0.0
    assert report.diagnostics["radiation_balance_scope"] == "full_radiative_flux"
    assert not report.violations


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_flux_is_rejected_before_successful_report(invalid):
    validator = RadiationEnergyConservationValidator()
    fluxes = _full_radiation_fluxes()
    fluxes["longwave_down"] = torch.tensor(invalid)
    with pytest.raises(ValueError, match="finite"):
        validator.validate_energy_conservation(fluxes)
    assert validator.total_checks == 0
    assert not validator.conservation_history


def test_canopy_absorption_fraction_does_not_require_complete_absorption():
    validator = RadiationEnergyConservationValidator(
        config={"parameters": {"enable_canopy_layer_check": True}}
    )
    report = validator.validate_energy_conservation(
        _full_radiation_fluxes(),
        canopy_outputs={"par_total": torch.tensor(100.0),
                        "apar_layer": torch.tensor([20.0, 30.0]),
                        "fpar": torch.tensor(0.5)},
    )
    assert report.canopy_balance["canopy_error"] == 0.0
    assert report.canopy_balance["fpar_error"] == 0.0
    assert report.diagnostics["canopy_balance_scope"] == "absorption_only"
    assert not report.violations


def test_canopy_energy_check_does_not_read_photon_flux_par_as_energy():
    """``par`` is a photon flux density [umol photons m-2 s-1] (its contract and
    readers); the canopy energy budget reads incident PAR energy only from
    ``par_total`` [W m-2] and never relabels the photon flux as energy."""
    validator = RadiationEnergyConservationValidator(
        config={"parameters": {"enable_canopy_layer_check": True}}
    )
    report = validator.validate_energy_conservation(
        _full_radiation_fluxes(),
        canopy_outputs={"par": torch.tensor(1500.0),
                        "apar_layer": torch.tensor([20.0, 30.0]),
                        "fpar": torch.tensor(0.5)},
    )
    assert "PAR total not available for canopy conservation check" in report.warnings
    assert not report.canopy_balance


def test_canopy_transmitted_port_closes_full_energy_budget():
    validator = RadiationEnergyConservationValidator(
        config={"parameters": {"enable_canopy_layer_check": True}}
    )
    canopy = {"par_total": torch.tensor(100.0),
              "apar_layer": torch.tensor([20.0, 30.0]),
              "par_transmitted": torch.tensor(50.0), "fpar": torch.tensor(0.5)}
    balanced = validator.validate_energy_conservation(_full_radiation_fluxes(), canopy_outputs=canopy)
    assert balanced.canopy_balance["canopy_error"] == 0.0
    assert balanced.diagnostics["canopy_balance_scope"] == "absorbed_and_transmitted"
    canopy["par_transmitted"] = torch.tensor(40.0)
    broken = validator.validate_energy_conservation(_full_radiation_fluxes(), canopy_outputs=canopy)
    assert broken.canopy_balance["canopy_error"] == 10.0
    assert broken.violations

