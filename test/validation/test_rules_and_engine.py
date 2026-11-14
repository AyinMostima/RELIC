# -*- coding: utf-8 -*-
"""
Unit tests for validation rules and engine aggregation.

Note: Use ASCII-only prints/comments to avoid encoding issues.
"""
from types import SimpleNamespace

import pytest
import torch

from relic.validation.framework.engine import ValidationEngine
from relic.validation.framework.layers import L1_UNIT, LAYER_METADATA
from relic.validation.framework.models import ValidationContext, ValidationCode
from relic.validation.framework.registry import get_global_registry, reset_global_registry
from relic.validation.framework.rules import cross_domain
from relic.validation.framework.rules.conservation import (
    ConservationRule,
    evaluate_element_mass_balance,
    validate_carbon_mass_conservation,
    validate_energy_balance_conservation,
    validate_nitrogen_mass_conservation,
    validate_water_mass_conservation,
)
from relic.coupling.core.conservation_checker import reduction_rounding_bound
from relic.parameters import get_global_manager
from relic.data.standards.unit_converter import unit_converter


def make_context_for_water_flux(precip, evap, transp, runoff, dstorage):
    """Helper to construct ValidationContext with required metadata."""
    metadata = {
        "water_fluxes": {
            "precipitation": torch.tensor(precip, dtype=torch.float32),
            "evaporation": torch.tensor(evap, dtype=torch.float32),
            "transpiration": torch.tensor(transp, dtype=torch.float32),
            "runoff": torch.tensor(runoff, dtype=torch.float32),
            "storage_change": torch.tensor(dstorage, dtype=torch.float32),
        }
    }
    ctx = ValidationContext(
        domain_filter=["hydrosphere"],
        tolerance_overrides={},
        metadata=metadata,
    )
    return ctx


def test_water_mass_conservation_pass():
    # Balanced: P - (E + T + R + dS) = 0
    ctx = make_context_for_water_flux(precip=2.0, evap=0.5, transp=0.5, runoff=1.0, dstorage=0.0)
    results = validate_water_mass_conservation(ctx)
    assert any(r.code == ValidationCode.MASS_BALANCE_OK_WATER and r.passed for r in results)


def test_water_mass_conservation_fail():
    # Unbalanced: residual = 1.0 - (1.0+1.0+0.0+0.0) = -1.0 -> should fail
    ctx = make_context_for_water_flux(precip=1.0, evap=1.0, transp=1.0, runoff=0.0, dstorage=0.0)
    results = validate_water_mass_conservation(ctx)
    assert any(r.code == ValidationCode.MASS_BALANCE_FAIL_WATER and r.is_failing() for r in results)


def test_engine_with_conservation_layer():
    # Use engine to execute L2 conservation for hydrosphere only
    ctx = make_context_for_water_flux(precip=2.0, evap=0.5, transp=0.5, runoff=1.0, dstorage=0.0)
    engine = ValidationEngine()
    report = engine.run(layers=["L2"], domains=["hydrosphere"], tags=["conservation"], context=ctx)
    # At least one conservation result exists and passed
    assert report.summary["total"] >= 1
    ok_items = [r for r in report.results if r.code in (ValidationCode.MASS_BALANCE_OK_WATER, ValidationCode.MASS_BALANCE_OK)]
    assert len(ok_items) >= 1
    assert all(item.passed for item in ok_items)


def test_phosphorus_mass_balance_uses_canonical_converter_basis():
    converter = unit_converter
    is_conserved, diagnostics = evaluate_element_mass_balance(
        inputs={
            "fertilizer_p_flux": (
                torch.tensor(1.0e-3, dtype=torch.float64),
                "kg P m-2 s-1",
            )
        },
        outputs={
            "phosphorus_export": (
                torch.tensor(0.75, dtype=torch.float64),
                "g P m-2 s-1",
            )
        },
        storage_change=(
            torch.tensor(0.25, dtype=torch.float64),
            "g P m-2 s-1",
        ),
        element="phosphorus",
        unit_converter=converter,
    )

    assert is_conserved is True
    assert diagnostics["target_unit"] == converter.get_element_budget_units(
        "phosphorus"
    )["flux"]
    assert diagnostics["total_input"] == pytest.approx(1.0, abs=1.0e-12)
    assert diagnostics["total_output"] == pytest.approx(1.0, abs=1.0e-12)


def test_engine_reset_restores_every_strict_structural_rule():
    """A registry reset must not turn strict structural validation into no-op."""
    required_rule_ids = {
        "dec_complex_structure",
        "hodge_passivity_structure",
        "iphs_structure",
        "chain_map_structure",
        "entropy_step",
        "port_power_units",
        "port_matching",
        "overflow_routing",
        "equivalence_certificate",
        "reaction_network_structure",
        "stochastic_sdae_structure",
        "state_domain_structure",
        "observation_diagnostic_structure",
    }
    reset_global_registry()

    engine = ValidationEngine()
    registered_rule_ids = set(get_global_registry().list_rules())

    assert engine.registry is get_global_registry()
    assert required_rule_ids <= registered_rule_ids

    assert engine.registry.unregister_rule("iphs_structure") is True
    assert "iphs_structure" not in engine.registry.list_rules()

    restored_engine = ValidationEngine()
    assert restored_engine.registry is engine.registry
    assert required_rule_ids <= set(restored_engine.registry.list_rules())


def _cf_syntax_context(schema):
    """Build a public StateManager-schema boundary for CF syntax tests."""
    return ValidationContext(state_manager=SimpleNamespace(schema=schema))


@pytest.mark.parametrize(
    "name",
    [
        "air_temperature",
        "vertical_distribution_fraction_of_emission",
        "air__temperature",
        "air_temperature_",
    ],
)
def test_cf_standard_name_syntax_accepts_relic_base_identifier_lexemes(name):
    """Syntax acceptance must not claim official CF table membership."""
    context = _cf_syntax_context(
        {"candidate": SimpleNamespace(cf_standard_name=name, units="1")}
    )

    results = cross_domain.validate_cf_standard_name_syntax(context)

    assert len(results) == 1
    assert results[0].code == ValidationCode.GENERIC_VALIDATION_OK
    assert results[0].passed is True
    assert "official membership not evaluated" in results[0].message.lower()
    assert results[0].metrics == {
        "variables_seen": 1,
        "explicit_names_syntax_checked": 1,
        "variables_without_cf_name": 0,
        "syntax_invalid_count": 0,
    }
    assert results[0].details[0].metadata == {
        "validation_scope": "lexical_syntax_only",
        "official_membership_checked": False,
    }


@pytest.mark.parametrize(
    "name",
    [
        "Air_temperature",
        "air-temperature",
        "_air_temperature",
        "1_air_temperature",
        "",
        "   ",
        17,
        "air_temperature standard_error",
    ],
)
def test_cf_standard_name_syntax_rejects_invalid_relic_identity_values(name):
    """The internal CF identity field accepts one exact base-name lexeme."""
    context = _cf_syntax_context(
        {"candidate": SimpleNamespace(cf_standard_name=name, units="1")}
    )

    results = cross_domain.validate_cf_standard_name_syntax(context)

    assert len(results) == 1
    assert results[0].code == ValidationCode.GENERIC_VALIDATION_FAILED
    assert results[0].is_failing()
    assert "lexical" in results[0].message.lower()
    assert "membership" in results[0].message.lower()
    assert results[0].metrics == {
        "variables_seen": 1,
        "explicit_names_syntax_checked": 1,
        "variables_without_cf_name": 0,
        "syntax_invalid_count": 1,
    }
    assert results[0].details[0].metadata["official_membership_checked"] is False


def test_cf_standard_name_syntax_treats_missing_optional_names_as_neutral():
    """Absent and explicit-null CF identities are optional, not warnings."""
    context = _cf_syntax_context(
        {
            "absent": SimpleNamespace(units="K"),
            "explicit_none": SimpleNamespace(cf_standard_name=None, units="K"),
        }
    )

    results = cross_domain.validate_cf_standard_name_syntax(context)

    assert len(results) == 1
    assert results[0].level.value == "INFO"
    assert results[0].code == ValidationCode.GENERIC_VALIDATION_OK
    assert results[0].metrics == {
        "variables_seen": 2,
        "explicit_names_syntax_checked": 0,
        "variables_without_cf_name": 2,
        "syntax_invalid_count": 0,
    }
    assert "no explicit cf standard names" in results[0].message.lower()
    assert "official membership was not evaluated" in results[0].message.lower()


def test_cf_standard_name_syntax_requires_state_manager_but_allows_empty_schema():
    """Missing audit input fails; an empty public schema is a scoped no-op."""
    missing_results = cross_domain.validate_cf_standard_name_syntax(
        ValidationContext()
    )
    empty_results = cross_domain.validate_cf_standard_name_syntax(
        _cf_syntax_context({})
    )

    assert len(missing_results) == 1
    assert missing_results[0].code == ValidationCode.MISSING_REQUIRED_DATA
    assert missing_results[0].is_failing()
    assert len(empty_results) == 1
    assert empty_results[0].code == ValidationCode.GENERIC_VALIDATION_OK
    assert empty_results[0].metrics["variables_seen"] == 0


def test_cf_standard_name_syntax_does_not_reclassify_programmer_errors():
    """Unexpected rule failures must remain visible to the caller."""

    class ExplodingSchema(dict):
        def items(self):
            raise AssertionError("unexpected schema failure")

    with pytest.raises(AssertionError, match="unexpected schema failure"):
        cross_domain.validate_cf_standard_name_syntax(
            _cf_syntax_context(ExplodingSchema())
        )


def test_cf_standard_name_syntax_has_one_registry_and_layer_identity():
    """The renamed rule must have one active taxonomy and dependency path."""
    assert "validate_cf_standard_name_syntax" in cross_domain.__all__
    assert "validate_cf_standard_name_compliance" not in cross_domain.__all__
    assert hasattr(cross_domain, "_matches_cf_standard_name_syntax")
    assert not hasattr(cross_domain, "_is_valid_cf_standard_name")
    assert not hasattr(cross_domain, "validate_cf_standard_name_compliance")

    rule_metadata = cross_domain.validate_cf_standard_name_syntax._validation_metadata
    dependent_metadata = (
        cross_domain.validate_variable_consistency_cross_domain._validation_metadata
    )
    assert rule_metadata.rule_id == "cf_standard_name_syntax"
    assert "lexical" in rule_metadata.description.lower()
    assert "membership" in rule_metadata.description.lower()
    assert dependent_metadata.dependencies == [
        "cf_standard_name_syntax",
        "unit_conversion_validity",
    ]
    assert "cf_standard_name_syntax" in LAYER_METADATA[L1_UNIT].typical_rules
    assert "cf_standard_name_compliance" not in LAYER_METADATA[L1_UNIT].typical_rules

    reset_global_registry()
    ValidationEngine()
    rule_ids = get_global_registry().list_rules()
    assert rule_ids.count("cf_standard_name_syntax") == 1
    assert "cf_standard_name_compliance" not in rule_ids


# References: LeVeque (1992), Numerical Methods for Conservation Laws; Higham (2002),
#   Accuracy and Stability of Numerical Algorithms, section 4.2 (rounding of a sum).
# Mathematical expression: r = P - (E + T + R + dS); rho = |r| / (|P| + |E| + |T| + |R| + |dS|);
#   the budget passes iff rho <= validation.conservation.water_conservation_tolerance.
# Parameter meanings: P precipitation, E evaporation, T transpiration, R runoff, dS storage change.
# Units: fluxes kg m-2 s-1; rho and the tolerance dimensionless.
# Domain/constraints: the same residual r = 1e-5 kg m-2 s-1 on a 1e-3 and on a 1e3 budget.
# Sign convention: r > 0 when the input exceeds the outputs and the storage change.
_WATER_KEY = "validation.conservation.water_conservation_tolerance"


def _water_budget(scale, residual):
    return ValidationContext(
        domain_filter=["hydrosphere"],
        tolerance_overrides={},
        metadata={
            "water_fluxes": {
                "precipitation": torch.tensor(scale, dtype=torch.float64),
                "evaporation": torch.tensor(scale - residual, dtype=torch.float64),
                "transpiration": torch.tensor(0.0, dtype=torch.float64),
                "runoff": torch.tensor(0.0, dtype=torch.float64),
                "storage_change": torch.tensor(0.0, dtype=torch.float64),
            }
        },
    )


def _with_parameter(key, value, run):
    manager = get_global_manager()
    original = float(manager.get(key))
    manager.set_value(key, value)
    try:
        return run()
    finally:
        manager.set_value(key, original)


@pytest.mark.parametrize(("scale", "closed"), [(1.0e-3, False), (1.0e3, True)])
def test_water_budget_verdict_compares_the_relative_residual(scale, closed):
    """The same 1e-5 residual is a 5e-3 closure error on a 1e-3 budget and 5e-9 on a 1e3 budget."""
    results = validate_water_mass_conservation(_water_budget(scale, 1.0e-5))
    codes = {result.code for result in results}

    assert (ValidationCode.MASS_BALANCE_OK_WATER in codes) is closed
    assert (ValidationCode.MASS_BALANCE_FAIL_WATER in codes) is (not closed)
    budget = next(result for result in results if result.metrics and "tolerance" in result.metrics)
    assert budget.metrics["tolerance"] == pytest.approx(float(get_global_manager().get(_WATER_KEY)))
    assert budget.metrics["water_balance_relative_residual"] == pytest.approx(1.0e-5 / (2.0 * scale - 1.0e-5))


@pytest.mark.parametrize(("scale", "closed"), [(1.0e-3, False), (1.0e3, True)])
def test_energy_budget_verdict_compares_the_relative_residual(scale, closed):
    """Rn - H - LE - G = 1e-5 W m-2 against |Rn| + |H| + |LE| + |G| of 2e-3 or 2e3 W m-2."""
    context = ValidationContext(
        tolerance_overrides={},
        metadata={
            "energy_fluxes": {
                "net_radiation": torch.tensor(scale, dtype=torch.float64),
                "sensible_heat_flux": torch.tensor(0.5 * scale, dtype=torch.float64),
                "latent_heat_flux": torch.tensor(0.5 * scale - 1.0e-5, dtype=torch.float64),
                "ground_heat_flux": torch.tensor(0.0, dtype=torch.float64),
            }
        },
    )
    codes = {result.code for result in validate_energy_balance_conservation(context)}

    assert (ValidationCode.ENERGY_BALANCE_OK in codes) is closed
    assert (ValidationCode.ENERGY_BALANCE_FAIL in codes) is (not closed)


def test_budget_tolerance_comes_only_from_the_budget_key():
    """A per-call override is refused and the budget key alone decides the verdict."""
    overridden = _water_budget(1.0e-3, 1.0e-5)
    overridden.tolerance_overrides = {"water_balance_tolerance": 1.0}
    refused = validate_water_mass_conservation(overridden)
    assert any(
        result.code == ValidationCode.GENERIC_VALIDATION_FAILED and _WATER_KEY in result.message
        for result in refused
    )

    # rho = 1e-5 / (0.2 - 1e-5) = 5e-5: above the packaged 1e-6, within 1e-4 (the key's upper bound).
    budget = lambda: validate_water_mass_conservation(_water_budget(0.1, 1.0e-5))  # noqa: E731
    assert any(result.code == ValidationCode.MASS_BALANCE_FAIL_WATER for result in budget())
    relaxed = _with_parameter(_WATER_KEY, 1.0e-4, budget)
    assert any(result.code == ValidationCode.MASS_BALANCE_OK_WATER for result in relaxed)


# A budget with a missing term cannot be evaluated: the rule reports it as not evaluable
# (ERROR, MISSING_REQUIRED_DATA, passed False, no residual) and names the missing terms; a
# missing term is never taken as zero. G = 0 for day-long steps (Allen et al. 1998, FAO-56
# Eq. 42) is a physical assumption the caller states by supplying ground_heat_flux = 0.
def _only_not_evaluable(results, missing):
    assert len(results) == 1, [result.message for result in results]
    (result,) = results
    assert result.passed is False
    assert result.code == ValidationCode.MISSING_REQUIRED_DATA
    assert "not evaluable" in result.message
    for name in missing:
        assert name in result.message
    assert result.details[0].value is None
    assert result.details[0].metadata["missing_terms"] == list(missing)
    assert not result.metrics
    return result


@pytest.mark.parametrize(
    "missing", ["precipitation", "evaporation", "transpiration", "runoff", "storage_change"]
)
def test_water_budget_with_a_missing_term_is_not_evaluable(missing):
    context = make_context_for_water_flux(precip=2.0, evap=0.5, transp=0.5, runoff=1.0, dstorage=0.0)
    del context.metadata["water_fluxes"][missing]

    _only_not_evaluable(validate_water_mass_conservation(context), [missing])


def test_energy_budget_without_ground_heat_flux_is_not_evaluable():
    fluxes = {
        "net_radiation": torch.tensor(100.0, dtype=torch.float64),
        "sensible_heat_flux": torch.tensor(30.0, dtype=torch.float64),
        "latent_heat_flux": torch.tensor(70.0, dtype=torch.float64),
    }
    context = ValidationContext(tolerance_overrides={}, metadata={"energy_fluxes": fluxes})
    _only_not_evaluable(validate_energy_balance_conservation(context), ["ground_heat_flux"])

    # The daily G ~ 0 assumption (FAO-56 Eq. 42) is valid only when the caller supplies it.
    fluxes["ground_heat_flux"] = torch.tensor(0.0, dtype=torch.float64)
    codes = {result.code for result in validate_energy_balance_conservation(context)}
    assert ValidationCode.ENERGY_BALANCE_OK in codes


def _carbon_context(**fluxes):
    return ValidationContext(
        tolerance_overrides={},
        metadata={
            "carbon_fluxes": {
                name: torch.tensor(value, dtype=torch.float64) for name, value in fluxes.items()
            }
        },
    )


def test_carbon_rule_evaluates_only_the_identities_its_terms_determine():
    """GPP, Ra and NPP determine NPP = GPP - Ra; the total balance needs its storage change."""
    results = validate_carbon_mass_conservation(
        _carbon_context(
            gross_primary_productivity=1.0e-5,
            autotrophic_respiration=4.0e-6,
            net_primary_productivity=6.0e-6,
        )
    )

    assert [result.code for result in results] == [ValidationCode.MASS_BALANCE_OK_CARBON]
    assert "npp_balance" in results[0].tags


def test_carbon_identity_with_a_missing_term_is_not_evaluable():
    results = validate_carbon_mass_conservation(
        _carbon_context(gross_primary_productivity=10.0, net_primary_productivity=6.0)
    )

    _only_not_evaluable(results, ["autotrophic_respiration"])


def test_carbon_terms_that_determine_no_identity_are_not_evaluable():
    results = validate_carbon_mass_conservation(
        _carbon_context(
            gross_primary_productivity=10.0,
            autotrophic_respiration=4.0,
            heterotrophic_respiration=3.0,
        )
    )

    result = _only_not_evaluable(results, [])
    for defining_term in ("net_primary_productivity", "net_ecosystem_exchange", "storage_change"):
        assert defining_term in result.message


@pytest.mark.parametrize(
    "missing", ["fertilizer", "net_mineralization", "plant_uptake", "storage_change"]
)
def test_nitrogen_budget_with_a_missing_term_is_not_evaluable(missing):
    fluxes = {
        name: torch.tensor(1.0, dtype=torch.float64)
        for name in (
            "deposition", "fixation", "fertilizer", "net_mineralization", "denitrification",
            "leaching", "volatilization", "plant_uptake", "storage_change",
        )
    }
    del fluxes[missing]
    context = ValidationContext(tolerance_overrides={}, metadata={"nitrogen_fluxes": fluxes})

    _only_not_evaluable(validate_nitrogen_mass_conservation(context), [missing])


# References: Wang, Y.P., Law, R.M., Pak, B. (2010), Biogeosciences 7:2261-2282 (CASA-CNP;
#   testbed manual: dNsoilmin/dt = Nsnet + Nmindep + Nminfix - Nminloss - Nminleach - Nupland);
#   RELIC docs/source/theory/nitrogen_cycling.md (NH4 and NO3 equations).
# Mathematical expression: dN_min = (deposition + fixation + fertilizer + net mineralization)
#   - (denitrification + leaching + volatilization + plant uptake); nitrification moves N from
#   NH4 to NO3 inside N_min = N_NH4 + N_NO3 and is not a term of the budget.
# Parameter meanings: the soil mineral N fluxes and dN_min the storage change of NH4 + NO3.
# Units: g N m-2 s-1.
# Domain/constraints: every term supplied; a supplied nitrification rate is not read.
# Sign convention: r > 0 when the inputs exceed the outputs and the storage change.
def test_nitrogen_budget_is_the_total_mineral_n_budget():
    fluxes = {
        "deposition": 1.0, "fixation": 0.5, "fertilizer": 2.0, "net_mineralization": 3.0,
        "denitrification": 1.0, "leaching": 1.5, "volatilization": 0.5, "plant_uptake": 2.5,
        "storage_change": 1.0, "nitrification": 4.0,
    }
    context = ValidationContext(
        tolerance_overrides={},
        metadata={
            "nitrogen_fluxes": {
                name: torch.tensor(value, dtype=torch.float64) for name, value in fluxes.items()
            }
        },
    )

    codes = {result.code for result in validate_nitrogen_mass_conservation(context)}

    assert ValidationCode.MASS_BALANCE_OK_NITROGEN in codes
    assert ValidationCode.MASS_BALANCE_FAIL_NITROGEN not in codes


# References: LeVeque (1992), Numerical Methods for Conservation Laws; Higham (2002), Accuracy
#   and Stability of Numerical Algorithms, section 4.2; budget_closure and ledger_acceptance.
# Mathematical expression: r_j = M_j - M_0 for every recorded step j; rho = max_j |r_j| /
#   (sum_i |m_j,i| + sum_i |m_0,i|); closed iff |r_j| <= max(tol * a_j, gamma_K(u) * a_j) with
#   K = 2 (max_j n_j + n_0 + steps), tol = validation.conservation.conservation_tolerance.
# Parameter meanings: M_j recorded amount of step j, m_j,i its cells, n_j the cell count.
# Units: amounts in one amount unit; rho, tol and gamma dimensionless.
# Domain/constraints: at least two finite records of one dtype; no literal tolerance, no
#   step-count loosening and no cap on the reported error.
# Sign convention: r_j > 0 when the amount grew.
_GENERAL_KEY = "validation.conservation.conservation_tolerance"


def test_conservation_rule_judges_the_history_with_the_general_key():
    """1000 -> 999 over ten steps: rho = 1 / 1999; the former rule passed it (tolerance 0.55)."""
    history = [torch.tensor(1000.0 - 0.1 * step, dtype=torch.float64) for step in range(11)]

    report = ConservationRule().validate({"mass_history": history})

    assert report["status"] == "failed"
    assert report["relative_error"] == pytest.approx(1.0 / 1999.0)
    assert report["residual"] == pytest.approx(1.0)
    assert report["tolerance_parameter"] == _GENERAL_KEY
    assert report["tolerance"] == pytest.approx(float(get_global_manager().get(_GENERAL_KEY)))
    relaxed = _with_parameter(
        _GENERAL_KEY, 1.0e-3, lambda: ConservationRule().validate({"mass_history": history})
    )
    assert relaxed["status"] == "passed"
    assert relaxed["relative_error"] == pytest.approx(1.0 / 1999.0)


def test_conservation_rule_counts_cells_and_steps_in_the_rounding_allowance():
    """float32 transfers between cells change the recorded total by rounding only."""
    generator = torch.Generator().manual_seed(11)
    field = 1.0 + 9.0 * torch.rand(64, generator=generator, dtype=torch.float32)
    history = [field.clone()]
    for step in range(10):
        transfer = torch.tensor(0.1234567 * (step + 1), dtype=torch.float32)
        source, target = step % 64, (7 * step + 3) % 64
        field = field.clone()
        field[source] = field[source] - transfer
        field[target] = field[target] + transfer
        history.append(field.clone())
    totals = torch.stack([record.sum() for record in history])
    assert bool(torch.any(totals != totals[0]))

    def judge(records):
        return _with_parameter(
            _GENERAL_KEY, 1.0e-10, lambda: ConservationRule().validate({"mass_history": records})
        )

    report = judge(history)
    worst = int(torch.argmax(torch.abs(totals[1:] - totals[0])).item()) + 1
    activity = float(totals[worst].abs() + totals[0].abs())
    assert report["status"] == "passed"
    assert report["accepted_residual"] == pytest.approx(
        reduction_rounding_bound(torch.float32, 2 * (64 + 64 + 10)) * activity
    )

    leaking = history[:-1] + [history[-1] - torch.eye(64, dtype=torch.float32)[0] * 0.1]
    assert judge(leaking)["status"] == "failed"


@pytest.mark.parametrize("key", ["tolerance", "n_steps", "initial_mass", "final_mass"])
def test_conservation_rule_reads_only_the_recorded_history(key):
    with pytest.raises(ValueError, match=key):
        ConservationRule().validate({"mass_history": [1.0, 1.0], key: 0.5})


def test_conservation_rule_has_no_tolerance_argument_or_alias():
    import inspect

    from relic.validation.framework.rules import conservation

    assert not inspect.signature(ConservationRule).parameters
    assert not hasattr(conservation, "MassConservationRule")
    with pytest.raises(ValueError, match="mass_history"):
        ConservationRule().validate({"mass_history": [1.0]})


# Plausibility limits of the carbon and energy rules are ParameterManager entries in
# validation/conservation.yaml (no literal threshold in the rule).
_PLAUSIBILITY_KEYS = {
    "validation.conservation.plausibility.gross_primary_productivity_max": "g C m-2 s-1",
    "validation.conservation.plausibility.net_radiation_abs_max": "W m-2",
    "validation.conservation.plausibility.latent_heat_flux_min": "W m-2",
    "validation.conservation.plausibility.bowen_ratio_abs_max": "dimensionless",
}


@pytest.mark.parametrize("key", sorted(_PLAUSIBILITY_KEYS))
def test_plausibility_limits_are_complete_parameter_entries(key):
    spec = get_global_manager().get_parameter(key)

    assert spec is not None, key
    assert str(spec.units) == _PLAUSIBILITY_KEYS[key]
    assert spec.bounds is not None and spec.bounds[0] <= spec.value <= spec.bounds[1]
    assert spec.description and spec.references and spec.provenance


def _energy_context(rn, h, le, g):
    return ValidationContext(
        tolerance_overrides={},
        metadata={
            "energy_fluxes": {
                "net_radiation": torch.tensor(rn, dtype=torch.float64),
                "sensible_heat_flux": torch.tensor(h, dtype=torch.float64),
                "latent_heat_flux": torch.tensor(le, dtype=torch.float64),
                "ground_heat_flux": torch.tensor(g, dtype=torch.float64),
            }
        },
    )


def _energy_warnings(context):
    return " ".join(
        result.message
        for result in validate_energy_balance_conservation(context)
        if result.code == ValidationCode.ENERGY_BALANCE_WARN
    )


@pytest.mark.parametrize(
    ("key", "value", "fluxes", "warning"),
    [
        (
            "validation.conservation.plausibility.net_radiation_abs_max",
            800.0,
            (900.0, 300.0, 500.0, 100.0),
            "net radiation",
        ),
        (
            "validation.conservation.plausibility.latent_heat_flux_min",
            -30.0,
            (-60.0, -10.0, -40.0, -10.0),
            "latent heat",
        ),
        (
            "validation.conservation.plausibility.bowen_ratio_abs_max",
            5.0,
            (400.0, 300.0, 50.0, 50.0),
            "Bowen ratio",
        ),
    ],
)
def test_energy_plausibility_limits_are_read_from_their_keys(key, value, fluxes, warning):
    context = _energy_context(*fluxes)

    assert warning not in _energy_warnings(context)
    assert warning in _with_parameter(key, value, lambda: _energy_warnings(context))


def test_bowen_ratio_check_needs_no_division_guard():
    """|H| > beta_max |LE|: a sensible flux with zero latent heat is an infinite Bowen ratio."""
    assert "Bowen ratio" in _energy_warnings(_energy_context(100.0, 50.0, 0.0, 50.0))
    assert "Bowen ratio" not in _energy_warnings(_energy_context(0.0, 0.0, 0.0, 0.0))


def test_gross_primary_productivity_limit_is_read_from_its_key():
    key = "validation.conservation.plausibility.gross_primary_productivity_max"
    # 8e-4 lies below the packaged limit and above 6e-4, a limit inside the key's bounds.
    context = _carbon_context(
        gross_primary_productivity=8.0e-4,
        autotrophic_respiration=3.0e-4,
        net_primary_productivity=5.0e-4,
    )

    def warnings():
        return " ".join(
            result.message
            for result in validate_carbon_mass_conservation(context)
            if result.code == ValidationCode.MASS_BALANCE_WARN_CARBON
        )

    assert "extremely high GPP" not in warnings()
    assert "extremely high GPP" in _with_parameter(key, 6.0e-4, warnings)


def test_budget_rules_hold_python_numbers_in_binary64():
    """1 - 4e-7 against 1: float32 storage would admit the residual by its rounding term."""
    context = ValidationContext(
        tolerance_overrides={},
        metadata={
            "water_fluxes": {
                "precipitation": 1.0,
                "evaporation": 1.0 - 4.0e-7,
                "transpiration": 0.0,
                "runoff": 0.0,
                "storage_change": 0.0,
            }
        },
    )

    results = _with_parameter(_WATER_KEY, 1.0e-8, lambda: validate_water_mass_conservation(context))

    assert ValidationCode.MASS_BALANCE_FAIL_WATER in {result.code for result in results}


@pytest.mark.parametrize(
    ("rule", "label"),
    [
        (validate_water_mass_conservation, "water_fluxes"),
        (validate_carbon_mass_conservation, "carbon_fluxes"),
        (validate_energy_balance_conservation, "energy_fluxes"),
        (validate_nitrogen_mass_conservation, "nitrogen_fluxes"),
    ],
)
def test_absent_flux_dictionary_is_not_applicable(rule, label):
    """The result keeps its WARNING level and passed flag but says that nothing was checked."""
    (result,) = rule(ValidationContext(tolerance_overrides={}, metadata={}))

    assert result.code == ValidationCode.MISSING_REQUIRED_DATA
    assert "not applicable (domain absent)" in result.message
    assert label in result.message
    assert result.details[0].value == "not applicable (domain absent)"


# Boundary handlers of the conservation rules (RULES 2, 2026-09-26): a data failure after a
# verdict leaves one explicit failed result (no partial OK code can be read as success); a
# configuration fault (a missing packaged key, KeyError) is not relabelled as a budget failure
# and reaches the validated_rule boundary.
def _closed_npp_context():
    return _carbon_context(
        gross_primary_productivity=1.0e-5,
        autotrophic_respiration=4.0e-6,
        net_primary_productivity=6.0e-6,
    )


def test_conservation_rule_failure_leaves_no_partial_verdict(monkeypatch):
    from relic.validation.framework.rules import conservation

    def failing_limit(key, units):
        raise ValueError("plausibility limit units failed")

    monkeypatch.setattr(conservation, "_plausibility_limit", failing_limit)
    results = validate_carbon_mass_conservation(_closed_npp_context())

    assert [result.code for result in results] == [ValidationCode.GENERIC_VALIDATION_FAILED]
    assert results[0].passed is False
    assert "plausibility limit units failed" in results[0].message


def test_conservation_rule_does_not_relabel_a_configuration_fault(monkeypatch):
    from relic.validation.framework.rules import conservation

    def missing_limit(key, units):
        raise KeyError(f"Required parameter '{key}' not found")

    monkeypatch.setattr(conservation, "_plausibility_limit", missing_limit)
    results = validate_carbon_mass_conservation(_closed_npp_context())

    assert [result.code for result in results] == [ValidationCode.GENERIC_VALIDATION_FAILED]
    assert results[0].passed is False
    assert results[0].message.startswith(
        "Validation rule 'carbon_mass_conservation' execution failed"
    )
