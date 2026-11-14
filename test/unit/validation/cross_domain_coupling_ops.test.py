from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relic.validation import cross_domain_coupling_ops as coupling_ops

# The water and energy checks compare the relative residual |r| / sum_k |t_k| with their
# budget keys validation.conservation.water_conservation_tolerance and
# validation.conservation.energy_conservation_tolerance (validation/conservation.yaml).


# References:
#   - Best et al. (2015), J. Hydrometeorol. 16:1425-1442.
# Mathematical expression:
#   ET = T + E_soil.
# Parameter meanings:
#   ET: evapotranspiration flux [kg m-2 s-1].
#   T: transpiration flux [kg m-2 s-1].
#   E_soil: soil evaporation flux [kg m-2 s-1].
def test_biosphere_water_partition_passes_for_closed_flux_ledger() -> None:
    result = coupling_ops.check_water_conservation_impl(
        "biosphere_atmosphere",
        {},
        {
            "evapotranspiration_flux": torch.tensor(1.0),
            "transpiration_rate": torch.tensor(0.7),
            "soil_evaporation_flux": torch.tensor(0.3),
        },
    )

    assert result["passed"] is True
    assert result["checked"] is True
    assert result["residual"] == pytest.approx(0.0, abs=1.0e-6)


def test_biosphere_water_partition_rejects_unclosed_flux_ledger() -> None:
    result = coupling_ops.check_water_conservation_impl(
        "biosphere_atmosphere",
        {},
        {
            "evapotranspiration_flux": 1.0,
            "transpiration_rate": 0.7,
            "soil_evaporation_flux": 0.5,
        },
    )

    assert result["passed"] is False
    assert result["checked"] is True
    assert result["residual"] == pytest.approx(0.2)


def test_biosphere_water_partition_fails_closed_when_flux_is_missing() -> None:
    result = coupling_ops.check_water_conservation_impl(
        "biosphere_atmosphere",
        {},
        {"evapotranspiration_flux": torch.tensor(1.0)},
    )

    assert result["passed"] is False
    assert result["checked"] is False
    assert "not evaluable" in result["message"]
    assert result["missing_terms"] == ["transpiration_rate", "soil_evaporation_flux"]


# A budget with a missing term is not evaluable: passed False, no residual, the missing terms
# named; a missing term is never taken as zero (the CNP element balance reports the same way).
@pytest.mark.parametrize(
    ("coupling_name", "input_state", "output_fluxes", "missing"),
    [
        (
            "biosphere_atmosphere",
            {},
            {"evapotranspiration_flux": 1.0, "soil_evaporation_flux": 0.3},
            ["transpiration_rate"],
        ),
        (
            "soil_vegetation",
            {},
            {"root_uptake_profile": [1.0], "effective_transpiration": 1.0},
            ["soil_depth_layers"],
        ),
        (
            "soil_vegetation",
            {"soil_depth_layers": [1.0]},
            {"root_uptake_profile": [1.0]},
            ["effective_transpiration"],
        ),
    ],
)
def test_water_budget_with_a_missing_term_is_not_evaluable(
    coupling_name, input_state, output_fluxes, missing
) -> None:
    result = coupling_ops.check_water_conservation_impl(coupling_name, input_state, output_fluxes)

    assert result["passed"] is False
    assert result["checked"] is False
    assert result["residual"] is None
    assert result["relative_residual"] is None
    assert result["missing_terms"] == missing
    assert "not evaluable" in result["message"]
    assert all(name in result["message"] for name in missing)


def test_water_budget_of_an_unknown_coupling_is_not_evaluable() -> None:
    result = coupling_ops.check_water_conservation_impl("unregistered_coupling", {}, {})

    assert result["passed"] is False
    assert result["checked"] is False
    assert result["residual"] is None
    assert "not evaluable" in result["message"]
    assert "unregistered_coupling" in result["message"]


def test_energy_balance_without_soil_heat_flux_is_not_evaluable() -> None:
    """Rn = H + LE closes only if G = 0 is stated; the check never assumes it (FAO-56 Eq. 42)."""
    result = coupling_ops.check_energy_conservation_impl(
        {"net_radiation": 500.0},
        {"sensible_heat_flux": 150.0, "latent_heat_flux": 350.0},
    )

    assert result["passed"] is False
    assert result["checked"] is False
    assert result["residual"] is None
    assert result["relative_residual"] is None
    assert result["missing_terms"] == ["soil_heat_flux"]
    assert "not evaluable" in result["message"]
    assert "soil_heat_flux" in result["message"]

    stated = coupling_ops.check_energy_conservation_impl(
        {"net_radiation": 500.0, "soil_heat_flux": 0.0},
        {"sensible_heat_flux": 150.0, "latent_heat_flux": 350.0},
    )
    assert stated["passed"] is True
    assert stated["residual"] == pytest.approx(0.0)


# References:
#   - Feddes et al. (1978), Simulation of Field Water Use.
# Mathematical expression:
#   T_eff = sum_i(u_i * Delta z_i).
# Parameter meanings:
#   u_i: root uptake density in layer i [kg m-3 s-1].
#   Delta z_i: layer thickness [m].
#   T_eff: effective transpiration [kg m-2 s-1].
def test_soil_water_uptake_integrates_layer_profile() -> None:
    result = coupling_ops.check_water_conservation_impl(
        "soil_vegetation",
        {"soil_depth_layers": torch.tensor([0.25, 1.0])},
        {
            "root_uptake_profile": torch.tensor([2.0, 4.0]),
            "effective_transpiration": torch.tensor(3.5),
        },
    )

    assert result["passed"] is True
    assert result["checked"] is True
    assert result["residual"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("input_state", "output_fluxes", "message"),
    [
        (
            {},
            {
                "root_uptake_profile": [1.0],
                "effective_transpiration": 1.0,
            },
            "not evaluable",
        ),
        (
            {"soil_depth_layers": [0.25, 0.5, 1.0]},
            {
                "root_uptake_profile": [1.0],
                "effective_transpiration": 1.0,
            },
            "shape",
        ),
        (
            {"soil_depth_layers": [float("nan")]},
            {
                "root_uptake_profile": [1.0],
                "effective_transpiration": 1.0,
            },
            "finite",
        ),
    ],
)
def test_soil_water_uptake_fails_closed_for_invalid_contract(
    input_state,
    output_fluxes,
    message: str,
) -> None:
    result = coupling_ops.check_water_conservation_impl(
        "soil_vegetation",
        input_state,
        output_fluxes,
    )

    assert result["passed"] is False
    assert message.lower() in result["message"].lower()


# References:
#   - Foken (2008), Micrometeorology.
# Mathematical expression:
#   R_n = H + LE + G.
# Parameter meanings:
#   R_n: net radiation [W m-2].
#   H: sensible heat flux [W m-2].
#   LE: latent heat flux [W m-2].
#   G: ground heat flux [W m-2].
def test_energy_balance_uses_complete_surface_flux_ledger() -> None:
    passed = coupling_ops.check_energy_conservation_impl(
        {
            "net_radiation": torch.tensor(500.0),
            "soil_heat_flux": torch.tensor(50.0),
        },
        {
            "sensible_heat_flux": torch.tensor(150.0),
            "latent_heat_flux": torch.tensor(300.0),
        },
    )
    failed = coupling_ops.check_energy_conservation_impl(
        {"net_radiation": 500.0, "soil_heat_flux": 50.0},
        {"sensible_heat_flux": 100.0, "latent_heat_flux": 200.0},
    )

    assert passed["passed"] is True
    assert passed["checked"] is True
    assert failed["passed"] is False
    assert failed["residual"] == pytest.approx(150.0)


@pytest.mark.parametrize(
    ("input_state", "output_fluxes", "message"),
    [
        ({}, {"sensible_heat_flux": 1.0, "latent_heat_flux": 1.0}, "not evaluable"),
        (
            {"net_radiation": float("inf"), "soil_heat_flux": 0.0},
            {"sensible_heat_flux": 1.0, "latent_heat_flux": 1.0},
            "finite",
        ),
        (
            {"net_radiation": [1.0, 2.0], "soil_heat_flux": 0.0},
            {"sensible_heat_flux": [1.0], "latent_heat_flux": [0.0, 0.0, 0.0]},
            "shape",
        ),
    ],
)
def test_energy_balance_fails_closed_for_invalid_contract(
    input_state,
    output_fluxes,
    message: str,
) -> None:
    result = coupling_ops.check_energy_conservation_impl(
        input_state,
        output_fluxes,
    )

    assert result["passed"] is False
    assert message.lower() in result["message"].lower()


# References:
#   - Sitch et al. (2003), Global Change Biology 9:161-185.
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
        -0.1,
        1.1,
    ],
)
def test_stress_factor_validation_rejects_nonfinite_empty_or_unbounded(
    value,
) -> None:
    result = coupling_ops.check_stress_factor_bounds_impl(
        {"water_stress_factor": value}
    )

    assert result["passed"] is False
    assert result["checked"] is True
    assert result["violations"]


def test_stress_factor_validation_accepts_finite_closed_interval() -> None:
    result = coupling_ops.check_stress_factor_bounds_impl(
        {
            "water_stress_factor": [0.0, 0.5, 1.0],
            "temperature_stress_factor": torch.tensor(0.7),
        }
    )

    assert result == {
        "passed": True,
        "checked": True,
        "violations": [],
    }


def test_stress_factor_validation_marks_absent_factors_unchecked() -> None:
    result = coupling_ops.check_stress_factor_bounds_impl(
        {"unrelated_flux": torch.tensor(1.0)}
    )

    assert result == {
        "passed": True,
        "checked": False,
        "violations": [],
    }


def test_water_carbon_validation_checks_only_the_declared_coupling_outputs() -> None:
    """WaterCarbonCoupling emits dimensionless modifiers, not a carbon budget.

    The coupling reads gross_primary_productivity and autotrophic_respiration
    and writes stress factors and rate modifiers only, so a GPP - R_a - NPP
    balance cannot be evaluated from its outputs; that identity is published by
    its owner, autotrophic_respiration, as
    diagnostics.autotrophic_respiration.carbon_balance_residual.
    """
    from test.utils.functor_minimal_construction import construct_minimal

    coupling = construct_minimal("relic.coupling.internal.water_carbon:WaterCarbonCoupling")
    outputs = set(coupling.metadata.output_variables)
    assert not outputs & {"gpp", "npp", "autotrophic_respiration"}

    fluxes = {name: torch.tensor([0.5, 0.75]) for name in outputs}
    result = coupling_ops.validate_coupling_conservation_impl(
        object(),
        "water_carbon",
        {"gross_primary_productivity": torch.tensor([1.0e-7, 2.0e-7])},
        fluxes,
    )

    assert result["passed"] is True
    assert result["errors"] == []
    assert set(result["checks"]) == {"stress_factor_bounds"}
    assert result["checks"]["stress_factor_bounds"]["checked"] is True
    assert not hasattr(coupling_ops, "check_carbon_conservation_impl")


def test_coupling_aggregate_rejects_unknown_coupling() -> None:
    unknown = coupling_ops.validate_coupling_conservation_impl(
        object(),
        "unregistered_coupling",
        {},
        {},
    )

    assert unknown["passed"] is False
    assert unknown["errors"]


# References:
#   - Feddes et al. (1978), Simulation of Field Water Use and Crop Yield.
# Mathematical expression:
#   T_eff = sum_i(u_i * Delta z_i).
# Parameter meanings:
#   u_i: root uptake density [kg m-3 s-1]; Delta z_i: layer thickness [m];
#   T_eff: effective transpiration [kg m-2 s-1].
# Units: u_i [kg m-3 s-1]; Delta z_i [m]; T_eff and the residual [kg m-2 s-1].
# Domain/constraints: strictly increasing layer-bottom depths (top at 0); scalar T_eff.
# Sign convention: positive uptake and transpiration remove soil water.
def test_step_validation_reads_the_water_tolerance_key() -> None:
    """Uptake 1e-3 against transpiration 1.0001e-3 kg m-2 s-1: rho = 1e-7 / 2.0001e-3 = 5e-5."""
    input_state = {"soil_depth_layers": torch.tensor([0.5, 1.0], dtype=torch.float64)}
    output_fluxes = {
        "root_uptake_profile": torch.tensor([1.0e-3, 1.0e-3], dtype=torch.float64),
        "effective_transpiration": torch.tensor(1.0001e-3, dtype=torch.float64),
    }

    def run():
        return coupling_ops.validate_step_conservation_impl(
            object(), "soil_vegetation", input_state, output_fluxes
        )

    packaged = run()
    relaxed = _with_parameter(_WATER_KEY, 1.0e-4, run)

    assert packaged["conservation_passed"] is False
    assert relaxed["conservation_passed"] is True
    assert relaxed["detailed_results"]["checks"]["water_conservation"][
        "tolerance"
    ] == pytest.approx(1.0e-4)


def test_conservation_checks_take_no_tolerance_argument() -> None:
    import inspect

    for function, name in (
        (coupling_ops.check_water_conservation_impl, "tolerance"),
        (coupling_ops.check_energy_conservation_impl, "tolerance"),
        (coupling_ops.validate_coupling_conservation_impl, "tolerances"),
        (coupling_ops.validate_step_conservation_impl, "tolerances"),
    ):
        assert name not in inspect.signature(function).parameters, function.__name__


def test_state_manager_summary_conversion_handles_none_and_scalar_metrics() -> None:
    assert coupling_ops.from_state_manager_summary_impl(None) == {
        "status": "no_summary",
        "conservation_passed": None,
        "message": "No conservation summary available",
    }

    summary = SimpleNamespace(
        timestamp="2026-07-26T00:00:00",
        total_violations=1,
        balance_error={"carbon": torch.tensor(0.25)},
        custom_checks={
            "biosphere.water": torch.tensor(0.0),
            "unscoped_metric": torch.tensor(1.0),
        },
    )
    converted = coupling_ops.from_state_manager_summary_impl(summary)

    assert converted["status"] == "converted"
    assert converted["conservation_passed"] is False
    assert converted["element_balance_errors"] == {"carbon": pytest.approx(0.25)}
    assert converted["domain_diagnostics"] == {
        "biosphere": {"water": pytest.approx(0.0)}
    }
    assert converted["summary_metrics"] == {
        "checked_elements": ["carbon"],
        "checked_domains": ["biosphere"],
        "violation_count": 1,
    }


# References: LeVeque (1992), Numerical Methods for Conservation Laws.
# Mathematical expression: r = ET - (T + E_soil); rho = |r| / (|ET| + |T| + |E_soil|); energy
#   r = Rn - (H + LE + G), rho = |r| / (|Rn| + |H| + |LE| + |G|); pass iff rho <= budget key.
# Parameter meanings: ET, T, E_soil water fluxes; Rn, H, LE, G surface energy fluxes.
# Units: water kg m-2 s-1; energy W m-2; rho and the tolerances dimensionless.
# Domain/constraints: the same residual 1e-5 on budgets of magnitude 1e-3 and 1e3.
# Sign convention: r > 0 when the left-hand flux exceeds its partition.
from relic.parameters import get_global_manager  # noqa: E402

_WATER_KEY = "validation.conservation.water_conservation_tolerance"


def _with_parameter(key, value, run):
    manager = get_global_manager()
    original = float(manager.get(key))
    manager.set_value(key, value)
    try:
        return run()
    finally:
        manager.set_value(key, original)


@pytest.mark.parametrize(("scale", "closed"), [(1.0e-3, False), (1.0e3, True)])
def test_water_partition_verdict_compares_the_relative_residual(scale, closed) -> None:
    result = coupling_ops.check_water_conservation_impl(
        "biosphere_atmosphere",
        {},
        {
            "evapotranspiration_flux": torch.tensor(scale, dtype=torch.float64),
            "transpiration_rate": torch.tensor(0.5 * scale, dtype=torch.float64),
            "soil_evaporation_flux": torch.tensor(0.5 * scale - 1.0e-5, dtype=torch.float64),
        },
    )

    assert result["passed"] is closed
    assert result["residual"] == pytest.approx(1.0e-5, rel=1.0e-6)
    assert result["relative_residual"] == pytest.approx(1.0e-5 / (2.0 * scale - 1.0e-5))
    assert result["tolerance"] == pytest.approx(float(get_global_manager().get(_WATER_KEY)))


@pytest.mark.parametrize(("scale", "closed"), [(1.0e-3, False), (1.0e3, True)])
def test_energy_balance_verdict_compares_the_relative_residual(scale, closed) -> None:
    result = coupling_ops.check_energy_conservation_impl(
        {
            "net_radiation": torch.tensor(scale, dtype=torch.float64),
            "soil_heat_flux": torch.tensor(0.0, dtype=torch.float64),
        },
        {
            "sensible_heat_flux": torch.tensor(0.5 * scale, dtype=torch.float64),
            "latent_heat_flux": torch.tensor(0.5 * scale - 1.0e-5, dtype=torch.float64),
        },
    )

    assert result["passed"] is closed
    assert result["tolerance"] == pytest.approx(
        float(get_global_manager().get("validation.conservation.energy_conservation_tolerance"))
    )


def test_budget_closure_reads_the_tolerance_parameter_it_is_given() -> None:
    """One closure definition for every identity; the caller names its one tolerance key."""
    key = "coupling.observation.tower_flux.support_closure_tolerance"
    closure = coupling_ops.budget_closure(
        key,
        torch.tensor(1.0e-9, dtype=torch.float64),
        [torch.tensor(1.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)],
    )

    assert closure["tolerance_parameter"] == key
    assert closure["tolerance"] == pytest.approx(float(get_global_manager().get(key)))
    assert closure["relative_residual"] == pytest.approx(0.5e-9)
    assert closure["passed"] is True
    with pytest.raises(KeyError, match="water"):
        coupling_ops.budget_closure("water", 0.0, [1.0, 1.0])


def test_element_balance_does_not_compare_pools_with_fluxes() -> None:
    """A pool [g m-2] and a flux [g m-2 s-1] are different quantities; no residual is formed."""
    result = coupling_ops.check_element_balance_impl(
        "carbon",
        {"carbon_pool": torch.tensor(1000.0)},
        {"carbon_flux": torch.tensor(10.0)},
    )

    assert result["passed"] is False
    assert result["checked"] is False
    assert result["residual"] is None
    assert "not evaluable" in result["message"]
    assert result["pools"] == {"carbon_pool": 1000.0}
    assert result["fluxes"] == {"carbon_flux": 10.0}


# References: ledger_acceptance and reduction_rounding_bound
#   (relic/coupling/core/conservation_checker.py); Higham (2002), section 4.2.
# Mathematical expression: accepted iff |r| <= max(tol * a, gamma_K(u) * b), a = sum_k |t_k| the
#   tolerance scale and b the rounding activity (b = a when no rounding activity is given).
# Parameter meanings: r residual; t_k terms; b magnitude of every operand whose rounding enters r.
# Units: r, a, b in the budget unit; tol and gamma dimensionless.
# Domain/constraints: finite, non-negative rounding activity broadcastable to the residual.
# Sign convention: a, b and the accepted residual are non-negative.
def test_budget_closure_separates_the_rounding_activity() -> None:
    from relic.coupling.core.conservation_checker import reduction_rounding_bound

    residual = torch.tensor(1.0e-9, dtype=torch.float32)
    terms = [torch.tensor(1.0e-4, dtype=torch.float32), torch.tensor(1.0e-4, dtype=torch.float32)]
    gamma = reduction_rounding_bound(torch.float32, 4)

    plain = coupling_ops.budget_closure(_WATER_KEY, residual, terms)
    wide = coupling_ops.budget_closure(
        _WATER_KEY, residual, terms, rounding_activity=torch.tensor(0.2, dtype=torch.float32)
    )

    assert plain["passed"] is False
    assert wide["passed"] is True
    assert wide["relative_residual"] == pytest.approx(plain["relative_residual"])
    assert wide["accepted_residual"] == pytest.approx(gamma * 0.2, rel=1.0e-6)


def test_budget_closure_rounding_follows_the_operand_dtype() -> None:
    from relic.coupling.core.conservation_checker import reduction_rounding_bound

    def accepted(dtype):
        return _with_parameter(
            _WATER_KEY,
            1.0e-8,
            lambda: coupling_ops.budget_closure(
                _WATER_KEY,
                torch.tensor(0.0, dtype=dtype),
                [torch.tensor(1.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)],
            )["accepted_residual"],
        )

    assert accepted(torch.float32) == pytest.approx(2.0 * reduction_rounding_bound(torch.float32, 4))
    assert accepted(torch.float64) == pytest.approx(2.0e-8)


def test_cnp_stoichiometry_reads_one_tolerance_key() -> None:
    import inspect

    key = "validation.conservation.stoichiometry.ratio_tolerance"
    for function in (coupling_ops.check_cnp_stoichiometry_impl, coupling_ops.validate_cnp_cycle_impl):
        parameters = inspect.signature(function).parameters
        assert "tolerance" not in parameters and "stoich_tolerance" not in parameters
    # C:N = 25 lies outside [8, 20] * (1 +- tol): outside for tol 0.2, inside for tol 0.3.
    fluxes = {"carbon_flux": torch.tensor(25.0), "nitrogen_flux": torch.tensor(1.0)}

    assert coupling_ops.check_cnp_stoichiometry_impl({}, fluxes)["passed"] is False
    relaxed = _with_parameter(key, 0.3, lambda: coupling_ops.check_cnp_stoichiometry_impl({}, fluxes))
    assert relaxed["passed"] is True


# References: SoilVegetationCoupling.compute_root_distribution (relic/coupling/internal/
#   soil_vegetation.py): soil_depth_layers holds layer-bottom depths with the top at 0.
# Mathematical expression: Delta z_i = z_i - z_(i-1), z_0 = 0; sum_i u_i Delta z_i = T_eff.
# Parameter meanings: z_i bottom depth of layer i; u_i uptake density; T_eff transpiration.
# Units: z_i and Delta z_i [m]; u_i [kg m-3 s-1]; T_eff [kg m-2 s-1].
# Domain/constraints: the state and outputs of the producing coupling itself.
# Sign convention: positive uptake removes soil water.
def test_soil_water_check_reads_the_coupling_state_semantics() -> None:
    from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling

    state = {
        "soil_moisture_profile": torch.tensor([0.28, 0.25, 0.22]),
        "soil_matric_potential": torch.tensor([-5000.0, -6000.0, -7000.0]),
        "potential_transpiration": torch.tensor(3.0e-6),
        "soil_depth_layers": torch.tensor([0.1, 0.3, 0.6]),
    }
    outputs = SoilVegetationCoupling(name="soil_vegetation", config={}).forward(
        state, time=0.0, dt=3600.0
    )

    result = coupling_ops.check_water_conservation_impl(
        "soil_vegetation", state, {name: item.value for name, item in outputs.items()}
    )

    assert result["checked"] is True
    assert result["passed"] is True, result["message"]
