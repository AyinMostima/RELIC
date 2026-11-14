"""
Coupling and CNP validation helpers for CrossDomainValidator.

This module extracts large conservation and stoichiometry routines from
``cross_domain_validator.py`` to keep the validator focused on orchestration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from relic.coupling.core.conservation_checker import (
    coarsest_floating_dtype,
    ledger_acceptance,
    require_relative_tolerance,
)
from relic.parameters import get_global_manager


def _as_finite_tensor(value: Any, name: str) -> torch.Tensor:
    """Convert a validation input to a finite, non-empty floating tensor."""
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be tensor-convertible") from exc
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


WATER_TOLERANCE_PARAMETER = "validation.conservation.water_conservation_tolerance"
ENERGY_TOLERANCE_PARAMETER = "validation.conservation.energy_conservation_tolerance"


def closure_tolerance(tolerance_parameter: str) -> float:
    """The dimensionless relative closure tolerance stored under one parameter key."""
    value = get_global_manager().get(
        tolerance_parameter, units="dimensionless", required=True
    )
    return require_relative_tolerance(float(value), tolerance_parameter)


def budget_closure(
    tolerance_parameter: str,
    residual: Any,
    terms: Sequence[Any],
    *,
    summands: Optional[int] = None,
    rounding_activity: Any = None,
) -> Dict[str, Any]:
    """Judge one budget identity by its relative residual under the shared ledger rule.

    This is the one relative-residual definition of the validation rules, the coupling
    contracts, the conservation history rule and the G3 compile witnesses; each caller names
    the one tolerance key of its identity. ``rounding_activity`` is the magnitude of every
    operand whose rounding enters the residual when that exceeds the gross activity of the
    terms (a storage rate formed from two stored amounts); it widens only the rounding term.
    """
    # References: LeVeque, R.J. (1992), Numerical Methods for Conservation Laws; Higham, N.J.
    #   (2002), Accuracy and Stability of Numerical Algorithms, section 4.2; ledger_acceptance
    #   (relic/coupling/core/conservation_checker.py), the one acceptance rule of every ledger.
    # Mathematical expression: e_i = |r_i|; a_i = sum_k |t_k,i|; b_i = a_i or the rounding
    #   activity; closed iff e_i <= max(tol * a_i, gamma_2K(u) * b_i) for every element i
    #   (ledger_acceptance with a_i as relative scale and b_i as activity); rho = max_i e_i / a_i,
    #   0 where a_i = 0.
    # Parameter meanings: r residual of the budget identity; t_k its K terms (summands K when a
    #   term is itself a sum); b_i the magnitude of all operands whose rounding enters r_i;
    #   tol the value of tolerance_parameter (for a budget,
    #   validation.conservation.<budget>_conservation_tolerance); u unit roundoff of the
    #   coarsest floating dtype of r and the terms (the operands actually rounded).
    # Units: r, t_k, a_i and b_i in the budget's own unit (kg m-2 s-1, g C m-2 s-1, W m-2, ...);
    #   rho, tol and gamma dimensionless, so the verdict does not depend on the unit or magnitude.
    # Domain/constraints: finite r and terms (else ValueError); tol in (0, 1); b_i finite and
    #   b_i >= a_i (else ValueError); no floor: by the triangle inequality |r_i| <= a_i, so
    #   a_i = 0 implies r_i = 0, an exactly closed budget.
    # Sign convention: r follows the caller's identity; e_i, a_i, b_i and rho are non-negative.
    tolerance = closure_tolerance(tolerance_parameter)
    tensors = torch.broadcast_tensors(
        *(torch.as_tensor(value) for value in (residual, *terms))
    )
    for tensor in tensors:
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(f"The budget terms judged by {tolerance_parameter} must be finite")
    rounding_dtype = coarsest_floating_dtype(*(tensor.dtype for tensor in tensors))
    values = [tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1) for tensor in tensors]
    absolute_residual = torch.abs(values[0])
    activity = torch.zeros_like(absolute_residual)
    for term in values[1:]:
        activity = activity + torch.abs(term)
    ratio = torch.where(
        activity > 0.0,
        absolute_residual / torch.where(activity > 0.0, activity, torch.ones_like(activity)),
        torch.zeros_like(activity),
    )
    worst = int(torch.argmax(ratio).item())
    term_count = 2 * (summands if summands is not None else len(terms))
    if rounding_activity is None:
        decision = ledger_acceptance(
            absolute_error=float(absolute_residual[worst].item()),
            relative_scale=float(activity[worst].item()),
            activity=float(activity[worst].item()),
            tolerance=tolerance,
            rounding_dtype=rounding_dtype,
            term_count=term_count,
        )
    else:
        rounding = torch.broadcast_to(
            torch.as_tensor(rounding_activity).detach().to(device="cpu", dtype=torch.float64),
            tensors[0].shape,
        ).reshape(-1)
        if not bool(torch.all(torch.isfinite(rounding))) or bool(torch.any(rounding < activity)):
            raise ValueError(
                f"The rounding activity of {tolerance_parameter} must be finite and at least "
                "the gross activity of the terms"
            )
        # Each element has its own rounding activity, so every element is judged by the rule.
        decisions = [
            ledger_acceptance(
                absolute_error=float(absolute_residual[index].item()),
                relative_scale=float(activity[index].item()),
                activity=float(rounding[index].item()),
                tolerance=tolerance,
                rounding_dtype=rounding_dtype,
                term_count=term_count,
            )
            for index in range(absolute_residual.numel())
        ]
        failing = [index for index, item in enumerate(decisions) if not item["accepted"]]
        decision = decisions[failing[0] if failing else worst]
    return {
        "passed": decision["accepted"],
        "residual": float(torch.max(absolute_residual).item()),
        "relative_residual": float(ratio[worst].item()),
        "tolerance": tolerance,
        "tolerance_parameter": tolerance_parameter,
        "accepted_residual": decision["accepted_residual"],
    }


def _budget_result(tolerance_parameter: str) -> Dict[str, Any]:
    """The result record of one coupling budget check before its terms are read."""
    return {
        "passed": False,
        "checked": False,
        "message": "",
        "residual": None,
        "relative_residual": None,
        "tolerance": closure_tolerance(tolerance_parameter),
        "tolerance_parameter": tolerance_parameter,
        "missing_terms": [],
    }


def _not_evaluable(result: Dict[str, Any], budget: str, missing: Sequence[str]) -> Dict[str, Any]:
    """Mark a budget without all of its terms as not evaluable (no term is taken as zero)."""
    result["missing_terms"] = list(missing)
    result["message"] = (
        f"{budget} is not evaluable: missing budget terms {', '.join(missing)}; "
        "a missing term is not taken as zero"
    )
    return result


def validate_coupling_conservation_impl(
    validator: Any,
    coupling_name: str,
    input_state: Dict,
    output_fluxes: Dict,
) -> Dict[str, Any]:
    """
    Validate conservation laws for internal coupling modules.
    """
    diagnostics = {
        "coupling_name": coupling_name,
        "passed": True,
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    supported_couplings = {
        "biosphere_atmosphere",
        "soil_vegetation",
        "water_carbon",
    }
    if coupling_name not in supported_couplings:
        diagnostics["passed"] = False
        diagnostics["errors"].append(f"Unknown coupling: {coupling_name}")

    if coupling_name in ["biosphere_atmosphere", "soil_vegetation"]:
        water_check = check_water_conservation_impl(
            coupling_name,
            input_state,
            output_fluxes,
        )
        diagnostics["checks"]["water_conservation"] = water_check
        if not water_check["passed"]:
            diagnostics["passed"] = False
            diagnostics["errors"].append(water_check["message"])

    if coupling_name == "biosphere_atmosphere":
        energy_check = check_energy_conservation_impl(
            input_state,
            output_fluxes,
        )
        diagnostics["checks"]["energy_conservation"] = energy_check
        if not energy_check["passed"]:
            diagnostics["passed"] = False
            diagnostics["errors"].append(energy_check["message"])

    # water_carbon has no carbon-budget check: WaterCarbonCoupling outputs only
    # dimensionless stress factors and rate modifiers, so a GPP - R_a - NPP
    # balance cannot be evaluated from them. The identity is published by its
    # owner, autotrophic_respiration, as
    # diagnostics.autotrophic_respiration.carbon_balance_residual; only the
    # stress-factor bounds below apply to this coupling.

    stress_check = check_stress_factor_bounds_impl(output_fluxes)
    diagnostics["checks"]["stress_factor_bounds"] = stress_check
    if not stress_check["passed"]:
        diagnostics["passed"] = False
        diagnostics["warnings"].extend(stress_check["violations"])

    return diagnostics


def check_water_conservation_impl(
    coupling_name: str,
    input_state: Dict,
    output_fluxes: Dict,
) -> Dict[str, Any]:
    """Check water mass conservation against validation.conservation.water_conservation_tolerance.

    A missing budget term, or a coupling without a water budget, makes the check not
    evaluable: passed False, no residual, and the missing terms named in the message.
    """
    result = _budget_result(WATER_TOLERANCE_PARAMETER)

    if coupling_name == "biosphere_atmosphere":
        evaporation_key = next(
            (
                name
                for name in ("soil_evaporation_flux", "soil_evaporation")
                if name in output_fluxes
            ),
            None,
        )
        required = ("evapotranspiration_flux", "transpiration_rate")
        missing = [name for name in required if name not in output_fluxes]
        if evaporation_key is None:
            missing.append("soil_evaporation_flux")
        if missing:
            return _not_evaluable(result, "Biosphere water balance", missing)

        result["checked"] = True
        try:
            et = _as_finite_tensor(
                output_fluxes["evapotranspiration_flux"],
                "evapotranspiration_flux",
            )
            transp = _as_finite_tensor(
                output_fluxes["transpiration_rate"],
                "transpiration_rate",
            )
            evaporation = _as_finite_tensor(
                output_fluxes[evaporation_key],
                evaporation_key,
            )
            et, transp, evaporation = torch.broadcast_tensors(
                et,
                transp,
                evaporation,
            )
        except ValueError as exc:
            result["message"] = str(exc)
            return result
        except RuntimeError:
            result["message"] = (
                "Biosphere water flux shape mismatch for ET, transpiration, "
                "and soil evaporation"
            )
            return result

        # References:
        #   Best et al. (2015), J. Hydrometeorol. 16:1425-1442; budget_closure (this module).
        # Mathematical expression:
        #   r = ET - (T + E_soil); rho = max |r| / (|ET| + |T| + |E_soil|) <= tol_water.
        # Parameter meanings:
        #   ET, T, E_soil: evapotranspiration, transpiration, soil evaporation;
        #   tol_water = validation.conservation.water_conservation_tolerance.
        # Units: ET, T, E_soil and r [kg m-2 s-1]; rho and tol_water dimensionless.
        # Domain/constraints: all three fluxes supplied (else not evaluable), finite and
        #   broadcastable.
        # Sign convention: r > 0 when ET exceeds its partition.
        closure = budget_closure(
            WATER_TOLERANCE_PARAMETER, et - (transp + evaporation), [et, transp, evaporation]
        )
        result["residual"] = closure["residual"]
        result["relative_residual"] = closure["relative_residual"]
        result["passed"] = closure["passed"]
        if not closure["passed"]:
            result["message"] = (
                "Biosphere water balance violated: relative residual "
                f"{closure['relative_residual']:.6e} > {closure['tolerance']:.6e} "
                f"({WATER_TOLERANCE_PARAMETER}); "
                f"max residual={closure['residual']:.6e} kg m-2 s-1"
            )

    elif coupling_name == "soil_vegetation":
        missing = []
        if "soil_depth_layers" not in input_state:
            missing.append("soil_depth_layers")
        for name in ("root_uptake_profile", "effective_transpiration"):
            if name not in output_fluxes:
                missing.append(name)
        if missing:
            return _not_evaluable(result, "Soil water balance", missing)

        result["checked"] = True
        try:
            depths = _as_finite_tensor(
                input_state["soil_depth_layers"],
                "soil_depth_layers",
            )
            uptake = _as_finite_tensor(
                output_fluxes["root_uptake_profile"],
                "root_uptake_profile",
            )
            eff_transp = _as_finite_tensor(
                output_fluxes["effective_transpiration"],
                "effective_transpiration",
            )
        except ValueError as exc:
            result["message"] = str(exc)
            return result

        # soil_depth_layers holds the bottom depth of each layer with the top at 0, as
        # SoilVegetationCoupling reads it (compute_root_distribution, forward).
        if (
            depths.ndim != 1
            or uptake.ndim != 1
            or depths.numel() != uptake.numel()
            or eff_transp.numel() != 1
        ):
            result["message"] = (
                "Soil water shape contract requires one layer-bottom depth per "
                "uptake layer and scalar effective transpiration"
            )
            return result

        layer_thickness = torch.diff(depths, prepend=depths.new_zeros(1))
        if bool(torch.any(layer_thickness <= 0.0)):
            result["message"] = (
                "soil_depth_layers must be strictly increasing positive layer-bottom depths"
            )
            return result

        # References:
        #   Feddes et al. (1978), Simulation of Field Water Use; budget_closure (this module).
        # Mathematical expression:
        #   r = sum_i(u_i * Delta z_i) - T_eff;
        #   rho = |r| / (sum_i |u_i * Delta z_i| + |T_eff|) <= tol_water.
        # Parameter meanings:
        #   u_i: root uptake density; Delta z_i = z_i - z_(i-1) the thickness of layer i from
        #   the layer-bottom depths z_i (z_0 = 0); T_eff: effective transpiration;
        #   tol_water = validation.conservation.water_conservation_tolerance.
        # Units: u_i [kg m-3 s-1]; z_i and Delta z_i [m]; T_eff and r [kg m-2 s-1]; rho
        #   dimensionless.
        # Domain/constraints: all three terms supplied (else not evaluable); one strictly
        #   increasing positive bottom depth per uptake layer; finite values.
        # Sign convention: positive uptake and transpiration remove soil water; r > 0 when the
        #   layered uptake exceeds the transpiration.
        layer_uptake = uptake * layer_thickness
        total_uptake = torch.sum(layer_uptake)
        closure = budget_closure(
            WATER_TOLERANCE_PARAMETER,
            total_uptake - eff_transp.reshape(()),
            [torch.sum(torch.abs(layer_uptake)), eff_transp.reshape(())],
            summands=layer_uptake.numel() + 1,
        )
        result["residual"] = closure["residual"]
        result["relative_residual"] = closure["relative_residual"]
        result["passed"] = closure["passed"]
        if not closure["passed"]:
            result["message"] = (
                "Soil water conservation violated: "
                f"uptake={total_uptake.detach().cpu().item():.6e}, "
                f"transpiration={eff_transp.detach().cpu().item():.6e}"
            )

    else:
        result["message"] = (
            f"Water balance is not evaluable for coupling {coupling_name}: "
            "no water budget is defined for it"
        )

    return result


def check_energy_conservation_impl(
    input_state: Dict,
    output_fluxes: Dict,
) -> Dict[str, Any]:
    """Check the surface energy balance against validation.conservation.energy_conservation_tolerance.

    Rn, H, LE and the ground heat flux G are all budget terms. A missing term makes the check
    not evaluable; G = 0 (for example FAO-56 Eq. 42 for day-long steps) is a physical
    assumption the caller states by supplying soil_heat_flux = 0.
    """
    result = _budget_result(ENERGY_TOLERANCE_PARAMETER)

    missing = [
        name for name in ("net_radiation", "soil_heat_flux") if name not in input_state
    ]
    missing.extend(
        name for name in ("sensible_heat_flux", "latent_heat_flux") if name not in output_fluxes
    )
    if missing:
        return _not_evaluable(result, "Energy balance", missing)

    result["checked"] = True
    try:
        rn = _as_finite_tensor(input_state["net_radiation"], "net_radiation")
        h = _as_finite_tensor(
            output_fluxes["sensible_heat_flux"],
            "sensible_heat_flux",
        )
        le = _as_finite_tensor(
            output_fluxes["latent_heat_flux"],
            "latent_heat_flux",
        )
        g = _as_finite_tensor(input_state["soil_heat_flux"], "soil_heat_flux")
        rn, h, le, g = torch.broadcast_tensors(rn, h, le, g)
    except ValueError as exc:
        result["message"] = str(exc)
        return result
    except RuntimeError:
        result["message"] = (
            "Energy balance shape mismatch for net radiation and surface fluxes"
        )
        return result

    # References:
    #   Foken (2008), Micrometeorology; Allen et al. (1998), FAO-56, Eq. 42 (G ~ 0 for
    #   day-long periods, supplied by the caller); budget_closure (this module).
    # Mathematical expression:
    #   r = R_n - (H + LE + G); rho = max |r| / (|R_n| + |H| + |LE| + |G|) <= tol_energy.
    # Parameter meanings:
    #   R_n: net radiation; H: sensible heat flux; LE: latent heat flux; G: ground heat flux;
    #   tol_energy = validation.conservation.energy_conservation_tolerance.
    # Units: R_n, H, LE, G and r [W m-2]; rho and tol_energy dimensionless.
    # Domain/constraints: all four terms supplied (else not evaluable), finite and
    #   broadcastable.
    # Sign convention: R_n positive downward; H, LE and G positive away from the surface.
    closure = budget_closure(ENERGY_TOLERANCE_PARAMETER, rn - (h + le + g), [rn, h, le, g])
    result["residual"] = closure["residual"]
    result["relative_residual"] = closure["relative_residual"]
    result["passed"] = closure["passed"]
    if not closure["passed"]:
        result["message"] = (
            "Energy balance violated: relative residual "
            f"{closure['relative_residual']:.6e} > {closure['tolerance']:.6e} "
            f"({ENERGY_TOLERANCE_PARAMETER}); "
            f"max residual={closure['residual']:.6e} W m-2"
        )

    return result


def check_stress_factor_bounds_impl(output_fluxes: Dict) -> Dict[str, Any]:
    """Check that stress factors are within [0, 1]."""
    result = {"passed": True, "checked": False, "violations": []}

    stress_factor_keys = [
        "water_stress_factor",
        "temperature_stress_factor",
        "light_stress_factor",
        "nutrient_stress_factor",
        "gpp_water_stress_factor",
        "gpp_vpd_stress_factor",
        "autotrophic_respiration_modifier",
        "heterotrophic_respiration_modifier",
        "overall_water_stress",
        "soil_respiration_moisture_factor",
    ]

    for key in stress_factor_keys:
        if key in output_fluxes:
            result["checked"] = True
            try:
                value = _as_finite_tensor(output_fluxes[key], key)
            except ValueError as exc:
                result["passed"] = False
                result["violations"].append(str(exc))
                continue

            # References:
            #   Sitch et al. (2003), Global Change Biology 9:161-185.
            # Mathematical expression:
            #   0 <= f_stress <= 1.
            # Parameter meanings:
            #   f_stress: dimensionless environmental limitation factor [-].
            if bool(torch.any(value < 0.0)) or bool(torch.any(value > 1.0)):
                result["passed"] = False
                result["violations"].append(
                    f"{key} out of bounds: "
                    f"min={value.min().detach().cpu().item():.4f}, "
                    f"max={value.max().detach().cpu().item():.4f}"
                )

    return result


def validate_step_conservation_impl(
    validator: Any,
    coupling_name: str,
    input_state: Dict,
    output_fluxes: Dict,
) -> Dict[str, Any]:
    """Unified step-level conservation validation for simulator integration."""
    coupling_result = validate_coupling_conservation_impl(
        validator,
        coupling_name,
        input_state,
        output_fluxes,
    )
    return {
        "timestamp": getattr(input_state, "timestamp", None),
        "coupling_name": coupling_name,
        "conservation_passed": coupling_result["passed"],
        "checks_performed": list(coupling_result["checks"].keys()),
        "violations_found": len(coupling_result["errors"]),
        "warnings_found": len(coupling_result["warnings"]),
        "detailed_results": coupling_result,
    }


def from_state_manager_summary_impl(summary: Any) -> Dict[str, Any]:
    """Convert StateManager ConservationSummary to validator report format."""
    if summary is None:
        return {
            "status": "no_summary",
            "conservation_passed": None,
            "message": "No conservation summary available",
        }

    total_violations = getattr(summary, "total_violations", 0)
    balance_errors = getattr(summary, "balance_error", {})
    custom_checks = getattr(summary, "custom_checks", {})
    conservation_passed = total_violations == 0

    domain_results: Dict[str, Dict[str, Any]] = {}
    for check_name, check_value in custom_checks.items():
        if "." in check_name:
            domain, metric = check_name.split(".", 1)
            if domain not in domain_results:
                domain_results[domain] = {}
            domain_results[domain][metric] = (
                float(check_value) if hasattr(check_value, "item") else check_value
            )

    return {
        "status": "converted",
        "timestamp": getattr(summary, "timestamp", None),
        "conservation_passed": conservation_passed,
        "total_violations": total_violations,
        "element_balance_errors": {
            element: float(error) if hasattr(error, "item") else error
            for element, error in balance_errors.items()
        },
        "domain_diagnostics": domain_results,
        "summary_metrics": {
            "checked_elements": list(balance_errors.keys()),
            "checked_domains": list(domain_results.keys()),
            "violation_count": total_violations,
        },
    }


def validate_cnp_cycle_impl(
    validator: Any,
    input_state: Dict,
    output_fluxes: Dict,
    check_stoichiometry: bool = True,
) -> Dict[str, Any]:
    """Validate coupled C/N/P conservation and stoichiometry."""
    results = {
        "passed": True,
        "carbon_balance": {},
        "nitrogen_balance": {},
        "phosphorus_balance": {},
        "stoichiometry": {},
        "cross_domain_fluxes": {},
        "violations": [],
        "warnings": [],
    }

    for element in ["carbon", "nitrogen", "phosphorus"]:
        element_result = check_element_balance_impl(element, input_state, output_fluxes)
        results[f"{element}_balance"] = element_result
        if not element_result.get("passed", False):
            results["passed"] = False
            results["violations"].append(
                f"{element.capitalize()} balance violation: "
                f"{element_result.get('message', 'unknown')}"
            )

    if check_stoichiometry:
        stoich_result = check_cnp_stoichiometry_impl(input_state, output_fluxes)
        results["stoichiometry"] = stoich_result
        if not stoich_result.get("passed", True):
            results["warnings"].extend(stoich_result.get("warnings", []))

    flux_result = check_cross_domain_flux_consistency_impl(output_fluxes)
    results["cross_domain_fluxes"] = flux_result
    if not flux_result.get("passed", True):
        results["warnings"].extend(flux_result.get("warnings", []))

    return results


def check_element_balance_impl(
    element: str,
    input_state: Dict,
    output_fluxes: Dict,
) -> Dict[str, Any]:
    """Report the pools and fluxes of one element; no balance is formed from them.

    A pool (g m-2) and a flux (g m-2 s-1) are different quantities, so the difference of their
    sums is no budget residual. An element budget needs its storage change over the step,
    which these inputs do not carry; the result is therefore not evaluable and not passed.
    """
    result = {
        "passed": False,
        "checked": False,
        "message": (
            f"{element.capitalize()} balance is not evaluable: the inputs carry pools and fluxes "
            "but no storage change, and a pool cannot be compared with a flux"
        ),
        "residual": None,
        "pools": {},
        "fluxes": {},
    }
    for key, value in input_state.items():
        if element in key.lower() and "pool" in key.lower():
            result["pools"][key] = float(value.item() if isinstance(value, torch.Tensor) else value)
    for key, value in output_fluxes.items():
        if element in key.lower() and any(
            term in key.lower() for term in ["flux", "rate", "emission"]
        ):
            result["fluxes"][key] = float(value.item() if isinstance(value, torch.Tensor) else value)
    return result


STOICHIOMETRY_RATIO_TOLERANCE = "validation.conservation.stoichiometry.ratio_tolerance"
STOICHIOMETRY_RANGES = {
    "C:N": "validation.conservation.stoichiometry.cn_ratio_range",
    "C:P": "validation.conservation.stoichiometry.cp_ratio_range",
    "N:P": "validation.conservation.stoichiometry.np_ratio_range",
}


def check_cnp_stoichiometry_impl(
    input_state: Dict,
    output_fluxes: Dict,
) -> Dict[str, Any]:
    """Check C:N:P ratios of the summed output fluxes against their plausible ranges.

    The ranges and their relative widening are read from validation/conservation.yaml; there
    is no per-call tolerance.
    """
    manager = get_global_manager()
    tolerance = float(
        manager.get(STOICHIOMETRY_RATIO_TOLERANCE, units="dimensionless", required=True)
    )
    ranges = {
        name: [float(bound) for bound in manager.get(key, required=True)]
        for name, key in STOICHIOMETRY_RANGES.items()
    }
    result = {"passed": True, "warnings": [], "ratios": {}}
    c_flux = 0.0
    n_flux = 0.0
    p_flux = 0.0

    for key, value in output_fluxes.items():
        val = float(value.item() if isinstance(value, torch.Tensor) else value)
        key_lower = key.lower()
        if "carbon" in key_lower or key_lower.startswith("c_"):
            c_flux += abs(val)
        elif "nitrogen" in key_lower or key_lower.startswith("n_"):
            n_flux += abs(val)
        elif "phosphorus" in key_lower or key_lower.startswith("p_"):
            p_flux += abs(val)

    # References: validation/conservation.yaml entries
    #   validation.conservation.stoichiometry.{ratio_tolerance, cn_ratio_range, cp_ratio_range,
    #   np_ratio_range} (model plausibility limits; the N:P range contains the Redfield 16).
    # Mathematical expression: ratio X:Y = sum |F_X| / sum |F_Y|, defined when sum |F_Y| > 0;
    #   warn iff X:Y < low * (1 - tol) or X:Y > high * (1 + tol).
    # Parameter meanings: F_X the output fluxes whose names mark element X; [low, high] the
    #   plausible range; tol its relative widening.
    # Units: fluxes in one mass-flux unit; ratios, ranges and tol dimensionless.
    # Domain/constraints: plausibility warnings only; a ratio with a zero denominator is not
    #   formed.
    # Sign convention: magnitudes only.
    for name, numerator, denominator in (
        ("C:N", c_flux, n_flux),
        ("C:P", c_flux, p_flux),
        ("N:P", n_flux, p_flux),
    ):
        if denominator <= 0.0 or (name == "N:P" and n_flux <= 0.0):
            continue
        ratio = numerator / denominator
        low, high = ranges[name]
        result["ratios"][name] = ratio
        if ratio < low * (1.0 - tolerance) or ratio > high * (1.0 + tolerance):
            result["passed"] = False
            result["warnings"].append(
                f"{name} ratio {ratio:.2f} outside the plausible range [{low:g}, {high:g}] "
                f"widened by {tolerance:g} ({STOICHIOMETRY_RANGES[name]})"
            )

    return result


def check_cross_domain_flux_consistency_impl(output_fluxes: Dict) -> Dict[str, Any]:
    """Check consistency of fluxes across domain boundaries."""
    result = {"passed": True, "warnings": [], "comparisons": {}}
    bio_c_flux = 0.0
    atm_c_flux = 0.0

    for key, value in output_fluxes.items():
        val = float(value.item() if isinstance(value, torch.Tensor) else value)
        key_lower = key.lower()
        if "biosphere" in key_lower and "carbon" in key_lower:
            bio_c_flux += val
        elif "atmosphere" in key_lower and "carbon" in key_lower:
            atm_c_flux += val

    if abs(bio_c_flux) > 1e-12 or abs(atm_c_flux) > 1e-12:
        residual = abs(bio_c_flux + atm_c_flux)
        result["comparisons"]["biosphere_atmosphere_carbon"] = {
            "biosphere": bio_c_flux,
            "atmosphere": atm_c_flux,
            "residual": residual,
        }
        if residual > 1e-6:
            result["passed"] = False
            result["warnings"].append(
                "Biosphere-atmosphere carbon flux inconsistency: "
                f"bio={bio_c_flux:.6e}, atm={atm_c_flux:.6e}, residual={residual:.6e}"
            )

    return result
