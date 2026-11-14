"""
Conservation Law Validation Rules

This module contains validation rules for mass and energy conservation laws
across different domains and processes in RELIC. These rules ensure
physical consistency and thermodynamic principles are maintained.

Key Conservation Laws:
    - Water mass conservation
    - Carbon mass conservation  
    - Nitrogen mass conservation
    - Energy balance conservation
    - Momentum conservation (where applicable)

Mathematical Framework:
    Conservation principle: partialS/partialt + nabla*F = P - L
    where S is stored quantity, F is flux, P is production, L is loss
    
    For steady state: nabla*F = P - L (production balances loss)
    For water: P - E - T - R - DeltaS = 0 (precipitation, evaporation, transpiration, runoff, storage)
    For carbon: GPP - Ra - Rh - DeltaC = 0 (gross primary productivity, autotrophic/heterotrophic respiration, storage)
    For energy: Rn - H - LE - G = 0 (net radiation, sensible heat, latent heat, ground heat flux)

References:
    - Thermodynamic principles
    - Ecosystem energy balance theory
    - Biogeochemical cycling principles
    - Conservation of mass and energy laws
"""

from typing import Dict, List, Optional, Any, Tuple
import logging

import torch
from torch import Tensor

from ..models import (
    ValidationResultModel, ValidationLevel, ValidationCode,
    ValidationContext, ValidationResults, ValidationDetail
)
from ..decorators import validated_rule
from ....parameters import get_global_manager
from ...cross_domain_coupling_ops import budget_closure

logger = logging.getLogger(__name__)


def _to_tensor(x: Any) -> Tensor:
    """A budget term as a tensor: tensors keep their dtype, Python numbers stay binary64.

    Python floats are binary64; holding them in float32 would round every term and let the
    rounding allowance of ledger_acceptance admit float32 rounding the inputs never had.
    """
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.float64)
    return torch.tensor([x], dtype=torch.float64)


_WATER_TOLERANCE = "validation.conservation.water_conservation_tolerance"
_CARBON_TOLERANCE = "validation.conservation.carbon_conservation_tolerance"
_ENERGY_TOLERANCE = "validation.conservation.energy_conservation_tolerance"
_NITROGEN_TOLERANCE = "validation.conservation.nitrogen_conservation_tolerance"

_WATER_TERMS = ("precipitation", "evaporation", "transpiration", "runoff", "storage_change")
_ENERGY_TERMS = ("net_radiation", "sensible_heat_flux", "latent_heat_flux", "ground_heat_flux")
# The nitrogen budget is the total soil mineral N, N_min = N_NH4 + N_NO3 (CASA-CNP mineral N
# pool; RELIC nitrogen_cycling theory). Nitrification moves N from NH4 to NO3 inside N_min and
# is therefore no term of it; net mineralization is the organic-to-mineral input.
_NITROGEN_INPUTS = ("deposition", "fixation", "fertilizer", "net_mineralization")
_NITROGEN_OUTPUTS = ("denitrification", "leaching", "volatilization", "plant_uptake")
_NITROGEN_TERMS = _NITROGEN_INPUTS + _NITROGEN_OUTPUTS + ("storage_change",)

_GPP_MAX = "validation.conservation.plausibility.gross_primary_productivity_max"
_NET_RADIATION_ABS_MAX = "validation.conservation.plausibility.net_radiation_abs_max"
_LATENT_HEAT_FLUX_MIN = "validation.conservation.plausibility.latent_heat_flux_min"
_BOWEN_RATIO_ABS_MAX = "validation.conservation.plausibility.bowen_ratio_abs_max"

NOT_APPLICABLE = "not applicable (domain absent)"

# Each carbon identity is selected by the one term that belongs to it alone (its defining
# term); all of its other terms are then required.
_CARBON_IDENTITIES = (
    (
        "NPP_balance",
        "NPP = GPP - Ra",
        "net_primary_productivity",
        ("gross_primary_productivity", "autotrophic_respiration"),
    ),
    (
        "NEE_balance",
        "NEE = Ra + Rh - GPP",
        "net_ecosystem_exchange",
        ("autotrophic_respiration", "heterotrophic_respiration", "gross_primary_productivity"),
    ),
    (
        "total_carbon_balance",
        "GPP - Ra - Rh - DeltaC = 0",
        "storage_change",
        ("gross_primary_productivity", "autotrophic_respiration", "heterotrophic_respiration"),
    ),
)


def _plausibility_limit(key: str, units: str) -> float:
    """One plausibility limit of validation/conservation.yaml in the units the rule uses."""
    return float(get_global_manager().get(key, units=units, required=True))


def domain_absent_result(label: str, scope: str, tags: List[str], data_key: str) -> ValidationResultModel:
    """The context carries no data of this budget: the rule does not apply and checks nothing.

    The level and passed flag stay as before (WARNING, passed), and the message and detail say
    that no check was performed.
    """
    return ValidationResultModel(
        level=ValidationLevel.WARNING,
        code=ValidationCode.MISSING_REQUIRED_DATA,
        message=f"{label}: {NOT_APPLICABLE}; the context carries no {data_key}, no check was performed",
        layer="L2",
        scope=scope,
        tags=[*tags, "not_applicable"],
        details=[ValidationDetail(
            variable_name=data_key,
            value=NOT_APPLICABLE,
            metadata={"applicability": NOT_APPLICABLE},
        )],
    )


def _refuse_tolerance_override(
    context: ValidationContext, override_key: str, tolerance_parameter: str
) -> None:
    """A budget tolerance has one source, its budget key in validation/conservation.yaml."""
    overrides = context.tolerance_overrides or {}
    if override_key in overrides:
        raise ValueError(
            f"{override_key} is not accepted: the budget tolerance is read only from "
            f"{tolerance_parameter}"
        )


def _not_evaluable_result(
    budget: str,
    scope: str,
    tags: List[str],
    variable_name: str,
    missing: List[str],
    reason: Optional[str] = None,
) -> ValidationResultModel:
    """A budget without all of its terms is not evaluable; no term is taken as zero."""
    explanation = reason or (
        f"missing budget terms {', '.join(missing)}; a missing term is not taken as zero"
    )
    return ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.MISSING_REQUIRED_DATA,
        message=f"{budget} is not evaluable: {explanation}",
        passed=False,
        layer="L2",
        scope=scope,
        tags=[*tags, "not_evaluable"],
        details=[ValidationDetail(
            variable_name=variable_name,
            value=None,
            units="dimensionless",
            metadata={"missing_terms": list(missing)},
        )],
    )


def _closure_verdict(
    closure: Dict[str, Any],
    ok_code: ValidationCode,
    fail_code: ValidationCode,
) -> Tuple[ValidationLevel, ValidationCode]:
    """INFO with the OK code for a closed budget, ERROR with the failure code otherwise."""
    if closure["passed"]:
        return ValidationLevel.INFO, ok_code
    return ValidationLevel.ERROR, fail_code


def _closure_message(label: str, closure: Dict[str, Any], units: str) -> str:
    verdict = "satisfied" if closure["passed"] else "violation"
    return (
        f"{label} {verdict}: relative residual {closure['relative_residual']:.2e} against "
        f"{closure['tolerance']:.2e} ({closure['tolerance_parameter']}); "
        f"max residual {closure['residual']:.2e} {units}"
    )


def evaluate_element_mass_balance(
    inputs: Dict[str, Tuple[Tensor, str]],
    outputs: Dict[str, Tuple[Tensor, str]],
    storage_change: Optional[Tuple[Tensor, str]] = None,
    element: str = "water",
    unit_converter: Optional[Any] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate elemental mass balance from flux dictionaries.

    Parameters
    ----------
    inputs : Dict[str, Tuple[Tensor, str]]
        Input fluxes with units.
    outputs : Dict[str, Tuple[Tensor, str]]
        Output fluxes with units.
    storage_change : Optional[Tuple[Tensor, str]], optional
        Storage tendency term with units.
    element : str, default="water"
        Conserved element label; its relative closure tolerance is
        ``validation.conservation.<element>_conservation_tolerance``.
    unit_converter : Optional[Any], optional
        Unit converter exposing ``can_convert`` and ``convert``.

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_conserved, diagnostics).
    """
    if unit_converter is None:
        raise ValueError("unit_converter is required for mass balance evaluation")

    unit_spec = unit_converter.get_element_budget_units(element)
    target_unit = unit_spec["flux"]
    flux_var_name = unit_spec["flux_variable"]
    errors: List[str] = []
    from ....data.standards.unit_converter import ConversionContext

    def _accumulate_total(
        data: Dict[str, Tuple[Tensor, str]],
        flux_variable_name: str,
    ) -> Tuple[Tensor, Tensor, int]:
        total = torch.zeros((), dtype=torch.float64)
        activity = torch.zeros((), dtype=torch.float64)
        count = 0
        for name, (value, unit) in data.items():
            if not unit_converter.can_convert(
                unit,
                target_unit,
                variable_name=flux_variable_name,
            ):
                errors.append(f"Cannot convert {name} from {unit} to {target_unit}")
                continue
            converted = unit_converter.convert(
                data=value,
                variable_name=flux_variable_name,
                source_units=unit,
                target_units=target_unit,
                context=ConversionContext(),
            )
            converted_tensor = (
                converted
                if isinstance(converted, Tensor)
                else torch.as_tensor(converted)
            ).detach().to(device="cpu", dtype=torch.float64)
            total += converted_tensor.sum()
            activity += converted_tensor.abs().sum()
            count += converted_tensor.numel()
        return total, activity, count

    total_input, input_activity, input_count = _accumulate_total(inputs, flux_var_name)
    total_output, output_activity, output_count = _accumulate_total(outputs, flux_var_name)

    if storage_change is not None:
        storage_val, storage_unit = storage_change
        if not unit_converter.can_convert(
            storage_unit,
            target_unit,
            variable_name=flux_var_name,
        ):
            errors.append(
                f"Cannot convert storage from {storage_unit} to {target_unit}"
            )
        else:
            storage_converted = unit_converter.convert(
                data=storage_val,
                variable_name=flux_var_name,
                source_units=storage_unit,
                target_units=target_unit,
                context=ConversionContext(),
            )
            storage_tensor = (
                storage_converted
                if isinstance(storage_converted, Tensor)
                else torch.as_tensor(storage_converted)
            ).detach().to(device="cpu", dtype=torch.float64)
            total_output += storage_tensor.sum()
            output_activity += storage_tensor.abs().sum()
            output_count += storage_tensor.numel()

    if errors:
        return False, {
            "errors": errors,
            "target_unit": target_unit,
            "total_input": float(total_input.item()),
            "total_output": float(total_output.item()),
            "relative_error": float("inf"),
        }

    # References: LeVeque, R.J. (1992), Numerical Methods for Conservation Laws; Higham, N.J.
    #   (2002), Accuracy and Stability of Numerical Algorithms, section 4.2.
    # Mathematical expression: r = sum(inputs) - (sum(outputs) + dS); rho = |r| / (sum |inputs|
    #   + sum |outputs| + |dS|); conserved iff rho <= tol (budget_closure).
    # Parameter meanings: inputs, outputs and dS the converted budget terms; tol =
    #   validation.conservation.<element>_conservation_tolerance.
    # Units: terms and r in the element's budget flux unit; rho and tol dimensionless.
    # Domain/constraints: every term convertible to the budget unit (else reported as errors);
    #   finite terms; no literal tolerance or floor.
    # Sign convention: r > 0 when the inputs exceed the outputs and the storage change.
    closure = budget_closure(
        f"validation.conservation.{element}_conservation_tolerance",
        total_input - total_output,
        [input_activity, output_activity],
        summands=input_count + output_count,
    )
    return closure["passed"], {
        "errors": [],
        "target_unit": target_unit,
        "total_input": float(total_input.item()),
        "total_output": float(total_output.item()),
        "relative_error": closure["relative_residual"],
        "tolerance": closure["tolerance"],
        "tolerance_parameter": closure["tolerance_parameter"],
    }


class ConservationRule:
    """
    Closure of a conserved amount along a recorded history (no sources or sinks).

    Mathematical Expression:
        r_j = M_j - M_0 for every recorded step j >= 1, with M_j = sum_i m_j,i the amount of
        record j; the history is conserved iff budget_closure accepts every r_j with the terms
        sum_i |m_j,i| and sum_i |m_0,i| and the tolerance
        validation.conservation.conservation_tolerance.

    References:
        - LeVeque (1992) Numerical Methods for Conservation Laws.
        - Higham (2002) Accuracy and Stability of Numerical Algorithms, section 4.2.
    """

    TOLERANCE_PARAMETER = "validation.conservation.conservation_tolerance"

    def validate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Judge every recorded amount of a closed system against the first record.

        Parameters
        ----------
        context : Dict[str, Any]
            ``mass_history``: a list or tuple of at least two recorded amounts, each a tensor
            (a field whose cells sum to the amount, or a scalar) or a real number (binary64);
            all records share one floating dtype. No other key is read.

        Returns
        -------
        Dict[str, Any]
            status ('passed' or 'failed'), relative_error (relative residual of the worst
            record), residual, accepted_residual, tolerance, tolerance_parameter, message.
        """
        unsupported = sorted(set(context) - {"mass_history"})
        if unsupported:
            raise ValueError(
                "ConservationRule reads only mass_history; unsupported keys: "
                + ", ".join(unsupported)
            )
        history = context.get("mass_history")
        if not isinstance(history, (list, tuple)) or len(history) < 2:
            raise ValueError(
                "mass_history must be a list or tuple of at least two recorded amounts"
            )
        amounts = [
            record if isinstance(record, Tensor) else torch.as_tensor(record, dtype=torch.float64)
            for record in history
        ]
        dtypes = sorted({str(amount.dtype) for amount in amounts})
        if len(dtypes) != 1 or not amounts[0].dtype.is_floating_point:
            raise ValueError(f"mass_history records must share one floating dtype; got {dtypes}")
        totals = torch.stack([amount.sum() for amount in amounts])
        activities = torch.stack([amount.abs().sum() for amount in amounts])
        steps = len(amounts) - 1

        # References: LeVeque, R.J. (1992), Numerical Methods for Conservation Laws; Higham,
        #   N.J. (2002), Accuracy and Stability of Numerical Algorithms, section 4.2;
        #   budget_closure and ledger_acceptance, the shared closure definition and rule.
        # Mathematical expression: r_j = M_j - M_0 (j = 1..n); a_j = sum_i |m_j,i| +
        #   sum_i |m_0,i|; conserved iff |r_j| <= max(tol * a_j, gamma_K(u) * a_j) for every j,
        #   K = 2 (max_j N_j + N_0 + n): the rounded summands of the two compared records and
        #   one rounded update per step; rho = max_j |r_j| / a_j.
        # Parameter meanings: M_j amount of record j; m_j,i its N_j cells; n number of steps;
        #   tol = validation.conservation.conservation_tolerance; u unit roundoff of the dtype
        #   of the records.
        # Units: M_j, r_j and a_j in the amount unit of the records; rho, tol and gamma
        #   dimensionless.
        # Domain/constraints: at least two finite records of one floating dtype; only the
        #   history is read: no per-call tolerance, no step-count loosening, no cap on the
        #   reported error.
        # Sign convention: r_j > 0 when the amount grew since the first record.
        closure = budget_closure(
            self.TOLERANCE_PARAMETER,
            totals[1:] - totals[0],
            [activities[1:], activities[0]],
            summands=max(amount.numel() for amount in amounts[1:]) + amounts[0].numel() + steps,
        )
        passed = closure["passed"]
        return {
            "status": "passed" if passed else "failed",
            "relative_error": closure["relative_residual"],
            "residual": closure["residual"],
            "accepted_residual": closure["accepted_residual"],
            "tolerance": closure["tolerance"],
            "tolerance_parameter": closure["tolerance_parameter"],
            "message": (
                f"Conservation {'validated' if passed else 'violated'} over {steps} recorded "
                f"steps: relative residual {closure['relative_residual']:.3e} against "
                f"{closure['tolerance']:.3e} ({closure['tolerance_parameter']})"
            ),
        }


@validated_rule(
    rule_id="water_mass_conservation",
    layer="L2", 
    name="Water Mass Conservation",
    description="Validates water mass conservation across hydrological processes",
    domains=["hydrosphere", "water_cycle"],
    tags=["conservation", "water", "mass_balance"],
    units_contract={"water_fluxes": "kg m-2 s-1", "storage": "kg m-2"},
    references=[
        "Hydrological cycle principles",
        "Water balance equation: P - E - T - R - DeltaS = 0"
    ]
)
def validate_water_mass_conservation(context: ValidationContext) -> ValidationResults:
    """
    Validate water mass conservation.
    
    Mathematical Expression:
        Water balance: P - E - T - R - DeltaS = residual
        where P=precipitation, E=evaporation, T=transpiration, R=runoff, DeltaS=storage change
        Conservation satisfied if |residual| < tolerance
    
    Parameters:
        context: Validation context containing water flux data
        
    Returns:
        List of validation results for water mass conservation
    """
    results = []
    
    try:
        # Extract water balance components from context
        metadata = context.metadata or {}
        water_fluxes = metadata.get('water_fluxes', {})
        
        if not water_fluxes:
            return [domain_absent_result(
                "Water balance", "water_mass_conservation", ["conservation", "water"], "water_fluxes"
            )]
        
        missing = [name for name in _WATER_TERMS if name not in water_fluxes]
        if missing:
            return [_not_evaluable_result(
                "Water balance",
                "water_mass_conservation",
                ["conservation", "water"],
                "water_balance_relative_residual",
                missing,
            )]
        precipitation, evaporation, transpiration, runoff, storage_change = (
            _to_tensor(water_fluxes[name]) for name in _WATER_TERMS
        )

        # References: LeVeque, R.J. (1992), Numerical Methods for Conservation Laws; Higham, N.J.
        #   (2002), Accuracy and Stability of Numerical Algorithms, section 4.2.
        # Mathematical expression: r = P - (E + T + R + dS); rho = max_i |r_i| / sum_k |t_k,i| over
        #   the terms t_k of the identity; closed iff rho <= tol (budget_closure: the shared
        #   ledger_acceptance rule with the budget's gross activity sum_k |t_k,i| as its scale).
        # Parameter meanings: P precipitation, E evaporation, T transpiration, R runoff, dS storage
        #   change; tol = validation.conservation.water_conservation_tolerance.
        # Units: terms and r in kg m-2 s-1; rho and tol dimensionless.
        # Domain/constraints: every term supplied (a missing term makes the budget not evaluable;
        #   it is never taken as zero); finite terms (else the rule reports a validation failure);
        #   a per-call override of the budget tolerance is refused; no literal threshold is used.
        # Sign convention: r > 0 when precipitation exceeds the outputs and the storage change.
        _refuse_tolerance_override(context, "water_balance_tolerance", _WATER_TOLERANCE)
        residual = precipitation - (evaporation + transpiration + runoff + storage_change)
        closure = budget_closure(
            _WATER_TOLERANCE,
            residual,
            [precipitation, evaporation, transpiration, runoff, storage_change],
        )
        level, code = _closure_verdict(
            closure,
            ValidationCode.MASS_BALANCE_OK_WATER,
            ValidationCode.MASS_BALANCE_FAIL_WATER,
        )
        results.append(ValidationResultModel(
            level=level,
            code=code,
            message=_closure_message("Water conservation", closure, "kg m-2 s-1"),
            layer="L2",
            scope="water_mass_conservation",
            tags=["conservation", "water", "mass_balance"],
            details=[ValidationDetail(
                variable_name="water_balance_relative_residual",
                value=closure["relative_residual"],
                threshold=closure["tolerance"],
                units="dimensionless",
                statistics={
                    "max_residual": closure["residual"],
                    "total_precipitation": float(precipitation.mean()),
                    "total_outflow": float((evaporation + transpiration + runoff).mean()),
                },
            )],
            metrics={
                "water_balance_residual_max": closure["residual"],
                "water_balance_relative_residual": closure["relative_residual"],
                "tolerance": closure["tolerance"],
            },
        ))

        # Additional check for negative fluxes where inappropriate
        negative_flux_warnings = []
        if torch.any(precipitation < 0):
            negative_flux_warnings.append("precipitation")
        if torch.any(evaporation < 0):
            negative_flux_warnings.append("evaporation")  
        if torch.any(transpiration < 0):
            negative_flux_warnings.append("transpiration")
        if torch.any(runoff < 0):
            negative_flux_warnings.append("runoff")
        
        if negative_flux_warnings:
            results.append(ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.MASS_BALANCE_WARN_WATER,
                message=f"Negative water fluxes detected: {', '.join(negative_flux_warnings)}",
                layer="L2",
                scope="water_mass_conservation",
                tags=["conservation", "water", "negative_flux"],
                details=[ValidationDetail(
                    variable_name="negative_fluxes",
                    value=', '.join(negative_flux_warnings),
                    expected="All fluxes >= 0"
                )]
            ))
            
    # Boundary of this rule (RULES 2): a term that is not tensor-convertible, not
    # finite or not shape-compatible, a refused tolerance override or a failed unit
    # conversion (TypeError, ValueError, RuntimeError) is reported as one explicit
    # failed result; verdicts appended before the failure are dropped, so no result
    # of this rule can read as success. Context-schema and configuration faults
    # (KeyError, AttributeError) propagate to the validated_rule boundary.
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Water mass conservation validation failed: {str(e)}")
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Water mass conservation validation failed: {str(e)}",
            passed=False,
            layer="L2",
            scope="water_mass_conservation",
            tags=["conservation", "water", "execution_error"]
        )]
    
    return results


@validated_rule(
    rule_id="carbon_mass_conservation",
    layer="L2",
    name="Carbon Mass Conservation", 
    description="Validates carbon mass conservation across biogeochemical processes",
    domains=["biosphere", "carbon_cycle"],
    tags=["conservation", "carbon", "mass_balance"],
    units_contract={"carbon_fluxes": "g C m-2 s-1", "carbon_pools": "g C m-2"},
    references=[
        "Carbon cycle principles",
        "Carbon balance: GPP - Ra - Rh - DeltaC = 0"
    ]
)
def validate_carbon_mass_conservation(context: ValidationContext) -> ValidationResults:
    """
    Validate carbon mass conservation.
    
    Mathematical Expression:
        Carbon balance: GPP - Ra - Rh - DeltaC = residual
        where GPP=gross primary productivity, Ra=autotrophic respiration,
        Rh=heterotrophic respiration, DeltaC=carbon storage change
        
    Parameters:
        context: Validation context containing carbon flux data
        
    Returns:
        List of validation results for carbon mass conservation
    """
    results = []
    
    try:
        # Extract carbon balance components from context  
        metadata = context.metadata or {}
        carbon_fluxes = metadata.get('carbon_fluxes', {})
        
        if not carbon_fluxes:
            return [domain_absent_result(
                "Carbon balance", "carbon_mass_conservation", ["conservation", "carbon"], "carbon_fluxes"
            )]
        
        requested = [
            identity for identity in _CARBON_IDENTITIES if identity[2] in carbon_fluxes
        ]
        if not requested:
            return [_not_evaluable_result(
                "Carbon balance",
                "carbon_mass_conservation",
                ["conservation", "carbon"],
                "carbon_balance_relative_residual",
                [],
                reason=(
                    "no carbon identity is determined by the supplied terms "
                    f"({', '.join(sorted(carbon_fluxes))}); each identity needs its defining "
                    "term net_primary_productivity (NPP = GPP - Ra), net_ecosystem_exchange "
                    "(NEE = Ra + Rh - GPP) or storage_change (GPP - Ra - Rh - DeltaC = 0)"
                ),
            )]
        missing = sorted({
            term
            for _, _, _, others in requested
            for term in others
            if term not in carbon_fluxes
        })
        if missing:
            return [_not_evaluable_result(
                "Carbon balance",
                "carbon_mass_conservation",
                ["conservation", "carbon"],
                "carbon_balance_relative_residual",
                missing,
            )]
        values = {
            name: _to_tensor(carbon_fluxes[name])
            for _, _, defining, others in requested
            for name in (defining, *others)
        }
        gpp = values["gross_primary_productivity"]
        ra = values["autotrophic_respiration"]
        rh = values.get("heterotrophic_respiration")
        identities = {
            "NPP_balance": lambda: (
                values["net_primary_productivity"] - (gpp - ra),
                [values["net_primary_productivity"], gpp, ra],
            ),
            "NEE_balance": lambda: (
                values["net_ecosystem_exchange"] - (ra + rh - gpp),
                [values["net_ecosystem_exchange"], ra, rh, gpp],
            ),
            "total_carbon_balance": lambda: (
                gpp - ra - rh - values["storage_change"],
                [gpp, ra, rh, values["storage_change"]],
            ),
        }

        # References: LeVeque, R.J. (1992), Numerical Methods for Conservation Laws; Higham, N.J.
        #   (2002), Accuracy and Stability of Numerical Algorithms, section 4.2.
        # Mathematical expression: r = NPP - (GPP - Ra), r = NEE - (Ra + Rh - GPP),
        #   r = GPP - Ra - Rh - dC; rho = max_i |r_i| / sum_k |t_k,i| over the terms t_k of the
        #   identity; closed iff rho <= tol (budget_closure: the shared ledger_acceptance rule
        #   with the budget's gross activity sum_k |t_k,i| as its scale).
        # Parameter meanings: GPP gross primary productivity, Ra autotrophic and Rh heterotrophic
        #   respiration, NPP net primary productivity, NEE net ecosystem exchange, dC storage
        #   change; tol = validation.conservation.carbon_conservation_tolerance.
        # Units: terms and r in g C m-2 s-1; rho and tol dimensionless.
        # Domain/constraints: an identity is evaluated when its defining term (NPP, NEE or dC)
        #   is supplied, and then all of its terms are required (a missing term makes the
        #   budget not evaluable; it is never taken as zero); finite terms; a per-call override
        #   of the budget tolerance is refused; no literal threshold is used.
        # Sign convention: r > 0 when the left-hand side of the identity exceeds its right-hand
        #   side; NEE is positive for a net carbon source to the atmosphere.
        _refuse_tolerance_override(context, "carbon_balance_tolerance", _CARBON_TOLERANCE)
        for name, equation, _, _ in requested:
            residual, terms = identities[name]()
            closure = budget_closure(_CARBON_TOLERANCE, residual, terms)
            level, code = _closure_verdict(
                closure,
                ValidationCode.MASS_BALANCE_OK_CARBON,
                ValidationCode.MASS_BALANCE_FAIL_CARBON,
            )
            results.append(ValidationResultModel(
                level=level,
                code=code,
                message=_closure_message(f"Carbon conservation ({name})", closure, "g C m-2 s-1"),
                layer="L2",
                scope="carbon_mass_conservation",
                tags=["conservation", "carbon", name],
                details=[ValidationDetail(
                    variable_name=f"{name}_relative_residual",
                    value=closure["relative_residual"],
                    threshold=closure["tolerance"],
                    units="dimensionless",
                    metadata={"equation": equation, "max_residual_g_C_m-2_s-1": closure["residual"]},
                )],
                metrics={
                    f"{name}_residual": closure["residual"],
                    f"{name}_relative_residual": closure["relative_residual"],
                    f"{name}_tolerance": closure["tolerance"],
                },
            ))

        # References: validation/conservation.yaml entry
        #   validation.conservation.plausibility.gross_primary_productivity_max (a model
        #   plausibility limit; no literature value was found in the codeliteraturedatabase).
        # Mathematical expression: warn iff GPP < 0, Ra < 0, Rh < 0 or GPP > GPP_max.
        # Parameter meanings: GPP, Ra, Rh the supplied carbon fluxes; GPP_max the limit.
        # Units: g C m-2 s-1 (GPP_max about 83 umol CO2 m-2 s-1).
        # Domain/constraints: plausibility warnings only; they do not change the budget verdict.
        # Sign convention: GPP, Ra and Rh are non-negative magnitudes.
        gpp_max = _plausibility_limit(_GPP_MAX, "g C m-2 s-1")
        flux_warnings = []
        if torch.any(gpp < 0):
            flux_warnings.append("negative GPP")
        if torch.any(ra < 0):
            flux_warnings.append("negative autotrophic respiration")
        if rh is not None and torch.any(rh < 0):
            flux_warnings.append("negative heterotrophic respiration")
        if torch.any(gpp > gpp_max):
            flux_warnings.append("extremely high GPP")

        if flux_warnings:
            ranges = {"gpp_range": gpp, "ra_range": ra, "rh_range": rh}
            results.append(ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.MASS_BALANCE_WARN_CARBON,
                message=f"Unrealistic carbon fluxes detected: {', '.join(flux_warnings)}",
                layer="L2",
                scope="carbon_mass_conservation",
                tags=["conservation", "carbon", "unrealistic_flux"],
                details=[ValidationDetail(
                    variable_name="carbon_flux_warnings",
                    value=', '.join(flux_warnings),
                    metadata={
                        key: [float(value.min()), float(value.max())]
                        for key, value in ranges.items()
                        if value is not None
                    },
                )]
            ))

    # Boundary of this rule (RULES 2): a term that is not tensor-convertible, not
    # finite or not shape-compatible, a refused tolerance override or a failed unit
    # conversion (TypeError, ValueError, RuntimeError) is reported as one explicit
    # failed result; verdicts appended before the failure are dropped, so no result
    # of this rule can read as success. Context-schema and configuration faults
    # (KeyError, AttributeError) propagate to the validated_rule boundary.
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Carbon mass conservation validation failed: {str(e)}")
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Carbon mass conservation validation failed: {str(e)}",
            passed=False,
            layer="L2",
            scope="carbon_mass_conservation",
            tags=["conservation", "carbon", "execution_error"]
        )]
    
    return results


@validated_rule(
    rule_id="energy_balance_conservation",
    layer="L2",
    name="Energy Balance Conservation",
    description="Validates energy balance conservation at ecosystem surface",
    domains=["atmosphere", "energy_balance"],
    tags=["conservation", "energy", "surface_balance"],
    units_contract={"energy_fluxes": "W m-2"},
    references=[
        "Surface energy balance equation: Rn - H - LE - G = 0",
        "Monin-Obukhov similarity theory"
    ]
)
def validate_energy_balance_conservation(context: ValidationContext) -> ValidationResults:
    """
    Validate surface energy balance conservation.
    
    Mathematical Expression:
        Energy balance: Rn - H - LE - G = residual
        where Rn=net radiation, H=sensible heat flux, LE=latent heat flux, G=ground heat flux
        
    Parameters:
        context: Validation context containing energy flux data
        
    Returns:
        List of validation results for energy balance conservation
    """
    results = []
    
    try:
        # Extract energy balance components from context
        metadata = context.metadata or {}
        energy_fluxes = metadata.get('energy_fluxes', {})
        
        if not energy_fluxes:
            return [domain_absent_result(
                "Energy balance", "energy_balance_conservation", ["conservation", "energy"], "energy_fluxes"
            )]
        
        missing = [name for name in _ENERGY_TERMS if name not in energy_fluxes]
        if missing:
            return [_not_evaluable_result(
                "Energy balance",
                "energy_balance_conservation",
                ["conservation", "energy"],
                "energy_balance_relative_residual",
                missing,
            )]
        net_radiation, sensible_heat, latent_heat, ground_heat = (
            _to_tensor(energy_fluxes[name]) for name in _ENERGY_TERMS
        )

        # References: LeVeque, R.J. (1992), Numerical Methods for Conservation Laws; Higham, N.J.
        #   (2002), Accuracy and Stability of Numerical Algorithms, section 4.2; Allen et al.
        #   (1998), FAO-56, Eq. 42 (G ~ 0 for day-long periods, an assumption the caller states
        #   by supplying ground_heat_flux = 0).
        # Mathematical expression: r = Rn - (H + LE + G); rho = max_i |r_i| / sum_k |t_k,i| over
        #   the terms t_k of the identity; closed iff rho <= tol (budget_closure: the shared
        #   ledger_acceptance rule with the budget's gross activity sum_k |t_k,i| as its scale).
        # Parameter meanings: Rn net radiation, H sensible, LE latent and G ground heat flux;
        #   tol = validation.conservation.energy_conservation_tolerance.
        # Units: terms and r in W m-2; rho and tol dimensionless.
        # Domain/constraints: every term supplied, G included (a missing term makes the budget
        #   not evaluable; it is never taken as zero); finite terms; a per-call override of the
        #   budget tolerance is refused; no literal threshold is used.
        # Sign convention: Rn positive downward, H, LE and G positive away from the surface; r > 0
        #   when Rn exceeds the partition.
        _refuse_tolerance_override(context, "energy_balance_tolerance", _ENERGY_TOLERANCE)
        residual = net_radiation - (sensible_heat + latent_heat + ground_heat)
        closure = budget_closure(
            _ENERGY_TOLERANCE,
            residual,
            [net_radiation, sensible_heat, latent_heat, ground_heat],
        )
        level, code = _closure_verdict(
            closure,
            ValidationCode.ENERGY_BALANCE_OK,
            ValidationCode.ENERGY_BALANCE_FAIL,
        )
        results.append(ValidationResultModel(
            level=level,
            code=code,
            message=_closure_message("Energy balance", closure, "W m-2"),
            layer="L2",
            scope="energy_balance_conservation",
            tags=["conservation", "energy", "surface_balance"],
            details=[ValidationDetail(
                variable_name="energy_balance_relative_residual",
                value=closure["relative_residual"],
                threshold=closure["tolerance"],
                units="dimensionless",
                statistics={
                    "max_residual": closure["residual"],
                    "net_radiation_mean": float(net_radiation.mean()),
                    "sensible_heat_mean": float(sensible_heat.mean()),
                    "latent_heat_mean": float(latent_heat.mean()),
                    "ground_heat_mean": float(ground_heat.mean()),
                },
            )],
            metrics={
                "energy_balance_residual_max": closure["residual"],
                "energy_balance_relative_residual": closure["relative_residual"],
                "tolerance": closure["tolerance"],
            },
        ))

        # References: validation/conservation.yaml entries
        #   validation.conservation.plausibility.{net_radiation_abs_max, latent_heat_flux_min,
        #   bowen_ratio_abs_max} (model plausibility limits; no literature value was found in the
        #   codeliteraturedatabase); Peixoto and Oort (1992), chapter 10 (Bowen ratio H/LE).
        # Mathematical expression: warn iff |Rn| > Rn_max, LE < LE_min or |H| > beta_max |LE|
        #   (|H/LE| > beta_max without a division; LE = 0 with H != 0 is an infinite ratio).
        # Parameter meanings: Rn net radiation, H sensible and LE latent heat flux; Rn_max,
        #   LE_min and beta_max the limits.
        # Units: Rn, H, LE, Rn_max, LE_min in W m-2; beta_max dimensionless.
        # Domain/constraints: plausibility warnings only; they do not change the budget verdict.
        # Sign convention: Rn positive downward; H and LE positive away from the surface, so a
        #   negative LE is a downward (condensation) flux.
        net_radiation_max = _plausibility_limit(_NET_RADIATION_ABS_MAX, "W m-2")
        latent_heat_min = _plausibility_limit(_LATENT_HEAT_FLUX_MIN, "W m-2")
        bowen_ratio_max = _plausibility_limit(_BOWEN_RATIO_ABS_MAX, "dimensionless")
        flux_warnings = []
        if torch.any(torch.abs(net_radiation) > net_radiation_max):
            flux_warnings.append("extreme net radiation values")
        if torch.any(latent_heat < latent_heat_min):
            flux_warnings.append("significant downward latent heat flux")
        if torch.any(torch.abs(sensible_heat) > bowen_ratio_max * torch.abs(latent_heat)):
            flux_warnings.append(f"extreme Bowen ratio (|H/LE| > {bowen_ratio_max:g})")
        
        if flux_warnings:
            results.append(ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.ENERGY_BALANCE_WARN,
                message=f"Energy flux warnings: {', '.join(flux_warnings)}",
                layer="L2",
                scope="energy_balance_conservation",
                tags=["conservation", "energy", "flux_warnings"],
                details=[ValidationDetail(
                    variable_name="energy_flux_warnings",
                    value=', '.join(flux_warnings),
                    metadata={
                        "net_radiation_range": [float(net_radiation.min()), float(net_radiation.max())],
                        "sensible_heat_range": [float(sensible_heat.min()), float(sensible_heat.max())],
                        "latent_heat_range": [float(latent_heat.min()), float(latent_heat.max())]
                    }
                )]
            ))
        
    # Boundary of this rule (RULES 2): a term that is not tensor-convertible, not
    # finite or not shape-compatible, a refused tolerance override or a failed unit
    # conversion (TypeError, ValueError, RuntimeError) is reported as one explicit
    # failed result; verdicts appended before the failure are dropped, so no result
    # of this rule can read as success. Context-schema and configuration faults
    # (KeyError, AttributeError) propagate to the validated_rule boundary.
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Energy balance conservation validation failed: {str(e)}")
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Energy balance conservation validation failed: {str(e)}",
            passed=False,
            layer="L2",
            scope="energy_balance_conservation",
            tags=["conservation", "energy", "execution_error"]
        )]
    
    return results


@validated_rule(
    rule_id="nitrogen_mass_conservation",
    layer="L2",
    name="Nitrogen Mass Conservation",
    description="Validates nitrogen mass conservation across biogeochemical processes",
    domains=["pedosphere", "nitrogen_cycle"],
    tags=["conservation", "nitrogen", "mass_balance"],
    units_contract={"nitrogen_fluxes": "g N m-2 s-1", "nitrogen_pools": "g N m-2"},
    references=[
        "Wang, Y.P., Law, R.M. and Pak, B. (2010). Biogeosciences 7:2261-2282 (CASA-CNP mineral N pool)",
        "Total soil mineral N balance: inputs - outputs - DeltaN_min = 0 (nitrification is internal)"
    ]
)
def validate_nitrogen_mass_conservation(context: ValidationContext) -> ValidationResults:
    """
    Validate nitrogen mass conservation.
    
    Mathematical Expression:
        Total soil mineral nitrogen, N_min = N_NH4 + N_NO3:
        N_inputs - N_outputs - DeltaN_min = residual
        where N_inputs = deposition + fixation + fertilizer + net mineralization
        N_outputs = denitrification + leaching + volatilization + plant uptake
        Nitrification (NH4 -> NO3) is internal to N_min and is not a term.
        
    Parameters:
        context: Validation context containing nitrogen flux data
        
    Returns:
        List of validation results for nitrogen mass conservation
    """
    results = []
    
    try:
        # Extract nitrogen balance components from context
        metadata = context.metadata or {}
        nitrogen_fluxes = metadata.get('nitrogen_fluxes', {})
        
        if not nitrogen_fluxes:
            return [domain_absent_result(
                "Nitrogen balance", "nitrogen_mass_conservation", ["conservation", "nitrogen"], "nitrogen_fluxes"
            )]
        
        missing = [name for name in _NITROGEN_TERMS if name not in nitrogen_fluxes]
        if missing:
            return [_not_evaluable_result(
                "Nitrogen balance",
                "nitrogen_mass_conservation",
                ["conservation", "nitrogen"],
                "nitrogen_balance_relative_residual",
                missing,
            )]
        (
            deposition, fixation, fertilizer, net_mineralization, denitrification,
            leaching, volatilization, plant_uptake, storage_change,
        ) = (_to_tensor(nitrogen_fluxes[name]) for name in _NITROGEN_TERMS)

        # References: Wang, Y.P., Law, R.M. and Pak, B. (2010), Biogeosciences 7:2261-2282
        #   (CASA-CNP mineral N pool: dNsoilmin/dt = Nsnet + Nmindep + Nminfix - Nminloss -
        #   Nminleach - Nupland); RELIC docs/source/theory/nitrogen_cycling.md (the NH4 and NO3
        #   equations, whose sum cancels nitrification); LeVeque (1992); Higham (2002), 4.2.
        # Mathematical expression: r = (deposition + fixation + fertilizer + net mineralization)
        #   - (denitrification + leaching + volatilization + plant uptake) - dN_min;
        #   rho = max_i |r_i| / sum_k |t_k,i|; closed iff rho <= tol (budget_closure).
        # Parameter meanings: N_min = N_NH4 + N_NO3 the soil mineral N; net mineralization the
        #   gross mineralization minus immobilization; dN_min the storage change of N_min;
        #   tol = validation.conservation.nitrogen_conservation_tolerance. Nitrification moves
        #   N from NH4 to NO3 inside N_min and is no term of this budget.
        # Units: terms and r in g N m-2 s-1; rho and tol dimensionless.
        # Domain/constraints: every term supplied (a missing term makes the budget not
        #   evaluable; it is never taken as zero); finite terms; a per-call override of the
        #   budget tolerance is refused; no literal threshold is used.
        # Sign convention: inputs add to N_min and outputs remove it; net mineralization is
        #   negative under net immobilization; r > 0 when the inputs exceed the outputs and dN_min.
        _refuse_tolerance_override(context, "nitrogen_balance_tolerance", _NITROGEN_TOLERANCE)
        total_inputs = deposition + fixation + fertilizer + net_mineralization
        total_outputs = denitrification + leaching + volatilization + plant_uptake
        residual = total_inputs - total_outputs - storage_change
        closure = budget_closure(
            _NITROGEN_TOLERANCE,
            residual,
            [
                deposition, fixation, fertilizer, net_mineralization, denitrification,
                leaching, volatilization, plant_uptake, storage_change,
            ],
        )
        level, code = _closure_verdict(
            closure,
            ValidationCode.MASS_BALANCE_OK_NITROGEN,
            ValidationCode.MASS_BALANCE_FAIL_NITROGEN,
        )
        results.append(ValidationResultModel(
            level=level,
            code=code,
            message=_closure_message("Nitrogen conservation", closure, "g N m-2 s-1"),
            layer="L2",
            scope="nitrogen_mass_conservation",
            tags=["conservation", "nitrogen", "mass_balance"],
            details=[ValidationDetail(
                variable_name="nitrogen_balance_relative_residual",
                value=closure["relative_residual"],
                threshold=closure["tolerance"],
                units="dimensionless",
                statistics={
                    "max_residual": closure["residual"],
                    "total_inputs": float(total_inputs.mean()),
                    "total_outputs": float(total_outputs.mean()),
                    "storage_change": float(storage_change.mean()),
                },
            )],
            metrics={
                "nitrogen_balance_residual_max": closure["residual"],
                "nitrogen_balance_relative_residual": closure["relative_residual"],
                "tolerance": closure["tolerance"],
            },
        ))

        # Check for negative fluxes where inappropriate
        flux_warnings = []
        if torch.any(deposition < 0):
            flux_warnings.append("negative nitrogen deposition")
        if torch.any(fixation < 0):
            flux_warnings.append("negative nitrogen fixation")
        if torch.any(fertilizer < 0):
            flux_warnings.append("negative fertilizer application")
        if torch.any(plant_uptake < 0):
            flux_warnings.append("negative plant nitrogen uptake")
        
        if flux_warnings:
            results.append(ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.MASS_BALANCE_WARN_NITROGEN,
                message=f"Unrealistic nitrogen fluxes: {', '.join(flux_warnings)}",
                layer="L2",
                scope="nitrogen_mass_conservation",
                tags=["conservation", "nitrogen", "unrealistic_flux"]
            ))
        
    # Boundary of this rule (RULES 2): a term that is not tensor-convertible, not
    # finite or not shape-compatible, a refused tolerance override or a failed unit
    # conversion (TypeError, ValueError, RuntimeError) is reported as one explicit
    # failed result; verdicts appended before the failure are dropped, so no result
    # of this rule can read as success. Context-schema and configuration faults
    # (KeyError, AttributeError) propagate to the validated_rule boundary.
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Nitrogen mass conservation validation failed: {str(e)}")
        return [ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Nitrogen mass conservation validation failed: {str(e)}",
            passed=False,
            layer="L2",
            scope="nitrogen_mass_conservation",
            tags=["conservation", "nitrogen", "execution_error"]
        )]
    
    return results


# Export validation rule functions
__all__ = [
    "validate_water_mass_conservation",
    "validate_carbon_mass_conservation", 
    "validate_energy_balance_conservation",
    "validate_nitrogen_mass_conservation"
]
