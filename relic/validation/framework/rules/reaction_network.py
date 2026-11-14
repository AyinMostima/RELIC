"""Validation rules for reaction-network IPHS structure."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ....data.standards.dimensionality import dimensionality_from_unit
from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)


_OPEN_SYSTEM_ROUTES = {
    "boundary_export",
    "managed_export",
    "diagnostic_residual",
    "port",
    "observation",
}
_PHYSICAL_OPEN_SYSTEM_ROUTES = {
    "boundary_export",
    "managed_export",
    "port",
}
_AFFINITY_SIGN_CONVENTION = "affinity_times_rate_nonnegative"
_NONCANONICAL_REACTION_KEYS = {
    "A",
    "C",
    "N",
    "charge",
    "composition_matrix",
    "element_matrix",
    "r",
    "reaction_affinities",
    "reaction_rates",
    "stoichiometric_matrix",
    "subsystem_N",
    "subsystem_stoichiometric_N",
    "z_h",
}
_ROUTE_DECLARATION_FIELDS = (
    "route",
    "owner",
    "target",
    "exchange_variable",
    "ledger_variable",
)


def _contains_bool_or_complex(value: Any) -> bool:
    """Return whether a nested numeric payload contains bool or complex data."""

    if isinstance(value, torch.Tensor):
        return value.dtype == torch.bool or torch.is_complex(value)
    if isinstance(value, (bool, complex)):
        return True
    dtype_kind = getattr(getattr(value, "dtype", None), "kind", None)
    if dtype_kind in {"b", "c"}:
        return True
    if dtype_kind == "O":
        return any(_contains_bool_or_complex(item) for item in value.flat)
    if isinstance(value, Mapping):
        return any(_contains_bool_or_complex(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_bool_or_complex(item) for item in value)
    return False


def _tensor(value: Any, *, dtype: torch.dtype = torch.float64) -> torch.Tensor | None:
    if value is None:
        return None
    if _contains_bool_or_complex(value):
        return None
    try:
        tensor = torch.as_tensor(value, dtype=dtype)
    except (TypeError, ValueError, RuntimeError):
        return None
    if (
        tensor.device.type == "meta"
        or tensor.layout != torch.strided
        or tensor.is_quantized
        or tensor.dtype == torch.bool
        or torch.is_complex(tensor)
    ):
        return None
    try:
        return tensor.detach().to(device="cpu", dtype=dtype)
    except (TypeError, ValueError, RuntimeError):
        return None


def _ordered_labels(
    value: Any,
    *,
    expected_count: int,
) -> tuple[tuple[str, ...] | None, str]:
    """Validate a non-empty, unique, ordered declaration basis."""

    if not isinstance(value, (list, tuple)) or not value:
        return None, "must be a non-empty list or tuple"
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return None, "must contain only non-empty strings"
    labels = tuple(str(item).strip() for item in value)
    if len(labels) != expected_count:
        return None, f"must contain exactly {expected_count} entries"
    if len(set(labels)) != len(labels):
        return None, "must contain unique entries"
    return labels, ""


def _canonical_unit_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    if not bool(torch.isfinite(value).all().item()):
        return float("inf")
    return float(torch.max(torch.abs(value)).detach().cpu().item())


def _sum_subsystem_matrices(payload: Mapping[str, Any]) -> tuple[torch.Tensor | None, str]:
    if "subsystem_stoichiometric_matrices" not in payload:
        return None, ""
    raw = payload["subsystem_stoichiometric_matrices"]
    if not isinstance(raw, (list, tuple)) or not raw:
        return None, "subsystem stoichiometric matrices must be a non-empty sequence"
    matrices = [_tensor(item) for item in raw]
    if any(item is None for item in matrices):
        return None, "subsystem stoichiometric matrices must be numeric tensors"
    first = matrices[0]
    if first is None:
        return None, "subsystem stoichiometric matrices must not be empty"
    total = torch.zeros_like(first)
    for item in matrices:
        if item is None:
            return None, "subsystem stoichiometric matrices must be numeric tensors"
        if item.ndim != 2 or item.numel() == 0:
            return None, "subsystem stoichiometric matrices must be non-empty rank-2 tensors"
        if not bool(torch.isfinite(item).all().item()):
            return None, "subsystem stoichiometric matrices must be finite"
        if tuple(item.shape) != tuple(first.shape):
            return None, "all subsystem stoichiometric matrices must have the same shape"
        total = total + item
        if not bool(torch.isfinite(total).all().item()):
            return None, "subsystem stoichiometric matrix sum must remain finite"
    return total, ""


def validate_reaction_network_payload(
    payload: Mapping[str, Any],
    *,
    strict: bool = False,
    require_affinity: bool | None = None,
) -> ValidationResultModel:
    """Validate stoichiometric conservation and reaction affinity direction.

    References:
      - Feinberg (2019), Foundations of Chemical Reaction Network Theory.
      - Rao and Esposito (2016), nonequilibrium thermodynamics of chemical networks.
      - Grmela and Ottinger (1997), GENERIC formalism.
    Mathematical expression:
      C_h N_h = 0 for conserved elements and z_h^T N_h = 0 for charge.
      The minimum strict dissipation condition is A^T r = sum_j(A_j r_j) >= -tol.
      Elementwise A_j r_j >= -tol is a stronger optional detailed-balance profile.
      [r] = [x] [t]^-1 and [A^T r] = [A] [r].
      True entropy production is sigma = A^T r / T and remains [N] without
      an executable positive-temperature binding.
      Strict unit closure uses one homogeneous species basis and SI-second
      rate spelling; heterogeneous unit normalization remains [N].
    Parameter meanings:
      C_h: conserved-element composition matrix.
      N_h: stoichiometric matrix with species rows and reaction columns.
      r_j: reaction rate.
      A_j: thermodynamic affinity under the declared sign convention.
      strict: require complete basis, unit, rate, and physical ledger contracts.
      require_affinity: caller-owned affinity obligation; None follows strict.
    Units:
      N_h and C_h use the declared homogeneous species and ledger bases.
      Rates are species amount per second; affinity-rate pairing is power.
    Domain/constraints:
      Declared affinity fields always remain subject to their checks.
      An ordinary structural contract may omit the entire affinity declaration.
    Sign convention:
      N_h r is positive into a species pool; A^T r is dissipation power.
    """

    name = str(payload.get("name", "reaction_network"))
    details: list[ValidationDetail] = []
    metrics: dict[str, int | float] = {}
    if require_affinity is not None and type(require_affinity) is not bool:
        raise TypeError("require_affinity must be bool or None.")
    affinity_required = strict if require_affinity is None else require_affinity
    affinity_declared = any(
        key in payload
        for key in (
            "affinities", "affinity_sign_convention", "affinity_reaction_order",
            "affinity_units", "affinity_rate_power_units",
        )
    )
    check_affinity = affinity_required or affinity_declared
    for key in sorted(_NONCANONICAL_REACTION_KEYS.intersection(payload)):
        details.append(
            ValidationDetail(
                variable_name=name,
                expected="canonical reaction payload fields only",
                value=f"noncanonical field '{key}'",
            )
        )
    for key in (
        "stoichiometric_N",
        "conservation_matrix",
        "rates",
        "affinities",
        "charge_vector",
        "subsystem_stoichiometric_matrices",
        "open_system_exchange",
        "open_system_ledger_rate",
    ):
        if key in payload and _contains_bool_or_complex(payload[key]):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected=f"real-valued {key} without bool or complex entries",
                    value="bool or complex data",
                )
            )
    try:
        if isinstance(payload.get("tolerance", 1.0e-10), bool):
            raise TypeError("bool is not a tolerance")
        tolerance = float(payload.get("tolerance", 1.0e-10))
    except (TypeError, ValueError, OverflowError):
        tolerance = 1.0e-10
        details.append(
            ValidationDetail(
                variable_name=name,
                expected="finite non-negative tolerance",
                value=payload.get("tolerance"),
            )
        )
    if not torch.isfinite(torch.tensor(tolerance, dtype=torch.float64)) or tolerance < 0.0:
        details.append(
            ValidationDetail(
                variable_name=name,
                expected="finite non-negative tolerance",
                value=tolerance,
            )
        )
        tolerance = 1.0e-10

    N = _tensor(payload.get("stoichiometric_N"))
    C = _tensor(payload.get("conservation_matrix"))
    if N is None or N.ndim != 2:
        details.append(
            ValidationDetail(
                variable_name=name,
                expected="rank-2 stoichiometric matrix N",
                value="missing or invalid",
            )
        )
    else:
        metrics["species_count"] = int(N.shape[0])
        metrics["reaction_count"] = int(N.shape[1])
        if not torch.isfinite(N).all():
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="finite stoichiometric matrix N",
                    value="non-finite",
                )
            )
        if strict and (N.shape[0] == 0 or N.shape[1] == 0):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="non-empty species and reaction axes",
                    value=f"{N.shape[0]} x {N.shape[1]}",
                )
            )
        elif strict and bool(torch.any(torch.all(N == 0, dim=0)).item()):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="every reaction column changes at least one species",
                    value="one or more zero reaction columns",
                )
            )

    open_route = str(payload.get("open_system_route", "")).strip()
    if open_route and open_route not in _OPEN_SYSTEM_ROUTES:
        details.append(
            ValidationDetail(
                variable_name=name,
                expected="recognized open_system_route",
                value=open_route,
            )
        )
    open_ledger_declared = any(
        key in payload
        for key in (
            "open_system_exchange",
            "open_system_ledger_rate",
            "open_system_exchange_units",
            "open_system_ledger_rate_units",
            "conservation_ledger_profile",
            "open_exchange_domain",
            "open_ledger_domain",
            "open_exchange_species_order",
            "open_ledger_basis",
            "open_system_route_binding",
        )
    )
    if strict and open_ledger_declared and not open_route:
        details.append(
            ValidationDetail(
                variable_name=name,
                expected="open_system_route for every declared open-system ledger",
                value="missing open_system_route",
            )
        )
    if strict and open_route and open_route not in _PHYSICAL_OPEN_SYSTEM_ROUTES:
        details.append(
            ValidationDetail(
                variable_name=name,
                expected=(
                    "physical open-system route: port, boundary_export, "
                    "or managed_export"
                ),
                value=open_route,
            )
        )
    if strict and C is None:
        details.append(
            ValidationDetail(
                variable_name=name,
                expected="conservation_matrix C_h for every strict internal ledger",
                value="missing conservation_matrix",
            )
        )

    if C is not None:
        if C.ndim != 2:
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="rank-2 conservation matrix C_h",
                    value="invalid",
                )
            )
        elif not torch.isfinite(C).all():
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="finite conservation matrix C_h",
                    value="non-finite",
                )
            )
        elif strict and C.shape[0] == 0:
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="non-empty conservation ledger C_h",
                    value=f"{C.shape[0]} x {C.shape[1]}",
                )
            )
        elif strict and bool(torch.any(torch.all(C == 0, dim=1)).item()):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="every strict conservation ledger row is nonzero",
                    value="one or more zero rows",
                )
            )
        elif N is not None and C.shape[1] != N.shape[0]:
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected=f"C_h columns = {N.shape[0]} species",
                    value=int(C.shape[1]),
                )
            )
        elif N is not None:
            residual = C @ N
            residual_norm = _max_abs(residual)
            metrics["stoichiometric_conservation_residual"] = residual_norm
            if residual_norm > tolerance:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="C_h N_h = 0",
                        value=residual_norm,
                        threshold=tolerance,
                    )
                )

    species_basis: tuple[str, ...] | None = None
    reaction_basis: tuple[str, ...] | None = None
    conservation_basis: tuple[str, ...] | None = None
    if strict or check_affinity or any(
        key in payload for key in (
            "species", "reactions", "conservation_basis", "rate_reaction_order",
            "state_domain", "reaction_domain", "conservation_domain",
        )
    ):
        if (
            (strict and check_affinity or "affinity_sign_convention" in payload)
            and payload.get("affinity_sign_convention") != _AFFINITY_SIGN_CONVENTION
        ):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected=(
                        "affinity sign convention "
                        f"{_AFFINITY_SIGN_CONVENTION}"
                    ),
                    value=payload.get("affinity_sign_convention"),
                )
            )
        species_count = int(N.shape[0]) if N is not None and N.ndim == 2 else 0
        reaction_count = int(N.shape[1]) if N is not None and N.ndim == 2 else 0
        ledger_count = int(C.shape[0]) if C is not None and C.ndim == 2 else 0
        for field_name, expected_count, expected_text in (
            (
                "species",
                species_count,
                "non-empty unique species basis matching N_h rows",
            ),
            (
                "reactions",
                reaction_count,
                "non-empty unique reaction basis matching N_h columns",
            ),
            (
                "conservation_basis",
                ledger_count,
                "non-empty unique conservation basis matching C_h rows",
            ),
        ):
            if not strict and field_name not in payload:
                continue
            labels, error = _ordered_labels(
                payload.get(field_name),
                expected_count=expected_count,
            )
            if error:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=expected_text,
                        value=error,
                    )
                )
            elif field_name == "species":
                species_basis = labels
            elif field_name == "reactions":
                reaction_basis = labels
            else:
                conservation_basis = labels

        order_fields = ("rate_reaction_order",)
        if check_affinity:
            order_fields += ("affinity_reaction_order",)
        for field_name in order_fields:
            if not strict and field_name not in payload:
                continue
            labels, error = _ordered_labels(
                payload.get(field_name),
                expected_count=reaction_count,
            )
            if error or reaction_basis is None or labels != reaction_basis:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=f"{field_name.replace('_', ' ')} equals reactions basis",
                        value=error or str(labels),
                    )
                )

        for field_name in (
            "state_domain",
            "reaction_domain",
            "conservation_domain",
        ):
            if not strict and field_name not in payload:
                continue
            value = payload.get(field_name)
            if not isinstance(value, str) or not value.strip():
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=f"non-empty {field_name} declaration",
                        value=value,
                    )
                )

    charge_declared = "charge_vector" in payload
    charge = _tensor(payload.get("charge_vector"))
    if strict and charge_declared and charge is None:
        details.append(
            ValidationDetail(
                variable_name=name,
                expected="numeric charge_vector",
                value="invalid declared charge vector",
            )
        )
    elif charge is not None and N is not None:
        charge_row = charge.reshape(1, -1)
        if not bool(torch.isfinite(charge_row).all().item()):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="finite charge vector",
                    value="non-finite",
                )
            )
        elif charge_row.shape[1] != N.shape[0]:
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected=f"charge vector length = {N.shape[0]} species",
                    value=int(charge_row.shape[1]),
                )
            )
        else:
            charge_residual = _max_abs(charge_row @ N)
            metrics["charge_conservation_residual"] = charge_residual
            if charge_residual > tolerance:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="z_h^T N_h = 0",
                        value=charge_residual,
                        threshold=tolerance,
                    )
                )

    rates = _tensor(payload.get("rates"))
    affinities = _tensor(payload.get("affinities"))
    if not check_affinity and (strict or "rates" in payload):
        if (
            rates is None
            or rates.ndim != 1
            or rates.numel() == 0
            or not bool(torch.isfinite(rates).all().item())
            or (N is not None and N.ndim == 2 and rates.numel() != N.shape[1])
        ):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="one finite rate per declared reaction in a non-empty rank-1 vector",
                    value="missing or invalid rates",
                )
            )
    if check_affinity and (rates is None or affinities is None):
        details.append(
            ValidationDetail(
                variable_name=name,
                expected="both rates r and affinities A",
                value="one or both missing",
            )
        )
    elif check_affinity and rates is not None and affinities is not None:
        if rates.ndim != 1 or affinities.ndim != 1:
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="rank-1 rates and affinities",
                    value=f"{tuple(rates.shape)} vs {tuple(affinities.shape)}",
                )
            )
        elif tuple(rates.shape) != tuple(affinities.shape):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="rates and affinities have matching shape",
                    value=f"{tuple(rates.shape)} vs {tuple(affinities.shape)}",
                )
            )
        elif N is not None and rates.numel() != N.shape[1]:
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected=f"one rate and affinity per {N.shape[1]} reactions",
                    value=int(rates.numel()),
                )
            )
        elif rates.numel() == 0:
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="non-empty rates and affinities",
                    value="empty reaction vectors",
                )
            )
        elif not torch.isfinite(rates).all() or not torch.isfinite(affinities).all():
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="finite rates and affinities",
                    value="non-finite",
                )
            )
        else:
            production = rates * affinities
            if not bool(torch.isfinite(production).all().item()):
                metrics["min_rate_affinity_product"] = float("inf")
                metrics["affinity_rate_power_native"] = float("inf")
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="finite affinity-rate power A^T r",
                        value="non-finite",
                    )
                )
            else:
                min_production = float(torch.min(production).detach().cpu().item())
                metrics["min_rate_affinity_product"] = min_production
                total_production = float(torch.sum(production).detach().cpu().item())
                metrics["affinity_rate_power_native"] = total_production
                if not torch.isfinite(
                    torch.tensor(total_production, dtype=torch.float64)
                ):
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected="finite affinity-rate power A^T r",
                            value=total_production,
                        )
                    )
                elif total_production < -tolerance:
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected="A^T r >= 0",
                            value=total_production,
                            threshold=tolerance,
                        )
                    )

    if strict or check_affinity or "species_units" in payload or "rate_units" in payload:
        species_count = int(N.shape[0]) if N is not None and N.ndim == 2 else 0
        species_units = payload.get("species_units")
        valid_species_unit_list = (
            isinstance(species_units, (list, tuple))
            and len(species_units) == species_count
            and all(isinstance(unit, str) and unit.strip() for unit in species_units)
        )
        species_unit_values: list[str] = []
        if valid_species_unit_list and isinstance(species_units, (list, tuple)):
            species_unit_values = [str(unit) for unit in species_units]
        species_dimensions = []
        canonical_species_units: list[str] = []
        if not valid_species_unit_list and (strict or "species_units" in payload):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="one non-empty species unit per species row",
                    value=str(species_units),
                )
            )
        else:
            for unit in species_unit_values:
                canonical_species_units.append(_canonical_unit_text(unit))
                dimension = dimensionality_from_unit(unit)
                if dimension is None:
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected="parseable species units from the canonical unit owner",
                            value=unit,
                        )
                    )
                else:
                    species_dimensions.append(dimension)
            if species_dimensions and any(
                not dimension.is_compatible(species_dimensions[0])
                for dimension in species_dimensions[1:]
            ):
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="dimensionally compatible species units for dimensionless N_h",
                        value=str(species_units),
                    )
                )
            if canonical_species_units and len(set(canonical_species_units)) != 1:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=(
                            "identical canonical species units for dimensionless N_h"
                        ),
                        value=str(species_units),
                    )
                )

        unit_dimensions = {}
        unit_keys = ("rate_units",)
        if check_affinity:
            unit_keys += ("affinity_units", "affinity_rate_power_units")
        for unit_key in unit_keys:
            if not strict and unit_key not in payload:
                continue
            unit_value = payload.get(unit_key)
            if not isinstance(unit_value, str) or not unit_value.strip():
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=f"non-empty {unit_key}",
                        value=unit_value,
                    )
                )
                continue
            dimension = dimensionality_from_unit(unit_value)
            if dimension is None:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=f"parseable {unit_key} from the canonical unit owner",
                        value=unit_value,
                    )
                )
                continue
            unit_dimensions[unit_key] = dimension

        if species_dimensions and "rate_units" in unit_dimensions:
            expected_rate_dimension = dimensionality_from_unit(
                f"{species_unit_values[0]} s-1"
            )
            if (
                expected_rate_dimension is None
                or not unit_dimensions["rate_units"].is_compatible(
                    expected_rate_dimension
                )
            ):
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="rate_units compatible with species unit per time",
                        value=payload.get("rate_units"),
                    )
                )
            expected_rate_units = f"{canonical_species_units[0]} s-1"
            if _canonical_unit_text(payload.get("rate_units")) != expected_rate_units:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="rate_units use canonical species_units s-1 basis",
                        value=payload.get("rate_units"),
                    )
                )

        if species_dimensions and "affinity_units" in unit_dimensions:
            species_dimension = species_dimensions[0]
            expected_affinity_dimension = None
            if species_dimension.N == 1.0 and species_dimension.M == 0.0:
                expected_affinity_dimension = dimensionality_from_unit("J mol-1")
            elif species_dimension.M == 1.0 and species_dimension.N == 0.0:
                expected_affinity_dimension = dimensionality_from_unit("J kg-1")
            if (
                expected_affinity_dimension is None
                or not unit_dimensions["affinity_units"].is_compatible(
                    expected_affinity_dimension
                )
            ):
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=(
                            "affinity_units compatible with energy per mass or "
                            "amount reaction extent"
                        ),
                        value=payload.get("affinity_units"),
                    )
                )
            affinity_tokens = _canonical_unit_text(
                payload.get("affinity_units")
            ).split()
            power_tokens = _canonical_unit_text(
                payload.get("affinity_rate_power_units")
            ).split()
            species_unit_basis = canonical_species_units[0].split()[0]
            affinity_scale_bound = (
                len(affinity_tokens) == 2
                and affinity_tokens[1] == f"{species_unit_basis}-1"
                and bool(power_tokens)
                and affinity_tokens[0] == power_tokens[0]
                and power_tokens[-1] == "s-1"
            )
            if not affinity_scale_bound:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=(
                            "affinity and power units share canonical energy, "
                            "species, and second scales"
                        ),
                        value=(
                            f"{payload.get('affinity_units')} -> "
                            f"{payload.get('affinity_rate_power_units')}"
                        ),
                    )
                )

        if {
            "rate_units",
            "affinity_units",
            "affinity_rate_power_units",
        }.issubset(unit_dimensions):
            production_dimension = dimensionality_from_unit(
                f"{payload['rate_units']} {payload['affinity_units']}"
            )
            if (
                production_dimension is None
                or not production_dimension.is_compatible(
                    unit_dimensions["affinity_rate_power_units"]
                )
            ):
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=(
                            "affinity_rate_power_units compatible with "
                            "rate_units * affinity_units"
                        ),
                        value=payload.get("affinity_rate_power_units"),
                    )
                )

        if strict and open_route:
            if open_route in _PHYSICAL_OPEN_SYSTEM_ROUTES:
                route_declaration = payload.get("open_system_route_binding")
                declaration_valid = (
                    isinstance(route_declaration, Mapping)
                    and all(
                        isinstance(route_declaration.get(field_name), str)
                        and str(route_declaration.get(field_name)).strip()
                        for field_name in _ROUTE_DECLARATION_FIELDS
                    )
                    and str(route_declaration.get("route", "")).strip() == open_route
                )
                if not declaration_valid:
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected=(
                                "complete structural route declaration with matching "
                                "route, owner, target, exchange_variable, and ledger_variable"
                            ),
                            value=str(route_declaration),
                        )
                    )

                state_domain = str(payload.get("state_domain", "")).strip()
                conservation_domain = str(
                    payload.get("conservation_domain", "")
                ).strip()
                if (
                    not isinstance(payload.get("open_exchange_domain"), str)
                    or str(payload.get("open_exchange_domain", "")).strip()
                    != state_domain
                ):
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected="open exchange domain equals state_domain",
                            value=payload.get("open_exchange_domain"),
                        )
                    )
                if (
                    not isinstance(payload.get("open_ledger_domain"), str)
                    or str(payload.get("open_ledger_domain", "")).strip()
                    != conservation_domain
                ):
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected="open ledger domain equals conservation_domain",
                            value=payload.get("open_ledger_domain"),
                        )
                    )
                exchange_order = payload.get("open_exchange_species_order")
                if (
                    not isinstance(exchange_order, (list, tuple))
                    or species_basis is None
                    or tuple(exchange_order) != species_basis
                ):
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected="open exchange species basis equals species basis",
                            value=str(exchange_order),
                        )
                    )
                ledger_basis = payload.get("open_ledger_basis")
                if (
                    not isinstance(ledger_basis, (list, tuple))
                    or conservation_basis is None
                    or tuple(ledger_basis) != conservation_basis
                ):
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected="open ledger basis equals conservation basis",
                            value=str(ledger_basis),
                        )
                    )

            exchange = _tensor(payload.get("open_system_exchange"))
            ledger_rate = _tensor(payload.get("open_system_ledger_rate"))
            if (
                exchange is None
                or ledger_rate is None
                or C is None
                or C.ndim != 2
            ):
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=(
                            "open_system_exchange and open_system_ledger_rate "
                            "with conservation_matrix"
                        ),
                        value="missing or invalid",
                    )
                )
            elif (
                exchange.ndim != 1
                or ledger_rate.ndim != 1
                or exchange.numel() != C.shape[1]
                or ledger_rate.numel() != C.shape[0]
                or not torch.isfinite(exchange).all()
                or not torch.isfinite(ledger_rate).all()
            ):
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="open-system ledger shapes match C_h and are finite",
                        value=f"{tuple(exchange.shape)} vs {tuple(ledger_rate.shape)}",
                    )
                )
            else:
                ledger_residual = _max_abs(C @ exchange - ledger_rate)
                metrics["open_system_ledger_residual"] = ledger_residual
                if ledger_residual > tolerance:
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected=(
                                "C_h q_open = declared open-system ledger rate"
                            ),
                            value=ledger_residual,
                            threshold=tolerance,
                        )
                    )
            exchange_units = payload.get("open_system_exchange_units")
            if not isinstance(exchange_units, str) or not exchange_units.strip():
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="non-empty open_system_exchange_units",
                        value=exchange_units,
                    )
                )
            else:
                exchange_dimension = dimensionality_from_unit(exchange_units)
                if exchange_dimension is None:
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected=(
                                "parseable open_system_exchange_units from the "
                                "canonical unit owner"
                            ),
                            value=exchange_units,
                        )
                    )
                elif (
                    "rate_units" in unit_dimensions
                    and not exchange_dimension.is_compatible(
                        unit_dimensions["rate_units"]
                    )
                ):
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected=(
                                "open_system_exchange_units compatible with rate_units"
                            ),
                            value=exchange_units,
                        )
                    )
                elif _canonical_unit_text(exchange_units) != _canonical_unit_text(
                    payload.get("rate_units")
                ):
                    details.append(
                        ValidationDetail(
                            variable_name=name,
                            expected=(
                                "open_system_exchange_units use the same canonical "
                                "basis as rate_units"
                            ),
                            value=exchange_units,
                        )
                    )

            ledger_profile = _canonical_unit_text(
                payload.get("conservation_ledger_profile")
            )
            ledger_rate_units = _canonical_unit_text(
                payload.get("open_system_ledger_rate_units")
            )
            if (
                ledger_profile != "dimensionless_homogeneous"
                or ledger_rate_units != _canonical_unit_text(payload.get("rate_units"))
            ):
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected=(
                            "conservation_ledger_profile=dimensionless_homogeneous "
                            "with ledger-rate units equal to rate_units"
                        ),
                        value=(
                            f"profile={ledger_profile}, "
                            f"units={ledger_rate_units or 'missing'}"
                        ),
                    )
                )

    subsystem_total, subsystem_error = _sum_subsystem_matrices(payload)
    if subsystem_error:
        details.append(
            ValidationDetail(
                variable_name=name,
                expected=subsystem_error,
                value=subsystem_error,
            )
        )
    if subsystem_total is not None and C is not None:
        if N is not None and tuple(subsystem_total.shape) != tuple(N.shape):
            details.append(
                ValidationDetail(
                    variable_name=name,
                    expected="subsystem matrix sum shape matches N_h",
                    value=f"{tuple(subsystem_total.shape)} vs {tuple(N.shape)}",
                )
            )
        elif N is not None:
            reconstruction_residual = _max_abs(subsystem_total - N)
            metrics["subsystem_reconstruction_residual"] = reconstruction_residual
            if reconstruction_residual > tolerance:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="sum_i N_i = N_h",
                        value=reconstruction_residual,
                        threshold=tolerance,
                    )
                )
        if C.ndim == 2 and C.shape[1] == subsystem_total.shape[0]:
            residual = C @ subsystem_total
            cross_residual = _max_abs(residual)
            metrics["cross_subsystem_conservation_residual"] = cross_residual
            if not bool(torch.isfinite(residual).all().item()):
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="finite cross-subsystem conservation residual",
                        value="non-finite",
                    )
                )
            elif cross_residual > tolerance:
                details.append(
                    ValidationDetail(
                        variable_name=name,
                        expected="C_h sum_i N_i = 0",
                        value=cross_residual,
                        threshold=tolerance,
                    )
                )

    passed = not details
    return ValidationResultModel(
        level=ValidationLevel.INFO if passed else ValidationLevel.ERROR,
        code=(
            ValidationCode.GENERIC_VALIDATION_OK
            if passed
            else ValidationCode.GENERIC_VALIDATION_FAILED
        ),
        message=(
            f"Reaction network '{name}' satisfies structural checks."
            if passed
            else f"Reaction network '{name}' violates structural checks."
        ),
        passed=passed,
        layer="L1",
        scope="reaction_network",
        tags=["iphs", "reaction", "stoichiometry", "conservation"],
        details=details,
        metrics=metrics,
    )


@validated_rule(
    "reaction_network_structure",
    "L1",
    name="Reaction-network structure",
    description="Validate C_h N_h conservation, charge conservation, and affinity direction.",
    domains=["scientific", "computation"],
    tags=["iphs", "reaction", "stoichiometry", "conservation"],
    references=[
        "Feinberg (2019), Foundations of Chemical Reaction Network Theory",
        "Rao and Esposito (2016), Nonequilibrium thermodynamics of chemical reaction networks",
    ],
)
def validate_reaction_network_structure(context: ValidationContext) -> ValidationResults:
    """Validate reaction-network payloads passed through ValidationContext metadata."""

    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    strict = metadata.get("strict") is True
    require_affinity = metadata.get("require_affinity")
    if "reaction_networks" in metadata:
        payloads = metadata["reaction_networks"]
    elif "reaction_network" in metadata:
        payloads = metadata["reaction_network"]
    elif "stoichiometric_contracts" in metadata:
        payloads = metadata["stoichiometric_contracts"]
    else:
        payloads = metadata
    if isinstance(payloads, (list, tuple)):
        results: ValidationResults = []
        for index, item in enumerate(payloads):
            if isinstance(item, Mapping):
                results.append(
                    validate_reaction_network_payload(
                        item, strict=strict, require_affinity=require_affinity,
                    )
                )
            else:
                results.append(
                    validate_reaction_network_payload(
                        {"name": f"invalid_reaction_network_{index}"},
                        strict=strict,
                        require_affinity=require_affinity,
                    )
                )
        if results:
            return results
        return [
            validate_reaction_network_payload(
                {"name": "missing_reaction_network"},
                strict=strict,
                require_affinity=require_affinity,
            )
        ]
    if isinstance(payloads, Mapping):
        return [validate_reaction_network_payload(
            payloads, strict=strict, require_affinity=require_affinity,
        )]
    return [
        validate_reaction_network_payload(
            {"name": "invalid_reaction_network_payload"},
            strict=strict,
            require_affinity=require_affinity,
        )
    ]


__all__ = ["validate_reaction_network_payload", "validate_reaction_network_structure"]
