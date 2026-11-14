"""Mass-conservative anthropogenic emission sector allocation.

This module owns one algebraic allocation map. It consumes the six canonical
inventory mass-flux ports and splits each flux over a fixed sector axis without
changing the flux rate, spatial shape, or per-cell species mass.

References:
    - Crippa et al. (2020), Scientific Data 7, 121.
    - McDuffie et al. (2020), Earth System Science Data 12, 3413-3442.
    - CMAQ User Guide (2019), SMOKE emissions preprocessing.

Mathematical expression:
    a[s,k,x] = f_sector[s] f_species[s,k] P[s,x]
    W[s,k,x] = a[s,k,x] / sum_s a[s,k,x]
    E_sector[s,k,x] = W[s,k,x] E_input[k,x]
    sum_s E_sector[s,k,x] = E_input[k,x]

Parameter meanings:
    f_sector is a sector prior [1].
    f_species is a sector-species support coefficient [1].
    P is a dimensionless proxy normalized over the spatial domain.
    E_input and E_sector are mass fluxes [kg m-2 s-1].

Open-system route:
    The module is an algebraic internal split. It creates no source or sink;
    all input species mass remains in the sector_emission_flux output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from relic.core.state import StateWriteMode, UpdateItem, UpdateMode


class SectorAllocation(ScaleCouplingFunctor):
    """Split canonical inventory fluxes over a fixed sector/species tensor."""

    STRUCTURAL_SECTORS = (
        "power",
        "transport",
        "industry",
        "buildings",
        "agriculture",
    )
    CANONICAL_SPECIES = ("nox", "co", "so2", "voc", "nh3", "pm25")
    CANONICAL_INPUTS = (
        "anthropogenic_nox_flux",
        "anthropogenic_co_flux",
        "anthropogenic_so2_flux",
        "anthropogenic_voc_flux",
        "anthropogenic_nh3_flux",
        "anthropogenic_pm25_flux",
    )
    PROXY_INPUTS = (
        "population_density",
        "gdp_density",
        "road_density",
        "land_use_fractions",
    )
    SPECIES_PARAMETER_KEYS = {
        ("power", "nox"): "power_sector.species_fractions.nox",
        ("power", "so2"): "power_sector.species_fractions.so2",
        ("power", "pm25"): "power_sector.species_fractions.pm25",
        ("transport", "nox"): "transport_sector.species_fractions.nox",
        ("transport", "co"): "transport_sector.species_fractions.co",
        ("transport", "pm25"): "transport_sector.species_fractions.pm25",
        ("industry", "nox"): "industry_sector.species_fractions.nox",
        ("industry", "so2"): "industry_sector.species_fractions.so2",
        ("industry", "voc"): "industry_sector.species_fractions.nmvoc",
        ("buildings", "nox"): "buildings_sector.species_fractions.nox",
        ("buildings", "pm25"): "buildings_sector.species_fractions.pm25",
        ("agriculture", "nh3"): "agriculture_sector.species_fractions.nh3",
    }

    def __init__(
        self,
        name: str = "sector_allocation",
        config: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError(
                "SectorAllocation requires a string name; pass configuration via config=."
            )
        if isinstance(config, dict) and config.get("parameters"):
            raise ValueError(
                "SectorAllocation scientific parameters must come from ParameterManager YAML."
            )
        metadata = FunctorMetadata(
            name=name,
            version="3.0.0",
            description=(
                "Pointwise mass-conservative split of canonical inventory fluxes "
                "over fixed sector and species axes"
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.PREPROCESSING,
            mathematical_expression=(
                "a_s,k = f_s q_s,k P_s; W_s,k = a_s,k / sum_s a_s,k; "
                "E_s,k = W_s,k E_k; sum_s E_s,k = E_k"
            ),
            references=[
                "Crippa et al. (2020) DOI:10.1038/s41597-020-0462-2",
                "McDuffie et al. (2020) DOI:10.5194/essd-12-3413-2020",
                "CMAQ User Guide (2019), SMOKE spatial allocation",
            ],
            parameters={
                "anthroposphere.emissions.sector_fractions.power": (
                    "Power-sector fraction of total anthropogenic emissions [dimensionless]"
                ),
                "anthroposphere.emissions.sector_fractions.transport": (
                    "Transport-sector fraction of total anthropogenic emissions [dimensionless]"
                ),
                "anthroposphere.emissions.sector_fractions.industry": (
                    "Industry-sector fraction of total anthropogenic emissions [dimensionless]"
                ),
                "anthroposphere.emissions.sector_fractions.buildings": (
                    "Buildings-sector fraction of total anthropogenic emissions [dimensionless]"
                ),
                "anthroposphere.emissions.sector_fractions.agriculture": (
                    "Agriculture-sector fraction of total anthropogenic emissions [dimensionless]"
                ),
                "anthroposphere.emissions.power_sector.species_fractions.nox": (
                    "Fraction of NOx mass within power-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.power_sector.species_fractions.so2": (
                    "Fraction of SO2 mass within power-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.power_sector.species_fractions.pm25": (
                    "Fraction of PM2.5 mass within power-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.transport_sector.species_fractions.nox": (
                    "Fraction of NOx mass for transport-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.transport_sector.species_fractions.co": (
                    "Fraction of CO mass for transport-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.transport_sector.species_fractions.pm25": (
                    "Fraction of PM2.5 mass for transport-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.industry_sector.species_fractions.nox": (
                    "Fraction of NOx mass for industry-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.industry_sector.species_fractions.so2": (
                    "Fraction of SO2 mass for industry-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.industry_sector.species_fractions.nmvoc": (
                    "Fraction of NMVOC mass for industry-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.buildings_sector.species_fractions.nox": (
                    "Fraction of NOx mass for buildings-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.buildings_sector.species_fractions.pm25": (
                    "Fraction of PM2.5 mass for buildings-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.agriculture_sector.species_fractions.nh3": (
                    "Fraction of NH3 mass for agriculture-sector emissions [dimensionless]"
                ),
                "anthroposphere.emissions.sector_allocation.proxy_floor": (
                    "Positive floor of the spatial allocation proxy [dimensionless]"
                ),
                "anthroposphere.emissions.sector_allocation.cropland_class_index": (
                    "Index of the cropland class on the land-use axis [dimensionless]"
                ),
                "anthroposphere.emissions.conservation_tolerance": (
                    "Maximum relative residual of the allocated emission mass [dimensionless]"
                ),
            },
            input_variables=[
                "anthropogenic_nox_flux",
                "anthropogenic_co_flux",
                "anthropogenic_so2_flux",
                "anthropogenic_voc_flux",
                "anthropogenic_nh3_flux",
                "anthropogenic_pm25_flux",
                "population_density",
                "gdp_density",
                "road_density",
                "land_use_fractions",
            ],
            output_variables=[
                "sector_emission_flux",
                "diagnostics.sector_allocation.allocation_weights",
                "diagnostics.sector_allocation.spatial_distribution",
                "diagnostics.sector_allocation.temporal_factors",
                "diagnostics.sector_allocation.mass_residual",
            ],
            conservation_laws=["mass"],
            spatial_dimensions=2,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            coupling_interfaces=["inventory", "atmosphere"],
        )
        # Scale coupling: inventory species totals are split into sector components
        # on the same grid cell (E_sector = W E_input with sum_s W = 1), a category
        # hierarchy rather than a transfer between grid supports.
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
            source_scale="inventory_grid",
            target_scale="sector_species_grid",
            coupling_type="hierarchical",
        )

    @staticmethod
    def _validate_runtime_tensor(
        value: Any,
        key: str,
        *,
        min_ndim: int,
        ndim_message: str | None = None,
    ) -> Tensor:
        """Validate a runtime tensor without coercing its dtype, device, or graph."""

        if not isinstance(value, Tensor):
            raise TypeError(f"{key} must be a torch.Tensor.")
        if value.device.type == "meta":
            raise TypeError(f"{key} must be materialized, not meta.")
        if (
            value.layout != torch.strided
            or value.is_quantized
            or bool(getattr(value, "is_nested", False))
        ):
            raise TypeError(f"{key} must use a dense strided Tensor layout.")
        if value.dtype not in {torch.float32, torch.float64}:
            raise TypeError(f"{key} must use float32 or float64.")
        if value.ndim < min_ndim:
            raise ValueError(
                ndim_message
                or f"{key} must expose at least {min_ndim} semantic axes."
            )
        if any(size <= 0 for size in value.shape):
            raise ValueError(f"{key} must have non-empty semantic axes.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} must be finite.")
        return value

    @classmethod
    def _continuous_parameter_tensor(
        cls,
        value: Any,
        key: str,
        reference: Tensor,
    ) -> Tensor:
        """Project one validated continuous parameter while preserving autograd."""

        if isinstance(value, Tensor):
            cls._validate_runtime_tensor(value, f"Parameter '{key}'", min_ndim=0)
            if value.numel() != 1:
                raise ValueError(f"Parameter '{key}' must be scalar.")
            converted = value.reshape(()).to(
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise ValueError(f"Parameter '{key}' must be a finite real scalar.")
            converted = reference.new_tensor(value)
        if not bool(torch.isfinite(converted)):
            raise ValueError(f"Parameter '{key}' must remain finite after projection.")
        return converted

    @staticmethod
    def _read_parameter_scalar(
        value: Any,
        key: str,
        *,
        integral: bool,
    ) -> int | float:
        """Read one discrete or audit-only parameter with exact scalar semantics."""

        if isinstance(value, Tensor):
            if value.device.type == "meta":
                raise ValueError(f"Parameter '{key}' must be materialized.")
            if (
                value.layout != torch.strided
                or value.is_quantized
                or bool(getattr(value, "is_nested", False))
                or value.numel() != 1
            ):
                raise ValueError(f"Parameter '{key}' must be a dense scalar Tensor.")
            integral_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            real_dtypes = integral_dtypes | {torch.float32, torch.float64}
            allowed = integral_dtypes if integral else real_dtypes
            if value.dtype not in allowed:
                expected = "integral" if integral else "real"
                raise ValueError(f"Parameter '{key}' must be an exact {expected} scalar.")
            scalar = value.detach().reshape(()).cpu().item()
        else:
            allowed_types = {int} if integral else {int, float}
            if type(value) not in allowed_types:
                expected = "integer" if integral else "real"
                raise ValueError(f"Parameter '{key}' must be an exact {expected} scalar.")
            scalar = value
        if not math.isfinite(float(scalar)):
            raise ValueError(f"Parameter '{key}' must be finite.")
        return int(scalar) if integral else float(scalar)

    def _sector_fraction_vector(self, reference: Tensor) -> Tensor:
        """Load positive sector priors through the canonical Functor API."""
        values = []
        for sector in self.STRUCTURAL_SECTORS:
            key = f"sector_fractions.{sector}"
            raw = self.get_parameter_value(
                key,
                namespace="anthroposphere.emissions",
                source_order=("manager",),
                required=True,
            )
            value = self._continuous_parameter_tensor(raw, key, reference)
            if bool(value < 0.0):
                raise ValueError(f"Parameter '{key}' must be non-negative.")
            values.append(value)
        vector = torch.stack(values)
        if not bool(torch.isfinite(vector).all()) or bool(torch.all(vector <= 0.0)):
            raise ValueError("Sector fractions require finite positive support.")
        return vector

    def _species_fraction_matrix(self, reference: Tensor) -> Tensor:
        """Load the declared sector-species support matrix."""
        rows = []
        for sector in self.STRUCTURAL_SECTORS:
            row = []
            for species in self.CANONICAL_SPECIES:
                key = self.SPECIES_PARAMETER_KEYS.get((sector, species))
                if key is None:
                    value = reference.new_zeros(())
                else:
                    raw = self.get_parameter_value(
                        key,
                        namespace="anthroposphere.emissions",
                        source_order=("manager",),
                        required=True,
                    )
                    value = self._continuous_parameter_tensor(raw, key, reference)
                    if bool(value < 0.0):
                        raise ValueError(f"Parameter '{key}' must be non-negative.")
                row.append(value)
            rows.append(torch.stack(row))
        matrix = torch.stack(rows)
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("Sector-species parameters must be finite.")
        support = matrix.sum(dim=0)
        missing = [
            self.CANONICAL_SPECIES[index]
            for index in torch.nonzero(support <= 0.0).detach().cpu().flatten().tolist()
        ]
        if missing:
            raise ValueError(f"Sector-species parameters lack support for: {missing}")
        return matrix

    def _canonical_flux_stack(self, state: Mapping[str, Any]) -> Tensor:
        """Validate and stack canonical species fluxes as [species, ...]."""
        missing = [key for key in self.CANONICAL_INPUTS if key not in state]
        if missing:
            raise KeyError(f"Missing canonical inventory fluxes: {missing}")

        tensors = []
        reference: Tensor | None = None
        for key in self.CANONICAL_INPUTS:
            value = self._validate_runtime_tensor(
                state[key],
                f"State input '{key}'",
                min_ndim=2,
                ndim_message=(
                    f"State input '{key}' must expose trailing 2D spatial axes."
                ),
            )
            if bool(torch.any(value < 0.0)):
                raise ValueError(f"State input '{key}' must be non-negative.")
            if reference is None:
                reference = value
            elif (
                value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.device != reference.device
            ):
                raise ValueError(
                    "All canonical inventory fluxes must share shape, dtype, and device."
                )
            tensors.append(value)
        return torch.stack(tensors, dim=0)

    def _require_proxy(
        self,
        state: Mapping[str, Any],
        key: str,
        reference: Tensor,
    ) -> Tensor:
        value = self._validate_runtime_tensor(
            state.get(key),
            f"State proxy '{key}'",
            min_ndim=2,
            ndim_message=f"State proxy '{key}' shape must match the flux shape.",
        )
        if value.shape != reference.shape:
            raise ValueError(
                f"State proxy '{key}' shape {tuple(value.shape)} must match "
                f"flux shape {tuple(reference.shape)}."
            )
        if value.dtype != reference.dtype:
            raise ValueError(f"State proxy '{key}' dtype must match the flux dtype.")
        if value.device != reference.device:
            raise ValueError(f"State proxy '{key}' device must match the flux device.")
        if bool(torch.any(value < 0.0)):
            raise ValueError(f"State proxy '{key}' must be non-negative.")
        return value

    def _cropland_proxy(self, state: Mapping[str, Any], reference: Tensor) -> Tensor:
        land_use = self._validate_runtime_tensor(
            state.get("land_use_fractions"),
            "State proxy 'land_use_fractions'",
            min_ndim=3,
            ndim_message=(
                "land_use_fractions must add exactly one class axis before the "
                "trailing spatial axes."
            ),
        )
        if land_use.ndim != reference.ndim + 1:
            raise ValueError(
                "land_use_fractions must add exactly one class axis before the "
                "trailing spatial axes."
            )
        class_axis = land_use.ndim - 3
        expected_leading = reference.shape[:-2]
        if land_use.shape[:class_axis] != expected_leading:
            raise ValueError("land_use_fractions leading axes must match the flux.")
        if land_use.shape[-2:] != reference.shape[-2:]:
            raise ValueError("land_use_fractions spatial axes must match the flux.")
        if land_use.dtype != reference.dtype:
            raise ValueError("land_use_fractions dtype must match the flux dtype.")
        if land_use.device != reference.device:
            raise ValueError("land_use_fractions device must match the flux device.")
        if bool(torch.any(land_use < 0.0)):
            raise ValueError("land_use_fractions must be non-negative.")

        key = "sector_allocation.cropland_class_index"
        raw_index = self.get_parameter_value(
            key,
            namespace="anthroposphere.emissions",
            source_order=("manager",),
            required=True,
        )
        class_index = int(
            self._read_parameter_scalar(raw_index, key, integral=True)
        )
        if class_index < 0 or class_index >= land_use.shape[class_axis]:
            raise ValueError(
                f"Cropland class index {class_index} is outside the land-use class axis."
            )
        proxy = land_use.select(dim=class_axis, index=class_index)
        return proxy

    def _normalize_proxy(self, proxy: Tensor, floor: Tensor, name: str) -> Tensor:
        """Normalize only trailing spatial axes without mixing batches."""
        spatial_mean = proxy.mean(dim=(-2, -1), keepdim=True)
        if not bool(torch.isfinite(spatial_mean).all()):
            raise ValueError(f"Spatial proxy '{name}' mean must be finite.")
        has_support = spatial_mean > 0.0
        safe_mean = torch.where(has_support, spatial_mean, torch.ones_like(spatial_mean))
        normalized = proxy / safe_mean
        stabilized = (normalized + floor) / (1.0 + floor)
        result = torch.where(has_support, stabilized, torch.zeros_like(stabilized))
        if not bool(torch.isfinite(result).all()) or bool(torch.any(result < 0.0)):
            raise ValueError(f"Spatial proxy '{name}' normalization is invalid.")
        return result

    def _spatial_proxy_stack(self, state: Mapping[str, Any], reference: Tensor) -> Tensor:
        """Return normalized proxies as [sector, ...source_shape]."""
        floor_key = "sector_allocation.proxy_floor"
        raw_floor = self.get_parameter_value(
            floor_key,
            namespace="anthroposphere.emissions",
            source_order=("manager",),
            required=True,
        )
        floor = self._continuous_parameter_tensor(raw_floor, floor_key, reference)
        if bool(floor <= 0.0):
            raise ValueError(f"Parameter '{floor_key}' must be positive.")

        population = self._require_proxy(state, "population_density", reference)
        gdp = self._require_proxy(state, "gdp_density", reference)
        roads = self._require_proxy(state, "road_density", reference)
        cropland = self._cropland_proxy(state, reference)
        raw_proxies = (gdp, roads, gdp, population, cropland)
        proxies = torch.stack(
            [
                self._normalize_proxy(proxy, floor, sector)
                for sector, proxy in zip(
                    self.STRUCTURAL_SECTORS,
                    raw_proxies,
                    strict=True,
                )
            ],
            dim=0,
        )
        if not bool(torch.isfinite(proxies).all()) or bool(torch.any(proxies < 0.0)):
            raise ValueError("Derived sector proxy stack must be finite and non-negative.")
        return proxies

    def _allocation_weights(
        self,
        state: Mapping[str, Any],
        fluxes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build pointwise weights normalized strictly along the sector axis."""
        reference = fluxes[0]
        sector_fractions = self._sector_fraction_vector(reference)
        species_fractions = self._species_fraction_matrix(reference)
        proxies = self._spatial_proxy_stack(state, reference)

        # References: Crippa et al. (2020); EPA SMOKE spatial surrogate guidance.
        # Mathematical expression: W[s,k,x] = a[s,k,x] / sum_s a[s,k,x].
        # Parameter meanings and units: f_sector, f_species, P, and W are unitless.
        # Domain and sign: a >= 0, denominator > 0, and sum_s W = 1.
        trailing = (1,) * reference.ndim
        sector_term = sector_fractions.view(
            len(self.STRUCTURAL_SECTORS), 1, *trailing
        )
        species_term = species_fractions.view(
            len(self.STRUCTURAL_SECTORS),
            len(self.CANONICAL_SPECIES),
            *trailing,
        )
        proxy_term = proxies.unsqueeze(1)
        raw_weights = sector_term * species_term * proxy_term
        if not bool(torch.isfinite(raw_weights).all()):
            raise ValueError("Derived raw allocation weights must be finite.")
        if bool(torch.any(raw_weights < 0.0)):
            raise ValueError("Derived raw allocation weights must be non-negative.")
        denominator = raw_weights.sum(dim=0, keepdim=True)
        if not bool(torch.isfinite(denominator).all()):
            raise ValueError("Allocation weight denominator must be finite.")
        if bool(torch.any(denominator <= 0.0)):
            raise ValueError("Allocation weights lack positive support for a species or cell.")
        weights = raw_weights / denominator
        if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0.0)):
            raise ValueError("Allocation weights must be finite and non-negative.")
        normalization_error = torch.abs(weights.sum(dim=0) - 1.0)
        normalization_limit = (
            8.0
            * len(self.STRUCTURAL_SECTORS)
            * torch.finfo(weights.dtype).eps
        )
        if bool(torch.any(normalization_error > normalization_limit)):
            raise RuntimeError("Allocation weights do not sum to one along sectors.")
        return weights, proxies

    @staticmethod
    def _pointwise_mass_audit(
        fluxes: Tensor,
        sector_flux: Tensor,
    ) -> tuple[Tensor, float]:
        """Return the full pointwise residual and a dimensionless maximum error."""

        residual = fluxes - sector_flux.sum(dim=0)
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("Sector allocation mass residual must be finite.")
        absolute_residual = torch.abs(residual)
        nonzero_input = torch.abs(fluxes) > 0.0
        zero_input_error = bool(
            torch.any((~nonzero_input) & (absolute_residual != 0.0))
        )
        if zero_input_error:
            return residual, math.inf
        if bool(torch.any(nonzero_input)):
            relative = absolute_residual[nonzero_input] / torch.abs(
                fluxes[nonzero_input]
            )
            max_error = float(relative.detach().max().cpu().item())
        else:
            max_error = 0.0
        if not math.isfinite(max_error):
            raise ValueError("Sector allocation relative mass error must be finite.")
        return residual, max_error

    def forward(
        self,
        state: dict[str, Tensor],
        time: float,
        dt: float,
    ) -> dict[str, Any]:
        """Apply the single mass-conservative sector allocation map.

        References:
            Crippa et al. (2020); CMAQ/SMOKE spatial allocation guidance.
        Mathematical expression:
            E_sector[s,k,x] = W[s,k,x] E_input[k,x],
            sum_s W[s,k,x] = 1.
        Parameter meanings:
            Sector/species coefficients and proxy regularization are declared in
            ParameterManager YAML. Fluxes use kg m-2 s-1; weights are unitless.
        Open/closed route:
            This is a closed algebraic split. No mass leaves the declared
            sector_emission_flux output and no residual is discarded.
        """
        # References: Crippa et al. (2020); CMAQ/SMOKE allocation guidance.
        # Mathematical expression: E_sector[s,k,x] = W[s,k,x] E_input[k,x].
        # Parameter meanings: W is unitless and E uses kg m-2 s-1.
        _ = time, dt
        fluxes = self._canonical_flux_stack(state)
        weights, proxies = self._allocation_weights(state, fluxes)
        sector_flux = weights * fluxes.unsqueeze(0)
        if not bool(torch.isfinite(sector_flux).all()):
            raise ValueError("Derived sector emission flux must be finite.")
        if bool(torch.any(sector_flux < 0.0)):
            raise ValueError("Derived sector emission flux must be non-negative.")
        residual, max_error = self._pointwise_mass_audit(fluxes, sector_flux)

        tolerance_key = "conservation_tolerance"
        raw_tolerance = self.get_parameter_value(
            tolerance_key,
            namespace="anthroposphere.emissions",
            source_order=("manager",),
            required=True,
        )
        tolerance = self._read_parameter_scalar(
            raw_tolerance,
            tolerance_key,
            integral=False,
        )
        if tolerance < 0.0:
            raise ValueError("Parameter 'conservation_tolerance' must be non-negative.")
        if max_error > tolerance:
            raise RuntimeError(
                f"Sector allocation mass residual {max_error:.6e} exceeds {tolerance:.6e}."
            )

        temporal_factors = fluxes.new_ones((len(self.STRUCTURAL_SECTORS),))
        return {
            "sector_emission_flux": UpdateItem(
                value=sector_flux,
                mode=UpdateMode.REPLACE,
                write_mode=StateWriteMode.DERIVED,
                unit="kg m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.sector_allocation.allocation_weights": weights,
            "diagnostics.sector_allocation.spatial_distribution": proxies,
            "diagnostics.sector_allocation.temporal_factors": temporal_factors,
            "diagnostics.sector_allocation.mass_residual": residual,
        }

    def required_parameters(self) -> tuple[str, ...]:
        """Declare every ParameterManager key used by the canonical runtime."""
        sector_keys = tuple(
            f"anthroposphere.emissions.sector_fractions.{sector}"
            for sector in self.STRUCTURAL_SECTORS
        )
        species_keys = tuple(
            f"anthroposphere.emissions.{key}"
            for key in self.SPECIES_PARAMETER_KEYS.values()
        )
        return (
            *sector_keys,
            *species_keys,
            "anthroposphere.emissions.sector_allocation.proxy_floor",
            "anthroposphere.emissions.sector_allocation.cropland_class_index",
            "anthroposphere.emissions.conservation_tolerance",
        )

    def upscale(self, fine_data: Tensor) -> Tensor:
        """Aggregate a declared [sector, species, ...] tensor over sectors."""
        fine_data = self._validate_runtime_tensor(
            fine_data,
            "fine_data",
            min_ndim=4,
        )
        if (
            fine_data.shape[0] != len(self.STRUCTURAL_SECTORS)
            or fine_data.shape[1] != len(self.CANONICAL_SPECIES)
        ):
            raise ValueError(
                "fine_data must use the canonical sector axis and species axis."
            )
        if bool(torch.any(fine_data < 0.0)):
            raise ValueError("fine_data must be non-negative.")
        return fine_data.sum(dim=0)

    def downscale(self, coarse_data: Tensor) -> Tensor:
        """Reject proxy-free downscaling; canonical allocation requires state proxies."""
        _ = coarse_data
        raise RuntimeError(
            "SectorAllocation.downscale requires declared proxy state; use forward(...)."
        )

    def check_conservation_laws(
        self,
        input_state: dict[str, Tensor],
        output_updates: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate per-cell, per-species mass conservation from current payloads."""
        fluxes = self._canonical_flux_stack(input_state)
        item = output_updates.get("sector_emission_flux")
        if not isinstance(item, UpdateItem) or not isinstance(item.value, Tensor):
            raise KeyError("sector_emission_flux UpdateItem is required for conservation.")
        update_contract_ok = (
            item.mode is UpdateMode.REPLACE
            and item.write_mode is StateWriteMode.DERIVED
            and item.unit == "kg m-2 s-1"
            and item.basis == "per_second"
            and item.per_timestep is False
            and item.source_functor == self.name
        )
        if not update_contract_ok:
            raise ValueError("sector_emission_flux UpdateItem contract is invalid.")
        sector_flux = self._validate_runtime_tensor(
            item.value,
            "sector_emission_flux",
            min_ndim=4,
        )
        expected_shape = (
            len(self.STRUCTURAL_SECTORS),
            len(self.CANONICAL_SPECIES),
            *fluxes.shape[1:],
        )
        if tuple(sector_flux.shape) != expected_shape:
            raise ValueError(
                f"sector_emission_flux shape {tuple(sector_flux.shape)} does not "
                f"match {expected_shape}."
            )
        if sector_flux.dtype != fluxes.dtype:
            raise ValueError("sector_emission_flux dtype must match the input flux dtype.")
        if sector_flux.device != fluxes.device:
            raise ValueError("sector_emission_flux device must match the input flux device.")
        if bool(torch.any(sector_flux < 0.0)):
            raise ValueError("sector_emission_flux must be non-negative.")
        _, max_error = self._pointwise_mass_audit(fluxes, sector_flux)
        return {"mass": max_error}


__all__ = ["SectorAllocation"]
