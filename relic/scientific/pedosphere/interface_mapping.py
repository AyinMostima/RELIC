"""
Interface Mapping for Pedosphere-Hydrosphere Coupling

This module provides mapping functions to ensure consistent variable naming
and unit conversions between pedosphere and hydrosphere modules.

The primary purpose is to map between:
- Hydrosphere outputs (drainage_flux, surface_runoff_flux, subsurface_runoff_flux)
- Pedosphere inputs (leaching_flux, water_flux)

Units:
- Water fluxes: kg m^-2 s^-1 (equivalent to mm/s for water)
- Carbon fluxes: g C m-2 s^-1
- Nitrogen fluxes: g N m-2 s^-1
"""

import torch
from typing import Any, Dict, Optional
import logging
from ...core.functor import ScaleCouplingFunctor
from ...core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from ...core.state import UpdateItem, UpdateMode
from ...data.standards.unit_converter import unit_converter
from ...scientific.pedosphere.api_contract import PedosphereContractMixin
from ...utils.physical_constants import get_physical_constant

# Parameter management imports
# Type aliases
StateDict = Dict[str, torch.Tensor]
logger = logging.getLogger(__name__)


class PedosphereInterfaceMapper(PedosphereContractMixin, ScaleCouplingFunctor):
    """
    Maps between pedosphere internal names and StateManager/external module names.

    This ensures consistent coupling with hydrosphere and other modules.

    Mathematical Expression:
        Interface mapping for scale coupling between hydrosphere and pedosphere
        Port chain map: y_h = C_h x_h, with C_h = I_3 for the three
        unit-preserving hydrological flux projections.

    References:
        - Clark et al. (2011), GMD 4, 677-699, doi:10.5194/gmd-4-677-2011
        - Clark et al. (2011), GMD 4, 701-722, doi:10.5194/gmd-4-701-2011
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'


    def __init__(self, name: str = "pedosphere_interface_mapper", **kwargs):
        """Initialize the pedosphere interface mapper."""
        parameter_manager = kwargs.pop("parameter_manager", None)
        self._upscale_bridge = kwargs.pop("upscale_bridge", None)
        self._downscale_bridge = kwargs.pop("downscale_bridge", None)
        if self._upscale_bridge is not None and not callable(self._upscale_bridge):
            raise TypeError("upscale_bridge must be callable")
        if self._downscale_bridge is not None and not callable(self._downscale_bridge):
            raise TypeError("downscale_bridge must be callable")
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Interface mapper for hydrosphere-pedosphere coupling with consistent variable naming and unit conversions",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression="y_h = C_h x_h; C_h = I_3 for unit-preserving water-flux port projection",
            references=[
                "Clark et al. (2011) doi:10.5194/gmd-4-677-2011",
                "Clark et al. (2011) doi:10.5194/gmd-4-701-2011",
            ],
            parameters={
                "pedosphere.nutrient_cycling.n_noy_fraction": (
                    "Fraction of N deposition as NOy [dimensionless]"
                ),
                "pedosphere.nutrient_cycling.n_nhx_fraction": (
                    "Fraction of N deposition as NHx [dimensionless]"
                ),
                "pedosphere.phosphorus.p_po4_fraction": (
                    "Fraction of P deposition as PO4 [dimensionless]"
                ),
                "pedosphere.phosphorus.p_dust_fraction": (
                    "Fraction of P deposition as dust-P [dimensionless]"
                ),
                "pedosphere.transport.nitrate_distribution_coefficient": (
                    "Nitrate sorption distribution coefficient K_d [L kg-1]"
                ),
                "pedosphere.transport.doc_distribution_coefficient": (
                    "Dissolved organic carbon sorption distribution coefficient K_d [L kg-1]"
                ),
                "pedosphere.numerical.epsilon_concentration": (
                    "Concentration floor of the division guards [dimensionless]"
                ),
                "pedosphere.numerical.mass_balance_tolerance": (
                    "Relative tolerance of the mass balance checks [dimensionless]"
                ),
            },
            input_variables=["drainage_flux", "subsurface_runoff_flux", "surface_runoff_flux"],
            output_variables=["leaching_flux", "lateral_flow", "surface_loss"], 
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(
            name=name,
            source_scale="hydrosphere",
            target_scale="pedosphere",
            coupling_type="spatial",  # Spatial coupling between compartments
            parameter_manager=parameter_manager,
            metadata=metadata,
            **kwargs
        )
        self.seconds_per_year = float(
            get_physical_constant(
                "relic.physical.constants.time.seconds_per_year"
            )
        )
        self.year_to_second = 1.0 / self.seconds_per_year
        self.water_density = float(
            get_physical_constant(
                "relic.physical.constants.water.density_liquid"
            )
        )

    def forward(self, state: StateDict, time: float, dt: torch.Tensor | float, **kwargs) -> StateDict:
        """
        Apply the declared hydrosphere-to-pedosphere water-flux port map.

        The three inputs and outputs are instantaneous fluxes in identical
        units, so the runtime returns REPLACE updates rather than state
        increments.
        """
        # References:
        # - Clark et al. (2011), doi:10.5194/gmd-4-677-2011.
        # Mathematical expression:
        # - y_h = C_h x_h, C_h = I_3 for the declared port ordering.
        # Parameter meanings:
        # - x_h and y_h are water mass fluxes [kg m-2 s-1].
        required_inputs = (
            "drainage_flux",
            "subsurface_runoff_flux",
            "surface_runoff_flux",
        )
        missing = [key for key in required_inputs if key not in state]
        if missing:
            raise ValueError(f"Missing required coupling inputs: {missing}")
        for key in required_inputs:
            value = state[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{key} must be torch.Tensor")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{key} must be finite")
            if bool(torch.any(value < 0.0)):
                raise ValueError(f"{key} must be non-negative")

        def _port(value: torch.Tensor) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit="kg m-2 s-1",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            )

        return {
            "leaching_flux": _port(state["drainage_flux"]),
            "lateral_flow": _port(state["subsurface_runoff_flux"]),
            "surface_loss": _port(state["surface_runoff_flux"]),
        }

    def map_atmosphere_to_pedosphere(self, state: StateDict) -> Dict[str, torch.Tensor]:
        """
        Map atmospheric deposition to pedosphere inputs.

        Converts atmospheric deposition fluxes to soil inputs with proper
        unit conversion and species splitting based on deposition chemistry.

        Parameters:
        -----------
        state : StateDict
            Current state from StateManager containing atmospheric deposition

        Returns:
        --------
        Dict[str, torch.Tensor]
            Mapped atmospheric inputs for pedosphere [g element m^-2 s^-1]

        Units:
        ------
        Input deposition units: [g element m^-2 year^-1], any tensor shape (gridded
        fields are converted elementwise)
        Output soil input units: [g element m^-2 s^-1], same shape as the input

        Conversion factors draw from the unified parameter database:
        - Year to second scaling uses physical.constants.time.seconds_per_year
        - Mass/area conversions reuse entries from the conversions namespace

        Species splitting:
        - Nitrogen: NOy:NHx from pedosphere.nutrient_cycling (override: two-element
          state key deposition_n_split)
        - Phosphorus: PO4:Dust-P from pedosphere.phosphorus (override: two-element
          state key deposition_p_split)
        - Each pair must sum to one within validation.conservation.functor_tolerance

        Domain/constraints:
        - Non-finite or negative deposition inputs and inconsistent split overrides
          raise ValueError; inputs are never silently dropped.
        Sign convention:
        - Deposition is a positive input to the soil surface layer.

        References:
        ----------
        1. Dentener, F. et al. (2006). Nitrogen and sulfur deposition on regional scales
        2. Mahowald, N. et al. (2008). Global distribution of atmospheric phosphorus sources
        3. Vet, R. et al. (2014). A global assessment of precipitation chemistry
        """
        # References: Dentener et al. (2006); Mahowald et al. (2008); Vet et al. (2014).
        # Mathematical expression: F_x = D_x / T_year for x in {wet N, dry N, wet P, dry P};
        #   F_noy = f_noy F_N, F_nhx = f_nhx F_N; F_po4 = f_po4 F_P, F_dust = f_dust F_P.
        # Parameter meanings: D_x is the annual deposition field, T_year the SSOT seconds per
        #   year, f_* the species split fractions (ParameterManager defaults or the
        #   two-element deposition_n_split / deposition_p_split state overrides).
        # Units: D_x [g element m-2 year-1]; F_* [g element m-2 s-1]; f_* [1]; T_year [s year-1].
        # Domain/constraints: D_x is any finite non-negative tensor and is converted elementwise
        #   (gridded inputs are not dropped); f_noy + f_nhx = 1 and f_po4 + f_dust = 1 within
        #   validation.conservation.functor_tolerance; invalid inputs raise ValueError.
        # Sign convention: deposition is a positive input to the soil surface layer.
        mapped_inputs: Dict[str, torch.Tensor] = {}
        n_noy_fraction, n_nhx_fraction = self._resolve_deposition_split(
            state,
            "deposition_n_split",
            default_fractions=(
                self.get_parameter_value("n_noy_fraction", namespace="pedosphere.nutrient_cycling", required=True),
                self.get_parameter_value("n_nhx_fraction", namespace="pedosphere.nutrient_cycling", required=True),
            ),
        )
        p_po4_fraction, p_dust_fraction = self._resolve_deposition_split(
            state,
            "deposition_p_split",
            default_fractions=(
                self.get_parameter_value("p_po4_fraction", namespace="pedosphere.phosphorus", required=True),
                self.get_parameter_value("p_dust_fraction", namespace="pedosphere.phosphorus", required=True),
            ),
        )

        deposition_table = (
            ("wet_n_deposition", "wet_noy_deposition", n_noy_fraction, "wet_nhx_deposition", n_nhx_fraction, "total_wet_n_input"),
            ("dry_n_deposition", "dry_noy_deposition", n_noy_fraction, "dry_nhx_deposition", n_nhx_fraction, "total_dry_n_input"),
            ("wet_p_deposition", "wet_po4_deposition", p_po4_fraction, "wet_dust_p_deposition", p_dust_fraction, "total_wet_p_input"),
            ("dry_p_deposition", "dry_po4_deposition", p_po4_fraction, "dry_dust_p_deposition", p_dust_fraction, "total_dry_p_input"),
        )
        for state_key, first_key, first_fraction, second_key, second_fraction, total_key in deposition_table:
            if state_key not in state:
                continue
            rate = self._annual_deposition_rate(state[state_key], state_key)
            mapped_inputs[first_key] = rate * first_fraction
            mapped_inputs[second_key] = rate * second_fraction
            mapped_inputs[total_key] = rate

        # Calculate totals for mass balance checking
        total_n_deposition = None
        total_p_deposition = None
        for key, value in mapped_inputs.items():
            if '_n_input' in key:
                total_n_deposition = value if total_n_deposition is None else total_n_deposition + value
            elif '_p_input' in key:
                total_p_deposition = value if total_p_deposition is None else total_p_deposition + value

        if total_n_deposition is not None and torch.any(total_n_deposition > 0):
            mapped_inputs['total_atmospheric_n_input'] = total_n_deposition

        if total_p_deposition is not None and torch.any(total_p_deposition > 0):
            mapped_inputs['total_atmospheric_p_input'] = total_p_deposition

        # Add to soil surface layer inputs (assuming surface application)
        for key, value in list(mapped_inputs.items()):
            if 'deposition' in key:
                # Map to soil surface input
                soil_key = key.replace('deposition', 'soil_input')
                mapped_inputs[soil_key] = value

        return mapped_inputs

    def _annual_deposition_rate(self, value: Any, name: str) -> torch.Tensor:
        """Convert one annual deposition field to a per-second rate, elementwise.

        References:
          - Dentener et al. (2006), deposition fields on regional grids.
        Mathematical expression:
          F = D / T_year.
        Parameter meanings:
          D is the annual deposition field for variable `name`; T_year is the
          SSOT seconds per year.
        Units:
          D [g element m-2 year-1]; F [g element m-2 s-1].
        Domain/constraints:
          D must be a finite, non-negative tensor of any shape; a Python number
          is accepted as a scalar. Anything else raises ValueError.
        Sign convention:
          deposition is a positive input to the soil surface.
        """
        tensor = torch.as_tensor(value)
        if not torch.is_floating_point(tensor):
            tensor = tensor.to(torch.get_default_dtype())
        if not torch.all(torch.isfinite(tensor)):
            raise ValueError(f"Deposition input '{name}' must be finite.")
        if torch.any(tensor < 0.0):
            raise ValueError(f"Deposition input '{name}' must be non-negative.")
        return tensor * self.year_to_second

    def _resolve_deposition_split(
        self,
        state: StateDict,
        override_key: str,
        *,
        default_fractions: tuple[Any, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two species split fractions, validating an optional state override.

        References:
          - Dentener et al. (2006), NOy/NHx partition of nitrogen deposition.
          - Mahowald et al. (2008), soluble/mineral partition of phosphorus deposition.
        Mathematical expression:
          f_1 + f_2 = 1 within tol; f_i >= 0.
        Parameter meanings:
          f_1, f_2 are the species fractions from ParameterManager defaults or the
          two-element state override `override_key`; tol is
          validation.conservation.functor_tolerance.
        Units:
          f_i, tol [1].
        Domain/constraints:
          an override must hold exactly two finite non-negative values; the pair must
          close to one so the split conserves the deposited element mass.
        Sign convention:
          fractions are non-negative multipliers of a positive deposition rate.
        """
        if override_key in state:
            override = torch.as_tensor(state[override_key])
            if not torch.is_floating_point(override):
                override = override.to(torch.get_default_dtype())
            if override.numel() != 2:
                raise ValueError(
                    f"'{override_key}' must hold exactly two split fractions, got {override.numel()} values."
                )
            override = override.reshape(2)
            if not torch.all(torch.isfinite(override)) or torch.any(override < 0.0):
                raise ValueError(f"'{override_key}' fractions must be finite and non-negative.")
            first, second = override[0], override[1]
        else:
            first = torch.as_tensor(default_fractions[0])
            second = torch.as_tensor(default_fractions[1])
            if not torch.is_floating_point(first):
                first = first.to(torch.get_default_dtype())
            second = second.to(first.dtype)
        tolerance = float(
            self.get_parameter_value(
                "functor_tolerance",
                namespace="validation.conservation",
                required=True,
            )
        )
        closure = float((first + second).detach())
        if abs(closure - 1.0) > tolerance:
            raise ValueError(
                f"'{override_key}' fractions must sum to one within {tolerance}, got {closure}."
            )
        return first, second

    def linear_adsorption_retardation_term(
        self,
        distribution_coefficient: Any,
        bulk_density: Any,
        soil_water_content: torch.Tensor,
    ) -> torch.Tensor:
        """Return R - 1 for linear equilibrium adsorption of a solute in soil water.

        References:
          - Jury and Horton (2004), Soil Physics, retardation factor R = 1 + rho_b Kd / theta.
          - van Genuchten and Wierenga (1986), solute transport with linear sorption.
        Mathematical expression:
          R - 1 = rho_b * Kd * c_L / theta; mobile fraction = 1 / R.
        Parameter meanings:
          rho_b is soil bulk density, Kd the solute distribution coefficient in
          liters per kilogram, c_L the SSOT liters-to-cubic-meters conversion,
          theta the volumetric soil water content.
        Units:
          rho_b [kg m-3]; Kd [L kg-1]; c_L [m3 L-1]; theta [m3 m-3]; R - 1 [1].
        Domain/constraints:
          theta must be positive (the caller clamps it with the numerical epsilon);
          Kd and rho_b are non-negative, so R >= 1.
        Sign convention:
          larger values retard leaching; the mobile fraction 1/R multiplies a
          positive downward solute flux.
        """
        theta = torch.as_tensor(soil_water_content)
        if not torch.is_floating_point(theta):
            theta = theta.to(torch.get_default_dtype())
        if torch.any(theta <= 0.0):
            raise ValueError("Soil water content must be positive for the retardation term.")
        kd = torch.as_tensor(distribution_coefficient, device=theta.device, dtype=theta.dtype)
        rho_b = torch.as_tensor(bulk_density, device=theta.device, dtype=theta.dtype)
        if torch.any(kd < 0.0) or torch.any(rho_b < 0.0):
            raise ValueError("Distribution coefficient and bulk density must be non-negative.")
        liter_to_cubic_meter = torch.as_tensor(
            get_physical_constant("relic.physical.constants.conversions.liter_to_cubic_meter"),
            device=theta.device,
            dtype=theta.dtype,
        )
        return rho_b * kd * liter_to_cubic_meter / theta

    def calculate_solute_leaching(self, water_flux: torch.Tensor,
                                 concentration: torch.Tensor,
                                 adsorption_coefficient: Optional[float] = None) -> torch.Tensor:
        """
        Calculate solute leaching flux from water flux and concentration.

        Parameters:
        -----------
        water_flux : torch.Tensor
            Water drainage flux [kg m^-2 s^-1]
        concentration : torch.Tensor
            Solute concentration [g m^-3]
        adsorption_coefficient : Optional[float]
            Retardation term R - 1 = rho_b Kd c_L / theta [dimensionless]; see
            linear_adsorption_retardation_term. The mobile fraction is 1 / R.

        Returns:
        --------
        torch.Tensor
            Solute leaching flux [g m^-2 s^-1]

        Mathematical Expression:
        -----------------------
        F_solute = F_water * C_solute * (1 - f_ads)

        Where:
        - F_solute: solute flux [g m^-2 s^-1]
        - F_water: water flux [kg m^-2 s^-1] = [mm s-1]
        - C_solute: concentration [g m^-3] = [g L^-1]
        - f_ads: retardation term R - 1 [dimensionless], so (1 - f_ads) above is
          implemented as the mobile fraction 1 / (1 + f_ads)
        """
        # References: Hillel (1998) unit conversion for water flux.
        # Equation: q_m = F_water / rho_w.
        # Parameters: F_water [kg m-2 s-1], rho_w [kg m-3].
        water_flux_m_per_s = water_flux / self.water_density

        # Calculate solute flux
        solute_flux = water_flux_m_per_s * concentration

        # Apply adsorption if specified
        if adsorption_coefficient is not None:
            # Simple linear adsorption model
            mobile_fraction = 1.0 / (1.0 + adsorption_coefficient)
            solute_flux = solute_flux * mobile_fraction

        return solute_flux

    def ensure_mass_balance(self, pools_before: Dict[str, torch.Tensor],
                           pools_after: Dict[str, torch.Tensor],
                           fluxes: Dict[str, torch.Tensor],
                           dt: float,
                           tolerance: Optional[float] = None) -> Dict[str, float]:
        """
        Check mass balance for carbon and nitrogen pools.

        Parameters:
        -----------
        pools_before : Dict[str, torch.Tensor]
            Pool sizes before update [g m^-2]
        pools_after : Dict[str, torch.Tensor]
            Pool sizes after update [g m^-2]
        fluxes : Dict[str, torch.Tensor]
            All fluxes during time step [g m^-2 s^-1]
        dt : float
            Time step size [s]
        tolerance : Optional[float]
            Acceptable mass balance error. If None, uses pedosphere.numerical.mass_balance_tolerance

        Returns:
        --------
        Dict[str, float]
            Mass balance errors by element
        """
        errors = {}

        if tolerance is None:
            tolerance_value = self.get_parameter_value(
                "mass_balance_tolerance",
                namespace="pedosphere.numerical",
                source_order=("manager",),
            )
            if tolerance_value is None:
                raise KeyError("Missing required parameter 'pedosphere.numerical.mass_balance_tolerance'")
            if torch.is_tensor(tolerance_value):
                detached = tolerance_value.detach().reshape(-1)
                if detached.numel() == 0:
                    raise ValueError("mass_balance_tolerance tensor is empty")
                tolerance = detached[:1].to(dtype=torch.float64).tolist()[0]
            else:
                tolerance = float(tolerance_value)

        def _reference_tensor() -> torch.Tensor:
            for group in (pools_before, pools_after, fluxes):
                for value in group.values():
                    if torch.is_tensor(value):
                        return value
            return torch.as_tensor(0.0)

        reference = _reference_tensor()
        zero = torch.zeros((), device=reference.device, dtype=reference.dtype)

        def _scalar(value: Any) -> float:
            if torch.is_tensor(value):
                detached = value.detach().reshape(-1)
                if detached.numel() == 0:
                    raise ValueError("Cannot convert empty tensor to scalar.")
                return detached[:1].to(dtype=torch.float64).tolist()[0]
            return float(value)

        def _sum_selected(source: Dict[str, torch.Tensor], predicate) -> torch.Tensor:
            total = zero
            for key, value in source.items():
                if predicate(key):
                    item = value
                    if not torch.is_tensor(item):
                        item = torch.as_tensor(item, device=reference.device, dtype=reference.dtype)
                    total = total + item
            return total

        tolerance_tensor = torch.as_tensor(tolerance, device=reference.device, dtype=reference.dtype)
        dt_tensor = torch.as_tensor(dt, device=reference.device, dtype=reference.dtype)

        # Carbon balance
        carbon_pools_before = _sum_selected(pools_before, lambda k: "_c" in k.lower())
        carbon_pools_after = _sum_selected(pools_after, lambda k: "_c" in k.lower())
        carbon_fluxes = _sum_selected(
            fluxes,
            lambda k: any(x in k.lower() for x in ["respiration", "carbon", "_c"]),
        )

        carbon_change = (carbon_pools_after - carbon_pools_before).sum()
        carbon_flux_total = (carbon_fluxes * dt_tensor).sum()
        carbon_error = torch.abs(carbon_change + carbon_flux_total)

        if carbon_error > tolerance_tensor:
            errors["carbon_balance_error"] = _scalar(carbon_error)

        # Nitrogen balance
        nitrogen_pools_before = _sum_selected(pools_before, lambda k: "_n" in k.lower())
        nitrogen_pools_after = _sum_selected(pools_after, lambda k: "_n" in k.lower())
        nitrogen_fluxes = _sum_selected(
            fluxes,
            lambda k: any(x in k.lower() for x in ["nitrogen", "_n", "n2o", "nh4", "no3"]),
        )

        nitrogen_total_before = nitrogen_pools_before.sum()
        if _scalar(nitrogen_total_before) > 0:
            nitrogen_change = (nitrogen_pools_after - nitrogen_pools_before).sum()
            nitrogen_flux_total = (nitrogen_fluxes * dt_tensor).sum()
            nitrogen_error = torch.abs(nitrogen_change + nitrogen_flux_total)

            if nitrogen_error > tolerance_tensor:
                errors["nitrogen_balance_error"] = _scalar(nitrogen_error)

        return errors

    def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
        """
        Upscale fine-scale pedosphere data to coarse scale.

        Apply the explicitly configured fine-to-coarse chain map.
        """
        if self._upscale_bridge is None:
            raise RuntimeError("upscale requires an explicit typed chain-map bridge")
        return self._upscale_bridge(fine_data)

    def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
        """
        Downscale coarse-scale data to fine scale.

        Apply the explicitly configured coarse-to-fine chain map.
        """
        if self._downscale_bridge is None:
            raise RuntimeError("downscale requires an explicit typed chain-map bridge")
        return self._downscale_bridge(coarse_data)


class LitterfallInterfaceMapper(PedosphereContractMixin, ScaleCouplingFunctor):
    """Map plant litterfall boundary fluxes into canonical pedosphere ports.

    This is the unique biosphere-to-pedosphere litter chain map. It converts
    element mass units with the central ``UnitConverter`` and applies the
    ParameterManager-backed empirical litter-quality partition once, before a
    graph selects one mutually exclusive decomposition closure.
    """

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    REQUIRED_INPUT_KEYS = (
        "leaf_litterfall_carbon",
        "froot_litterfall_carbon",
        "leaf_litterfall_nitrogen",
        "froot_litterfall_nitrogen",
        "lignin_n_ratio",
    )
    OPTIONAL_GRAZING_INPUT_KEYS = (
        "dung_c_flux",
        "dung_n_flux",
    )
    INPUT_KEYS = REQUIRED_INPUT_KEYS + OPTIONAL_GRAZING_INPUT_KEYS
    OUTPUT_KEYS = (
        "metabolic_litter_input",
        "structural_litter_input",
        "organic_nitrogen_input",
    )
    PARAMETER_DESCRIPTIONS = {
        "pedosphere.decomposition.lignin_n_threshold": (
            "Lignin-to-nitrogen threshold for litter partitioning [dimensionless]"
        ),
        "pedosphere.decomposition.lignin_partitioning_coefficient": (
            "Empirical exponential litter-quality coefficient [dimensionless]"
        ),
        "pedosphere.decomposition.min_metabolic_fraction": (
            "Minimum metabolic litter fraction [dimensionless]"
        ),
        "pedosphere.decomposition.max_metabolic_fraction": (
            "Maximum metabolic litter fraction [dimensionless]"
        ),
        "pedosphere.decomposition.manure_metabolic_fraction": (
            "Metabolic fraction for the conservative two-pool manure projection "
            "[dimensionless]"
        ),
        "pedosphere.numerical.clamp_min": (
            "Lower bound for the non-negative lignin-quality argument [dimensionless]"
        ),
    }

    def __init__(
        self,
        name: str = "litterfall_interface_mapper",
        config: Optional[Dict[str, Any]] = None,
        parameter_manager: Optional[Any] = None,
    ) -> None:
        """Initialize the explicit litterfall C/N port map."""
        metadata = FunctorMetadata(
            name=name,
            version="1.1.0",
            description=(
                "Conservative biosphere/management-to-pedosphere litter and "
                "manure carbon-nitrogen chain map"
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression=(
                "F_C,g = 1000 * (F_leaf,C,kg + F_root,C,kg); "
                "f_met = clip(exp(-k_part * max(LN - LN0, 0)), f_min, f_max); "
                "F_dung,C,g = 1000 * F_dung,C,kg; "
                "F_met = f_met * F_C,g + f_man * F_dung,C,g; "
                "F_str = (1 - f_met) * F_C,g + "
                "(1 - f_man) * F_dung,C,g; "
                "F_N,g = 1000 * (F_leaf,N,kg + F_root,N,kg + F_dung,N,kg)"
            ),
            references=[
                "Parton et al. (1987) Soil Science Society of America Journal 51, 1173-1179",
                "Metherell et al. (1993) CENTURY Soil Organic Matter Model documentation",
                "Thornton et al. (2002) Agricultural and Forest Meteorology 113, 185-222",
                "Coleman and Jenkinson (1996) RothC-26.3 model guide",
            ],
            parameters=dict(self.PARAMETER_DESCRIPTIONS),
            input_variables=list(self.INPUT_KEYS),
            output_variables=list(self.OUTPUT_KEYS),
            conservation_laws=["carbon", "nitrogen"],
            spatial_dimensions=0,
            coupling_interfaces=[
                "phenology_litterfall_boundary",
                "grazing_dung_boundary",
                "pedosphere_litter_carbon_input",
                "pedosphere_litter_nitrogen_input",
            ],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
            source_scale="biosphere",
            target_scale="pedosphere",
            coupling_type="spatial",
            conservation_properties=["carbon", "nitrogen"],
            parameter_manager=parameter_manager,
        )
        self.metadata.parameters = dict(self.PARAMETER_DESCRIPTIONS)
        self._grazing_dung_selected = (
            "grazing_management" in self.get_dependencies()
        )
        self.optional_input_variables = (
            set()
            if self._grazing_dung_selected
            else set(self.OPTIONAL_GRAZING_INPUT_KEYS)
        )
        self.lignin_n_threshold = self._require_dimensionless_parameter(
            "lignin_n_threshold",
            namespace="pedosphere.decomposition",
        )
        self.lignin_partitioning_coefficient = self._require_dimensionless_parameter(
            "lignin_partitioning_coefficient",
            namespace="pedosphere.decomposition",
        )
        self.min_metabolic_fraction = self._require_dimensionless_parameter(
            "min_metabolic_fraction",
            namespace="pedosphere.decomposition",
        )
        self.max_metabolic_fraction = self._require_dimensionless_parameter(
            "max_metabolic_fraction",
            namespace="pedosphere.decomposition",
        )
        self.manure_metabolic_fraction = self._require_dimensionless_parameter(
            "manure_metabolic_fraction",
            namespace="pedosphere.decomposition",
        )
        self.clamp_min = self._require_dimensionless_parameter(
            "clamp_min",
            namespace="pedosphere.numerical",
        )
        if not (
            0.0 <= self.clamp_min
            and 0.0 <= self.min_metabolic_fraction
            <= self.max_metabolic_fraction
            <= 1.0
            and self.lignin_partitioning_coefficient >= 0.0
            and 0.0 <= self.manure_metabolic_fraction <= 1.0
        ):
            raise ValueError("Invalid litterfall interface partition parameters")

    def _require_dimensionless_parameter(self, name: str, *, namespace: str) -> float:
        """Return one required dimensionless mapping parameter."""
        value = self.get_parameter_value(
            name,
            namespace=namespace,
            units="dimensionless",
            validate=True,
            required=True,
        )
        return float(self._to_scalar(value))

    @staticmethod
    def _require_nonnegative_tensor(state: StateDict, key: str) -> torch.Tensor:
        """Return one required finite non-negative StateManager tensor."""
        if key not in state:
            raise KeyError(key)
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be torch.Tensor")
        if value.numel() == 0:
            raise ValueError(f"{key} must not be empty")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{key} must be finite")
        if bool(torch.any(value < 0.0)):
            raise ValueError(f"{key} must be non-negative")
        return value

    @staticmethod
    def _require_port_tensor(
        updates: Dict[str, Any],
        key: str,
    ) -> torch.Tensor:
        """Return one finite tensor carried by a canonical output port."""
        if key not in updates:
            raise KeyError(key)
        value = updates[key]
        if isinstance(value, UpdateItem):
            value = value.value
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must carry torch.Tensor")
        if value.numel() == 0:
            raise ValueError(f"{key} must not be empty")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{key} must be finite")
        return value

    def _get_optional_grazing_dung(
        self,
        state: StateDict,
        *,
        reference_shape: torch.Size,
        reference_device: torch.device,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return the explicit paired grazing dung boundary, when selected."""
        present = {key: key in state for key in self.OPTIONAL_GRAZING_INPUT_KEYS}
        if self._grazing_dung_selected:
            for key in self.OPTIONAL_GRAZING_INPUT_KEYS:
                if not present[key]:
                    raise KeyError(key)
        elif any(present.values()):
            raise ValueError(
                "grazing dung input requires the grazing_management dependency"
            )
        else:
            return None

        dung_carbon = self._require_nonnegative_tensor(state, "dung_c_flux")
        dung_nitrogen = self._require_nonnegative_tensor(state, "dung_n_flux")
        for key, value in (
            ("dung_c_flux", dung_carbon),
            ("dung_n_flux", dung_nitrogen),
        ):
            if value.shape != reference_shape:
                raise ValueError(f"{key} must match the litterfall flux shape")
            if value.device != reference_device:
                raise ValueError(f"{key} must use the litterfall flux device")
        return dung_carbon, dung_nitrogen

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: Dict[str, Any],
    ) -> Dict[str, float]:
        """Return element-wise residuals for the open litterfall chain map."""
        # References:
        # - Parton et al. (1987); Metherell et al. (1993); Thornton et al. (2002).
        # Mathematical expression:
        # - r_C = (F_met + F_str)
        #   - convert(F_leaf,C + F_root,C + F_dung,C, kg C -> g C).
        # - r_N = F_org,N
        #   - convert(F_leaf,N + F_root,N + F_dung,N, kg N -> g N).
        # Parameter meanings:
        # - F_met and F_str are the metabolic and structural carbon ports.
        # - F_org,N is the organic nitrogen port; no empirical parameter enters this check.
        # Units and route:
        # - Residuals are evaluated separately in g element m-2 s-1.
        # - The mapper preserves open boundary flows and does not advance a local pool.
        leaf_carbon = self._require_nonnegative_tensor(
            input_state,
            "leaf_litterfall_carbon",
        )
        root_carbon = self._require_nonnegative_tensor(
            input_state,
            "froot_litterfall_carbon",
        )
        leaf_nitrogen = self._require_nonnegative_tensor(
            input_state,
            "leaf_litterfall_nitrogen",
        )
        root_nitrogen = self._require_nonnegative_tensor(
            input_state,
            "froot_litterfall_nitrogen",
        )
        grazing_dung = self._get_optional_grazing_dung(
            input_state,
            reference_shape=leaf_carbon.shape,
            reference_device=leaf_carbon.device,
        )
        if grazing_dung is not None:
            dung_carbon, dung_nitrogen = grazing_dung
            expected_carbon_kg = leaf_carbon + root_carbon + dung_carbon
            expected_nitrogen_kg = (
                leaf_nitrogen + root_nitrogen + dung_nitrogen
            )
        else:
            expected_carbon_kg = leaf_carbon + root_carbon
            expected_nitrogen_kg = leaf_nitrogen + root_nitrogen
        expected_carbon = unit_converter.convert(
            expected_carbon_kg,
            "carbon_flux",
            "kg C m-2 s-1",
            "g C m-2 s-1",
        )
        expected_nitrogen = unit_converter.convert(
            expected_nitrogen_kg,
            "nitrogen_flux",
            "kg N m-2 s-1",
            "g N m-2 s-1",
        )
        actual_carbon = self._require_port_tensor(
            output_updates,
            "metabolic_litter_input",
        ) + self._require_port_tensor(
            output_updates,
            "structural_litter_input",
        )
        actual_nitrogen = self._require_port_tensor(
            output_updates,
            "organic_nitrogen_input",
        )

        def _maximum_absolute_residual(
            actual: torch.Tensor,
            expected: torch.Tensor,
        ) -> float:
            residual = torch.max(torch.abs(actual - expected))
            return float(residual.detach().cpu().item())

        return {
            "carbon": _maximum_absolute_residual(actual_carbon, expected_carbon),
            "nitrogen": _maximum_absolute_residual(
                actual_nitrogen,
                expected_nitrogen,
            ),
        }

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: torch.Tensor | float,
        **kwargs: Any,
    ) -> StateDict:
        """Apply the conservative litterfall C/N chain map."""
        # References:
        # - Parton et al. (1987); Metherell et al. (1993); Thornton et al. (2002).
        # Mathematical expression:
        # - F_C,g = convert(F_leaf,C + F_root,C, kg C -> g C).
        # - f_met = clip(exp(-k_part * max(LN - LN0, 0)), f_min, f_max).
        # - F_dung,C,g = convert(F_dung,C, kg C -> g C).
        # - F_met = f_met * F_C,g + f_man * F_dung,C,g.
        # - F_str = (1 - f_met) * F_C,g + (1 - f_man) * F_dung,C,g.
        # - F_N,g = convert(F_leaf,N + F_root,N + F_dung,N, kg N -> g N).
        # Parameter meanings:
        # - LN is the litter lignin-to-nitrogen ratio [1].
        # - LN0 is its empirical partition threshold [1].
        # - k_part is the empirical partition coefficient [1].
        # - f_min and f_max bound the metabolic fraction [1].
        # - f_man is the ParameterManager-backed manure metabolic fraction [1].
        # Units and route:
        # - Inputs are kg element m-2 s-1; outputs are g element m-2 s-1.
        # - The RothC FYM 49/49 active split is projected conservatively onto
        #   RELIC's two litter input pools; strict RothC equivalence is not claimed.
        # - Carbon and nitrogen remain explicit open boundary ports; no pool is advanced here.
        del time, dt, kwargs
        inputs = {
            key: self._require_nonnegative_tensor(state, key)
            for key in self.REQUIRED_INPUT_KEYS
        }
        flux_shape = inputs["leaf_litterfall_carbon"].shape
        for key in self.INPUT_KEYS[:4]:
            if inputs[key].shape != flux_shape:
                raise ValueError(f"{key} must match the litterfall flux shape")
        lignin_n_ratio = inputs["lignin_n_ratio"]
        if lignin_n_ratio.numel() != 1 and lignin_n_ratio.shape != flux_shape:
            raise ValueError(
                "lignin_n_ratio must be scalar or match the litterfall flux shape"
            )

        carbon_kg = (
            inputs["leaf_litterfall_carbon"]
            + inputs["froot_litterfall_carbon"]
        )
        nitrogen_kg = (
            inputs["leaf_litterfall_nitrogen"]
            + inputs["froot_litterfall_nitrogen"]
        )
        grazing_dung = self._get_optional_grazing_dung(
            state,
            reference_shape=flux_shape,
            reference_device=inputs["leaf_litterfall_carbon"].device,
        )
        carbon_g = unit_converter.convert(
            carbon_kg,
            "carbon_flux",
            "kg C m-2 s-1",
            "g C m-2 s-1",
        )
        nitrogen_g = unit_converter.convert(
            nitrogen_kg,
            "nitrogen_flux",
            "kg N m-2 s-1",
            "g N m-2 s-1",
        )
        quality_argument = torch.clamp(
            lignin_n_ratio.to(device=carbon_g.device, dtype=carbon_g.dtype)
            - self.lignin_n_threshold,
            min=self.clamp_min,
        )
        metabolic_fraction = torch.clamp(
            torch.exp(-self.lignin_partitioning_coefficient * quality_argument),
            min=self.min_metabolic_fraction,
            max=self.max_metabolic_fraction,
        )
        metabolic_input = carbon_g * metabolic_fraction
        structural_input = carbon_g - metabolic_input
        if grazing_dung is not None:
            dung_carbon, dung_nitrogen = grazing_dung
            dung_carbon_g = unit_converter.convert(
                dung_carbon,
                "carbon_flux",
                "kg C m-2 s-1",
                "g C m-2 s-1",
            ).to(device=carbon_g.device, dtype=carbon_g.dtype)
            dung_nitrogen_g = unit_converter.convert(
                dung_nitrogen,
                "nitrogen_flux",
                "kg N m-2 s-1",
                "g N m-2 s-1",
            ).to(device=nitrogen_g.device, dtype=nitrogen_g.dtype)
            metabolic_input = (
                metabolic_input
                + self.manure_metabolic_fraction * dung_carbon_g
            )
            structural_input = (
                structural_input
                + (1.0 - self.manure_metabolic_fraction) * dung_carbon_g
            )
            nitrogen_g = nitrogen_g + dung_nitrogen_g

        def _port(value: torch.Tensor, unit: str, element: str) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                basis="instantaneous",
                per_timestep=False,
                element=element,
                source_functor=self.name,
            )

        return {
            "metabolic_litter_input": _port(
                metabolic_input,
                "g C m-2 s-1",
                "C",
            ),
            "structural_litter_input": _port(
                structural_input,
                "g C m-2 s-1",
                "C",
            ),
            "organic_nitrogen_input": _port(
                nitrogen_g,
                "g N m-2 s-1",
                "N",
            ),
        }


def get_pedosphere_interface_mapper(
    parameter_manager: Optional[Any] = None,
    *,
    upscale_bridge: Optional[Any] = None,
    downscale_bridge: Optional[Any] = None,
) -> PedosphereInterfaceMapper:
    """
    Create a PedosphereInterfaceMapper instance.

    Parameters
    ----------
    parameter_manager : Optional[Any]
        Optional parameter manager to inject for this mapper.

    Returns
    -------
    PedosphereInterfaceMapper
        Mapper instance with appropriate parameter manager wiring.
    """
    return PedosphereInterfaceMapper(
        parameter_manager=parameter_manager,
        upscale_bridge=upscale_bridge,
        downscale_bridge=downscale_bridge,
    )


def get_litterfall_interface_mapper(
    parameter_manager: Optional[Any] = None,
    *,
    dependencies: Optional[list[str]] = None,
) -> LitterfallInterfaceMapper:
    """Create the unique biosphere-to-pedosphere litterfall chain map."""
    return LitterfallInterfaceMapper(
        parameter_manager=parameter_manager,
        config={"dependencies": list(dependencies or [])},
    )


# Helper functions
def _require_mapper_param(
    mapper: PedosphereInterfaceMapper,
    name: str,
    *,
    namespace: str,
) -> Any:
    value = mapper.get_parameter_value(
        name,
        namespace=namespace,
        source_order=("manager",),
    )
    if value is None:
        raise KeyError(f"Missing required parameter '{namespace}.{name}'")
    return value


# Convenience functions
def get_leaching_flux(state: StateDict) -> torch.Tensor:
    """
    Return the canonical drainage flux drainage_flux [kg m-2 s-1] of the state.

    The flux is a required input: a missing drainage_flux raises; no alias
    (leaching_flux, percolation, subsurface_runoff_flux) and no zero stands in for it.
    """
    if "drainage_flux" not in state:
        raise ValueError("Pedosphere leaching requires the input drainage_flux")
    return state["drainage_flux"]


def _require_soil_geometry(state: StateDict, reference: torch.Tensor) -> tuple:
    """Return the bulk_density [kg m-3] and soil_depth [m] forcing inputs of the state.

    Both are required: a missing or non-positive value raises (no default).
    """
    missing = [name for name in ("bulk_density", "soil_depth") if name not in state]
    if missing:
        raise ValueError("Pedosphere leaching requires the inputs " + ", ".join(missing))
    values = []
    for name in ("bulk_density", "soil_depth"):
        value = torch.as_tensor(state[name], device=reference.device, dtype=reference.dtype)
        if not bool(torch.all(torch.isfinite(value) & (value > 0.0))):
            raise ValueError(f"Input '{name}' must be finite and positive")
        values.append(value)
    return values[0], values[1]


def calculate_doc_leaching(state: StateDict,
                          doc_pool: torch.Tensor,
                          soil_water_content: torch.Tensor,
                          doc_solubility: Optional[float] = None) -> torch.Tensor:
    """
    Calculate dissolved organic carbon leaching.

    Parameters:
    -----------
    state : StateDict
        Current state with water fluxes
    doc_pool : torch.Tensor
        Dissolved organic carbon pool [g C m-2]
    soil_water_content : torch.Tensor
        Soil water content [m3 m-3]
    doc_solubility : Optional[float]
        DOC solubility fraction [dimensionless], defaults to parameter from manager

    Returns:
    --------
    torch.Tensor
        DOC leaching flux [g C m-2 s^-1]
    """
    mapper = get_pedosphere_interface_mapper()

    # Get default DOC solubility from parameter manager if not provided
    if doc_solubility is None:
        doc_solubility = _require_mapper_param(
            mapper,
            "doc_solubility",
            namespace="pedosphere.nutrient_cycling",
        )

    # Get drainage flux
    drainage = get_leaching_flux(state)

    # Soil geometry forcing (bulk_density, soil_depth contracts): required state inputs;
    # no default depth or bulk density stands in for them.
    bulk_density, soil_depth = _require_soil_geometry(state, soil_water_content)
    soil_water_volume = soil_water_content * soil_depth  # m^3 m^-2

    # DOC concentration [g C m^-3]
    epsilon_concentration = _require_mapper_param(
        mapper,
        "epsilon_concentration",
        namespace="pedosphere.numerical",
    )
    epsilon_tensor = torch.as_tensor(
        epsilon_concentration,
        device=soil_water_content.device,
        dtype=soil_water_content.dtype,
    )
    doc_concentration = (doc_pool * doc_solubility) / torch.clamp(
        soil_water_volume,
        min=epsilon_tensor,
    )

    # References: Jury and Horton (2004); van Genuchten and Wierenga (1986).
    # Mathematical expression: R - 1 = rho_b Kd_DOC c_L / theta; F_DOC = q_w C_DOC / R.
    # Parameter meanings: Kd_DOC DOC distribution coefficient, rho_b bulk density,
    #   c_L liters to cubic meters, theta soil water content, q_w volumetric drainage.
    # Units: Kd_DOC [L kg-1]; rho_b [kg m-3]; c_L [m3 L-1]; theta [m3 m-3]; F_DOC [g C m-2 s-1].
    # Domain/constraints: theta clamped by pedosphere.numerical.epsilon_concentration; R >= 1.
    # Sign convention: leaching is a positive loss from the soil column.
    doc_kd = _require_mapper_param(
        mapper,
        "doc_distribution_coefficient",
        namespace="pedosphere.transport",
    )
    doc_adsorption = mapper.linear_adsorption_retardation_term(
        doc_kd,
        bulk_density,
        torch.clamp(soil_water_content, min=epsilon_tensor),
    )

    # Calculate leaching using interface mapper
    doc_leaching = mapper.calculate_solute_leaching(
        water_flux=drainage,
        concentration=doc_concentration,
        adsorption_coefficient=doc_adsorption
    )

    # Ensure we don't leach more than available
    # Get seconds_per_day from the physical constants SSOT.
    seconds_per_day = get_physical_constant(
        "relic.physical.constants.time.seconds_per_day"
    )
    seconds_per_day = torch.as_tensor(seconds_per_day, device=doc_leaching.device, dtype=doc_leaching.dtype)
    # Availability cap: at most the soluble DOC pool can leave within one day.
    max_leaching = doc_pool * doc_solubility / seconds_per_day
    doc_leaching = torch.min(doc_leaching, max_leaching)

    return doc_leaching


def calculate_nitrate_leaching(state: StateDict,
                              no3_pool: torch.Tensor,
                              soil_water_content: torch.Tensor) -> torch.Tensor:
    """
    Calculate nitrate leaching (highly mobile).

    Parameters:
    -----------
    state : StateDict
        Current state with water fluxes
    no3_pool : torch.Tensor
        Nitrate nitrogen pool [g N m-2]
    soil_water_content : torch.Tensor
        Soil water content [m3 m-3]

    Returns:
    --------
    torch.Tensor
        NO3 leaching flux [g N m-2 s^-1]
    """
    mapper = get_pedosphere_interface_mapper()
    # Get drainage flux
    drainage = get_leaching_flux(state)

    # Soil geometry forcing (bulk_density, soil_depth contracts): required state inputs.
    bulk_density, soil_depth = _require_soil_geometry(state, soil_water_content)
    soil_water_volume = soil_water_content * soil_depth  # m^3 m^-2
    epsilon_concentration = _require_mapper_param(
        mapper,
        "epsilon_concentration",
        namespace="pedosphere.numerical",
    )
    epsilon_tensor = torch.as_tensor(
        epsilon_concentration,
        device=soil_water_content.device,
        dtype=soil_water_content.dtype,
    )

    # NO3 is highly soluble - most is in solution
    no3_concentration = no3_pool / torch.clamp(soil_water_volume, min=epsilon_tensor)

    # References: Jury and Horton (2004); van Genuchten and Wierenga (1986).
    # Mathematical expression: R - 1 = rho_b Kd c_L / theta; f_mobile = 1 / R.
    # Parameter meanings: Kd nitrate distribution coefficient, rho_b bulk density,
    #   c_L SSOT liters-to-cubic-meters conversion, theta soil water content.
    # Units: Kd [L kg-1]; rho_b [kg m-3]; c_L [m3 L-1]; theta [m3 m-3]; R - 1 [1].
    # Domain/constraints: theta clamped by pedosphere.numerical.epsilon_concentration; R >= 1.
    # Sign convention: leaching is a positive loss from the soil column.
    no3_kd = _require_mapper_param(
        mapper,
        "nitrate_distribution_coefficient",
        namespace="pedosphere.transport",
    )
    no3_adsorption = mapper.linear_adsorption_retardation_term(
        no3_kd,
        bulk_density,
        torch.clamp(soil_water_content, min=epsilon_tensor),
    )

    # Calculate leaching - NO3 is very mobile (low adsorption)
    no3_leaching = mapper.calculate_solute_leaching(
        water_flux=drainage,
        concentration=no3_concentration,
        adsorption_coefficient=no3_adsorption
    )

    # Ensure we don't leach more than available
    seconds_per_hour = get_physical_constant(
        "relic.physical.constants.time.seconds_per_hour"
    )
    seconds_per_hour = torch.as_tensor(
        seconds_per_hour,
        device=no3_leaching.device,
        dtype=no3_leaching.dtype,
    )
    max_leaching = no3_pool / seconds_per_hour
    no3_leaching = torch.min(no3_leaching, max_leaching)

    return no3_leaching


# Export main components
__all__ = [
    'PedosphereInterfaceMapper',
    'LitterfallInterfaceMapper',
    'get_pedosphere_interface_mapper',
    'get_litterfall_interface_mapper',
    'get_leaching_flux',
    'calculate_doc_leaching',
    'calculate_nitrate_leaching'
]
