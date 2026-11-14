"""
Land Use to Parameter Mapping Module

This module reports area-weighted diagnostic summaries of land-use-class
parameters. The summaries are observations only: downstream scientific
processes continue to obtain their executable parameters from ParameterManager.

References:
    - Pongratz, J., et al. (2018). Models meet data: Challenges and opportunities in implementing land management in Earth system models. GCB 24, 1470-1487.
    - Lawrence, D.M., et al. (2016). The Land Use Model Intercomparison Project (LUMIP). GMD 9, 2973-2998.
    - FAO (2017). Voluntary Guidelines for Sustainable Soil Management.

Mathematical Expression:
    Diagnostic mapping function:
    theta_bar = sum_i (f_i * theta_i)
    
    where:
    - f_i is the fraction of land use type i [1]
    - theta_i is the parameter value for land use type i
    - theta_bar is an area-weighted diagnostic summary

Scientific boundary:
    The arithmetic mean is not asserted to be an effective nonlinear process
    coefficient, a conserved quantity, or a replacement for sub-grid tile/PFT
    calculations.
"""

import torch
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from dataclasses import dataclass

from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.abstractions import (
    CoordinateTensor,
    FieldTensor,
    GradientTensor,
    EnvironmentDict,
    Tensor
)
from .transitions import LandUseType
from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.core.functor import PhysicalFieldFunctor
from relic.core.state import UpdateItem, UpdateMode

@dataclass
class VegetationParameters:
    """
    Vegetation parameters for each land use type
    
    Attributes:
        lai_max: Maximum leaf area index [m^2/m^2]
        lai_min: Minimum leaf area index [m^2/m^2]
        vcmax25: Maximum carboxylation rate at 25degC [umol CO2 m-2 s-1]
        jmax25: Maximum electron transport rate at 25degC [umol e- m-2 s-1]
        root_depth: Effective rooting depth [m]
        canopy_height: Canopy height [m]
        wood_density: Wood density [kg m-3]
        sla: Specific leaf area [m^2 kg-1]
        leaf_lifespan: Leaf lifespan [days]
        c3_fraction: Fraction of C3 photosynthesis [1]
        c4_fraction: Fraction of C4 photosynthesis [1]
    """
    lai_max: float
    lai_min: float
    vcmax25: float
    jmax25: float
    root_depth: float
    canopy_height: float
    wood_density: float
    sla: float
    leaf_lifespan: float
    c3_fraction: float
    c4_fraction: float


@dataclass
class SoilParameters:
    """
    Soil parameters modified by land use
    
    Attributes:
        decomposition_rate: Base decomposition rate modifier [1]
        nitrification_rate: Nitrification rate modifier [1]
        denitrification_rate: Denitrification rate modifier [1]
        soil_porosity: Soil porosity [m^3/m^3]
        infiltration_rate: Infiltration rate [mm/hr]
        soil_carbon_input: Soil carbon input rate [kg C m-2 yr-1]
        tillage_factor: Tillage disturbance factor [1]
        erosion_rate: Erosion rate [kg m-2 yr-1]
    """
    decomposition_rate: float
    nitrification_rate: float
    denitrification_rate: float
    soil_porosity: float
    infiltration_rate: float
    soil_carbon_input: float
    tillage_factor: float
    erosion_rate: float


@dataclass
class HydrologicalParameters:
    """
    Hydrological parameters affected by land use
    
    Attributes:
        surface_roughness: Surface roughness length [m]
        albedo: Surface albedo [1]
        emissivity: Surface emissivity [1]
        interception_capacity: Canopy interception capacity [mm]
        runoff_coefficient: Surface runoff coefficient [1]
        evaporation_resistance: Soil evaporation resistance [s m-1]
    """
    surface_roughness: float
    albedo: float
    emissivity: float
    interception_capacity: float
    runoff_coefficient: float
    evaporation_resistance: float


LAND_USE_PARAMETER_ROOT = "anthroposphere.land_use"
VEGETATION_PARAMETER_NAMES: Tuple[str, ...] = (
    "lai_max",
    "lai_min",
    "vcmax25",
    "jmax25",
    "root_depth",
    "canopy_height",
    "wood_density",
    "sla",
    "leaf_lifespan",
    "c3_fraction",
    "c4_fraction",
)
SOIL_PARAMETER_NAMES: Tuple[str, ...] = (
    "decomposition_rate",
    "nitrification_rate",
    "denitrification_rate",
    "soil_porosity",
    "infiltration_rate",
    "soil_carbon_input",
    "tillage_factor",
    "erosion_rate",
)
HYDROLOGICAL_PARAMETER_NAMES: Tuple[str, ...] = (
    "surface_roughness",
    "albedo",
    "emissivity",
    "interception_capacity",
    "runoff_coefficient",
    "evaporation_resistance",
)

DIAGNOSTIC_PREFIX = "diagnostics.land_use_parameter_mapping"
VEGETATION_DIAGNOSTIC_OUTPUTS: Tuple[str, ...] = tuple(
    f"{DIAGNOSTIC_PREFIX}.vegetation.{name}"
    for name in VEGETATION_PARAMETER_NAMES
)
SOIL_DIAGNOSTIC_OUTPUTS: Tuple[str, ...] = tuple(
    f"{DIAGNOSTIC_PREFIX}.soil.{name}"
    for name in SOIL_PARAMETER_NAMES
)
HYDROLOGY_DIAGNOSTIC_OUTPUTS: Tuple[str, ...] = tuple(
    f"{DIAGNOSTIC_PREFIX}.hydrology.{name}"
    for name in HYDROLOGICAL_PARAMETER_NAMES
)
RUNTIME_DIAGNOSTIC = f"{DIAGNOSTIC_PREFIX}.canonical_iphs_runtime"


class LandUseParameterMapping(PhysicalFieldFunctor):
    """
    Land Use Parameter Mapping Functor
    
    Maps land use types to ecosystem model parameters, providing
    weighted parameter values based on land use fractions.
    """

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    # Horizontal (x, y) support: spatial queries take coordinates of shape [n, 2]
    # (_validate_query_coordinates and the spatial chain maps). The field identity
    # is the declared outputs and their contracts, not a constructor field name.
    SPATIAL_DIMS = 2
    DIAGNOSTIC_OUTPUTS: Tuple[str, ...] = (
        VEGETATION_DIAGNOSTIC_OUTPUTS
        + SOIL_DIAGNOSTIC_OUTPUTS
        + HYDROLOGY_DIAGNOSTIC_OUTPUTS
        + (RUNTIME_DIAGNOSTIC,)
    )
    
    def __init__(
        self,
        name: str = "land_use_mapping",
        config: Optional[Dict[str, Any]] = None,
        *,
        field_query_bridge: Optional[Any] = None,
        field_gradient_bridge: Optional[Any] = None,
    ):
        """
        Initialize parameter mapping functor
        
        Args:
            name: Functor name
            config: Configuration dictionary
            field_query_bridge: Bound typed chain map from source field support
                to query coordinates.
            field_gradient_bridge: Bound typed chain map from source field
                support to gradients at query coordinates.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if field_query_bridge is not None and not callable(field_query_bridge):
            raise TypeError("field_query_bridge must be callable")
        if field_gradient_bridge is not None and not callable(field_gradient_bridge):
            raise TypeError("field_gradient_bridge must be callable")
        # Set up metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Reports ParameterManager-backed area-weighted land-use "
                "parameter diagnostics without overriding process parameters"
            ),
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.PREPROCESSING,
            mathematical_expression="theta_bar = sum_i f_i theta_i",
            references=[
                "Pongratz et al. (2018) DOI:10.1111/gcb.13988",
                "Lawrence et al. (2016) DOI:10.5194/gmd-9-2973-2016",
                "FAO (2017) Sustainable Soil Management Guidelines"
            ],
            parameters={
                "anthroposphere.land_use.vegetation.*": (
                    "Vegetation parameters (LAI range, Vcmax25, Jmax25, rooting depth, canopy "
                    "height, wood density, SLA, leaf lifespan, C3 and C4 fractions) per land-use "
                    "type [units per key]"
                ),
                "anthroposphere.land_use.soil.*": (
                    "Soil parameters (decomposition, nitrification, denitrification, erosion and "
                    "infiltration rates, carbon input, porosity, tillage factor) per land-use "
                    "type [units per key]"
                ),
                "anthroposphere.land_use.hydrological.*": (
                    "Hydrological parameters (albedo, emissivity, evaporation resistance, "
                    "interception capacity, runoff coefficient, surface roughness) per land-use "
                    "type [units per key]"
                ),
                "validation.conservation.conservation_tolerance": (
                    "General relative tolerance of the conservation checks [dimensionless]"
                ),
            },
            input_variables=["land_use_fraction"],
            output_variables=[
                "diagnostics.land_use_parameter_mapping.vegetation.lai_max",
                "diagnostics.land_use_parameter_mapping.vegetation.lai_min",
                "diagnostics.land_use_parameter_mapping.vegetation.vcmax25",
                "diagnostics.land_use_parameter_mapping.vegetation.jmax25",
                "diagnostics.land_use_parameter_mapping.vegetation.root_depth",
                "diagnostics.land_use_parameter_mapping.vegetation.canopy_height",
                "diagnostics.land_use_parameter_mapping.vegetation.wood_density",
                "diagnostics.land_use_parameter_mapping.vegetation.sla",
                "diagnostics.land_use_parameter_mapping.vegetation.leaf_lifespan",
                "diagnostics.land_use_parameter_mapping.vegetation.c3_fraction",
                "diagnostics.land_use_parameter_mapping.vegetation.c4_fraction",
                "diagnostics.land_use_parameter_mapping.soil.decomposition_rate",
                "diagnostics.land_use_parameter_mapping.soil.nitrification_rate",
                "diagnostics.land_use_parameter_mapping.soil.denitrification_rate",
                "diagnostics.land_use_parameter_mapping.soil.soil_porosity",
                "diagnostics.land_use_parameter_mapping.soil.infiltration_rate",
                "diagnostics.land_use_parameter_mapping.soil.soil_carbon_input",
                "diagnostics.land_use_parameter_mapping.soil.tillage_factor",
                "diagnostics.land_use_parameter_mapping.soil.erosion_rate",
                "diagnostics.land_use_parameter_mapping.hydrology.surface_roughness",
                "diagnostics.land_use_parameter_mapping.hydrology.albedo",
                "diagnostics.land_use_parameter_mapping.hydrology.emissivity",
                "diagnostics.land_use_parameter_mapping.hydrology.interception_capacity",
                "diagnostics.land_use_parameter_mapping.hydrology.runoff_coefficient",
                "diagnostics.land_use_parameter_mapping.hydrology.evaporation_resistance",
                "diagnostics.land_use_parameter_mapping.canonical_iphs_runtime",
            ],
            conservation_laws=[],
            spatial_dimensions=2,
            temporal_scale=TemporalScale.DAY,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
            coupling_interfaces=["land_use_transitions"],
        )
        
        # Initialize parent class with proper metadata and config
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
        )

        self.runtime_path = "canonical_land_use_parameter_diagnostic_observation"
        self._field_query_bridge = field_query_bridge
        self._field_gradient_bridge = field_gradient_bridge
        self._mapped_parameter_units: Dict[str, str] = {}
        
        # Initialize parameter lookup tables
        self.vegetation_params = self._initialize_vegetation_parameters()
        self.soil_params = self._initialize_soil_parameters()
        self.hydro_params = self._initialize_hydrological_parameters()
        
        # Initialize field state for PhysicalField functionality
        self._current_field = None
        self._current_coordinates = None

    def _fetch_parameter_group(self, namespace: str, parameter_names: Iterable[str]) -> Dict[str, float]:
        """Load a group of parameters from the parameter manager."""
        manager = self.get_parameter_manager()
        if manager is None:
            raise RuntimeError("Parameter manager unavailable for land use parameter lookup.")
        values: Dict[str, float] = {}
        parameter_group = namespace.split(".")[-2]
        diagnostic_group = "hydrology" if parameter_group == "hydrological" else parameter_group
        for name in parameter_names:
            full_key = f"{namespace}.{name}"
            spec = manager.get_parameter(full_key)
            if spec is None:
                raise KeyError(f"Missing parameter specification for '{full_key}'.")
            if not spec.units:
                raise ValueError(f"Parameter '{full_key}' must declare units.")
            diagnostic_name = f"{DIAGNOSTIC_PREFIX}.{diagnostic_group}.{name}"
            existing_unit = self._mapped_parameter_units.get(diagnostic_name)
            if existing_unit is not None and existing_unit != spec.units:
                raise ValueError(
                    f"Inconsistent units for mapped parameter '{name}': "
                    f"'{existing_unit}' and '{spec.units}'."
                )
            self._mapped_parameter_units[diagnostic_name] = spec.units
            values[name] = self.get_parameter_value(
                name,
                namespace=namespace,
                source_order=("manager",),
                required=True,
            )
        return values

    @property
    def mapped_parameter_units(self) -> Dict[str, str]:
        """Return the validated diagnostic-unit map."""

        return dict(self._mapped_parameter_units)

    
    def _initialize_vegetation_parameters(self) -> Dict[LandUseType, VegetationParameters]:
        """Load vegetation parameters for each land use type."""
        params: Dict[LandUseType, VegetationParameters] = {}
        for land_use_type in LandUseType:
            namespace = f"{LAND_USE_PARAMETER_ROOT}.vegetation.{land_use_type.value}"
            values = self._fetch_parameter_group(namespace, VEGETATION_PARAMETER_NAMES)
            params[land_use_type] = VegetationParameters(**values)
        return params
    
    def _initialize_soil_parameters(self) -> Dict[LandUseType, SoilParameters]:
        """Load soil parameters for each land use type."""
        params: Dict[LandUseType, SoilParameters] = {}
        for land_use_type in LandUseType:
            namespace = f"{LAND_USE_PARAMETER_ROOT}.soil.{land_use_type.value}"
            values = self._fetch_parameter_group(namespace, SOIL_PARAMETER_NAMES)
            params[land_use_type] = SoilParameters(**values)
        return params
    
    def _initialize_hydrological_parameters(self) -> Dict[LandUseType, HydrologicalParameters]:
        """Load hydrological parameters for each land use type."""
        params: Dict[LandUseType, HydrologicalParameters] = {}
        for land_use_type in LandUseType:
            namespace = f"{LAND_USE_PARAMETER_ROOT}.hydrological.{land_use_type.value}"
            values = self._fetch_parameter_group(namespace, HYDROLOGICAL_PARAMETER_NAMES)
            params[land_use_type] = HydrologicalParameters(**values)
        return params
    
    def get_dependencies(self) -> List[str]:
        """Get functor dependencies."""
        return self._merge_declared_dependencies(["land_use_transitions"])

    def _structural_n_dofs(self, n_dofs: Optional[int] = None) -> int:
        """Resolve structural audit size without changing runtime tensor shape."""

        value = len(LandUseType) if n_dofs is None else int(n_dofs)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare the observed land-use-fraction support.

        References:
          - Lawrence et al. (2016), LUMIP land-use fractions.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          theta_bar = sum_i f_i theta_i.
        Parameter meanings:
          f_i is the observed land-use fraction [1] and theta_i is a
          ParameterManager-backed class value. Mixed-unit diagnostic outputs
          are not represented as a single DEC form or Hodge operator.
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="anthroposphere_land_use_parameter_mapping",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="1",
                extensivity="intensive",
                physical_domain="anthroposphere",
                semantic_name="land_use_fraction_observation",
                orientation_convention="area_fraction_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager namespaces used by runtime and structure."""

        return (
            "anthroposphere.land_use.vegetation.*",
            "anthroposphere.land_use.soil.*",
            "anthroposphere.land_use.hydrological.*",
        )

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Any, ...]:
        """Return no Hodge term for mixed-unit diagnostic observations."""

        _ = complex_bundle, resolved_parameters
        return ()

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no stored energy term for algebraic parameter mapping."""

        _ = state
        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no standalone entropy term for algebraic parameter mapping."""

        _ = state
        return {}

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return no reversible dynamics for an algebraic observation."""

        _ = state, complex_bundle
        return {"state_vars": (), "ports": ()}

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return no dissipative dynamics for an algebraic observation."""

        _ = state, complex_bundle
        return {"state_vars": (), "ports": ()}

    def ports(self) -> tuple[Any, ...]:
        """Return no public power port for diagnostic parameter summaries."""

        return ()

    def compute(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
    ) -> Mapping[str, Any]:
        """Run the canonical calculation through the existing Functor lifecycle."""

        return self.forward(dict(state), time, dt)
    
    def _land_use_type_axis(self, fractions: torch.Tensor) -> int:
        """Return the class axis for the canonical land-use tensor layouts."""

        if fractions.dim() == 1:
            return 0
        if fractions.dim() == 2:
            return 1
        if fractions.dim() == 3:
            candidates = [
                axis
                for axis in (0, 1)
                if fractions.shape[axis] == len(LandUseType)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Three-dimensional land_use_fraction must have one unique "
                    "nine-class axis in position 0 or 1"
                )
            return candidates[0]
        if fractions.dim() == 4:
            return 1
        raise ValueError(
            "land_use_fraction must use [type], [batch,type], [type,y,x], "
            "[batch,type,space], or [batch,type,y,x] layout"
        )

    def _validate_land_use_fraction(
        self,
        land_use_fractions: torch.Tensor,
        bring_type_dim_first: bool = False,
    ) -> torch.Tensor:
        """Validate the canonical nine-class fractional-area observation.

        References:
            Hurtt et al. (2020), LUH2 land-use fractions.
        Mathematical expression:
            f_i >= 0 and sum_i f_i = 1 on every spatial support.
        Parameter meanings:
            f_i is the fractional area of canonical land-use class i [1].
            The conservation tolerance is ParameterManager-backed.
        """

        if not isinstance(land_use_fractions, torch.Tensor):
            raise TypeError("land_use_fraction must be a torch.Tensor")
        if not torch.is_floating_point(land_use_fractions):
            raise TypeError("land_use_fraction must use a floating-point dtype")
        if land_use_fractions.numel() == 0:
            raise ValueError("land_use_fraction must not be empty")
        if not torch.isfinite(land_use_fractions).all():
            raise ValueError("land_use_fraction must contain only finite values")
        if torch.any(land_use_fractions < 0.0):
            raise ValueError("land_use_fraction must be non-negative")

        type_dim = self._land_use_type_axis(land_use_fractions)
        if land_use_fractions.shape[type_dim] != len(LandUseType):
            raise ValueError(
                f"land_use_fraction must contain exactly {len(LandUseType)} "
                "land-use classes"
            )
        tolerance = float(
            self.get_parameter_value(
                "conservation_tolerance",
                namespace="validation.conservation",
                source_order=("manager",),
                required=True,
            )
        )
        totals = land_use_fractions.sum(dim=type_dim)
        if torch.max(torch.abs(totals - 1.0)) > tolerance:
            raise ValueError("land_use_fraction must sum to one on every support")

        if bring_type_dim_first and type_dim != 0:
            return land_use_fractions.movedim(type_dim, 0)
        return land_use_fractions
    
    def calculate_weighted_parameters(self,
                                     land_use_fractions: torch.Tensor) -> Tuple[Dict, Dict, Dict]:
        """
        Calculate weighted parameters based on land use fractions
        
        Args:
            land_use_fractions: Fractions for each land use type [n_types]
            
        Returns:
            Tuple of (vegetation_params, soil_params, hydro_params) dictionaries
        """
        veg_out: Dict[str, torch.Tensor] = {}
        soil_out: Dict[str, torch.Tensor] = {}
        hydro_out: Dict[str, torch.Tensor] = {}

        land_use_types = list(LandUseType)
        land_use_fractions = self._validate_land_use_fraction(
            land_use_fractions,
            bring_type_dim_first=True
        )

        # References: Lawrence et al. (2016); Le Moigne (2018).
        # Mathematical expression: theta_bar = sum_i f_i theta_i.
        # Parameter meanings: f_i is land-use fractional area [1]; theta_i is
        # a ParameterManager class value with its declared unit. These values
        # are diagnostic summaries, not executable nonlinear coefficients.
        def _weighted_group(
            parameter_table: Mapping[LandUseType, Any],
            names: Iterable[str],
        ) -> Dict[str, torch.Tensor]:
            result: Dict[str, torch.Tensor] = {}
            expand_shape = (len(land_use_types),) + (1,) * (
                land_use_fractions.dim() - 1
            )
            for attr in names:
                coefficients = torch.stack(
                    [
                        torch.as_tensor(
                            getattr(parameter_table[land_use_type], attr),
                            dtype=land_use_fractions.dtype,
                            device=land_use_fractions.device,
                        )
                        for land_use_type in land_use_types
                    ]
                ).reshape(expand_shape)
                result[attr] = (land_use_fractions * coefficients).sum(dim=0)
            return result

        veg_out.update(_weighted_group(self.vegetation_params, VEGETATION_PARAMETER_NAMES))
        soil_out.update(_weighted_group(self.soil_params, SOIL_PARAMETER_NAMES))
        hydro_out.update(_weighted_group(self.hydro_params, HYDROLOGICAL_PARAMETER_NAMES))

        return veg_out, soil_out, hydro_out
    
    def forward(self,
        # References: Pongratz et al. (2018); Lawrence et al. (2016); FAO (2017).
        # Mathematical expression: theta_bar = sum_i f_i theta_i over the declared land-use classes.
        # Parameter meanings: f_i are the land-use fractions on the class axis; theta_i the
        #   class-specific ecosystem parameter values from ParameterManager; theta_bar the
        #   area-weighted mapped parameter.
        # Units: f_i [1]; theta_i and theta_bar carry the unit declared per mapped parameter
        #   (_mapped_parameter_units); fraction diagnostics [1].
        # Domain/constraints: fractions are finite, non-negative and sum to one on the class
        #   axis; mapped parameters are REPLACE observations and advance no state.
        # Sign convention: not applicable (weighted average).
                state: Dict[str, torch.Tensor],
                time: float,
                dt: float) -> Dict[str, UpdateItem]:
        """
        Map land use fractions to ecosystem parameters

        Args:
            state: Current state dictionary holding the declared land_use_fraction
            time: Current simulation time [s]
            dt: Time step [s]

        Returns:
            Dictionary with mapped parameters
        """
        _ = time, dt
        if "land_use_fraction" not in state:
            raise KeyError("Missing required input: land_use_fraction")
        land_use_fractions = state["land_use_fraction"]

        # Calculate weighted parameters
        veg_params, soil_params, hydro_params = self.calculate_weighted_parameters(land_use_fractions)

        def _diagnostic(output_name: str, value: torch.Tensor) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit=self._mapped_parameter_units[output_name],
                source_functor=self.name,
            )

        outputs: Dict[str, UpdateItem] = {
            "diagnostics.land_use_parameter_mapping.vegetation.lai_max": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.lai_max",
                veg_params["lai_max"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.lai_min": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.lai_min",
                veg_params["lai_min"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.vcmax25": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.vcmax25",
                veg_params["vcmax25"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.jmax25": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.jmax25",
                veg_params["jmax25"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.root_depth": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.root_depth",
                veg_params["root_depth"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.canopy_height": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.canopy_height",
                veg_params["canopy_height"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.wood_density": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.wood_density",
                veg_params["wood_density"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.sla": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.sla",
                veg_params["sla"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.leaf_lifespan": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.leaf_lifespan",
                veg_params["leaf_lifespan"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.c3_fraction": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.c3_fraction",
                veg_params["c3_fraction"],
            ),
            "diagnostics.land_use_parameter_mapping.vegetation.c4_fraction": _diagnostic(
                "diagnostics.land_use_parameter_mapping.vegetation.c4_fraction",
                veg_params["c4_fraction"],
            ),
            "diagnostics.land_use_parameter_mapping.soil.decomposition_rate": _diagnostic(
                "diagnostics.land_use_parameter_mapping.soil.decomposition_rate",
                soil_params["decomposition_rate"],
            ),
            "diagnostics.land_use_parameter_mapping.soil.nitrification_rate": _diagnostic(
                "diagnostics.land_use_parameter_mapping.soil.nitrification_rate",
                soil_params["nitrification_rate"],
            ),
            "diagnostics.land_use_parameter_mapping.soil.denitrification_rate": _diagnostic(
                "diagnostics.land_use_parameter_mapping.soil.denitrification_rate",
                soil_params["denitrification_rate"],
            ),
            "diagnostics.land_use_parameter_mapping.soil.soil_porosity": _diagnostic(
                "diagnostics.land_use_parameter_mapping.soil.soil_porosity",
                soil_params["soil_porosity"],
            ),
            "diagnostics.land_use_parameter_mapping.soil.infiltration_rate": _diagnostic(
                "diagnostics.land_use_parameter_mapping.soil.infiltration_rate",
                soil_params["infiltration_rate"],
            ),
            "diagnostics.land_use_parameter_mapping.soil.soil_carbon_input": _diagnostic(
                "diagnostics.land_use_parameter_mapping.soil.soil_carbon_input",
                soil_params["soil_carbon_input"],
            ),
            "diagnostics.land_use_parameter_mapping.soil.tillage_factor": _diagnostic(
                "diagnostics.land_use_parameter_mapping.soil.tillage_factor",
                soil_params["tillage_factor"],
            ),
            "diagnostics.land_use_parameter_mapping.soil.erosion_rate": _diagnostic(
                "diagnostics.land_use_parameter_mapping.soil.erosion_rate",
                soil_params["erosion_rate"],
            ),
            "diagnostics.land_use_parameter_mapping.hydrology.surface_roughness": _diagnostic(
                "diagnostics.land_use_parameter_mapping.hydrology.surface_roughness",
                hydro_params["surface_roughness"],
            ),
            "diagnostics.land_use_parameter_mapping.hydrology.albedo": _diagnostic(
                "diagnostics.land_use_parameter_mapping.hydrology.albedo",
                hydro_params["albedo"],
            ),
            "diagnostics.land_use_parameter_mapping.hydrology.emissivity": _diagnostic(
                "diagnostics.land_use_parameter_mapping.hydrology.emissivity",
                hydro_params["emissivity"],
            ),
            "diagnostics.land_use_parameter_mapping.hydrology.interception_capacity": _diagnostic(
                "diagnostics.land_use_parameter_mapping.hydrology.interception_capacity",
                hydro_params["interception_capacity"],
            ),
            "diagnostics.land_use_parameter_mapping.hydrology.runoff_coefficient": _diagnostic(
                "diagnostics.land_use_parameter_mapping.hydrology.runoff_coefficient",
                hydro_params["runoff_coefficient"],
            ),
            "diagnostics.land_use_parameter_mapping.hydrology.evaporation_resistance": _diagnostic(
                "diagnostics.land_use_parameter_mapping.hydrology.evaporation_resistance",
                hydro_params["evaporation_resistance"],
            ),
        }
        reference = veg_params["lai_max"]
        outputs[RUNTIME_DIAGNOSTIC] = UpdateItem(
            value=torch.ones((), dtype=reference.dtype, device=reference.device),
            mode=UpdateMode.REPLACE,
            per_timestep=False,
            unit="dimensionless",
            source_functor=self.name,
        )

        return outputs
    
    def _validate_query_coordinates(self, coordinates: CoordinateTensor) -> None:
        """Validate explicit target support for a spatial chain-map call."""
        if not isinstance(coordinates, torch.Tensor):
            raise TypeError("coordinates must be a torch.Tensor")
        if coordinates.dim() != 2 or coordinates.shape[1] != self.SPATIAL_DIMS:
            raise ValueError(
                f"coordinates must have shape [n, {self.SPATIAL_DIMS}]"
            )
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must contain only finite values")

    def _require_current_spatial_field(self) -> Tuple[CoordinateTensor, FieldTensor]:
        """Return the declared source support or fail before spatial evaluation."""
        if self._current_field is None:
            raise RuntimeError(
                "spatial evaluation requires a current spatial field; call update_field "
                "with land_use_fraction and coordinates first"
            )
        if self._current_coordinates is None:
            raise RuntimeError(
                "spatial evaluation requires current spatial field coordinates"
            )

        source_coordinates = self._current_coordinates
        source_values = self._current_field
        self._validate_query_coordinates(source_coordinates)
        if not isinstance(source_values, torch.Tensor):
            raise TypeError("current spatial field must be a torch.Tensor")
        source_values = source_values.reshape(-1)
        if source_values.shape[0] != source_coordinates.shape[0]:
            raise RuntimeError(
                "current spatial field and coordinates must have the same support size"
            )
        if source_values.device != source_coordinates.device:
            raise RuntimeError(
                "current spatial field and coordinates must use the same device"
            )
        if not torch.isfinite(source_values).all():
            raise ValueError("current spatial field must contain only finite values")
        return source_coordinates, source_values

    @staticmethod
    def _validate_spatial_bridge_output(
        output: Any,
        *,
        target_coordinates: CoordinateTensor,
        expected_shape: Tuple[int, ...],
        expected_dtype: torch.dtype,
        bridge_name: str,
    ) -> torch.Tensor:
        """Validate a result returned by an existing bound spatial bridge."""
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{bridge_name} must return a torch.Tensor")
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"{bridge_name} returned shape {tuple(output.shape)}; "
                f"expected {expected_shape}"
            )
        if output.device != target_coordinates.device:
            raise RuntimeError(
                f"{bridge_name} output must use the target coordinate device"
            )
        if output.dtype != expected_dtype:
            raise RuntimeError(
                f"{bridge_name} output must preserve the source field dtype"
            )
        if not torch.isfinite(output).all():
            raise ValueError(f"{bridge_name} output must contain only finite values")
        return output

    # PhysicalField abstract method implementations
    def get_value_at(self, coordinates: CoordinateTensor) -> FieldTensor:
        """
        Get parameter field values at specified spatial coordinates.
        
        This method provides spatially-interpolated land use parameter values
        based on coordinate locations within the domain.
        
        Parameters:
            coordinates : CoordinateTensor
                Spatial coordinates [batch_size, 2]
                Units: [m, m] (x, y coordinates)
                
        Returns:
            values : FieldTensor
                Interpolated parameter values at coordinates [batch_size]
                Units: [dimensionless] (aggregated parameter index)
                
        Mathematical Expression:
            phi(x,y) = sum_i f_i(x,y) * theta_i
            where f_i(x,y) is land use fraction at location (x,y)
            and theta_i is the parameter value for land use type i
        """
        # References: Lawrence et al. (2016); Pongratz et al. (2018).
        # Mathematical expression: theta_target = R_h theta_source.
        # Parameter meanings: R_h is the externally bound typed chain map;
        # source and target coordinates declare its spatial supports.
        self._validate_query_coordinates(coordinates)
        source_coordinates, source_values = self._require_current_spatial_field()
        if self._field_query_bridge is None:
            raise RuntimeError(
                "get_value_at requires an explicit typed chain-map bridge"
            )
        if coordinates.device != source_coordinates.device:
            raise RuntimeError(
                "source and target coordinates must use the same device"
            )
        result = self._field_query_bridge(
            source_coordinates,
            source_values,
            coordinates,
        )
        return self._validate_spatial_bridge_output(
            result,
            target_coordinates=coordinates,
            expected_shape=(coordinates.shape[0],),
            expected_dtype=source_values.dtype,
            bridge_name="field_query_bridge",
        )
    
    def compute_gradient(self, coordinates: CoordinateTensor) -> GradientTensor:
        """
        Compute spatial gradient of parameter field.
        
        Calculates spatial derivatives of land use parameter fields,
        useful for identifying transition zones and heterogeneity patterns.
        
        Parameters:
            coordinates : CoordinateTensor
                Spatial coordinates [batch_size, 2]
                
        Returns:
            gradient : GradientTensor
                Parameter field gradients [batch_size, 2]
                Units: [dimensionless m-1]
                
        Mathematical Expression:
            grad(phi) = (partial phi / partial x, partial phi / partial y)
            where phi is the parameter field
        """
        # References: Lawrence et al. (2016); Pongratz et al. (2018).
        # Mathematical expression: grad(theta)_target = G_h theta_source.
        # Parameter meanings: G_h is the externally bound typed gradient map;
        # source and target coordinates declare its spatial supports.
        self._validate_query_coordinates(coordinates)
        source_coordinates, source_values = self._require_current_spatial_field()
        if self._field_gradient_bridge is None:
            raise RuntimeError(
                "compute_gradient requires an explicit typed chain-map bridge"
            )
        if coordinates.device != source_coordinates.device:
            raise RuntimeError(
                "source and target coordinates must use the same device"
            )
        result = self._field_gradient_bridge(
            source_coordinates,
            source_values,
            coordinates,
        )
        return self._validate_spatial_bridge_output(
            result,
            target_coordinates=coordinates,
            expected_shape=(coordinates.shape[0], self.SPATIAL_DIMS),
            expected_dtype=source_values.dtype,
            bridge_name="field_gradient_bridge",
        )
    
    def interpolate(self, 
                   source_coordinates: CoordinateTensor,
                   target_coordinates: CoordinateTensor,
                   *,
                   source_values: Optional[FieldTensor] = None) -> FieldTensor:
        """
        Interpolate parameter values between coordinate systems.

        Parameters:
            source_coordinates : CoordinateTensor
                Source grid coordinates [n_source, 2]
            source_values : FieldTensor
                Source field values aligned with source_coordinates
            target_coordinates : CoordinateTensor  
                Target coordinates [n_target, 2]

        The injected field_query_bridge (a typed chain map) owns the interpolation
        algorithm and its support contract; there is no method to select.
                
        Returns:
            interpolated_values : FieldTensor
                Interpolated parameter values [n_target]
        """
        self._validate_query_coordinates(source_coordinates)
        self._validate_query_coordinates(target_coordinates)
        if self._field_query_bridge is None:
            raise RuntimeError("interpolate requires an explicit typed chain-map bridge")
        if source_coordinates.device != target_coordinates.device:
            raise RuntimeError("source and target coordinates must use the same device")

        if source_values is None:
            current_coordinates, values = self._require_current_spatial_field()
            if not torch.equal(source_coordinates, current_coordinates):
                raise RuntimeError(
                    "source_coordinates must match the current spatial field support"
                )
        else:
            if not isinstance(source_values, torch.Tensor):
                raise TypeError("source_values must be a torch.Tensor")
            values = source_values
            if values.dim() != 1:
                raise ValueError("source_values must have shape [n_source]")
            if values.shape[0] != source_coordinates.shape[0]:
                raise ValueError(
                    "source_values and source_coordinates must have the same support size"
                )
            if values.device != source_coordinates.device:
                raise RuntimeError(
                    "source_values and source_coordinates must use the same device"
                )
            if not torch.isfinite(values).all():
                raise ValueError("source_values must contain only finite values")

        result = self._field_query_bridge(
            source_coordinates,
            values,
            target_coordinates,
        )
        return self._validate_spatial_bridge_output(
            result,
            target_coordinates=target_coordinates,
            expected_shape=(target_coordinates.shape[0],),
            expected_dtype=values.dtype,
            bridge_name="field_query_bridge",
        )
    
    def update_field(self, 
                    time_step: float,
                    source_terms: EnvironmentDict,
                    boundary_conditions: Optional[Dict[str, Tensor]] = None) -> None:
        """
        Update parameter field based on land use transitions.
        
        Parameters:
            time_step : float
                Time step for update [s]
            source_terms : EnvironmentDict
                Land use change drivers
            boundary_conditions : Optional[Dict[str, Tensor]]
                Boundary conditions (typically not used for parameter fields)
                
        Mathematical Expression:
            theta_bar = sum_i f_i theta_i.
        """
        _ = time_step, boundary_conditions
        if "land_use_fraction" not in source_terms:
            raise KeyError("Missing required input: land_use_fraction")
        if "coordinates" not in source_terms:
            raise KeyError("Missing required spatial support: coordinates")

        veg_params, _, _ = self.calculate_weighted_parameters(
            source_terms["land_use_fraction"]
        )
        # References: Lawrence et al. (2016); Pongratz et al. (2018).
        # Mathematical expression: LAI_bar = sum_i f_i LAI_i.
        # Parameter meanings: f_i is fractional area [1] and LAI_i is the
        # ParameterManager-backed class maximum leaf area index [1].
        next_field = veg_params["lai_max"].reshape(-1)
        next_coordinates = source_terms["coordinates"]
        self._validate_query_coordinates(next_coordinates)
        if next_field.shape[0] != next_coordinates.shape[0]:
            raise RuntimeError(
                "mapped LAI field and coordinates must have the same support size"
            )
        if next_field.device != next_coordinates.device:
            raise RuntimeError(
                "mapped LAI field and coordinates must use the same device"
            )
        if not torch.isfinite(next_field).all():
            raise ValueError("mapped LAI field must contain only finite values")
        self._current_field = next_field
        self._current_coordinates = next_coordinates
