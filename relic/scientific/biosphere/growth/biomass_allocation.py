"""
Biomass Allocation Model for RELIC

This module implements carbon allocation to different plant organs (leaves, stems, roots)
with support for Plant Functional Type (PFT) parameterization, environmental controls,
and allometric relationships.

Mathematical Framework
----------------------
1. Allocation fractions: f_leaf + f_stem + f_root = 1
2. Growth rates: dB_i/dt = f_i * (NPP - F_BVOC,C)
3. Environmental modification: f_i = f_i,base * M_env(light, water, nutrients)
4. Allometric constraints: f_stem = a * (B_leaf/B_stem)^b
5. Pipe model: f_stem ? f_leaf * height_growth_demand

Implementation Details
----------------------
Based on PnET, CASA, and 3-PG model approaches:
- PnET temperature-dependent root allocation: TMult = exp(0.1 * (T - 7.1)) * 0.68
- CASA phenology-based allocation switching
- Environmental stress modifications
- LAI-dependent leaf allocation constraints
- Growth respiration is owned upstream by GrowthRespiration

References
----------
- Gao, Y. et al. (2025). pnetr: An R package for the PnET model family
- Zhang, Q. et al. (2011). CASA-CNP carbon allocation routines  
- Sands, P.J. & Landsberg, J.J. (2002). 3-PG allocation strategies
- Poorter, H., et al. (2012). Biomass allocation to leaves, stems and roots. 
  New Phytologist 193, 30-50
- Franklin, O., et al. (2012). Modeling carbon allocation in trees. 
  Tree Physiology 32, 648-666
- Litton, C.M., et al. (2007). Carbon allocation in forest ecosystems.
  Global Change Biology 13, 2089-2109
- Kesselmeier, J. et al. (2002). Volatile organic compound emissions
  in relation to plant carbon fixation and the terrestrial carbon budget.
  Global Biogeochemical Cycles 16, 1126

Units
-----
- Allocation fractions: dimensionless [0-1]
- Growth rates: kg C m-2 s^-1
- NPP: kg C m-2 s^-1
- Biomass: kg C m-2
"""

from typing import Dict, Optional, Union, Any, Tuple, List
import torch
from torch import Tensor
import logging

from relic.core.functor import FunctorMetadata, ProcessCategory, ExecutionPhase, SpatialScale, VariableKind
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.computation.operators.iphs import advance_reaction_network

logger = logging.getLogger(__name__)


def _tensor_range_text(value: Tensor) -> str:
    """Format a tensor for log messages as a scalar or a min..max range without altering the data."""
    detached = value.detach()
    if detached.numel() == 1:
        return f"{detached.reshape(-1).tolist()[0]:.3f}"
    return f"[{detached.min().tolist():.3f}, {detached.max().tolist():.3f}] (n={detached.numel()})"


class BiomassAllocation(BiologicalProcessFunctor):
    """Biomass allocation functor with ParameterManager-driven defaults."""
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    """
    Biomass Allocation Model for Carbon Distribution.
    
    This Functor implements comprehensive carbon allocation strategies based on
    PnET, CASA, and 3-PG model approaches. It distributes net primary production
    (NPP) among different plant organs (leaves, stems, roots) considering:
    
    - Plant Functional Type (PFT) specific allocation patterns  
    - Environmental controls (temperature, water, nutrients)
    - Phenological state (growth phase, LAI constraints)
    - Allometric relationships and size constraints
    - Explicit biomass-stock updates from canonical NPP
    
    Mathematical Framework
    ----------------------
    1. Base allocation fractions from PFT: f_leaf, f_stem, f_root (sum = 1.0)
    2. Environmental modification: f_i = f_i,base * M_env(T, W, N, light)
    3. Phenological control: allocation shifts based on phenology phase
    4. LAI constraints: reduce leaf allocation when LAI > LAI_max
    5. Allometric constraints: maintain structural relationships
    6. Growth rates: dB_i/dt = f_i * (NPP - F_BVOC,C)
    
    Implementation Strategy
    -----------------------
    Based on CASA-CNP and PnET allocation algorithms:
    - Fixed base fractions modified by environmental conditions
    - Temperature-dependent root allocation (PnET approach)
    - LAI-dependent leaf allocation (CASA approach)
    - Seasonal allocation switching based on phenology
    - Growth respiration remains owned by the upstream respiration ledger
    
    References
    ----------
    - Gao, Y. et al. (2025). pnetr: An R package for the PnET model family
    - Zhang, Q. et al. (2011). CASA-CNP carbon allocation routines
    - Sands, P.J. & Landsberg, J.J. (2002). 3-PG allocation strategies
    - Franklin, O. et al. (2012). Modeling carbon allocation in trees
    - Litton, C.M. et al. (2007). Carbon allocation in forest ecosystems
    
    Examples
    --------
    >>> allocation = BiomassAllocation(
    ...     name="biomass_allocation",
    ...     pft_name="broadleaf_deciduous_temperate",
    ...     enable_environmental_control=True,
    ...     enable_phenology_control=True
    ... )
    >>> state = {
    ...     'net_primary_productivity': torch.tensor(1.7361e-7),
    ...     'air_temperature': torch.tensor(298.15),
    ...     'leaf_area_index': 4.5,            # m^2 m^2
    ...     'phenology_phase': 2,              # Growth phase
    ...     'water_stress_factor': torch.tensor(0.8),
    ... }
    >>> result = allocation.forward(state, time=0, dt=1.0)
    >>> print(result['growth_rate_leaf'].value)
    """
    
    def __init__(
        self,
        name: str = "biomass_allocation",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        pft_name: Optional[str] = None,
        enable_environmental_control: bool = True,
        enable_phenology_control: bool = True,
        enable_allometry_control: bool = False,
        phenology_model_owner: str = "gsi_phenology",
        bvoc_carbon_allocation_owner: Optional[str] = None,
    ):
        """
        Initialize biomass allocation model.
        
        Parameters
        ----------
        name : str, optional
            Functor name, default "biomass_allocation"
        config : Optional[Dict[str, Any]]
            Configuration parameters that can override defaults
        device : Optional[torch.device]
            Computation device
        pft_name : Optional[str]
            Plant Functional Type name for parameter loading
        enable_environmental_control : bool, optional
            Enable environmental modification of allocation, default True
        enable_phenology_control : bool, optional
            Enable phenological control of allocation, default True
        enable_allometry_control : bool, optional
            Enable allometric constraints (experimental), default False
        phenology_model_owner : str, optional
            Selected compile-time phenology owner, default "gsi_phenology"
        bvoc_carbon_allocation_owner : Optional[str], optional
            Selected compile-time owner of the shared BVOC plant-carbon
            allocation port. None disables this interconnection.
        """
        if bvoc_carbon_allocation_owner is None:
            selected_bvoc_owner = None
        elif (
            not isinstance(bvoc_carbon_allocation_owner, str)
            or not bvoc_carbon_allocation_owner.strip()
        ):
            raise ValueError(
                "bvoc_carbon_allocation_owner must be None or a non-empty string"
            )
        else:
            selected_bvoc_owner = bvoc_carbon_allocation_owner.strip()

        input_variables = [
            'net_primary_productivity',
            'air_temperature',
            'water_stress_factor',
            'leaf_area_index',
            'phenology_phase',
            'leaf_biomass_c',
            'wood_biomass_c',
            'root_biomass_c',
        ]
        if selected_bvoc_owner is not None:
            input_variables.append('bvoc_carbon_allocation_flux')

        # Create FunctorBase metadata first
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Carbon allocation to leaves, stems, and roots with PFT and environmental controls",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "dB_i/dt = f_i * (NPP - F_BVOC,C); sum_i f_i = 1"
            ),
            references=[
                "Gao et al. (2025). pnetr: An R package for the PnET model family",
                "Zhang et al. (2011). CASA-CNP allocation routines", 
                "Sands & Landsberg (2002). 3-PG allocation strategies",
                "Franklin et al. (2012). Modeling carbon allocation in trees",
                "Kesselmeier et al. (2002). VOC emissions and the terrestrial carbon budget",
            ],
            parameters={
                # Allocation fractions
                'f_leaf_base': {
                    'description': 'Base fraction of NPP allocated to leaves',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.f_leaf_base'
                },
                'f_stem_base': {
                    'description': 'Base fraction of NPP allocated to stems',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.f_stem_base'
                },
                'f_root_base': {
                    'description': 'Base fraction of NPP allocated to roots',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.f_root_base'
                },
                # Environmental response parameters
                'root_alloc_temp_coeff': {
                    'description': 'Temperature coefficient for root allocation',
                    'units': 'K-1',
                    'namespace': 'biosphere.growth.allocation.root_alloc_temp_coeff'
                },
                'root_alloc_temp_ref': {
                    'description': 'Reference temperature for root allocation',
                    'units': 'K',
                    'namespace': 'biosphere.growth.allocation.root_alloc_temp_ref'
                },
                'root_alloc_temp_mult': {
                    'description': 'Temperature multiplier for root allocation',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.root_alloc_temp_mult'
                },
                'water_stress_threshold': {
                    'description': 'Threshold for water stress response',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.water_stress_threshold'
                },
                'temperature_stress_factor': {
                    'description': 'Temperature stress response factor',
                    'units': 'K-1',
                    'namespace': 'biosphere.growth.allocation.temperature_stress_factor'
                },
                'temperature_opt': {
                    'description': 'Optimal temperature for allocation',
                    'units': 'K',
                    'namespace': 'biosphere.growth.allocation.temperature_opt'
                },
                # LAI control parameters
                'lai_max_threshold': {
                    'description': 'Maximum LAI for leaf allocation control',
                    'units': 'm2 m-2',
                    'namespace': 'biosphere.growth.allocation.lai_max_threshold'
                },
                'lai_min_threshold': {
                    'description': 'Minimum LAI threshold',
                    'units': 'm2 m-2',
                    'namespace': 'biosphere.growth.allocation.lai_min_threshold'
                },
                # Phenology parameters
                'greenup_f_leaf': {
                    'description': 'Leaf allocation fraction during greenup',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.greenup_f_leaf'
                },
                'dormant_phase_code': {
                    'description': 'Integer code for the dormant phenology phase',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.dormant_phase_code'
                },
                'greenup_phase_code': {
                    'description': 'Integer code for the greenup phenology phase',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.greenup_phase_code'
                },
                'mature_phase_code': {
                    'description': 'Integer code for the mature phenology phase',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.mature_phase_code'
                },
                'senescence_phase_code': {
                    'description': 'Integer code for the senescence phenology phase',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.senescence_phase_code'
                },
                'dormant_phase_modifier': {
                    'description': 'Allocation modifier for the dormant phase',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.dormant_phase_modifier'
                },
                'greenup_phase_modifier': {
                    'description': 'Allocation modifier for the greenup phase',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.greenup_phase_modifier'
                },
                'mature_phase_modifier': {
                    'description': 'Allocation modifier for the mature phase',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.mature_phase_modifier'
                },
                'senescence_phase_modifier': {
                    'description': 'Allocation modifier for the senescence phase',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.senescence_phase_modifier'
                },
                'dormant_leaf_allocation': {
                    'description': 'Leaf allocation fraction used during dormancy',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.dormant_leaf_allocation'
                },
                'lai_max_leaf_allocation': {
                    'description': 'Leaf allocation fraction at the maximum LAI threshold',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.lai_max_leaf_allocation'
                },
                'herbaceous_stem_allocation': {
                    'description': 'Stem allocation fraction for herbaceous vegetation',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.herbaceous_stem_allocation'
                },
                'herbaceous_lai_min_stem_allocation': {
                    'description': 'Herbaceous stem allocation below the minimum LAI threshold',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.phenology.herbaceous_lai_min_stem_allocation'
                },
                'emergency_stem_fraction': {
                    'description': 'Emergency stem fraction used when allocation degenerates',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.emergency_stem_fraction'
                },
                'emergency_root_fraction': {
                    'description': 'Emergency root fraction used when allocation degenerates',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.emergency_root_fraction'
                },
                'default_environmental_modifier': {
                    'description': 'Neutral environmental diagnostic value when control is disabled',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.defaults.default_environmental_modifier'
                },
                'default_phenological_modifier': {
                    'description': 'Neutral phenology diagnostic value when control is disabled',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.defaults.default_phenological_modifier'
                },
                'temp_root_factor_min': {
                    'description': 'Lower bound for the temperature root-allocation factor',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.defaults.temp_root_factor_min'
                },
                'temp_root_factor_max': {
                    'description': 'Upper bound for the temperature root-allocation factor',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.defaults.temp_root_factor_max'
                },
                'biosphere.pft.<pft>.allocation.f_leaf': {
                    'description': 'PFT-specific leaf allocation fraction, read for the configured PFT',
                    'units': 'dimensionless',
                },
                'biosphere.pft.<pft>.allocation.f_stem': {
                    'description': 'PFT-specific stem allocation fraction, read for the configured PFT',
                    'units': 'dimensionless',
                },
                'biosphere.pft.<pft>.allocation.f_root': {
                    'description': 'PFT-specific root allocation fraction, read for the configured PFT',
                    'units': 'dimensionless',
                },
                # Allometric parameters
                'allometry_leaf_stem_exp': {
                    'description': 'Leaf-stem allometric exponent',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.allometry_leaf_stem_exp'
                },
                'allometry_root_shoot_exp': {
                    'description': 'Root-shoot allometric exponent',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.allometry_root_shoot_exp'
                },
                'allometric_blend_weight': {
                    'description': 'Blending weight for allometric constraint adjustment',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.allometric_blend_weight'
                },
                # Validation parameters
                'min_allocation_frac': {
                    'description': 'Minimum allocation fraction',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.min_allocation_frac'
                },
                'max_allocation_frac': {
                    'description': 'Maximum allocation fraction',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.max_allocation_frac'
                },
                'allocation_sum_tolerance': {
                    'description': 'Tolerance for allocation sum check',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.allocation_sum_tolerance'
                },
                'normalization_epsilon': {
                    'description': 'Positive denominator guard for allocation normalization',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.normalization_epsilon'
                },
                'fraction_zero_tolerance': {
                    'description': 'Tolerance for identifying zero allocation fractions',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.fraction_zero_tolerance'
                },
                'woody_stem_fraction_threshold': {
                    'description': 'Threshold for woody stem allocation behavior',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.woody_stem_fraction_threshold'
                },
                'allometry_biomass_floor': {
                    'description': 'Positive biomass floor for allometric ratios',
                    'units': 'kg C m-2',
                    'namespace': 'biosphere.growth.allocation.allometry_biomass_floor'
                },
                'allometry_ratio_max': {
                    'description': 'Maximum dimensionless allometric biomass ratio',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.growth.allocation.allometry_ratio_max'
                },
                'conservation_tolerance': {
                    'description': 'Relative carbon-ledger tolerance',
                    'units': 'dimensionless',
                    'namespace': 'validation.conservation.conservation_tolerance'
                },
            },
            input_variables=input_variables,
            output_variables=[
                'growth_rate_leaf',
                'growth_rate_stem',
                'growth_rate_root',
                'leaf_biomass_c',
                'wood_biomass_c',
                'root_biomass_c',
                'diagnostics.biomass_allocation.total_growth_rate',
                'diagnostics.biomass_allocation.allocation_fraction_leaf',
                'diagnostics.biomass_allocation.allocation_fraction_stem',
                'diagnostics.biomass_allocation.allocation_fraction_root',
                'diagnostics.biomass_allocation.allocation_sum',
                'diagnostics.biomass_allocation.environmental_modifier',
                'diagnostics.biomass_allocation.phenology_modifier',
                'diagnostics.biomass_allocation.carbon_balance_residual',
            ],
            conservation_laws=['carbon_partition'],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            spatial_dimensions=0
        )
        
        # Initialize parent class first
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
            device=device
        )
        
        # Production-ready implementation - no warnings needed
        
        # Store attributes
        self.pft_name = pft_name
        self.enable_environmental_control = enable_environmental_control
        self.enable_phenology_control = enable_phenology_control
        self.enable_allometry_control = enable_allometry_control
        if not isinstance(phenology_model_owner, str) or not phenology_model_owner.strip():
            raise ValueError("phenology_model_owner must be a non-empty string")
        self.phenology_model_owner = phenology_model_owner.strip()
        self.bvoc_carbon_allocation_owner = selected_bvoc_owner

        # Initialize parameters (registers them with parameter manager)
        self._initialize_parameters()

        # Load PFT parameters if provided (this registers PFT-specific parameters)
        if pft_name:
            self._load_pft_parameters(pft_name)
        else:
            # Default to global base allocation fractions
            self.f_leaf_base = self._ensure_tensor(
                self.get_parameter_value(
                    "f_leaf_base",
                    namespace="biosphere.growth.allocation", required=True)
            )
            self.f_stem_base = self._ensure_tensor(
                self.get_parameter_value(
                    "f_stem_base",
                    namespace="biosphere.growth.allocation", required=True)
            )
            self.f_root_base = self._ensure_tensor(
                self.get_parameter_value(
                    "f_root_base",
                    namespace="biosphere.growth.allocation", required=True)
            )

        if not hasattr(self, "min_allocation_frac"):
            self.min_allocation_frac = self._ensure_tensor(
                self.get_parameter_value(
                    "min_allocation_frac",
                    namespace="biosphere.growth.allocation", required=True)
            )
        if not hasattr(self, "max_allocation_frac"):
            self.max_allocation_frac = self._ensure_tensor(
                self.get_parameter_value(
                    "max_allocation_frac",
                    namespace="biosphere.growth.allocation", required=True)
            )
        
        # Phenology phase definitions (following CASA model, sourced from ParameterManager)
        self.DORMANT_PHASE = int(
            self.get_parameter_value(
                "dormant_phase_code",
                namespace="biosphere.growth.allocation.phenology", required=True)
        )
        self.GREENUP_PHASE = int(
            self.get_parameter_value(
                "greenup_phase_code",
                namespace="biosphere.growth.allocation.phenology", required=True)
        )
        self.MATURE_PHASE = int(
            self.get_parameter_value(
                "mature_phase_code",
                namespace="biosphere.growth.allocation.phenology", required=True)
        )
        self.SENESCENCE_PHASE = int(
            self.get_parameter_value(
                "senescence_phase_code",
                namespace="biosphere.growth.allocation.phenology", required=True)
        )

        # Inputs read only by a disabled control are optional; the required-input
        # rule (FunctorBase.get_required_inputs) reads this declaration.
        optional_inputs: List[str] = []
        if not self.enable_environmental_control:
            optional_inputs.extend(['air_temperature', 'water_stress_factor'])
        if not self.enable_phenology_control:
            optional_inputs.extend(['leaf_area_index', 'phenology_phase'])
        if not self.enable_allometry_control:
            optional_inputs.extend(['leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c'])
        self.optional_input_variables = optional_inputs
        logger.info(f"Initialized BiomassAllocation Functor with PFT: {pft_name}")

    def _initialize_parameters(self) -> None:
        """Initialize allocation parameters with defaults from PnET/CASA models."""

        # Note: No longer using register_buffer for parameters since get_parameter_value
        # provides tensors directly from ParameterManager. This maintains single source of truth.

        # Eagerly load key parameters during initialization to register them with parameter manager
        # This ensures tests can verify proper parameter system integration

        # Load temperature parameters (CF-1.8 compliant: all in Kelvin)
        _ = self.get_parameter_value(
            "root_alloc_temp_ref",
            namespace="biosphere.growth.allocation",
        )
        _ = self.get_parameter_value(
            "temperature_stress_factor",
            namespace="biosphere.growth.allocation",
        )
        _ = self.get_parameter_value(
            "temperature_opt",
            namespace="biosphere.growth.allocation",
        )

        # Load phenology phase modifiers
        _ = self.get_parameter_value(
            "dormant_phase_modifier",
            namespace="biosphere.growth.allocation.phenology",
        )
        _ = self.get_parameter_value(
            "greenup_phase_modifier",
            namespace="biosphere.growth.allocation.phenology",
        )
        _ = self.get_parameter_value(
            "mature_phase_modifier",
            namespace="biosphere.growth.allocation.phenology",
        )
        _ = self.get_parameter_value(
            "senescence_phase_modifier",
            namespace="biosphere.growth.allocation.phenology",
        )
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of Functor dependencies (not state variables).
        
        Returns
        -------
        List[str]
            Required Functor names that must execute before this one
        """
        dependencies = ["autotrophic_respiration"]
        if self.enable_environmental_control:
            dependencies.append("water_stress")
        if self.enable_phenology_control:
            dependencies.append(self.phenology_model_owner)
        if self.bvoc_carbon_allocation_owner is not None:
            dependencies.append(self.bvoc_carbon_allocation_owner)
        return self._merge_declared_dependencies(dependencies)
    
    def get_provides(self) -> List[str]:
        """
        Get list of output variables provided by this Functor.

        Returns
        -------
        List[str]
            Output variable names
        """
        return list(self.metadata.output_variables)

    _RUNTIME_DTYPE_KEYS = (
        "net_primary_productivity",
        "bvoc_carbon_allocation_flux",
        "air_temperature",
        "water_stress_factor",
        "leaf_area_index",
        "leaf_biomass_c",
        "wood_biomass_c",
        "root_biomass_c",
    )

    def _runtime_dtype(self, state: Dict[str, Union[Tensor, float]]) -> torch.dtype:
        """Return the floating dtype owned by the runtime state for this step.

        References:
          - PyTorch type promotion rules.
          - RELIC runtime-precision contract: the runtime state owns dtype and device.
        Mathematical expression:
          dtype_run = promote_types over the floating tensors found under the
          declared input keys, or torch.get_default_dtype() when none is present.
        Parameter meanings:
          state is the caller's state and forcing mapping.
        Units:
          none; dtype resolution carries no physical unit.
        Domain/constraints:
          only floating tensors participate; integer or missing entries are ignored,
          so a float64 simulation is never silently truncated to float32 here.
        Sign convention:
          not applicable.
        """
        resolved: Optional[torch.dtype] = None
        for key in self._RUNTIME_DTYPE_KEYS:
            value = state.get(key)
            if isinstance(value, Tensor) and value.is_floating_point():
                resolved = value.dtype if resolved is None else torch.promote_types(resolved, value.dtype)
        return resolved if resolved is not None else torch.get_default_dtype()

    def _ensure_tensor(self, value: Union[Tensor, float], dtype: torch.dtype = torch.float32) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(float(value), dtype=dtype, device=self.device)

    def _require_state_tensor(
        self,
        state: Dict[str, Union[Tensor, float]],
        key: str,
        *,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> Tensor:
        """Read one canonical state tensor and fail on missing or invalid values."""
        if key not in state:
            raise KeyError(f"Missing required state '{key}' for biomass allocation.")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"Required state '{key}' must be a torch.Tensor")
        value = value.to(device=self.device, dtype=self._runtime_dtype(state))
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"Required state '{key}' must be finite")
        if lower is not None and torch.any(value < lower):
            if lower == 0.0:
                raise ValueError(f"Required state '{key}' must be non-negative")
            raise ValueError(f"Required state '{key}' must be >= {lower}")
        if upper is not None and torch.any(value > upper):
            raise ValueError(f"Required state '{key}' must be <= {upper}")
        return value

    @staticmethod
    def _scalar_from_value(value: Union[Tensor, float, int]) -> float:
        """Convert scalar tensor or numeric value to Python float without direct scalar extraction sync calls."""
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError("Expected scalar tensor value.")
            detached = value.detach().reshape(-1)
            values = detached.tolist()
            if not values:
                raise ValueError("Expected non-empty scalar tensor value.")
            return float(values[0])
        return float(value)
    
    def forward(
        # References: Potter et al. (1993); Franklin et al. (2012); Kesselmeier et al. (2002),
        #   existing RELIC carbon partition.
        # Mathematical expression: A=NPP-F_BVOC_C when a BVOC owner is explicitly selected, else
        #   A=NPP; g_i=A*f_i; Delta_B_i=dt*g_i; r=NPP-F_BVOC_C-sum_i(g_i).
        # Parameter meanings: f_i are the final normalized leaf, stem and root fractions;
        #   respiration was already deducted upstream.
        # Units: NPP,g_i,F_BVOC_C [kg C m-2 s-1]; B_i,Delta_B_i [kg C m-2]; dt [s]; fractions [1].
        # Domain/constraints: Nonnegative NPP; selected BVOC debit cannot exceed NPP without a
        #   declared reserve; finite positive scalar dt; final rate residual must satisfy the
        #   existing tolerance.
        # Sign convention: Biomass increments are positive carbon input; BVOC is deducted once.
        #   Environmental/phenology modifiers are diagnostics, not extra multipliers of already
        #   allocated NPP.
        self,
        state: Dict[str, Union[Tensor, float]],
        time: float,
        dt: float,
        **kwargs
    ) -> Dict[str, UpdateItem]:
        """
        Carbon allocation core solver (BiologicalProcess forward pass).

        References
        ----------
        - Sands & Landsberg (2002). 3-PG allocation strategies.
        - Poorter et al. (2012). Biomass allocation to leaves, stems and roots.
        - Franklin et al. (2012). Modeling carbon allocation in trees.
        - Kesselmeier et al. (2002). VOC emissions and the terrestrial
          carbon budget.

        Mathematical Expressions
        ------------------------
        dB_i/dt = f_i * (NPP - F_BVOC,C)
            NPP is net_primary_productivity [kg C m-2 s-1].
            F_BVOC,C is the selected BVOC carbon allocation flux
            [kg C m-2 s-1].
        sum_i f_i = 1
            Environmental and phenological controls redistribute the fractions.

        Parameter Meaning
        -----------------
        - f_leaf_base/f_stem_base/f_root_base: baseline allocation fractions.
        - default_environmental_modifier/default_phenological_modifier: unity modifiers for diagnostics.
        - validation.conservation.*: tolerance configuration for conservation checks.
        - GrowthRespiration owns construction respiration before NPP is formed.
        """
        del time, kwargs  # unused here but retained for interface compliance
        state = dict(state)
        npp = self._require_state_tensor(
            state,
            "net_primary_productivity",
            lower=0.0,
        )
        if self.bvoc_carbon_allocation_owner is None:
            if "bvoc_carbon_allocation_flux" in state:
                raise ValueError(
                    "bvoc_carbon_allocation_owner must select the upstream "
                    "BVOC closure before bvoc_carbon_allocation_flux is consumed"
                )
            allocatable_npp = npp
            bvoc_carbon_allocation = None
        else:
            bvoc_carbon_allocation = self._require_state_tensor(
                state,
                "bvoc_carbon_allocation_flux",
                lower=0.0,
            )
            if torch.any(bvoc_carbon_allocation > npp):
                raise ValueError(
                    "bvoc_carbon_allocation_flux cannot exceed "
                    "net_primary_productivity without an explicit "
                    "non-structural carbon reserve"
                )
            allocatable_npp = npp - bvoc_carbon_allocation

        # References: Franklin et al. (2012); PFT allocation tables.
        # Mathematical expression: f = [f_leaf, f_stem, f_root].
        # Parameter meanings: f_i are instance-resolved allocation
        # fractions loaded from ParameterManager or PFT registry at
        # initialization, then used as coefficients in r_h.
        # References: RELIC runtime-precision contract; PyTorch type promotion.
        # Mathematical expression: all coefficients are cast to dtype_run of the state.
        # Parameter meanings: f_i are stored allocation fractions; dtype_run from _runtime_dtype.
        # Units: f_i [1].
        # Domain/constraints: casting widens float32 parameters to a float64 runtime and
        #   never narrows the state.
        # Sign convention: not applicable.
        runtime_dtype = self._runtime_dtype(state)
        f_leaf = self.f_leaf_base.to(device=self.device, dtype=runtime_dtype)
        f_stem = self.f_stem_base.to(device=self.device, dtype=runtime_dtype)
        f_root = self.f_root_base.to(device=self.device, dtype=runtime_dtype)

        default_environmental_modifier = self.get_parameter_value(
            "default_environmental_modifier",
            namespace="biosphere.growth.allocation.defaults",
        )
        env_modifier = torch.as_tensor(default_environmental_modifier, dtype=runtime_dtype, device=self.device)
        if self.enable_environmental_control:
            env_modifier, f_leaf, f_stem, f_root = self._apply_environmental_controls(state, f_leaf, f_stem, f_root)

        default_phenological_modifier = self.get_parameter_value(
            "default_phenological_modifier",
            namespace="biosphere.growth.allocation.defaults",
        )
        pheno_modifier = torch.as_tensor(default_phenological_modifier, dtype=runtime_dtype, device=self.device)
        if self.enable_phenology_control:
            pheno_modifier, f_leaf, f_stem, f_root = self._apply_phenological_controls(state, f_leaf, f_stem, f_root)

        if self.enable_allometry_control:
            f_leaf, f_stem, f_root = self._apply_allometric_constraints(state, f_leaf, f_stem, f_root)

        f_leaf, f_stem, f_root = self._normalize_allocation_fractions(f_leaf, f_stem, f_root)

        dt_tensor = self._ensure_tensor(dt, dtype=runtime_dtype)
        if dt_tensor.numel() != 1 or not torch.all(torch.isfinite(dt_tensor)):
            raise ValueError("dt must be one finite scalar")
        if torch.any(dt_tensor <= 0):
            raise ValueError("dt must be positive")

        # References: Potter et al. (1993); Franklin et al. (2012);
        # Poorter et al. (2012); ELM/CLM NPP allocation implementations.
        # Mathematical expression:
        # NPP = F_BVOC,C + sum_i(r_i), r_i = f_i * (NPP - F_BVOC,C),
        # sum_i(f_i) = 1, x_dot = N_h r_h, and delta B_i = r_i * dt.
        # Parameter meanings: f_i are ParameterManager/PFT allocation fractions;
        # NPP is the upstream post-respiration carbon flux; F_BVOC,C is the
        # selected BVOC synthesis or no-storage allocation debit; dt is the
        # step length.
        # Units: r_i, NPP, and F_BVOC,C are kg C m-2 s-1;
        # delta B_i is kg C m-2.
        # Open-system route: NPP enters through the autotrophic-respiration port
        # and is routed to the selected BVOC allocation port plus leaf, wood,
        # and root biomass stocks.
        growth_leaf = allocatable_npp * f_leaf
        growth_stem = allocatable_npp * f_stem
        growth_root = allocatable_npp * f_root
        allocation_network = advance_reaction_network(
            name=f"{self.name}.biomass_allocation_reaction_network",
            species=(
                "growth_rate_leaf",
                "growth_rate_stem",
                "growth_rate_root",
            ),
            reactions=(
                "leaf_npp_allocation",
                "stem_npp_allocation",
                "root_npp_allocation",
            ),
            stoichiometric_N=torch.as_tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=growth_leaf.device,
                dtype=growth_leaf.dtype,
            ),
            rates=(
                growth_leaf,
                growth_stem,
                growth_root,
            ),
            dt=dt_tensor,
            open_system_route="port",
        )
        growth_leaf = allocation_network.species_rate("growth_rate_leaf")
        growth_stem = allocation_network.species_rate("growth_rate_stem")
        growth_root = allocation_network.species_rate("growth_rate_root")
        total_growth = growth_leaf + growth_stem + growth_root
        carbon_balance_residual = npp - total_growth
        if bvoc_carbon_allocation is not None:
            carbon_balance_residual = (
                carbon_balance_residual - bvoc_carbon_allocation
            )
        conservation_tolerance_param = self.get_parameter_value(
            "conservation_tolerance",
            namespace="validation.conservation", required=True)
        conservation_tolerance = self._scalar_from_value(conservation_tolerance_param)
        conservation_scale = torch.where(
            torch.abs(npp) > 0,
            torch.abs(npp),
            torch.ones_like(npp),
        )
        relative_error = torch.abs(carbon_balance_residual) / conservation_scale
        if not torch.all(relative_error <= conservation_tolerance):
            raise ValueError(
                "Biomass allocation carbon ledger does not close within "
                f"relative tolerance {conservation_tolerance}."
            )

        def replace(value: Tensor, unit: str) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=False,
                source_functor=self.name,
            )

        def accumulate(value: Tensor) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.ACCUMULATE,
                unit="kg C m-2",
                per_timestep=True,
                element="C",
                source_functor=self.name,
            )

        state_delta = allocation_network.state_delta
        return {
            "growth_rate_leaf": replace(growth_leaf, "kg C m-2 s-1"),
            "growth_rate_stem": replace(growth_stem, "kg C m-2 s-1"),
            "growth_rate_root": replace(growth_root, "kg C m-2 s-1"),
            "leaf_biomass_c": accumulate(state_delta[0]),
            "wood_biomass_c": accumulate(state_delta[1]),
            "root_biomass_c": accumulate(state_delta[2]),
            "diagnostics.biomass_allocation.total_growth_rate": replace(
                total_growth,
                "kg C m-2 s-1",
            ),
            "diagnostics.biomass_allocation.allocation_fraction_leaf": replace(f_leaf, "1"),
            "diagnostics.biomass_allocation.allocation_fraction_stem": replace(f_stem, "1"),
            "diagnostics.biomass_allocation.allocation_fraction_root": replace(f_root, "1"),
            "diagnostics.biomass_allocation.allocation_sum": replace(
                f_leaf + f_stem + f_root,
                "1",
            ),
            "diagnostics.biomass_allocation.environmental_modifier": replace(env_modifier, "1"),
            "diagnostics.biomass_allocation.phenology_modifier": replace(pheno_modifier, "1"),
            "diagnostics.biomass_allocation.carbon_balance_residual": replace(
                carbon_balance_residual,
                "kg C m-2 s-1",
            ),
        }

    def check_conservation_laws(
        self,
        input_state: Dict[str, Union[Tensor, float]],
        output_updates: Dict[str, Union[Tensor, UpdateItem]],
    ) -> Dict[str, float]:
        """Audit the open NPP-to-growth carbon partition.

        References:
            - Sands and Landsberg (2002), 3-PG biomass allocation.
            - Franklin et al. (2012), tree carbon allocation.
            - Kesselmeier et al. (2002), BVOC carbon export.
        Mathematical expression:
            R_C = NPP - F_BVOC,C - F_leaf - F_stem - F_root.
            carbon_partition = max(|R_C| / |NPP|) for non-zero NPP;
            the absolute residual is used when NPP is exactly zero.
        Parameter meanings:
            NPP is upstream net primary productivity, F_BVOC,C is the
            compile-time selected BVOC carbon debit, and F_leaf, F_stem,
            and F_root are organ growth rates. All terms use
            [kg C m-2 s-1].
        Open-system route:
            NPP enters from the autotrophic-respiration ledger and is routed
            to explicit plant biomass stocks plus the selected BVOC boundary
            port. This audit does not claim a closed ecosystem carbon stock.
        """
        # References: Franklin et al. (2012); Kesselmeier et al. (2002); explicit plant-carbon
        #   partition contract.
        # Mathematical expression: r=NPP-F_BVOC_C-g_leaf-g_stem-g_root; report=max(abs(r)/scale),
        #   scale=abs(NPP) where positive and 1 otherwise.
        # Parameter meanings: F_BVOC_C is zero only when no upstream BVOC owner is selected; g_i
        #   are required canonical growth rates.
        # Units: Raw r and rates [kg C m-2 s-1]; normalized report [1].
        # Domain/constraints: Finite required terms with matching shape, dtype and device; do not
        #   substitute biomass pool amounts for rate ports.
        # Sign convention: Positive r is unallocated carbon supply; the returned magnitude does not
        #   close a physical budget by itself.

        def require_input(key: str) -> Tensor:
            if key not in input_state:
                raise KeyError(f"Missing conservation input '{key}'")
            value = input_state[key]
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Biomass allocation conservation input '{key}' "
                    "must be a torch.Tensor."
                )
            if value.numel() == 0:
                raise ValueError(
                    f"Biomass allocation conservation input '{key}' "
                    "must be non-empty."
                )
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"Biomass allocation conservation input '{key}' "
                    "must be floating."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"Biomass allocation conservation input '{key}' "
                    "must be finite."
                )
            if bool(torch.any(value < 0.0)):
                raise ValueError(
                    f"Biomass allocation conservation input '{key}' "
                    "must be non-negative."
                )
            return value.to(device=self.device, dtype=self._runtime_dtype(input_state))

        def require_output(key: str) -> Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Biomass allocation conservation output '{key}' "
                    "must be a torch.Tensor."
                )
            if value.numel() == 0:
                raise ValueError(
                    f"Biomass allocation conservation output '{key}' "
                    "must be non-empty."
                )
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"Biomass allocation conservation output '{key}' "
                    "must be floating."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"Biomass allocation conservation output '{key}' "
                    "must be finite."
                )
            if bool(torch.any(value < 0.0)):
                raise ValueError(
                    f"Biomass allocation conservation output '{key}' "
                    "must be non-negative."
                )
            return value

        npp = require_input('net_primary_productivity')
        if self.bvoc_carbon_allocation_owner is None:
            bvoc_carbon_allocation = torch.zeros_like(npp)
        else:
            bvoc_carbon_allocation = require_input(
                'bvoc_carbon_allocation_flux'
            )

        growth_rates = tuple(
            require_output(key)
            for key in (
                'growth_rate_leaf',
                'growth_rate_stem',
                'growth_rate_root',
            )
        )
        evidence = (bvoc_carbon_allocation, *growth_rates)
        if any(value.shape != npp.shape for value in evidence):
            raise ValueError(
                "Biomass allocation conservation terms must have identical shapes."
            )
        if any(
            value.device != npp.device or value.dtype != npp.dtype
            for value in evidence
        ):
            raise ValueError(
                "Biomass allocation conservation terms must share dtype and device."
            )

        residual = (
            npp
            - bvoc_carbon_allocation
            - growth_rates[0]
            - growth_rates[1]
            - growth_rates[2]
        )
        scale = torch.where(
            torch.abs(npp) > 0.0,
            torch.abs(npp),
            torch.ones_like(npp),
        )
        normalized_residual = torch.abs(residual) / scale
        return {
            'carbon_partition': float(
                torch.max(normalized_residual).detach().item()
            )
        }
    
    def _apply_environmental_controls(
        self,
        state: Dict[str, Union[Tensor, float]],
        f_leaf: Tensor,
        f_stem: Tensor,
        f_root: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Apply environmental controls to allocation fractions.
        
        Based on PnET temperature-dependent root allocation and water stress effects.
        
        References
        ----------
        - Gao et al. (2025). pnetr: An R package for the PnET model family
        - Aber & Federer (1992). A generalized, lumped-parameter model of photosynthesis, 
          evapotranspiration and net primary production. Ecological Modelling 92:1-32
        
        Mathematical Expression
        -----------------------
        TMult = exp(0.1 * (Tavg - 7.1)) * 0.68
        where Tavg is mean temperature in Celsius, 7.1 degC is the reference temperature,
        0.1 is the temperature sensitivity coefficient, and 0.68 is the baseline multiplier
        
        This temperature function increases root allocation in warmer conditions to
        support increased water and nutrient demand. The exponential response reflects
        Q10-like temperature sensitivity of root growth processes.
        
        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state variables including temperature (K) and water_stress (0-1)
        f_leaf, f_stem, f_root : Tensor
            Base allocation fractions (dimensionless, sum to 1.0)
            
        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            (environmental_modifier, f_leaf_modified, f_stem_modified, f_root_modified)
            All fractions adjusted to maintain sum = 1.0
        """
        # References: Gao et al. (2025), PnET model family; Poorter et al. (2012); existing RELIC
        #   calibrated allocation.
        # Mathematical expression: qT=clip(a*exp(b*(T-Tref))*exp(-c*abs(T-Topt)),qmin,qmax);
        #   qW=1+max(0,(wcrit-fw)/wcrit); fr_new=fr*qT*qW;
        #   fl_new,fs_new=(fl,fs)*(1-fr_new)/(fl+fs+eps) when aboveground allocation is positive.
        # Parameter meanings: a,b,c,Tref,Topt and bounds are parameter-owned; fw is the required
        #   water-stress control.
        # Units: T,Tref,Topt [K]; b,c [K-1]; fractions and q factors [1].
        # Domain/constraints: Finite positive-K input and fw in [0,1]; the following canonical
        #   normalization/ledger checks remain necessary. The no-aboveground branch preserves
        #   existing fractions.
        # Sign convention: Water limitation raises relative root investment; no carbon is created.
        #   Returned env_modifier=(qT+fw)/2 is descriptive only.
        temperature = self._require_state_tensor(
            state,
            'air_temperature',
            lower=0.0,
        )
        water_stress = self._require_state_tensor(
            state,
            'water_stress_factor',
            lower=0.0,
            upper=1.0,
        )
        
        # PnET temperature effect on root allocation
        # All temperature parameters use the canonical Kelvin unit contract.
        # TMult = exp(root_alloc_temp_coeff * (Tavg - reference_temp)) * root_alloc_temp_mult
        # Reference: Aber & Federer (1992), implemented in pnetr package
        root_alloc_temp_coeff = self.get_parameter_value(
            "root_alloc_temp_coeff",
            namespace="biosphere.growth.allocation",
        )
        root_alloc_temp_ref = self.get_parameter_value(
            "root_alloc_temp_ref",
            namespace="biosphere.growth.allocation",
        )
        root_alloc_temp_mult = self.get_parameter_value(
            "root_alloc_temp_mult",
            namespace="biosphere.growth.allocation",
        )

        temp_mult = torch.exp(
            root_alloc_temp_coeff * (temperature - root_alloc_temp_ref)
        ) * root_alloc_temp_mult

        # Modify root allocation based on temperature
        # Higher temperatures increase root allocation (drought response)
        # Apply additional temperature stress based on deviation from optimum
        temperature_opt = self.get_parameter_value(
            "temperature_opt",
            namespace="biosphere.growth.allocation",
        )
        temperature_stress_factor = self.get_parameter_value(
            "temperature_stress_factor",
            namespace="biosphere.growth.allocation",
        )
        temp_root_factor_min = self.get_parameter_value(
            "temp_root_factor_min",
            namespace="biosphere.growth.allocation.defaults",
        )
        temp_root_factor_max = self.get_parameter_value(
            "temp_root_factor_max",
            namespace="biosphere.growth.allocation.defaults",
        )
        
        temp_deviation = torch.abs(temperature - temperature_opt)
        temp_stress = torch.exp(-temperature_stress_factor * temp_deviation)
        temp_root_factor = torch.clamp(
            temp_mult * temp_stress, 
            min=temp_root_factor_min, 
            max=temp_root_factor_max
        )
        
        # Water stress effect: increase root allocation under water stress
        # Following CASA approach: more root allocation when water stressed
        water_stress_threshold = self.get_parameter_value(
            "water_stress_threshold",
            namespace="biosphere.growth.allocation",
        )
        water_stress_effect = torch.where(
            water_stress < water_stress_threshold,
            (water_stress_threshold - water_stress) / water_stress_threshold,
            torch.zeros_like(water_stress)
        )
        water_root_factor = 1.0 + water_stress_effect  # Increase root allocation under stress
        
        # Combined environmental effect on root allocation
        root_env_factor = temp_root_factor * water_root_factor
        
        # Apply environmental modification
        f_root_mod = f_root * root_env_factor
        
        # Adjust leaf and stem allocation to maintain balance
        # Reduce leaf allocation more than stem under stress (CASA logic)
        remaining_fraction = 1.0 - f_root_mod
        original_aboveground = f_leaf + f_stem
        normalization_epsilon = self.get_parameter_value(
            "normalization_epsilon",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        
        if torch.any(original_aboveground > 0):
            scale_factor = remaining_fraction / (
                original_aboveground + normalization_epsilon
            )
            f_leaf_mod = f_leaf * scale_factor
            f_stem_mod = f_stem * scale_factor
        else:
            f_leaf_mod = f_leaf
            f_stem_mod = f_stem
        
        # Calculate overall environmental modifier
        env_modifier = (temp_root_factor + water_stress) / 2.0
        
        return env_modifier, f_leaf_mod, f_stem_mod, f_root_mod
    
    def _apply_phenological_controls(
        self,
        state: Dict[str, Union[Tensor, float]],
        f_leaf: Tensor,
        f_stem: Tensor,
        f_root: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Apply phenological controls following CASA model approach.
        
        References
        ----------
        - Potter et al. (1993). Terrestrial ecosystem production: A process model based on
          global satellite and surface data. Global Biogeochem Cycles 7:811-841
        - Randerson et al. (1996). Contribution of increasing CO2 and climate to carbon 
          storage by ecosystems. Science 274:1170-1173
        - Zhang et al. (2011). CASA-CNP carbon allocation routines. 
          Technical Documentation, CSIRO Marine and Atmospheric Research
        
        Mathematical Expression
        -----------------------
        Phenological phases control allocation priority:
        Phase 0 (Dormant): f_leaf = 0, carbon to storage/roots
        Phase 1 (Greenup): f_leaf = greenup_f_leaf (0.8), rapid leaf expansion  
        Phase 2 (Growing): Standard allocation with LAI feedback
        Phase 3 (Senescence): f_leaf -> 0, carbon to storage organs
        
        LAI feedback: When LAI > LAI_max, suppress leaf allocation
        
        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state variables including phenology_phase and leaf_area_index
        f_leaf, f_stem, f_root : Tensor
            Current allocation fractions (dimensionless)
            
        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            (phenology_modifier, f_leaf_modified, f_stem_modified, f_root_modified)
            Fractions adjusted for seasonal growth priorities
        """
        # References: Zhang et al. (2011), CASA-CNP allocation routines; existing RELIC phase/LAI
        #   controls.
        # Mathematical expression: Dormant/high-LAI phases set leaf allocation from their parameter
        #   and redistribute stem/root by fs/(fs+fr+eps),fr/(fs+fr+eps). Greenup/low-LAI set
        #   leaf=g; woody stem/root=(1-g)*(e_s,e_r), herbaceous stem=a_h and root=1-g. Other cells
        #   keep input fractions.
        # Parameter meanings: Phase codes and LAI thresholds select the existing parameter-defined
        #   branches; the phase modifier is a separate diagnostic.
        # Units: LAI [m2 m-2]; all fractions/modifiers [1]; phase is an integer code.
        # Domain/constraints: Exactly one supported phase code per cell; nonnegative finite LAI;
        #   final normalization is required after piecewise allocation.
        # Sign convention: This changes partition fractions only; it does not apply a second
        #   phenological multiplier to total carbon supply.
        phenology_phase = self._require_state_tensor(state, 'phenology_phase')
        lai = self._require_state_tensor(state, 'leaf_area_index', lower=0.0)
        rounded_phase = torch.round(phenology_phase)
        valid_codes = torch.as_tensor(
            [
                self.DORMANT_PHASE,
                self.GREENUP_PHASE,
                self.MATURE_PHASE,
                self.SENESCENCE_PHASE,
            ],
            device=phenology_phase.device,
            dtype=torch.long,
        )
        if not torch.all(phenology_phase == rounded_phase):
            raise ValueError("phenology_phase must contain integer phase codes")
        if not torch.all(torch.isin(rounded_phase.to(dtype=torch.long), valid_codes)):
            raise ValueError("phenology_phase contains an unsupported phase code")
        
        # Initialize modified fractions
        f_leaf_mod = f_leaf.clone()
        f_stem_mod = f_stem.clone()
        f_root_mod = f_root.clone()
        normalization_epsilon = self.get_parameter_value(
            "normalization_epsilon",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        
        # Phase-based allocation (CASA approach)
        phase = rounded_phase.long()
        
        # Dormant phase (phase == 0): no leaf allocation
        dormant_mask = (phase == self.DORMANT_PHASE)
        dormant_leaf_allocation = self.get_parameter_value(
            "dormant_leaf_allocation",
            namespace="biosphere.growth.allocation.phenology",
        )
        dormant_leaf_tensor = torch.full_like(f_leaf_mod, dormant_leaf_allocation)

        f_leaf_mod = torch.where(dormant_mask, dormant_leaf_tensor, f_leaf_mod)
        remaining_dormant = torch.where(dormant_mask, f_leaf + f_stem + f_root, f_stem + f_root)
        f_stem_mod = torch.where(
            dormant_mask,
            f_stem / (f_stem + f_root + normalization_epsilon) * remaining_dormant,
            f_stem_mod,
        )
        f_root_mod = torch.where(
            dormant_mask,
            f_root / (f_stem + f_root + normalization_epsilon) * remaining_dormant,
            f_root_mod,
        )
        
        # Greenup phase (phase == 1): high leaf allocation
        greenup_mask = (phase == self.GREENUP_PHASE)
        greenup_f_leaf = self.get_parameter_value(
            "greenup_f_leaf",
            namespace="biosphere.growth.allocation",
        )
        f_leaf_greenup = greenup_f_leaf
        remaining_greenup = 1.0 - f_leaf_greenup
        woody_threshold = self.get_parameter_value(
            "woody_stem_fraction_threshold",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        emergency_stem_fraction = self.get_parameter_value(
            "emergency_stem_fraction",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        emergency_root_fraction = self.get_parameter_value(
            "emergency_root_fraction",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        
        # Check if woody (has stem allocation) vs herbaceous
        is_woody = f_stem > woody_threshold
        woody_greenup_mask = greenup_mask & is_woody
        herb_greenup_mask = greenup_mask & ~is_woody
        
        # Woody plants: split remaining between stem and root
        f_leaf_mod = torch.where(woody_greenup_mask, f_leaf_greenup, f_leaf_mod)
        f_stem_mod = torch.where(
            woody_greenup_mask,
            remaining_greenup * emergency_stem_fraction,
            f_stem_mod,
        )
        f_root_mod = torch.where(
            woody_greenup_mask,
            remaining_greenup * emergency_root_fraction,
            f_root_mod,
        )
        
        # Herbaceous plants: allocate remaining to roots only
        f_leaf_mod = torch.where(herb_greenup_mask, f_leaf_greenup, f_leaf_mod)
        herbaceous_stem_allocation = self.get_parameter_value(
            "herbaceous_stem_allocation",
            namespace="biosphere.growth.allocation.phenology",
        )
        herb_stem_tensor = torch.full_like(f_stem_mod, herbaceous_stem_allocation)

        f_stem_mod = torch.where(herb_greenup_mask, herb_stem_tensor, f_stem_mod)
        f_root_mod = torch.where(herb_greenup_mask, remaining_greenup, f_root_mod)
        
        # Mature/Senescence phases: LAI-based control
        mature_mask = (phase == self.MATURE_PHASE) | (phase == self.SENESCENCE_PHASE)
        
        # Get LAI threshold parameters
        lai_max_threshold = self.get_parameter_value(
            "lai_max_threshold",
            namespace="biosphere.growth.allocation",
        )
        lai_min_threshold = self.get_parameter_value(
            "lai_min_threshold",
            namespace="biosphere.growth.allocation",
        )
        
        # If LAI exceeds maximum, stop leaf allocation
        lai_max_mask = mature_mask & (lai >= lai_max_threshold)
        lai_max_leaf_allocation = self.get_parameter_value(
            "lai_max_leaf_allocation",
            namespace="biosphere.growth.allocation.phenology",
        )
        lai_max_tensor = torch.full_like(f_leaf_mod, lai_max_leaf_allocation)

        f_leaf_mod = torch.where(lai_max_mask, lai_max_tensor, f_leaf_mod)
        remaining_lai_max = torch.where(lai_max_mask, f_leaf + f_stem + f_root, f_stem + f_root)
        f_stem_mod = torch.where(
            lai_max_mask,
            f_stem / (f_stem + f_root + normalization_epsilon) * remaining_lai_max,
            f_stem_mod,
        )
        f_root_mod = torch.where(
            lai_max_mask,
            f_root / (f_stem + f_root + normalization_epsilon) * remaining_lai_max,
            f_root_mod,
        )
        
        # If LAI below minimum, increase leaf allocation
        lai_min_mask = mature_mask & (lai < lai_min_threshold)
        f_leaf_mod = torch.where(lai_min_mask, f_leaf_greenup, f_leaf_mod)
        remaining_lai_min = 1.0 - f_leaf_greenup
        
        # Woody vs herbaceous for low LAI case
        woody_lai_min_mask = lai_min_mask & is_woody
        herb_lai_min_mask = lai_min_mask & ~is_woody

        f_stem_mod = torch.where(woody_lai_min_mask, remaining_lai_min * emergency_stem_fraction, f_stem_mod)
        f_root_mod = torch.where(woody_lai_min_mask, remaining_lai_min * emergency_root_fraction, f_root_mod)
        
        herbaceous_lai_min_stem_allocation = self.get_parameter_value(
            "herbaceous_lai_min_stem_allocation",
            namespace="biosphere.growth.allocation.phenology",
        )
        herb_lai_min_tensor = torch.full_like(f_stem_mod, herbaceous_lai_min_stem_allocation)

        f_stem_mod = torch.where(herb_lai_min_mask, herb_lai_min_tensor, f_stem_mod)
        f_root_mod = torch.where(herb_lai_min_mask, remaining_lai_min, f_root_mod)
        
        # Calculate phenology modifier
        # Higher modifier during active growth phases
        pheno_modifier = torch.ones_like(phenology_phase)
        
        dormant_phase_modifier = self.get_parameter_value(
            "dormant_phase_modifier",
            namespace="biosphere.growth.allocation.phenology",
        )
        greenup_phase_modifier = self.get_parameter_value(
            "greenup_phase_modifier",
            namespace="biosphere.growth.allocation.phenology",
        )
        mature_phase_modifier = self.get_parameter_value(
            "mature_phase_modifier",
            namespace="biosphere.growth.allocation.phenology",
        )
        senescence_phase_modifier = self.get_parameter_value(
            "senescence_phase_modifier",
            namespace="biosphere.growth.allocation.phenology",
        )
        
        dormant_modifier_tensor = torch.full_like(pheno_modifier, dormant_phase_modifier)

        
        pheno_modifier = torch.where(phase == self.DORMANT_PHASE, dormant_modifier_tensor, pheno_modifier)
        greenup_modifier_tensor = torch.full_like(pheno_modifier, greenup_phase_modifier)
        pheno_modifier = torch.where(phase == self.GREENUP_PHASE, greenup_modifier_tensor, pheno_modifier)
        mature_modifier_tensor = torch.full_like(pheno_modifier, mature_phase_modifier)
        pheno_modifier = torch.where(phase == self.MATURE_PHASE, mature_modifier_tensor, pheno_modifier)
        senescence_modifier_tensor = torch.full_like(pheno_modifier, senescence_phase_modifier)
        pheno_modifier = torch.where(phase == self.SENESCENCE_PHASE, senescence_modifier_tensor, pheno_modifier)

        return pheno_modifier, f_leaf_mod, f_stem_mod, f_root_mod
    
    def _apply_allometric_constraints(
        self,
        state: Dict[str, Union[Tensor, float]],
        f_leaf: Tensor,
        f_stem: Tensor,
        f_root: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply allometric constraints based on pipe model and root-shoot balance.

        References
        ----------
        - Shinozaki, K., et al. (1964). A quantitative analysis of plant form
          - the pipe model theory. I. Basic Analyses. Japanese Journal of Ecology.
        - Enquist, B.J. & Niklas, K.J. (2002). Global allocation rules for
          patterns of biomass partitioning in seed plants. Science 295, 1517-1520.
        - Franklin, O., et al. (2012). Modeling carbon allocation in trees:
          a search for principles. Tree Physiology 32, 648-666.
        - Poorter, H., et al. (2012). Biomass allocation to leaves, stems
          and roots. New Phytologist 193, 30-50.

        Mathematical Expressions
        ------------------------
        1. Pipe model constraint (Shinozaki et al. 1964):
           f_stem_target = f_leaf * (B_leaf / B_stem)^allometry_leaf_stem_exp
           Sapwood cross-sectional area scales with leaf area, so stem investment
           must track leaf biomass ratio.

        2. Root-shoot functional balance (Enquist & Niklas 2002):
           R_target = (B_root / B_shoot)^allometry_root_shoot_exp
           f_root_target = R_target / (1 + R_target)
           Root allocation adjusts to maintain optimal root:shoot ratio.

        3. Blending (Franklin et al. 2012):
           f_i = (1 - w) * f_i_current + w * f_i_target
           where w = allometric_blend_weight from parameter system.

        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state variables. Requires leaf_biomass_c, wood_biomass_c,
            and root_biomass_c when allometric control is enabled.
        f_leaf, f_stem, f_root : Tensor
            Current allocation fractions
        allometry_leaf_stem_exp : float
            Exponent for leaf-stem allometric scaling (from parameter system)
        allometry_root_shoot_exp : float
            Exponent for root-shoot allometric scaling (from parameter system)
        allometric_blend_weight : float
            Blending weight [0-1] controlling constraint strength (from parameter system)

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            (f_leaf_modified, f_stem_modified, f_root_modified)
        """
        # References: Shinozaki et al. (1964); Enquist and Niklas (2002); Franklin et al. (2012).
        # Mathematical expression: fs_target=clip(fl*clip(Bl/max(Bs,Bmin),eps,rmax)^a,fmin,fmax);
        #   R=clip(Br/max(Bl+Bs,Bmin),eps,rmax)^b; fr_target=clip(R/(1+R),fmin,fmax);
        #   fl_target=clip(1-fs_target-fr_target,fmin,fmax); normalize targets by sum+eps;
        #   f_out=(1-w)*f+w*f_target.
        # Parameter meanings: Bl,Bs,Br are current carbon stocks; a,b are fitted exponents; w is
        #   blend weight; all floors/bounds use existing parameters.
        # Units: Stocks and Bmin [kg C m-2]; ratios,fractions,exponents,w [1].
        # Domain/constraints: Nonnegative finite stocks; this reduced allometric adjustment is not
        #   a universal mechanistic theorem. Current debug argument conversion fails on
        #   multi-element target tensors; see the recorded blocking probe.
        # Sign convention: Positive target fractions redistribute the same supplied carbon; this
        #   helper adds no source or sink.
        b_leaf_t = self._require_state_tensor(state, 'leaf_biomass_c', lower=0.0)
        b_stem_t = self._require_state_tensor(state, 'wood_biomass_c', lower=0.0)
        b_root_t = self._require_state_tensor(state, 'root_biomass_c', lower=0.0)

        # Retrieve allometric parameters from parameter system
        leaf_stem_exp = self.get_parameter_value(
            "allometry_leaf_stem_exp",
            namespace="biosphere.growth.allocation",
        )
        root_shoot_exp = self.get_parameter_value(
            "allometry_root_shoot_exp",
            namespace="biosphere.growth.allocation",
        )
        blend_weight = self.get_parameter_value(
            "allometric_blend_weight",
            namespace="biosphere.growth.allocation",
        )

        leaf_stem_exp = torch.as_tensor(leaf_stem_exp, device=f_leaf.device, dtype=f_leaf.dtype)
        root_shoot_exp = torch.as_tensor(root_shoot_exp, device=f_leaf.device, dtype=f_leaf.dtype)
        blend_weight = torch.as_tensor(blend_weight, device=f_leaf.device, dtype=f_leaf.dtype)

        biomass_floor = self.get_parameter_value(
            "allometry_biomass_floor",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        ratio_max = self.get_parameter_value(
            "allometry_ratio_max",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        normalization_epsilon = self.get_parameter_value(
            "normalization_epsilon",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        min_fraction = self.get_parameter_value(
            "min_allocation_frac",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        max_fraction = self.get_parameter_value(
            "max_allocation_frac",
            namespace="biosphere.growth.allocation",
            required=True,
        )

        # --- Pipe model constraint: stem must support leaf area ---
        # Reference: Shinozaki et al. (1964)
        # Math: f_stem_target = f_leaf * (B_leaf / B_stem)^leaf_stem_exp
        # When B_leaf/B_stem is high, stem investment increases to support canopy.
        b_stem_safe = torch.clamp(b_stem_t, min=biomass_floor)
        leaf_stem_ratio = torch.clamp(
            b_leaf_t / b_stem_safe,
            min=normalization_epsilon,
            max=ratio_max,
        )
        pipe_model_factor = torch.pow(leaf_stem_ratio, leaf_stem_exp)
        f_stem_target = torch.clamp(
            f_leaf * pipe_model_factor,
            min=min_fraction,
            max=max_fraction,
        )

        # --- Root-shoot functional balance ---
        # Reference: Enquist & Niklas (2002)
        # Math: R = (B_root / B_shoot)^root_shoot_exp; f_root_target = R / (1 + R)
        b_shoot = b_leaf_t + b_stem_t
        b_shoot_safe = torch.clamp(b_shoot, min=biomass_floor)
        root_shoot_ratio = torch.clamp(
            b_root_t / b_shoot_safe,
            min=normalization_epsilon,
            max=ratio_max,
        )
        r_target = torch.pow(root_shoot_ratio, root_shoot_exp)
        f_root_target = torch.clamp(
            r_target / (1.0 + r_target),
            min=min_fraction,
            max=max_fraction,
        )

        # --- Derive f_leaf_target from the constraint that fractions sum to 1 ---
        f_leaf_target = torch.clamp(
            1.0 - f_stem_target - f_root_target,
            min=min_fraction,
            max=max_fraction,
        )

        # Re-normalize targets to sum to 1.0
        target_total = f_leaf_target + f_stem_target + f_root_target
        f_leaf_target = f_leaf_target / (target_total + normalization_epsilon)
        f_stem_target = f_stem_target / (target_total + normalization_epsilon)
        f_root_target = f_root_target / (target_total + normalization_epsilon)

        # --- Blend current fractions with allometric targets ---
        # Reference: Franklin et al. (2012) - gradual adjustment approach
        # Math: f_i = (1 - w) * f_i_current + w * f_i_target
        f_leaf_out = (1.0 - blend_weight) * f_leaf + blend_weight * f_leaf_target
        f_stem_out = (1.0 - blend_weight) * f_stem + blend_weight * f_stem_target
        f_root_out = (1.0 - blend_weight) * f_root + blend_weight * f_root_target

        # Logging only: the fractions stay gridded tensors; the summary is
        # computed only when debug logging is enabled to avoid device syncs.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Allometric constraints applied: blend_w=%.2f, "
                "f_leaf %s->%s, f_stem %s->%s, f_root %s->%s",
                self._scalar_from_value(blend_weight),
                _tensor_range_text(f_leaf), _tensor_range_text(f_leaf_out),
                _tensor_range_text(f_stem), _tensor_range_text(f_stem_out),
                _tensor_range_text(f_root), _tensor_range_text(f_root_out),
            )

        return f_leaf_out, f_stem_out, f_root_out
    
    def _normalize_allocation_fractions(
        self,
        f_leaf: Tensor,
        f_stem: Tensor,
        f_root: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normalize allocation fractions to ensure they sum to 1.0.
        
        Parameters
        ----------
        f_leaf, f_stem, f_root : Tensor
            Allocation fractions to normalize
            
        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            Normalized allocation fractions
        """
        # References: Franklin et al. (2012); Poorter et al. (2012); existing bounded allocation
        #   contract.
        # Mathematical expression: Preserve fractions numerically close to zero; bound other
        #   fractions by fmin/fmax. If abs(sum(f)-1)>sum_tolerance, set f_i=f_i/(sum(f)+eps).
        # Parameter meanings: fraction_zero_tolerance, allocation_sum_tolerance, numerical epsilon
        #   and bounds are existing parameters.
        # Units: Fractions, sum, epsilon and tolerances [1].
        # Domain/constraints: This uses finite configured bounds and a tolerance-based
        #   normalization; sum(f) is not an exact algebraic one because the denominator includes
        #   eps.
        # Sign convention: Fractions are nonnegative; any remaining imbalance is tested by the
        #   existing carbon ledger, not hidden by a new source.
        # Apply bounds only where fractions are not intentionally set to zero
        # Allow zero allocation in special cases (e.g., dormant phase, high LAI)
        
        # References: Franklin et al. (2012); Poorter et al. (2012).
        # Mathematical expression: f_i = clip(f_i, f_min, f_max);
        # sum_i f_i = 1 for i in {leaf, stem, root}.
        # Parameter meanings: f_i are organ allocation fractions,
        # f_min and f_max are ParameterManager-backed physical bounds.
        # Get allocation bounds from parameter system
        min_allocation_frac = self.get_parameter_value(
            "min_allocation_frac",
            namespace="biosphere.growth.allocation",
        )
        max_allocation_frac = self.get_parameter_value(
            "max_allocation_frac",
            namespace="biosphere.growth.allocation",
        )
        allocation_sum_tolerance = self.get_parameter_value(
            "allocation_sum_tolerance",
            namespace="biosphere.growth.allocation",
        )
        fraction_zero_tolerance = self.get_parameter_value(
            "fraction_zero_tolerance",
            namespace="biosphere.growth.allocation",
            required=True,
        )
        normalization_epsilon = self.get_parameter_value(
            "normalization_epsilon",
            namespace="biosphere.growth.allocation",
            required=True,
        )

        min_allocation_frac = torch.as_tensor(
            min_allocation_frac, device=f_leaf.device, dtype=f_leaf.dtype
        )
        max_allocation_frac = torch.as_tensor(
            max_allocation_frac, device=f_leaf.device, dtype=f_leaf.dtype
        )
        allocation_sum_tolerance = torch.as_tensor(
            allocation_sum_tolerance, device=f_leaf.device, dtype=f_leaf.dtype
        )
        fraction_zero_tolerance = torch.as_tensor(
            fraction_zero_tolerance, device=f_leaf.device, dtype=f_leaf.dtype
        )
        normalization_epsilon = torch.as_tensor(
            normalization_epsilon, device=f_leaf.device, dtype=f_leaf.dtype
        )

        zero_leaf_allowed = torch.isclose(
            f_leaf,
            torch.zeros_like(f_leaf),
            atol=float(fraction_zero_tolerance),
        )
        zero_stem_allowed = torch.isclose(
            f_stem,
            torch.zeros_like(f_stem),
            atol=float(fraction_zero_tolerance),
        )
        zero_root_allowed = torch.isclose(
            f_root,
            torch.zeros_like(f_root),
            atol=float(fraction_zero_tolerance),
        )
        
        # Apply minimum bounds only where zero is not allowed
        f_leaf = torch.where(zero_leaf_allowed, torch.zeros_like(f_leaf), torch.clamp(f_leaf, min=min_allocation_frac))
        f_stem = torch.where(zero_stem_allowed, torch.zeros_like(f_stem), torch.clamp(f_stem, min=min_allocation_frac))
        f_root = torch.where(zero_root_allowed, torch.zeros_like(f_root), torch.clamp(f_root, min=min_allocation_frac))
        
        # Apply maximum bounds everywhere
        f_leaf = torch.clamp(f_leaf, max=max_allocation_frac)
        f_stem = torch.clamp(f_stem, max=max_allocation_frac)  
        f_root = torch.clamp(f_root, max=max_allocation_frac)
        
        # Calculate sum and normalize
        total = f_leaf + f_stem + f_root
        
        # Check if normalization is needed
        needs_normalization = torch.abs(total - 1.0) > allocation_sum_tolerance
        
        if torch.any(needs_normalization):
            f_leaf = torch.where(
                needs_normalization,
                f_leaf / (total + normalization_epsilon),
                f_leaf,
            )
            f_stem = torch.where(
                needs_normalization,
                f_stem / (total + normalization_epsilon),
                f_stem,
            )
            f_root = torch.where(
                needs_normalization,
                f_root / (total + normalization_epsilon),
                f_root,
            )
        
        return f_leaf, f_stem, f_root
    
    def _load_pft_parameters(self, pft_name: str) -> None:
        """
        Load PFT-specific allocation parameters via parameter manager.
        
        Parameters
        ----------
        pft_name : str
            Plant Functional Type name
        """
        pft_namespace = f"biosphere.pft.{pft_name}.allocation"

        def _get_optional_pft_parameter(name: str) -> Optional[Any]:
            return self.get_parameter_value(
                name,
                namespace=pft_namespace,
                source_order=("manager",),
            )

        f_leaf = _get_optional_pft_parameter("f_leaf")
        f_stem = _get_optional_pft_parameter("f_stem")
        f_root = _get_optional_pft_parameter("f_root")
        manager_values = (f_leaf, f_stem, f_root)
        if all(value is None for value in manager_values):
            # An unknown PFT: the ParameterManager is the one source of the triple, so
            # there is nothing to fall back to.
            from relic.scientific.biosphere.pft.pft_registry import list_available_pfts

            raise KeyError(
                f"Unknown PFT '{pft_name}': the ParameterManager holds no allocation "
                f"keys in namespace '{pft_namespace}'. Known PFTs: "
                f"{', '.join(sorted(list_available_pfts()))}"
            )
        if any(value is None for value in manager_values):
            raise KeyError(
                f"Incomplete PFT allocation parameters in namespace '{pft_namespace}'."
            )

        fractions = torch.as_tensor(
            [f_leaf, f_stem, f_root],
            device=self.device,
            dtype=torch.float32,
        )
        if not torch.all(torch.isfinite(fractions)):
            raise ValueError(f"PFT '{pft_name}' allocation fractions must be finite")
        if torch.any(fractions < 0):
            raise ValueError(f"PFT '{pft_name}' allocation fractions must be non-negative")
        total_allocation = fractions.sum()
        tolerance = self._scalar_from_value(
            self.get_parameter_value(
                "allocation_sum_tolerance",
                namespace="biosphere.growth.allocation",
                required=True,
            )
        )
        if not torch.isfinite(total_allocation) or torch.any(total_allocation <= 0):
            raise ValueError(f"PFT '{pft_name}' allocation fractions must have positive sum")
        if abs(self._scalar_from_value(total_allocation) - 1.0) > tolerance:
            raise ValueError(
                f"PFT '{pft_name}' allocation fractions must sum to 1 within {tolerance}"
            )
        fractions = fractions / total_allocation
        self.f_leaf_base, self.f_stem_base, self.f_root_base = fractions.unbind()

        logger.info(f"Successfully loaded allocation PFT parameters for {pft_name}")
        logger.debug(f"Allocation fractions - Leaf: {float(self.f_leaf_base):.3f}, "
                    f"Stem: {float(self.f_stem_base):.3f}, Root: {float(self.f_root_base):.3f}")
    
    def get_allocation_summary(self) -> Dict[str, float]:
        """
        Get summary of current allocation parameters.
        
        Returns
        -------
        Dict[str, float]
            Summary of allocation parameters
        """
        return {
            'base_allocation_leaf': self._scalar_from_value(self.f_leaf_base),
            'base_allocation_stem': self._scalar_from_value(self.f_stem_base),
            'base_allocation_root': self._scalar_from_value(self.f_root_base),
            'environmental_control': self.enable_environmental_control,
            'phenology_control': self.enable_phenology_control,
            'allometry_control': self.enable_allometry_control,
            'bvoc_carbon_allocation_owner': self.bvoc_carbon_allocation_owner,
            'pft_name': self.pft_name or 'default'
        }


__all__ = ['BiomassAllocation']
