"""Canonical plant growth-respiration carbon flux for RELIC.

References
----------
- Penning de Vries et al. (1974), Journal of Theoretical Biology 45, 339-377.
- Gifford (2003), Functional Plant Biology 30, 171-186.
- E3SM Land Model carbon allocation and available-carbon implementation.

Mathematical framework
----------------------
``C_avail = max(GPP - R_m, 0)``
``Y_g = clamp(Y_0 + a_np * NP / (NP + K_np), Y_min, Y_max)``
``R_g,potential = (1 - Y_g) * C_avail * f_T * f_S``
``R_g = min(R_g,potential, C_avail)``

All carbon fluxes use ``kg C m-2 s-1``. The explicit carbon-limit residual is
``R_g,potential - R_g`` and represents unrealized potential, not a hidden
source or sink. This level-2 process is an open carbon export port and does not
claim a strict Hamiltonian or GENERIC closure.
"""

from typing import Any, Dict, Optional, Union
import torch
from torch import Tensor

from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.unified_response_functions import (
    q10_temperature_response,
    michaelis_menten_response,
)
from relic.utils.physical_constants import get_physical_constant
from relic.computation.operators.iphs import advance_reaction_network


REFERENCE_TEMPERATURE_K = get_physical_constant("reference_temperature")


class GrowthRespiration(BiologicalProcessFunctor):
    """
    Growth Respiration Model following CASA framework.
    
    Architecture-compliant implementation using ParameterManager as single source of truth.
    
    Implements construction cost-based growth respiration with:
    - CASA growth efficiency (Ygrow) framework
    - Temperature dependence via Q10
    - Substrate limitation effects
    - N:P ratio influence on efficiency
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    
    def __init__(
        self,
        name: str = "growth_respiration",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        pft_name: Optional[str] = None
    ):
        """
        Initialize growth respiration model.
        
        Parameters
        ----------
        name : str, optional
            Model name for identification
        config : Dict[str, Any], optional
            Configuration parameters
        device : torch.device, optional
            Computation device (CPU/CUDA)
        pft_name : str, optional
            Plant Functional Type name for specialized parameters
        """
        # Device setup first
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
            
        # Set up parameter namespace BEFORE parent initialization
        self.base_namespace = "biosphere.respiration.growth"
        if pft_name:
            # Use PFT-specific namespace if provided
            self.namespace = f"biosphere.pft.{pft_name}.respiration.growth"
            self.pft_name = pft_name
        else:
            # Use general namespace
            self.namespace = self.base_namespace
            self.pft_name = None
            
        # Create metadata
        from relic.core.functor import FunctorMetadata, ProcessCategory, ExecutionPhase, SpatialScale, VariableKind
        metadata = FunctorMetadata(
            name=name,
            version="2.0.0",
            description="Carbon-bounded growth respiration in kg C m-2 s-1",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "R_g = min((1 - Y_g) * max(GPP - R_m, 0) * f_T * f_S, "
                "max(GPP - R_m, 0))"
            ),
            references=[
                "Gifford, R.M. (2003). Plant respiration in productivity models",
                "CASA-CNP model documentation"
            ],
            parameters={
                "biosphere.respiration.growth.ygrow_base": (
                    "Baseline growth efficiency [dimensionless]"
                ),
                "biosphere.respiration.growth.np_response_coeff": (
                    "N:P sensitivity of the growth efficiency [dimensionless]"
                ),
                "biosphere.respiration.growth.np_half_sat": (
                    "N:P half-saturation of the growth efficiency response [dimensionless]"
                ),
                "biosphere.respiration.growth.q10_growth": (
                    "Q10 of growth respiration [dimensionless]"
                ),
                "biosphere.respiration.growth.km_substrate_growth": (
                    "Soluble sugar half-saturation of growth [g C m-2]"
                ),
                "biosphere.respiration.growth.min_growth_efficiency": (
                    "Minimum growth efficiency [dimensionless]"
                ),
                "biosphere.respiration.growth.max_growth_efficiency": (
                    "Maximum growth efficiency [dimensionless]"
                ),
                "biosphere.validation.epsilon.nutrient": (
                    "Phosphorus denominator guard of the N:P ratio [g P m-2]"
                ),
            },
            input_variables=[
                "gross_primary_productivity",
                "maintenance_respiration_total",
                "air_temperature",
                "plant_nitrogen",
                "plant_phosphorus",
                "soluble_sugar",
            ],
            output_variables=[
                "growth_respiration_total",
                "diagnostics.growth_respiration.growth_efficiency",
                "diagnostics.growth_respiration.available_carbon",
                "diagnostics.growth_respiration.np_ratio_effect",
                "diagnostics.growth_respiration.temperature_factor",
                "diagnostics.growth_respiration.substrate_factor",
                "diagnostics.growth_respiration.carbon_limit_residual",
            ],
            conservation_laws=["carbon"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )

        # Initialize BiologicalProcessFunctor. The config goes to FunctorBase,
        # the single reader of config["dependencies"] and config parameters.
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
            device=device
        )
        # Default scheduled producers of the two carbon input fluxes. They are
        # instance-level declarations: add_dependency merges further names and
        # an explicit composition (set_dependencies, GraphExecutor links) rebinds
        # them. Plant nutrient and soluble-sugar pools are StateManager inputs,
        # not a synthetic carbon_pools Functor edge.
        for provider in ("farquhar_photosynthesis", "maintenance_respiration"):
            self.add_dependency(provider)

        # Set biological process parameters
        self.process_name = name
        self.t_ref = self._to_parameter_tensor(REFERENCE_TEMPERATURE_K).to(
            device=self.device,
            dtype=torch.float32,
        )
            
        # Initialize parameters - but do NOT store values, only register in ParameterManager
        self._register_parameters()
        
    def _register_parameters(self) -> None:
        """
        Ensure growth respiration parameters are registered.
        
        Note: Parameter bounds are now read from YAML instead of hardcoded.
        This follows the strict parameter management architecture where all
        parameter metadata (including bounds) must be defined in YAML files.
        """
        required_parameters = [
            ("ygrow_base", self.namespace),
            ("np_response_coeff", self.namespace),
            ("np_half_sat", self.namespace),
            ("q10_growth", self.namespace),
            ("km_substrate_growth", self.namespace),
            ("min_growth_efficiency", self.namespace),
            ("max_growth_efficiency", self.namespace),
            ("nutrient", "biosphere.validation.epsilon"),
        ]

        for name, namespace in required_parameters:
            secondary_namespace = None
            if namespace == self.namespace and self.base_namespace != self.namespace:
                secondary_namespace = self.base_namespace
            self._ensure_parameter_exists(
                name,
                namespace=namespace,
                secondary_namespace=secondary_namespace,
            )

    def _ensure_parameter_exists(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
        secondary_namespace: Optional[str] = None,
    ) -> None:
        """
        Verify parameter exists in ParameterManager.
        
        Bounds are automatically read from YAML parameter definitions.
        No need to pass bounds explicitly - they are part of the parameter metadata.
        """
        parameter_namespace = namespace or self.namespace
        value = self.get_parameter_value(
            name,
            namespace=parameter_namespace,
        )
        if value is None and secondary_namespace:
            value = self.get_parameter_value(
                name,
                namespace=secondary_namespace,
            )

        if value is None:
            full_key = f"{parameter_namespace}.{name}"
            raise KeyError(f"Required parameter '{full_key}' is missing from the ParameterManager")
        return value

    def _get_growth_parameter(
        self,
        name: str,
    ) -> Any:
        """Fetch growth respiration parameter with namespace cascade when needed."""
        primary_namespace = self.namespace
        secondary_namespace = None
        if self.base_namespace and self.base_namespace != primary_namespace:
            secondary_namespace = self.base_namespace
        value = self.get_parameter_value(
            name,
            namespace=primary_namespace,
        )
        if value is None and secondary_namespace:
            value = self.get_parameter_value(
                name,
                namespace=secondary_namespace,
            )
        if value is None:
            full_key = f"{primary_namespace}.{name}"
            raise KeyError(f"Required parameter '{full_key}' is missing from the ParameterManager")
        return value

    def _to_parameter_tensor(self, value: Any) -> Tensor:
        """Convert parameter values to tensors without forcing dtype changes."""
        if isinstance(value, Tensor):
            return value.to(device=self.device)
        return torch.as_tensor(value, device=self.device)
    
    @property
    def ygrow_base(self) -> torch.Tensor:
        """Get base growth efficiency."""
        if not hasattr(self, '_ygrow_base'):
            self._ygrow_base = self._to_parameter_tensor(
                self._get_growth_parameter(
                    "ygrow_base",
                ),
            )
        return self._ygrow_base
    
    @property
    def q10_growth(self) -> torch.Tensor:
        """Get Q10 temperature coefficient for growth."""
        if not hasattr(self, '_q10_growth'):
            self._q10_growth = self._to_parameter_tensor(
                self._get_growth_parameter(
                    "q10_growth",
                ),
            )
        return self._q10_growth
    
    @property
    def np_response_coeff(self) -> torch.Tensor:
        """Get N:P ratio response coefficient."""
        if not hasattr(self, '_np_response_coeff'):
            self._np_response_coeff = self._to_parameter_tensor(
                self._get_growth_parameter(
                    "np_response_coeff",
                ),
            )
        return self._np_response_coeff
    
    @property
    def np_half_sat(self) -> torch.Tensor:
        """Get N:P half-saturation constant."""
        if not hasattr(self, '_np_half_sat'):
            self._np_half_sat = self._to_parameter_tensor(
                self._get_growth_parameter(
                    "np_half_sat",
                ),
            )
        return self._np_half_sat
    
    @property
    def km_substrate_growth(self) -> torch.Tensor:
        """Get substrate half-saturation for growth."""
        if not hasattr(self, '_km_substrate_growth'):
            self._km_substrate_growth = self._to_parameter_tensor(
                self._get_growth_parameter(
                    "km_substrate_growth",
                ),
            )
        return self._km_substrate_growth
    
    @property
    def min_growth_efficiency(self) -> torch.Tensor:
        """Get minimum growth efficiency."""
        if not hasattr(self, '_min_growth_efficiency'):
            self._min_growth_efficiency = self._to_parameter_tensor(
                self._get_growth_parameter(
                    "min_growth_efficiency",
                ),
            )
        return self._min_growth_efficiency
    
    @property
    def max_growth_efficiency(self) -> torch.Tensor:
        """Get maximum growth efficiency."""
        if not hasattr(self, '_max_growth_efficiency'):
            self._max_growth_efficiency = self._to_parameter_tensor(
                self._get_growth_parameter(
                    "max_growth_efficiency",
                ),
            )
        return self._max_growth_efficiency
        
    def get_provides(self) -> list:
        """Get provided outputs."""
        return [
            "growth_respiration_total",
            "diagnostics.growth_respiration.growth_efficiency",
            "diagnostics.growth_respiration.available_carbon",
            "diagnostics.growth_respiration.np_ratio_effect",
            "diagnostics.growth_respiration.temperature_factor",
            "diagnostics.growth_respiration.substrate_factor",
            "diagnostics.growth_respiration.carbon_limit_residual",
        ]
        
    def forward(
        # References: Penning de Vries et al. (1974); Gifford (2003); existing E3SM-inspired
        #   available-carbon accounting.
        # Mathematical expression: Cavail=max(GPP-Rm,0); NP=N/(P+eps);
        #   Y=clip(Y0+a*NP/(NP+Knp),Ymin,Ymax); fT=Q10^((T-Tref)/10); fS=S/(Km+S);
        #   Rpot=(1-Y)*Cavail*fT*fS; Rg=min(Rpot,Cavail).
        # Parameter meanings: Y is growth efficiency; N/P is the declared tissue nutrient proxy; S
        #   is soluble sugar; existing parameters supply all scales and bounds.
        # Units: GPP,Rm,Cavail,Rpot,Rg [kg C m-2 s-1]; T [K]; N [g N m-2]; P [g P m-2];
        #   S,Km [g C m-2]; the N:P mass ratio and remaining factors [1].
        # Domain/constraints: Finite nonnegative required inputs, positive Kelvin temperature and
        #   valid parameter bounds. This is an open respiration port; Rpot-Rg is unrealized
        #   potential, not a new physical export.
        # Sign convention: Rg is positive outward carbon; the carbon cap prevents growth
        #   respiration exceeding available photosynthetic supply.
        self,
        state: Dict[str, Union[Tensor, float]],
        time: float,
        dt: float,
        **kwargs
    ) -> Dict[str, UpdateItem]:
        """
        Forward pass through growth respiration model.
        
        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state containing required variables (NO DEFAULTS PROVIDED)
        time : float
            Current simulation time [s]
        dt : float
            Time step [s]
        **kwargs
            Additional keyword arguments
            
        Returns
        -------
        Dict[str, UpdateItem]
            Canonical flux and namespaced diagnostics.
        """
        gpp = self._extract_tensor(state, "gross_primary_productivity")
        maintenance_resp = self._extract_tensor(
            state,
            "maintenance_respiration_total",
        )
        temperature = self._extract_tensor(state, "air_temperature")
        n_content = self._extract_tensor(state, "plant_nitrogen")
        p_content = self._extract_tensor(state, "plant_phosphorus")
        soluble_sugar = self._extract_tensor(state, "soluble_sugar")

        required_tensors = {
            "gross_primary_productivity": gpp,
            "maintenance_respiration_total": maintenance_resp,
            "air_temperature": temperature,
            "plant_nitrogen": n_content,
            "plant_phosphorus": p_content,
            "soluble_sugar": soluble_sugar,
        }
        for variable_name, value in required_tensors.items():
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"Input '{variable_name}' must contain only finite values"
                )
        for variable_name in (
            "gross_primary_productivity",
            "maintenance_respiration_total",
            "plant_nitrogen",
            "plant_phosphorus",
            "soluble_sugar",
        ):
            if torch.any(required_tensors[variable_name] < 0):
                raise ValueError(f"Input '{variable_name}' must be non-negative")
        if torch.any(temperature <= 0):
            raise ValueError("Input 'air_temperature' must be greater than 0 K")
        
        # Get parameters from ParameterManager (runtime access, always current values)
        ygrow_base_tensor = self.ygrow_base.to(device=gpp.device, dtype=gpp.dtype)
        np_response_coeff_tensor = self.np_response_coeff.to(device=gpp.device, dtype=gpp.dtype)
        np_half_sat_tensor = self.np_half_sat.to(device=gpp.device, dtype=gpp.dtype)
        q10_growth_tensor = self.q10_growth.to(device=gpp.device, dtype=gpp.dtype)
        km_substrate_tensor = self.km_substrate_growth.to(device=gpp.device, dtype=gpp.dtype)
        min_eff_tensor = self.min_growth_efficiency.to(device=gpp.device, dtype=gpp.dtype)
        max_eff_tensor = self.max_growth_efficiency.to(device=gpp.device, dtype=gpp.dtype)
        epsilon_nutrient_tensor = torch.as_tensor(
            self._ensure_parameter_exists(
                "nutrient",
                namespace="biosphere.validation.epsilon",
            ),
            device=gpp.device,
            dtype=gpp.dtype,
        )
        
        available_carbon = torch.clamp(gpp - maintenance_resp, min=0.0)
        
        # Calculate N:P ratio effect on growth efficiency
        # epsilon_nutrient has units g P m-2 to match p_content units
        np_ratio = n_content / (p_content + epsilon_nutrient_tensor)
        np_effect = np_response_coeff_tensor * np_ratio / (np_ratio + np_half_sat_tensor)
        
        # Calculate growth efficiency (Ygrow) with bounds
        growth_efficiency = torch.clamp(
            ygrow_base_tensor + np_effect, 
            min=min_eff_tensor, 
            max=max_eff_tensor
        )
        
        # Temperature response using Q10
        temp_factor = q10_temperature_response(
            torch.ones_like(gpp, dtype=gpp.dtype, device=gpp.device),
            temperature,
            q10_growth_tensor,
            self.t_ref
        )
        
        # Substrate limitation
        vmax_unit = torch.ones_like(soluble_sugar, dtype=soluble_sugar.dtype, device=soluble_sugar.device)
        substrate_factor = michaelis_menten_response(soluble_sugar, vmax_unit, km_substrate_tensor)
        
        # References: Penning de Vries et al. (1974); Gifford (2003); E3SM Land Model.
        # Mathematical expression: C_avail=max(GPP-R_m,0),
        # R_g,p=(1-Y_g)*C_avail*f_T*f_S, and R_g=min(R_g,p,C_avail).
        # Parameter meanings: Y_g is carbon-use efficiency; f_T is the Q10 response;
        # f_S is the sugar response; C_avail is carbon remaining after maintenance.
        # Units: GPP, R_m, C_avail, R_g,p, and R_g are kg C m-2 s-1.
        # Open-system route: R_g is a positive plant-to-atmosphere carbon port;
        # excess potential is reported as an observation diagnostic.
        potential_growth_respiration = (
            (1.0 - growth_efficiency)
            * available_carbon
            * temp_factor
            * substrate_factor
        )
        growth_respiration = torch.minimum(
            potential_growth_respiration,
            available_carbon,
        )
        carbon_limit_residual = potential_growth_respiration - growth_respiration

        # References: Penning de Vries et al. (1974); Gifford (2003).
        # Mathematical expression: x_dot = N_h r_h with r_h = [R_g] and N_h = [1].
        # Parameter meanings: N_h maps the unique carbon export to its open port.
        # Units: r_h and x_dot use kg C m-2 s-1.
        # Open-system route: port.
        growth_network = advance_reaction_network(
            name=f"{self.name}.growth_respiration_reaction_network",
            species=("growth_respiration_total",),
            reactions=("construction_growth_respiration",),
            stoichiometric_N=torch.ones((1, 1), device=growth_respiration.device, dtype=growth_respiration.dtype),
            rates=(growth_respiration,),
            open_system_route="port",
        )
        growth_respiration = growth_network.species_rate("growth_respiration_total")

        def replace(value: Tensor, unit: str) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=False,
                element=None,
                source_functor=self.name,
            )

        return {
            "growth_respiration_total": replace(
                growth_respiration,
                "kg C m-2 s-1",
            ),
            "diagnostics.growth_respiration.growth_efficiency": replace(
                growth_efficiency,
                "1",
            ),
            "diagnostics.growth_respiration.available_carbon": replace(
                available_carbon,
                "kg C m-2 s-1",
            ),
            "diagnostics.growth_respiration.np_ratio_effect": replace(
                np_effect,
                "1",
            ),
            "diagnostics.growth_respiration.temperature_factor": replace(
                temp_factor,
                "1",
            ),
            "diagnostics.growth_respiration.substrate_factor": replace(
                substrate_factor,
                "1",
            ),
            "diagnostics.growth_respiration.carbon_limit_residual": replace(
                carbon_limit_residual,
                "kg C m-2 s-1",
            ),
        }
    
    def _extract_tensor(
        self,
        state: Dict[str, Union[Tensor, float]],
        key: str,
    ) -> Tensor:
        """Extract one required canonical input without aliases or defaults."""
        if key not in state:
            raise KeyError(
                f"Required state variable '{key}' not found in input state. "
                f"Available keys: {list(state.keys())}"
            )

        value = state[key]
        
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        else:
            value = value.to(device=self.device)
            if not value.is_floating_point():
                value = value.to(dtype=torch.float32)
            
        return value


__all__ = ['GrowthRespiration']

