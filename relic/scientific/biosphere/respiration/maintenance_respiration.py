"""
Maintenance Respiration Model for RELIC

This module provides a complete implementation of maintenance respiration
with Q10 temperature response, thermal acclimation, and substrate limitation.
Architecture-compliant version using ParameterManager as single source of truth.

References
----------
- Lloyd, J., Taylor, J.A. (1994). On the temperature dependence of soil respiration
- Atkin, O.K., Tjoelker, M.G. (2003). Thermal acclimation and dynamic response
- Ryan, M.G. (1991). Effects of climate change on plant respiration
- Reich, P.B., et al. (2016). Boreal and temperate trees show strong acclimation

Mathematical Expressions
------------------------
1. Q10 response: R(T) = R_ref * Q10^((T-T_ref)/10)
2. Acclimation: R_ref,acc = R_ref,base * exp[-a * (T_growth - T_standard)]
3. Substrate limitation: f_sub = S / (Km + S)
4. Total: R_m = R_tissue * f_T * f_sub * f_H2O

Units
-----
- R_m: kg C m-2 s-1
- Base rates: s-1
- Temperature: K
- Biomass: g C m-2
- Substrate: g C m-2
"""

from numbers import Integral
from typing import Dict, Mapping, Optional, Union, Any
import torch
from torch import Tensor
import logging

# Import from core framework
from relic.core.functor import BiologicalProcessFunctor
from relic.utils.unified_response_functions import (
    q10_temperature_response,
    michaelis_menten_response,
    stress_response_functions,
    validate_temperature_range,
)
from relic.utils.physical_constants import get_physical_constant
from relic.data.standards.unit_converter import unit_converter, ConversionContext
from relic.computation.operators.iphs import advance_reaction_network
from relic.core.state import UpdateItem, UpdateMode

# Initialize logger
logger = logging.getLogger(__name__)

REFERENCE_TEMPERATURE_K = get_physical_constant("reference_temperature")


class MaintenanceRespiration(BiologicalProcessFunctor):
    """
    Maintenance Respiration Model with Q10 response.
    
    Architecture-compliant implementation using ParameterManager as single source of truth.
    
    Features
    --------
    - Q10 temperature response with literature parameters
    - Thermal acclimation based on growth temperature history
    - Substrate (soluble sugar) limitation
    - Water stress effects
    - Tissue-specific respiration rates
    - Temperature history tracking for acclimation
    
    Architecture
    ------------
    - Inherits from BiologicalProcess for framework integration
    - Uses unified response function library
    - Supports multiple tissue types
    - Pure runtime parameter access (no static caching)
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    optional_input_variables = ["growing_temperature"]
    # The acclimation temperature history is replay-critical hidden state: it
    # is carried as module extra state so that the accepted-step snapshot and
    # durable checkpoints restore the exact window (G08 memory contract).
    EXTRA_STATE_FORMAT = "relic.maintenance_respiration.acclimation_history"
    EXTRA_STATE_VERSION = 1
    
    def __init__(
        self,
        name: str = "maintenance_respiration",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        enable_acclimation: bool = True,
        acclimation_days: int = 10,
        pft_name: Optional[str] = None
    ):
        """
        Initialize maintenance respiration model.
        
        Parameters
        ----------
        name : str, optional
            Model name for identification
        config : Dict[str, Any], optional
            Configuration parameters
        device : torch.device, optional
            Computation device (CPU/CUDA)
        enable_acclimation : bool, optional
            Enable temperature acclimation, default True
        acclimation_days : int, optional
            Days for temperature history averaging, default 10
        pft_name : str, optional
            Plant Functional Type name for specialized parameters
        """
        # Initialize base classes
        conf = config or {}
        
        # Device setup first
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
            
        # Set up parameter namespace BEFORE parent initialization
        self.base_namespace = "biosphere.respiration.maintenance"
        if pft_name:
            # Use PFT-specific namespace if provided
            self.namespace = f"biosphere.pft.{pft_name}.respiration.maintenance"
            self.pft_name = pft_name
        else:
            # Use general namespace
            self.namespace = self.base_namespace
            self.pft_name = None
        
        # Create metadata
        from relic.core.functor import (
            ExecutionPhase,
            FunctorMetadata,
            ProcessCategory,
            SpatialScale,
            VariableKind,
        )
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Maintenance respiration with Q10 response and acclimation",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="R(T) = R_ref * Q10^((T-T_ref)/10)",
            references=[
                "Ryan, M.G. (1991). Effects of climate change on plant respiration",
                "Atkin & Tjoelker (2003). Thermal acclimation and dynamic response"
            ],
            parameters={
                "biosphere.respiration.maintenance.rm_leaf_25": (
                    "Leaf maintenance respiration coefficient at 25 degC [s-1]"
                ),
                "biosphere.respiration.maintenance.rm_stem_25": (
                    "Stem maintenance respiration coefficient at 25 degC [s-1]"
                ),
                "biosphere.respiration.maintenance.rm_root_25": (
                    "Root maintenance respiration coefficient at 25 degC [s-1]"
                ),
                "biosphere.respiration.maintenance.q10_respiration": (
                    "Q10 of maintenance respiration [dimensionless]"
                ),
                "biosphere.respiration.maintenance.q10_leaf_factor": (
                    "Leaf-specific Q10 multiplier [dimensionless]"
                ),
                "biosphere.respiration.maintenance.q10_stem_factor": (
                    "Stem-specific Q10 multiplier [dimensionless]"
                ),
                "biosphere.respiration.maintenance.q10_root_factor": (
                    "Root-specific Q10 multiplier [dimensionless]"
                ),
                "biosphere.respiration.maintenance.acclimation_rate_coeff": (
                    "Thermal acclimation coefficient [K-1]"
                ),
                "biosphere.respiration.maintenance.km_substrate": (
                    "Soluble-carbon half-saturation pool [g C m-2]"
                ),
                "biosphere.respiration.maintenance.water_stress_wilting_point": (
                    "Water potential of zero respiration response [MPa]"
                ),
                "biosphere.respiration.maintenance.water_stress_field_capacity": (
                    "Water potential of full respiration response [MPa]"
                ),
            },
            input_variables=[
                "air_temperature",
                "biomass_leaf",
                "biomass_stem",
                "biomass_root",
                "soluble_sugar",
                "soil_water_potential",
                "growing_temperature",
            ],
            output_variables=[
                "maintenance_respiration_total",
                "diagnostics.maintenance_respiration.leaf_flux",
                "diagnostics.maintenance_respiration.stem_flux",
                "diagnostics.maintenance_respiration.root_flux",
                "diagnostics.maintenance_respiration.leaf_rate_coefficient",
                "diagnostics.maintenance_respiration.substrate_factor",
                "diagnostics.maintenance_respiration.water_stress_factor",
                "diagnostics.maintenance_respiration.growth_temperature",
            ],
            conservation_laws=["carbon"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        
        # Initialize BiologicalProcessFunctor. The config goes to FunctorBase,
        # the single reader of config["dependencies"] and config parameters.
        super().__init__(
            name=name,
            config=conf,
            metadata=metadata,
            device=device
        )

        # Set biological process parameters
        self.process_name = name

        # Acclimation settings
        if (
            isinstance(acclimation_days, bool)
            or not isinstance(acclimation_days, Integral)
            or int(acclimation_days) < 1
        ):
            raise ValueError("acclimation_days must be a positive integer window length")
        self.enable_acclimation = enable_acclimation
        self.acclimation_days = int(acclimation_days)

        # Initialize parameters and expose snapshots for diagnostics/tests
        self._register_parameters()

        # Rolling growth-temperature window (spatial mean per sample) as a
        # fixed ring buffer: oldest sample overwritten once the window is full.
        self._history_buffer = torch.zeros(self.acclimation_days, dtype=torch.float64)
        self._history_count = 0
        self._history_index = 0


    # ------------------------------------------------------------------
    # Acclimation history ownership (module extra state)
    # ------------------------------------------------------------------
    @property
    def temperature_history(self) -> Tensor:
        """Return the valid growth-temperature samples, oldest first [K]."""
        if self._history_count < self.acclimation_days:
            return self._history_buffer[: self._history_count].clone()
        return torch.roll(self._history_buffer, shifts=-self._history_index, dims=0).clone()

    def get_extra_state(self) -> Dict[str, Any]:
        """Return the replay-critical acclimation window for rollback and restart.

        References:
          - Atkin and Tjoelker (2003): acclimation follows the growth
            temperature of the preceding days.
          - PyTorch Module extra-state contract; revised RELIC foundations,
            memory/history restart contract (only the current state is not an
            exact restart).
        Mathematical expression:
          T_growth = mean(T_k, k in valid window); the window buffer, its
          valid count, and the next write index identify the history exactly.
        Parameter meanings:
          window_days N, count <= N valid samples, index next write slot,
          buffer the N spatial-mean temperatures.
        Units: buffer K; count, index, window_days 1.
        Domain/constraints: exact copy; restoring into a model with another
          window length is rejected.
        Sign convention: not applicable.
        """
        return {
            "format": self.EXTRA_STATE_FORMAT,
            "version": self.EXTRA_STATE_VERSION,
            "window_days": self.acclimation_days,
            "count": self._history_count,
            "index": self._history_index,
            "buffer": self._history_buffer.detach().cpu().clone(),
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Maintenance respiration extra state must be a mapping.")
        expected = {"format", "version", "window_days", "count", "index", "buffer"}
        if set(state) != expected:
            raise ValueError("Maintenance respiration extra-state keys do not match the schema.")
        if state["format"] != self.EXTRA_STATE_FORMAT:
            raise ValueError("Maintenance respiration extra-state format is unsupported.")
        if state["version"] != self.EXTRA_STATE_VERSION:
            raise ValueError("Maintenance respiration extra-state version is unsupported.")
        window = state["window_days"]
        count = state["count"]
        index = state["index"]
        buffer = state["buffer"]
        for name, value in (("window_days", window), ("count", count), ("index", index)):
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"Maintenance respiration extra-state {name} must be a non-negative integer.")
        if int(window) != self.acclimation_days:
            raise ValueError(
                "Maintenance respiration extra-state window length does not match acclimation_days."
            )
        if int(count) > self.acclimation_days or int(index) >= self.acclimation_days:
            raise ValueError("Maintenance respiration extra-state count/index exceed the window.")
        if (
            not isinstance(buffer, Tensor)
            or buffer.shape != (self.acclimation_days,)
            or not torch.is_floating_point(buffer)
            or not bool(torch.isfinite(buffer).all())
        ):
            raise TypeError("Maintenance respiration extra-state buffer must be a finite window tensor.")
        self._history_buffer = buffer.detach().cpu().to(torch.float64).clone()
        self._history_count = int(count)
        self._history_index = int(index)
        
        
    def _register_parameters(self) -> None:
        """
        Ensure all required parameters exist in the ParameterManager.

        Note: Bounds are read from YAML definitions; no hardcoded bounds here.
        """
        required_parameters = [
            ("rm_leaf_25", self.namespace),
            ("rm_stem_25", self.namespace),
            ("rm_root_25", self.namespace),
            ("q10_respiration", self.namespace),
            ("q10_leaf_factor", self.namespace),
            ("q10_stem_factor", self.namespace),
            ("q10_root_factor", self.namespace),
            ("acclimation_rate_coeff", self.namespace),
            ("km_substrate", self.namespace),
            ("water_stress_wilting_point", self.namespace),
            ("water_stress_field_capacity", self.namespace),
        ]

        self.parameter_snapshots: Dict[str, Tensor] = {}
        for name, namespace in required_parameters:
            secondary_namespace = None
            if namespace == self.namespace and self.base_namespace != self.namespace:
                secondary_namespace = self.base_namespace
            value = self._ensure_parameter_exists(
                name,
                namespace=namespace,
                secondary_namespace=secondary_namespace,
            )
            snapshot = self._to_tensor(value)
            self.parameter_snapshots[name] = snapshot
            # Expose snapshot as attribute for diagnostics/tests
            setattr(self, name, snapshot)

    def _ensure_parameter_exists(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
        secondary_namespace: Optional[str] = None,
    ) -> Any:
        """
        Verify that a required parameter is registered.

        Bounds are read from YAML definitions; no hardcoded bounds here.
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

    def _get_maintenance_parameter(self, name: str) -> Any:
        """Fetch maintenance respiration parameter with namespace cascade when needed."""
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
    
    def get_provides(self) -> list:
        """Get provided outputs."""
        return list(self.metadata.output_variables)
    
    def forward(
        # References: Ryan (1991); Atkin and Tjoelker (2003); Reich et al. (2016).
        # Mathematical expression:
        #   Rm_i=rref_i*exp(-a*(Tgrowth-Tref))*(Q10_base*Q10_i)^((T-Tref)/10)*B_i*S/(Km+S)*fw;
        #   total=convert(sum_i(Rm_i),g C to kg C), omitting acclimation when disabled/unready.
        # Parameter meanings: i denotes leaf, stem and root; fw is the existing linear
        #   water-potential response; externally supplied growing_temperature overrides the history
        #   estimate.
        # Units: rref_i [s-1]; B_i,S,Km [g C m-2]; internal rates [g C m-2 s-1]; public total [kg C
        #   m-2 s-1]; temperatures [K].
        # Domain/constraints: Finite required inputs, nonnegative biomass/substrate and the current
        #   temperature-range guard; acclimation activation follows the existing external-driver or
        #   minimum-sample rule.
        # Sign convention: Maintenance respiration is a positive carbon export rate; this Functor
        #   does not independently debit plant stocks.
        self,
        state: Dict[str, Union[Tensor, float]],
        time: float,
        dt: float,
        **kwargs
    ) -> Dict[str, UpdateItem]:
        """
        Forward pass through maintenance respiration model.
        
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
        Dict[str, Tensor]
            Respiration outputs with all components
        """
        working_state = dict(state)
        temperature = self._extract_tensor(working_state, 'air_temperature')
        biomass_leaf = self._extract_tensor(working_state, 'biomass_leaf')
        biomass_stem = self._extract_tensor(working_state, 'biomass_stem')
        biomass_root = self._extract_tensor(working_state, 'biomass_root')
        soluble_sugar = self._extract_tensor(working_state, 'soluble_sugar')
        water_potential = self._extract_tensor(working_state, 'soil_water_potential')

        required_tensors = {
            'air_temperature': temperature,
            'biomass_leaf': biomass_leaf,
            'biomass_stem': biomass_stem,
            'biomass_root': biomass_root,
            'soluble_sugar': soluble_sugar,
            'soil_water_potential': water_potential,
        }
        for variable_name, value in required_tensors.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Input '{variable_name}' must contain only finite values")
        for variable_name in ('biomass_leaf', 'biomass_stem', 'biomass_root', 'soluble_sugar'):
            if torch.any(required_tensors[variable_name] < 0):
                raise ValueError(f"Input '{variable_name}' must be non-negative")

        validate_temperature_range(temperature, min_temp=233.15, max_temp=333.15)
        
        # Get growth temperature from state if provided, otherwise calculate from history
        if 'growing_temperature' in working_state:
            growth_temperature = self._extract_tensor(working_state, 'growing_temperature')
            if not torch.isfinite(growth_temperature).all():
                raise ValueError("Input 'growing_temperature' must contain only finite values")
        else:
            # Update temperature history for acclimation
            self._update_temperature_history(temperature)
            growth_temperature = self._calculate_growth_temperature()
        
        # Get parameters from ParameterManager (runtime access, always current values)
        # Base respiration rates at 25C reference temperature
        # Reference: Ryan et al. (1996), Reich et al. (2008)
        rm_leaf_25_tensor = self._to_tensor(
            self._get_maintenance_parameter("rm_leaf_25")
        )
        rm_stem_25_tensor = self._to_tensor(
            self._get_maintenance_parameter("rm_stem_25")
        )
        rm_root_25_tensor = self._to_tensor(
            self._get_maintenance_parameter("rm_root_25")
        )

        # Get Q10 parameters
        # Reference: Atkin et al. (2005) - Q10 typically ranges 1.5-4.0
        q10_base = self._to_tensor(
            self._get_maintenance_parameter("q10_respiration")
        )
        q10_leaf_factor = self._to_tensor(
            self._get_maintenance_parameter("q10_leaf_factor")
        )
        q10_stem_factor = self._to_tensor(
            self._get_maintenance_parameter("q10_stem_factor")
        )
        q10_root_factor = self._to_tensor(
            self._get_maintenance_parameter("q10_root_factor")
        )

        q10_leaf = q10_base * q10_leaf_factor
        q10_stem = q10_base * q10_stem_factor
        q10_root = q10_base * q10_root_factor

        # Calculate acclimated parameters if enabled
        if self.enable_acclimation and (
            'growing_temperature' in working_state or len(self.temperature_history) >= 3
        ):
            rm_ref_acclimated = self._apply_acclimation(
                growth_temperature, rm_leaf_25_tensor, rm_stem_25_tensor, rm_root_25_tensor
            )
        else:
            rm_ref_acclimated = {
                'leaf': rm_leaf_25_tensor,
                'stem': rm_stem_25_tensor,
                'root': rm_root_25_tensor
            }

        # Calculate temperature response using Q10
        temp_factor_leaf = q10_temperature_response(
            rm_ref_acclimated['leaf'], temperature, q10_leaf, REFERENCE_TEMPERATURE_K
        )
        temp_factor_stem = q10_temperature_response(
            rm_ref_acclimated['stem'], temperature, q10_stem, REFERENCE_TEMPERATURE_K
        )
        temp_factor_root = q10_temperature_response(
            rm_ref_acclimated['root'], temperature, q10_root, REFERENCE_TEMPERATURE_K
        )

        # Calculate substrate limitation
        # Michaelis-Menten half-saturation constant for soluble sugars
        km_substrate = self._to_tensor(
            self._get_maintenance_parameter("km_substrate")
        )
        vmax_unit = torch.ones_like(soluble_sugar, dtype=torch.float32, device=self.device)
        substrate_factor = michaelis_menten_response(soluble_sugar, vmax_unit, km_substrate)

        # Calculate water stress
        # Water stress parameters (soil water potential thresholds)
        wilting_point = float(
            self._get_maintenance_parameter("water_stress_wilting_point")
        )
        field_capacity = float(
            self._get_maintenance_parameter("water_stress_field_capacity")
        )
        water_stress_factor = stress_response_functions.water_stress_linear(
            water_potential, wilting_point=wilting_point, field_capacity=field_capacity
        )

        # References: Ryan (1991); Atkin and Tjoelker (2003); CLM plant carbon allocation.
        # Mathematical expression: R_i = k_i(T) B_i f_sub f_water and R_m = sum_i R_i.
        # Parameter meanings: k_i is the tissue rate coefficient, B_i is tissue carbon,
        # f_sub is substrate limitation, and f_water is the water-stress response.
        # Units: k_i [s-1], B_i [g C m-2], R_i [g C m-2 s-1], R_m [kg C m-2 s-1].
        # Open-system route: maintenance carbon leaves the plant ledger through the
        # maintenance_respiration_total atmosphere-facing carbon port.
        resp_leaf = temp_factor_leaf * biomass_leaf * substrate_factor * water_stress_factor
        resp_stem = temp_factor_stem * biomass_stem * substrate_factor * water_stress_factor
        resp_root = temp_factor_root * biomass_root * substrate_factor * water_stress_factor

        # Convert the mass fluxes through the existing unit-conversion SSOT.
        context = ConversionContext()
        resp_leaf_kg = unit_converter.convert(
            data=resp_leaf,
            variable_name='carbon_respiration_flux',
            source_units='g C m-2 s-1',
            target_units='kg C m-2 s-1',
            context=context
        )
        resp_stem_kg = unit_converter.convert(
            data=resp_stem,
            variable_name='carbon_respiration_flux',
            source_units='g C m-2 s-1',
            target_units='kg C m-2 s-1',
            context=context
        )
        resp_root_kg = unit_converter.convert(
            data=resp_root,
            variable_name='carbon_respiration_flux',
            source_units='g C m-2 s-1',
            target_units='kg C m-2 s-1',
            context=context
        )
        total_resp = resp_leaf_kg + resp_stem_kg + resp_root_kg

        # References: Ryan (1991); Atkin and Tjoelker (2003); Grmela and Ottinger (1997).
        # Mathematical expression: x_dot = N_h r_h with r_h = [R_m] and N_h = [1].
        # Parameter meanings: N_h maps the unique total carbon export to its port;
        # r_h is the ParameterManager-backed maintenance-respiration mass flux.
        respiration_network = advance_reaction_network(
            name=f"{self.name}.maintenance_respiration_reaction_network",
            species=("maintenance_respiration_total",),
            reactions=("total_maintenance_respiration",),
            stoichiometric_N=torch.ones(
                (1, 1),
                device=total_resp.device,
                dtype=total_resp.dtype,
            ),
            rates=(total_resp,),
            open_system_route="port",
        )

        def replace(value: Tensor, unit: str) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=False,
                source_functor=self.name,
            )

        return {
            'maintenance_respiration_total': replace(
                respiration_network.species_rate("maintenance_respiration_total"),
                "kg C m-2 s-1",
            ),
            'diagnostics.maintenance_respiration.leaf_flux': replace(
                resp_leaf_kg,
                "kg C m-2 s-1",
            ),
            'diagnostics.maintenance_respiration.stem_flux': replace(
                resp_stem_kg,
                "kg C m-2 s-1",
            ),
            'diagnostics.maintenance_respiration.root_flux': replace(
                resp_root_kg,
                "kg C m-2 s-1",
            ),
            'diagnostics.maintenance_respiration.leaf_rate_coefficient': replace(
                temp_factor_leaf,
                "s-1",
            ),
            'diagnostics.maintenance_respiration.substrate_factor': replace(
                substrate_factor,
                "1",
            ),
            'diagnostics.maintenance_respiration.water_stress_factor': replace(
                water_stress_factor,
                "1",
            ),
            'diagnostics.maintenance_respiration.growth_temperature': replace(
                growth_temperature,
                "K",
            ),
        }
    
    def _update_temperature_history(self, temperature: Tensor):
        """Append the spatial-mean temperature of this step to the window.

        References: Atkin and Tjoelker (2003); Wilks (2011) running means.
        Mathematical expression: buffer[index] = mean_cells(T);
          index' = (index + 1) mod N; count' = min(count + 1, N).
        Parameter meanings: T air temperature of the step [K]; N window.
        Units: K.
        Domain/constraints: exactly one sample per emitted step; the update
          happens before the growth temperature of the same step is read.
        Sign convention: not applicable.
        """
        if isinstance(temperature, Tensor):
            temp_value = float(temperature.detach().to(torch.float64).mean())
        else:
            temp_value = float(temperature)
        self._history_buffer[self._history_index] = temp_value
        self._history_index = (self._history_index + 1) % self.acclimation_days
        self._history_count = min(self._history_count + 1, self.acclimation_days)

    def _calculate_growth_temperature(self) -> Tensor:
        """Return the window-mean growth temperature or the reference temperature."""
        # References: Atkin and Tjoelker (2003); Wilks (2011), running means.
        # Mathematical expression: Tgrowth=mean(buffer[0:count]) before the window fills, otherwise
        #   mean(buffer); empty history returns the existing reference temperature.
        # Parameter meanings: The buffer contains one spatial-mean temperature sample per history
        #   update; acclimation_days currently sets sample count N.
        # Units: Tgrowth and samples [K]; count,N are dimensionless sample counts.
        # Domain/constraints: This unweighted window is over calls, not elapsed days or a
        #   dt-weighted temporal integral; the current result is returned as float32.
        # Sign convention: Temperature is an unsigned physical state diagnostic; no carbon flux is
        #   generated.
        if self._history_count > 0:
            mean_temp = float(self._history_buffer[: self._history_count].mean()) if (
                self._history_count < self.acclimation_days
            ) else float(self._history_buffer.mean())
            return torch.as_tensor(mean_temp, device=self.device, dtype=torch.float32)

        return self._to_tensor(REFERENCE_TEMPERATURE_K)
    
    def _apply_acclimation(
        self, 
        growth_temperature: Tensor, 
        base_rm_leaf: Tensor, 
        base_rm_stem: Tensor, 
        base_rm_root: Tensor
    ) -> dict:
        """
        Apply thermal acclimation to respiration parameters.
        
        Parameters
        ----------
        growth_temperature : Tensor
            Mean growth temperature [K]
        base_rm_leaf, base_rm_stem, base_rm_root : Tensor
            Base respiration rates for each tissue type
            
        Returns
        -------
        dict
            Acclimated base respiration rates
        """
        # References: Atkin and Tjoelker (2003); Reich et al. (2016), existing reduced acclimation
        #   response.
        # Mathematical expression: factor=exp(-a*(Tgrowth-Tstandard));
        #   rref_acc_i=rref_base_i*factor for leaf, stem and root.
        # Parameter meanings: a is acclimation_rate_coeff from the parameter system; Tstandard is
        #   the physical reference temperature.
        # Units: a [K-1]; temperatures [K]; rref [s-1]; factor [1].
        # Domain/constraints: Uses the caller/history growth-temperature estimate; it does not
        #   implement a second acclimation clock or estimate exposure duration.
        # Sign convention: Warmer growth temperature lowers the reference rate when a>0; the factor
        #   itself carries no stock change.
        # Get acclimation parameters
        # Acclimation rate coefficient - Reference: Atkin & Tjoelker (2003)
        acclimation_rate_coeff = self._to_tensor(
            self._get_maintenance_parameter("acclimation_rate_coeff")
        )
        t_standard = self._to_tensor(REFERENCE_TEMPERATURE_K)
        
        # Type I acclimation: Adjust base respiration rate
        temp_diff = growth_temperature - t_standard
        acclimation_factor = torch.exp(-acclimation_rate_coeff * temp_diff)
        
        rm_ref_acclimated = {
            'leaf': base_rm_leaf * acclimation_factor,
            'stem': base_rm_stem * acclimation_factor,
            'root': base_rm_root * acclimation_factor
        }
        
        return rm_ref_acclimated
    
    def _extract_tensor(
        self,
        state: Dict[str, Union[Tensor, float]],
        key: str,
    ) -> Tensor:
        """
        Extract and convert input to tensor.
        
        State inputs are canonical-only and never receive runtime defaults.
        """
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

    def _to_tensor(self, value: Union[Tensor, float, int]) -> Tensor:
        """
        Convert scalar parameter values into device-aware tensors for snapshots.
        """
        if isinstance(value, Tensor):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(float(value), dtype=torch.float32, device=self.device)


# Export class
__all__ = ['MaintenanceRespiration']

