"""
Carbon Pool Transfer Matrix Module for RELIC Soil BGC

This module implements the transfer matrix approach for soil carbon dynamics,
ensuring mass balance and providing a unified framework for carbon flow between
pools. It follows the mathematical framework established in RothC, CENTURY, 
and other pool-based models.

Scientific Basis:
-----------------
The model uses matrix algebra to represent carbon transfers between pools,
following the compartmental modeling approach common in soil biogeochemistry.
This provides a clear mathematical structure and ensures mass balance.

References:
- Coleman, K. & Jenkinson, D.S. (1996). RothC-26.3 - A Model for the turnover 
  of carbon in soil. In: Evaluation of Soil Organic Matter Models. NATO ASI Series I 38.
- Parton, W.J. et al. (1987). Analysis of factors controlling soil organic matter 
  levels in Great Plains grasslands. Soil Science Society of America Journal, 51, 1173-1179.
- Sierra, C.A. et al. (2012). Models of soil organic matter decomposition: the SoilR package. 
  Geoscientific Model Development, 5, 1045-1060.
- Luo, Y. & Weng, E. (2011). Dynamic disequilibrium of the terrestrial carbon cycle
  under global change. Trends in Ecology & Evolution, 26, 96-104.

Mathematical Expression:
-----------------------
Matrix formulation:
dC/dt = I - DC + TC

Where:
- C: vector of carbon pool contents [g C m^-2]
- I: vector of external inputs [g C m^-2 s^-1]
- D: diagonal matrix of decomposition rates [s^-1]
- T: transfer matrix [dimensionless]

Transfer matrix formulation:
T_ij = transfer efficiency from pool j to pool i

Conservation constraint:
Sigma_i(T_ij) <= 1 for all j (respiration = 1 - Sigma_i(T_ij))
"""

from typing import Any, Dict
import logging
import torch

# Import core framework
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.abstractions import StateDict, FluxDict

# Import unified response functions
from relic.utils.unified_response_functions import (
    validate_tensor_shapes
)
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.computation.operators.iphs import advance_reaction_network
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin
from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)


class PoolsMatrix(PedosphereContractMixin, BiologicalProcessFunctor):
    """
    Carbon pool transfer matrix functor for soil biogeochemistry.
    
    Implements matrix-based carbon transfer between soil carbon pools
    with mass balance checking and quality assurance.
    
    References:
    - Sierra et al. (2012) SoilR decomposition framework
    - Luo & Weng (2011) Dynamic disequilibrium approach
    - Coleman & Jenkinson (1996) RothC transfer structure
    
    Mathematical Expression:
    -----------------------
    Pool dynamics:
    dC_i/dt = I_i + Sigma_j(T_ij * k_j * f_env * C_j) - k_i * f_env * C_i
    
    Matrix form:
    dC/dt = I + (T - Identity) * K_eff * C
    
    Where:
    - C: carbon pool vector [g C m^-2]
    - I: input vector [g C m^-2 s^-1]
    - T: transfer matrix [dimensionless]
    - K: decomposition rate matrix [s^-1]
    - f_env: environmental modifier [dimensionless]
    - K_eff: diag(k_i * f_env) [s^-1]
    
    Mass balance constraint:
    sum_i(Delta C_i) + R_total * dt - sum_i(I_i) * dt = 0
    
    Units:
    - Carbon pools: g C m^-2
    - Transfer rates: g C m^-2 s^-1
    - Decomposition rates: s^-1
    - Transfer efficiencies: dimensionless
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    CANONICAL_POOL_NAMES = (
        "metabolic_litter",
        "structural_litter",
        "fast_som",
        "slow_som",
        "passive_som",
    )
    POOL_STATE_KEYS = (
        "metabolic_litter_c",
        "structural_litter_c",
        "fast_som_c",
        "slow_som_c",
        "passive_som_c",
    )
    BOUNDARY_INPUT_KEYS = (
        "metabolic_litter_input",
        "structural_litter_input",
        "fast_som_input",
        "slow_som_input",
        "passive_som_input",
    )
    RUNTIME_INPUT_KEYS = POOL_STATE_KEYS + (
        "environmental_modifier",
    ) + BOUNDARY_INPUT_KEYS
    PARAMETER_DESCRIPTIONS = {
        "pedosphere.decomposition.pool_names": "Canonical five-pool ordering [dimensionless]",
        "pedosphere.numerical.mass_balance_tolerance": "Relative carbon-ledger tolerance [dimensionless]",
        "pedosphere.numerical.epsilon": "Numerical denominator guard [dimensionless]",
        "pedosphere.numerical.max_pool_size": "Maximum accepted carbon stock [g C m-2]",
        "pedosphere.decomposition.t_met_fast": "Metabolic litter to fast SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_met_slow": "Metabolic litter to slow SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_str_fast": "Structural litter to fast SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_str_slow": "Structural litter to slow SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_fast_slow": "Fast SOM to slow SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_fast_passive": "Fast SOM to passive SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_slow_fast": "Slow SOM to fast SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_slow_passive": "Slow SOM to passive SOM transfer fraction [dimensionless]",
        "pedosphere.carbon_pools.turnover_rate_metabolic": "Metabolic litter turnover coefficient [s-1]",
        "pedosphere.carbon_pools.turnover_rate_structural": "Structural litter turnover coefficient [s-1]",
        "pedosphere.carbon_pools.turnover_rate_fast_som": "Fast SOM turnover coefficient [s-1]",
        "pedosphere.carbon_pools.turnover_rate_slow_som": "Slow SOM turnover coefficient [s-1]",
        "pedosphere.carbon_pools.turnover_rate_passive_som": "Passive SOM turnover coefficient [s-1]",
        "pedosphere.numerical.default_environmental_modifier": "Reference modifier used only for turnover-time reporting [dimensionless]",
    }
    
    def __init__(
        self,
        name: str = "pools_matrix",
        config: Dict[str, Any] | None = None,
    ):
        """
        Initialize carbon pools matrix functor.
        
        Parameters:
        -----------
        name : str
            Unique identifier for this functor instance
        config : Dict
            Configuration dictionary containing parameters and settings
        """
        config = {} if config is None else config

        # Define comprehensive metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Carbon pool transfer matrix with mass balance for soil biogeochemistry",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="dC/dt = (T - I) K_eff C + I_ext",
            references=[
                "Sierra, C.A. et al. (2012). SoilR decomposition framework",
                "Coleman, K. & Jenkinson, D.S. (1996). RothC transfer matrix",
                "Parton, W.J. et al. (1987). CENTURY pool structure",
                "Luo, Y. & Weng, E. (2011). Dynamic disequilibrium"
            ],
            parameters=dict(self.PARAMETER_DESCRIPTIONS),
            input_variables=list(self.RUNTIME_INPUT_KEYS),
            output_variables=[
                "metabolic_litter_c",
                "structural_litter_c",
                "fast_som_c",
                "slow_som_c",
                "passive_som_c",
                "carbon_decomposition_rate",
                "heterotrophic_respiration",
                "diagnostics.pools_matrix.total_respiration_step",
                "diagnostics.pools_matrix.total_transfer_step",
                "diagnostics.pools_matrix.net_carbon_change_step",
                "diagnostics.pools_matrix.total_carbon_stock",
                "diagnostics.pools_matrix.total_litter_carbon_stock",
                "diagnostics.pools_matrix.total_som_carbon_stock",
                "diagnostics.pools_matrix.carbon_balance_residual",
                "diagnostics.pools_matrix.carbon_balance_relative_residual",
                "diagnostics.pools_matrix.carbon_balance_ok",
            ],
            conservation_laws=["carbon", "mass"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=[
                "environmental_modifier",
                *self.BOUNDARY_INPUT_KEYS,
                "heterotrophic_respiration",
            ]
        )
        
        super().__init__(name=name, config=config, metadata=metadata, device=None)
        # The generic biological bridge advertises optional response helpers.
        # PoolsMatrix uses only the exact ParameterManager keys listed above.
        self.metadata.parameters = dict(self.PARAMETER_DESCRIPTIONS)
        
        # Add FunctorBase consistency attributes
        self.name = name
        self._config_raw = dict(config) if config else {}
        self.metadata = metadata
        self.logger = logger

        self._execution_count = 0
        
        # Load pool configuration
        self.pool_names = self.get_parameter_value(
            'pool_names',
            namespace='pedosphere.decomposition',
            required=True)
        if tuple(self.pool_names) != self.CANONICAL_POOL_NAMES:
            raise ValueError(
                "PoolsMatrix requires the unique canonical five-pool basis "
                f"{self.CANONICAL_POOL_NAMES}; received {tuple(self.pool_names)}."
            )
        self.n_pools = len(self.pool_names)
        
        # Pool indices for easy access - needed before initializing transfer matrix
        self.pool_indices = {name: idx for idx, name in enumerate(self.pool_names)}
        
        # Quality control parameters - needed before initializing transfer matrix
        self.mass_balance_tolerance = self.get_parameter_value(
            'mass_balance_tolerance',
            namespace='pedosphere.numerical',
            required=True)
        self.epsilon = self.get_parameter_value(
            'epsilon',
            namespace='pedosphere.numerical',
            required=True)
        
        # Initialize transfer matrix
        self.register_buffer("transfer_matrix", self._initialize_transfer_matrix())
        
        # Additional quality control parameters
        self.max_pool_size = self.get_parameter_value(
            'max_pool_size',
            namespace='pedosphere.numerical',
            required=True)
        self._base_decomposition_rates = {
            pool_name: self.get_parameter_value(
                f"turnover_rate_{pool_name.replace('_litter', '')}",
                namespace="pedosphere.carbon_pools",
                required=True,
            )
            for pool_name in self.pool_names
        }
        # Performance tracking
        self.total_inputs = 0.0
        self.total_outputs = 0.0
        self.total_transfers = 0.0
        self.cumulative_mass_error = 0.0
        self.max_mass_error = 0.0
    
    def _initialize_transfer_matrix(self) -> torch.Tensor:
        """
        Initialize transfer matrix between carbon pools.
        
        Returns:
        --------
        torch.Tensor
            Transfer efficiency matrix [dimensionless]
            T_ij = efficiency of transfer from pool j to pool i
        """
        # Initialize transfer matrix (n_pools x n_pools)
        transfer_matrix = torch.zeros(self.n_pools, self.n_pools, dtype=torch.float32)
        
        # Default transfer efficiencies based on CENTURY/RothC structure
        # These can be overridden by configuration
        
        if 'metabolic_litter' in self.pool_names and 'fast_som' in self.pool_names:
            met_idx = self.pool_indices['metabolic_litter']
            fast_idx = self.pool_indices['fast_som']
            slow_idx = self.pool_indices.get('slow_som', fast_idx)
            
            # Metabolic litter transfers
            transfer_matrix[fast_idx, met_idx] = self.get_parameter_value(
                't_met_fast',
                namespace='pedosphere.decomposition',
                required=True)
            transfer_matrix[slow_idx, met_idx] = self.get_parameter_value(
                't_met_slow',
                namespace='pedosphere.decomposition',
                required=True)
        
        if 'structural_litter' in self.pool_names and 'fast_som' in self.pool_names:
            str_idx = self.pool_indices['structural_litter']
            fast_idx = self.pool_indices['fast_som']
            slow_idx = self.pool_indices.get('slow_som', fast_idx)
            
            # Structural litter transfers
            transfer_matrix[fast_idx, str_idx] = self.get_parameter_value(
                't_str_fast',
                namespace='pedosphere.decomposition',
                required=True)
            transfer_matrix[slow_idx, str_idx] = self.get_parameter_value(
                't_str_slow',
                namespace='pedosphere.decomposition',
                required=True)
        
        if 'fast_som' in self.pool_names and 'slow_som' in self.pool_names:
            fast_idx = self.pool_indices['fast_som']
            slow_idx = self.pool_indices['slow_som']
            passive_idx = self.pool_indices.get('passive_som', slow_idx)
            
            # Fast SOM transfers
            transfer_matrix[slow_idx, fast_idx] = self.get_parameter_value(
                't_fast_slow',
                namespace='pedosphere.decomposition',
                required=True)
            transfer_matrix[passive_idx, fast_idx] = self.get_parameter_value(
                't_fast_passive',
                namespace='pedosphere.decomposition',
                required=True)
            
            # Slow SOM transfers
            transfer_matrix[fast_idx, slow_idx] = self.get_parameter_value(
                't_slow_fast',
                namespace='pedosphere.decomposition',
                required=True)
            transfer_matrix[passive_idx, slow_idx] = self.get_parameter_value(
                't_slow_passive',
                namespace='pedosphere.decomposition',
                required=True)
        
        # Validate transfer matrix
        self._validate_transfer_matrix(transfer_matrix)
        
        return transfer_matrix
    
    def _validate_transfer_matrix(self, transfer_matrix: torch.Tensor) -> None:
        """
        Validate transfer matrix for conservation constraints.
        
        Parameters:
        -----------
        transfer_matrix : torch.Tensor
            Transfer efficiency matrix to validate
            
        Raises:
        -------
        ValueError
            If matrix violates conservation constraints
        """
        # Check matrix dimensions
        if transfer_matrix.shape != (self.n_pools, self.n_pools):
            raise ValueError(f"Transfer matrix must be {self.n_pools}x{self.n_pools}")
        
        # Check non-negative transfers
        if torch.any(transfer_matrix < 0):
            raise ValueError("Transfer efficiencies must be non-negative")
        
        # Check conservation constraint (column sums <= 1)
        column_sums = torch.sum(transfer_matrix, dim=0)
        if torch.any(column_sums > 1.0 + self.mass_balance_tolerance):
            raise ValueError(f"Transfer efficiencies exceed 1.0: {column_sums}")
        
        # Check for self-transfers (should be zero on diagonal)
        diagonal = torch.diag(transfer_matrix)
        if torch.any(diagonal > self.epsilon):
            self.logger.warning(f"Non-zero diagonal elements in transfer matrix: {diagonal}")
    
# get_parameter_value removed - using BiologicalProcess implementation
    
    @staticmethod
    def _runtime_tensor(value: Any, template: torch.Tensor | float) -> torch.Tensor:
        """Coerce a scalar or tensor to a runtime-compatible tensor."""
        template_tensor = torch.as_tensor(template)
        return torch.as_tensor(
            value,
            dtype=template_tensor.dtype,
            device=template_tensor.device,
        )

    @staticmethod
    def _require_state_tensor(state: StateDict, key: str) -> torch.Tensor:
        """Return one required finite runtime tensor without a fallback."""
        if key not in state:
            raise KeyError(key)
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        if value.numel() == 0:
            raise ValueError(f"{key} must not be empty")
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} must contain only finite values")
        return value

    def _collect_runtime_inputs(
        self,
        state: StateDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and stack the declared state and open-boundary inputs."""
        pool_values = [
            self._require_state_tensor(state, key) for key in self.POOL_STATE_KEYS
        ]
        reference = pool_values[0]
        for key, value in zip(self.POOL_STATE_KEYS[1:], pool_values[1:]):
            if value.shape != reference.shape:
                raise ValueError(f"{key} must match {self.POOL_STATE_KEYS[0]} shape")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(
                    f"{key} must match {self.POOL_STATE_KEYS[0]} dtype and device"
                )
        current_pools = torch.stack(pool_values)
        if torch.any(current_pools < 0):
            raise ValueError("Carbon pool inputs must be non-negative")

        modifier = self._require_state_tensor(state, "environmental_modifier")
        if modifier.shape != reference.shape:
            raise ValueError("environmental_modifier must match carbon pool shape")
        if modifier.device != reference.device or modifier.dtype != reference.dtype:
            raise ValueError(
                "environmental_modifier must match carbon pool dtype and device"
            )
        if torch.any(modifier < 0):
            raise ValueError("environmental_modifier must be non-negative")

        boundary_values = [
            self._require_state_tensor(state, key) for key in self.BOUNDARY_INPUT_KEYS
        ]
        for key, value in zip(self.BOUNDARY_INPUT_KEYS, boundary_values):
            if value.shape != reference.shape:
                raise ValueError(f"{key} must match carbon pool shape")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{key} must match carbon pool dtype and device")
        external_input_vector = torch.stack(boundary_values)
        if torch.any(external_input_vector < 0):
            raise ValueError("Pool boundary inputs must be non-negative")

        return current_pools, modifier, external_input_vector

    def _decomposition_rate_tensor(self, pools: torch.Tensor) -> torch.Tensor:
        """Broadcast the five ParameterManager turnover rates to pool shape."""
        spatial_template = pools[0]
        rates = []
        for pool_name in self.pool_names:
            rate = self._runtime_tensor(
                self._base_decomposition_rates[pool_name],
                spatial_template,
            )
            if rate.numel() == 1 and spatial_template.numel() != 1:
                rate = rate.expand_as(spatial_template)
            if rate.shape != spatial_template.shape:
                raise ValueError(
                    f"turnover_rate_{pool_name} must be scalar or match pool shape"
                )
            rates.append(rate)
        return torch.stack(rates)

    def _evaluate_step_fluxes(
        self,
        pools: torch.Tensor,
        decomposition_rates: torch.Tensor,
        environmental_modifier: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the five-pool compartment rates without advancing state.

        References:
        - Sierra et al. (2012), SoilR compartment-model framework.
        - Sierra and Mueller (2015), mass-balanced SOM dynamical systems.
        Mathematical expression:
        - D_j = min(k_j f_env C_j, C_j / dt)
        - F_ij = T_ij D_j
        - R_j = (1 - sum_i T_ij) D_j
        - dC/dt = (T - I) D
        Parameter meanings:
        - C_j: source-pool carbon [g C m-2].
        - k_j: first-order decomposition coefficient [s-1].
        - f_env: environmental modifier [dimensionless].
        - T_ij: source-j to destination-i transfer fraction [dimensionless].
        - dt: integration interval [s].
        Units:
        - D, F, R, and dC/dt are g C m-2 s-1.
        Open-system route:
        - The fraction 1 - sum_i T_ij exits through heterotrophic respiration.
        """
        validate_tensor_shapes([pools, decomposition_rates], "pools_and_rates")
        if pools.shape[0] != self.n_pools:
            raise ValueError(f"Pool vector must have {self.n_pools} elements")
        if not torch.isfinite(pools).all():
            raise ValueError("carbon_pools must contain only finite values")
        if not torch.isfinite(decomposition_rates).all():
            raise ValueError("decomposition_rates must contain only finite values")
        if torch.any(pools < 0):
            raise ValueError("carbon_pools must be non-negative")
        if torch.any(decomposition_rates < 0):
            raise ValueError("decomposition_rates must be non-negative")

        modifier = self._runtime_tensor(environmental_modifier, pools)
        dt_tensor = self._runtime_tensor(dt, pools)
        if not torch.isfinite(modifier).all() or torch.any(modifier < 0):
            raise ValueError("environmental_modifier must be finite and non-negative")
        if not torch.isfinite(dt_tensor).all() or torch.any(dt_tensor <= 0):
            raise ValueError("dt must be finite and positive")

        effective_rates = decomposition_rates * modifier
        decomposition_fluxes = effective_rates * pools
        requested_step = decomposition_fluxes * dt_tensor
        roundoff = torch.finfo(pools.dtype).eps * torch.maximum(
            torch.ones_like(pools),
            torch.abs(pools),
        )
        if torch.any(requested_step > pools + roundoff):
            raise ValueError(
                "The requested timestep would decompose more carbon than available"
            )

        matrix_shape = (self.n_pools, self.n_pools) + (1,) * (pools.ndim - 1)
        transfer_matrix = self.transfer_matrix.to(
            device=pools.device,
            dtype=pools.dtype,
        ).reshape(matrix_shape)
        transfer_flux_matrix = transfer_matrix * decomposition_fluxes.unsqueeze(0)
        transfers_in = torch.sum(transfer_flux_matrix, dim=1)
        transfers_out = torch.sum(transfer_flux_matrix, dim=0)
        respiration_fluxes = decomposition_fluxes - transfers_out
        if torch.any(respiration_fluxes < -self.epsilon):
            raise ValueError("Transfer fractions produce negative respiration")

        return {
            "decomposition_fluxes": decomposition_fluxes,
            "transfer_flux_matrix": transfer_flux_matrix,
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "respiration_fluxes": respiration_fluxes,
            "total_respiration": torch.sum(respiration_fluxes, dim=0),
            "net_pool_changes": transfers_in - decomposition_fluxes,
            "effective_rates": effective_rates,
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Evaluate only the dimensionally consistent per-step carbon ledger.

        References:
        - Sierra and Mueller (2015), SOM mass-balance framework.
        Mathematical expression:
        - epsilon_C = sum_i Delta C_i + R*dt - sum_i I_i*dt.
        Parameter meanings:
        - Delta C_i: pool increment; R: respiration rate; I_i: boundary input.
        Units:
        - All supplied ledger terms are integrated amounts [g C m-2].
        Open-system route:
        - ``litter_input_flux`` is the signed boundary term and respiration is
          the atmosphere-facing port term.
        """
        del input_state

        def _tensor_value(value: Any) -> torch.Tensor:
            if isinstance(value, UpdateItem):
                value = value.value
            return torch.as_tensor(value)

        if "litter_input_flux" in output_updates:
            required = tuple(f"{name}_c" for name in self.pool_names) + (
                "heterotrophic_respiration",
                "litter_input_flux",
            )
            if any(key not in output_updates for key in required):
                return {}
            residual = sum(
                (_tensor_value(output_updates[key]) for key in required),
                start=torch.zeros_like(_tensor_value(output_updates[required[0]])),
            )
        else:
            residual_key = "diagnostics.pools_matrix.carbon_balance_residual"
            if residual_key not in output_updates:
                return {}
            residual = _tensor_value(output_updates[residual_key])

        magnitude = float(torch.max(torch.abs(residual)).detach().cpu().item())
        return {law: magnitude for law in self.metadata.conservation_laws}
    
    def forward(self,
        # References:
        # - Sierra et al. (2012); Sierra and Mueller (2015).
        # Mathematical expression:
        # - dC/dt = (T - I) D + I_ext; R = (1 - 1^T T) D.
        # Parameter meanings:
        # - T is the transfer matrix, D_j = k_j*f_env*C_j, and I_ext is input.
        # Units:
        # - C and Delta C use g C m-2; D, I_ext, and R use g C m-2 s-1.
        # Open-system route:
        # - Litter enters through boundary inputs and respiration exits by port.
                state: StateDict,
                time: float,
                dt: torch.Tensor | float) -> Dict[str, UpdateItem]:
        """
        Advance the selected matrix closure exactly once through ``N_h r_h``.
        """
        del time
        current_pools, environmental_modifier, external_input_vector = (
            self._collect_runtime_inputs(state)
        )
        decomposition_rates = self._decomposition_rate_tensor(current_pools)
        dt_tensor = self._runtime_tensor(dt, current_pools[0])
        if (
            dt_tensor.numel() != 1
            or not torch.isfinite(dt_tensor).all()
            or torch.any(dt_tensor <= 0)
        ):
            raise ValueError("dt must be finite and positive scalar")

        fluxes = self._evaluate_step_fluxes(
            current_pools,
            decomposition_rates,
            environmental_modifier,
            dt_tensor,
        )

        identity = torch.eye(
            self.n_pools,
            device=current_pools.device,
            dtype=current_pools.dtype,
        )
        transfer = self.transfer_matrix.to(
            device=current_pools.device,
            dtype=current_pools.dtype,
        )
        stoichiometric_N = torch.cat((transfer - identity, identity), dim=1)
        pool_network = advance_reaction_network(
            name="carbon_pool_matrix_network",
            species=tuple(f"{pool_name}_c" for pool_name in self.pool_names),
            reactions=(
                *(f"{pool_name}_decomposition" for pool_name in self.pool_names),
                *(f"{pool_name}_boundary_input" for pool_name in self.pool_names),
            ),
            stoichiometric_N=stoichiometric_N,
            rates=torch.cat(
                (fluxes["decomposition_fluxes"], external_input_vector),
                dim=0,
            ),
            dt=dt_tensor,
            **self._iphs_reaction_conservation_kwargs(open_system_route="port"),
        )
        raw_delta_pools = pool_network.state_delta
        if raw_delta_pools is None:
            raise RuntimeError("Reaction network did not produce a per-step state delta")
        # Numerical state contract:
        # Delta C_repr = (C + Delta C_raw) - C. This is the exact increment
        # representable by the current stock tensor and applied by StateManager.
        updated_pools = current_pools + raw_delta_pools
        delta_pools = updated_pools - current_pools
        if not torch.isfinite(updated_pools).all():
            raise ValueError("Compartment advance produced non-finite carbon pools")
        if torch.any(updated_pools < 0):
            raise ValueError("Compartment advance produced a negative carbon pool")
        if torch.any(updated_pools > self.max_pool_size):
            raise ValueError("Compartment advance exceeds max_pool_size")

        total_input_rate = torch.sum(external_input_vector, dim=0)
        total_respiration_rate = fluxes["total_respiration"]
        gross_decomposition_rate = torch.sum(fluxes["decomposition_fluxes"], dim=0)
        total_respiration_step = total_respiration_rate * dt_tensor
        total_transfer_step = torch.sum(fluxes["transfers_in"], dim=0) * dt_tensor
        net_carbon_change_step = torch.sum(delta_pools, dim=0)
        carbon_balance_residual = (
            net_carbon_change_step
            + total_respiration_step
            - total_input_rate * dt_tensor
        )
        epsilon = self._runtime_tensor(self.epsilon, carbon_balance_residual)
        balance_scale = (
            torch.abs(total_respiration_step)
            + torch.abs(total_input_rate * dt_tensor)
            + epsilon
        )
        carbon_balance_relative_residual = carbon_balance_residual / balance_scale
        tolerance = self._runtime_tensor(
            self.mass_balance_tolerance,
            carbon_balance_relative_residual,
        )
        carbon_balance_ok = torch.abs(carbon_balance_relative_residual) <= tolerance

        # References: Sierra et al. (2012); Sierra and Mueller (2015).
        # Mathematical expression: sum_i Delta C_i + R*dt - sum_i I_i*dt = 0.
        # Parameter meanings: Delta C_i are pool changes, R is respiration export,
        # I_i are boundary inputs, and dt is the time interval.
        # Units: every ledger term is g C m-2.
        # Open-system route: input enters at the boundary and R exits by port.
        conservation_terms = {
            **{
                f"{pool_name}_c": delta_pools[index]
                for index, pool_name in enumerate(self.pool_names)
            },
            "heterotrophic_respiration": total_respiration_step,
            "litter_input_flux": -total_input_rate * dt_tensor,
        }
        self._enforce_forward_conservation(state, conservation_terms)

        self.total_inputs += float(
            (total_input_rate * dt_tensor).detach().sum().cpu().item()
        )
        self.total_outputs += float(total_respiration_step.detach().sum().cpu().item())
        self.total_transfers += float(total_transfer_step.detach().sum().cpu().item())
        self._execution_count += 1

        output: Dict[str, UpdateItem] = {}
        for index, pool_name in enumerate(self.pool_names):
            pool_key = f"{pool_name}_c"
            output[pool_key] = UpdateItem(
                value=delta_pools[index],
                mode=UpdateMode.ACCUMULATE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=True,
                element="C",
                source_functor=self.name,
            )

        output.update({
            "carbon_decomposition_rate": UpdateItem(
                value=gross_decomposition_rate,
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "heterotrophic_respiration": UpdateItem(
                value=total_respiration_rate,
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                element="C",
                source_functor=self.name,
            ),
            "diagnostics.pools_matrix.total_respiration_step": UpdateItem(
                value=total_respiration_step,
                mode=UpdateMode.REPLACE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.pools_matrix.total_transfer_step": UpdateItem(
                value=total_transfer_step,
                mode=UpdateMode.REPLACE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.pools_matrix.net_carbon_change_step": UpdateItem(
                value=net_carbon_change_step,
                mode=UpdateMode.REPLACE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.pools_matrix.total_carbon_stock": UpdateItem(
                value=torch.sum(updated_pools, dim=0),
                mode=UpdateMode.REPLACE,
                unit="g C m-2",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.pools_matrix.total_litter_carbon_stock": UpdateItem(
                value=torch.sum(updated_pools[:2], dim=0),
                mode=UpdateMode.REPLACE,
                unit="g C m-2",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.pools_matrix.total_som_carbon_stock": UpdateItem(
                value=torch.sum(updated_pools[2:], dim=0),
                mode=UpdateMode.REPLACE,
                unit="g C m-2",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.pools_matrix.carbon_balance_residual": UpdateItem(
                value=carbon_balance_residual,
                mode=UpdateMode.REPLACE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.pools_matrix.carbon_balance_relative_residual": UpdateItem(
                value=carbon_balance_relative_residual,
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                basis="per_step",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.pools_matrix.carbon_balance_ok": UpdateItem(
                value=carbon_balance_ok,
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
        })
        return output
    
    def get_pool_turnover_times(self) -> Dict[str, float]:
        """
        Calculate mean residence times for each carbon pool.
        
        Returns:
        --------
        Dict[str, float]
            Mean residence times in years for each pool
            
        Mathematical Expression:
        -----------------------
        tau_i = C_i / (k_i * f_env * C_i) = 1 / (k_i * f_env)
        
        Where environmental modifier is assumed at typical conditions
        """
        turnover_times = {}
        
        # References: RELIC numerical defaults.
        # Mathematical expression: k_effective = k_base * f_env_typical
        # Parameters: default_environmental_modifier [dimensionless]
        typical_env_modifier = float(
            self.get_parameter_value(
                "default_environmental_modifier",
                namespace="pedosphere.numerical",
                required=True)
        )
        seconds_per_year = float(get_physical_constant("seconds_per_year"))
        
        for pool_name in self.pool_names:
            # Get base decomposition rate from parameter system
            # These correspond to turnover_rate_* parameters in carbon_pools_parameters.yaml
            turnover_name = pool_name.replace("_litter", "") if pool_name.endswith("_litter") else pool_name
            k_base = self.get_parameter_value(
                f'turnover_rate_{turnover_name}',
                namespace='pedosphere.carbon_pools', required=True)
            k_effective = k_base * typical_env_modifier
            
            turnover_time_years = 1.0 / (k_effective * seconds_per_year)
            turnover_times[f"{pool_name}_turnover_years"] = turnover_time_years
            
        return turnover_times
    
    def get_transfer_statistics(self) -> Dict[str, float]:
        """
        Get transfer matrix statistics and performance metrics.
        
        Returns:
        --------
        Dict[str, float]
            Statistics on transfer processes
        """
        # Calculate matrix properties
        matrix_sum = float(torch.sum(self.transfer_matrix).detach().reshape(-1)[0].tolist())
        max_efficiency = float(torch.max(self.transfer_matrix).detach().reshape(-1)[0].tolist())
        
        # Column sums (total efficiency out of each pool)
        column_sums = torch.sum(self.transfer_matrix, dim=0)
        mean_efficiency = float(torch.mean(column_sums).detach().reshape(-1)[0].tolist())
        
        # Respiration fractions (1 - column_sum)
        respiration_fractions = 1.0 - column_sums
        mean_respiration_fraction = float(torch.mean(respiration_fractions).detach().reshape(-1)[0].tolist())
        
        return {
            'total_transfer_efficiency': matrix_sum,
            'max_transfer_efficiency': max_efficiency,
            'mean_transfer_efficiency': mean_efficiency,
            'mean_respiration_fraction': mean_respiration_fraction,
            'total_inputs_processed': self.total_inputs,
            'total_transfers_processed': self.total_transfers,
            'total_outputs_generated': self.total_outputs,
            'cumulative_mass_error': self.cumulative_mass_error,
            'max_mass_error_observed': self.max_mass_error,
            'execution_count': self._execution_count,
            **self.get_pool_turnover_times()
        }
    
    def reset_performance_tracking(self) -> None:
        """Reset performance tracking counters."""
        self._execution_count = 0
        self.total_inputs = 0.0
        self.total_outputs = 0.0
        self.total_transfers = 0.0
        self.cumulative_mass_error = 0.0
        self.max_mass_error = 0.0


# Export main classes
__all__ = ['PoolsMatrix']

