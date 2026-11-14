"""
L1 Unit Tests for PoolsMatrix Module

This module provides comprehensive unit tests for the carbon pools transfer
matrix implementation, covering matrix operations, mass balance, conservation
laws, and numerical accuracy.

Test Categories:
- L1.1: Basic instantiation and configuration
- L1.2: Transfer matrix validation and properties
- L1.3: Mass balance and conservation laws
- L1.4: Matrix operations and numerical accuracy
- L1.5: Boundary condition handling
- L1.6: Parameter sensitivity analysis  
- L1.7: Performance and stability tests
"""

import pytest
import torch
from builtins import float as builtin_float
from typing import Dict

# Import the module under test
from relic.scientific.pedosphere.carbon_pools import PoolsMatrix
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager


POOL_OUTPUTS = (
    "metabolic_litter_c",
    "structural_litter_c",
    "fast_som_c",
    "slow_som_c",
    "passive_som_c",
)
BOUNDARY_INPUTS = (
    "metabolic_litter_input",
    "structural_litter_input",
    "fast_som_input",
    "slow_som_input",
    "passive_som_input",
)
EXPECTED_INPUTS = set(POOL_OUTPUTS + ("environmental_modifier",) + BOUNDARY_INPUTS)
FORMAL_FLUX_OUTPUTS = (
    "carbon_decomposition_rate",
    "heterotrophic_respiration",
)
POOLS_MATRIX_DIAGNOSTICS = (
    "diagnostics.pools_matrix.total_respiration_step",
    "diagnostics.pools_matrix.total_transfer_step",
    "diagnostics.pools_matrix.net_carbon_change_step",
    "diagnostics.pools_matrix.total_carbon_stock",
    "diagnostics.pools_matrix.total_litter_carbon_stock",
    "diagnostics.pools_matrix.total_som_carbon_stock",
    "diagnostics.pools_matrix.carbon_balance_residual",
    "diagnostics.pools_matrix.carbon_balance_relative_residual",
    "diagnostics.pools_matrix.carbon_balance_ok",
)
EXPECTED_OUTPUTS = set(POOL_OUTPUTS + FORMAL_FLUX_OUTPUTS + POOLS_MATRIX_DIAGNOSTICS)
EXPECTED_PARAMETER_KEYS = {
    "pedosphere.decomposition.pool_names",
    "pedosphere.numerical.mass_balance_tolerance",
    "pedosphere.numerical.epsilon",
    "pedosphere.numerical.max_pool_size",
    "pedosphere.decomposition.t_met_fast",
    "pedosphere.decomposition.t_met_slow",
    "pedosphere.decomposition.t_str_fast",
    "pedosphere.decomposition.t_str_slow",
    "pedosphere.decomposition.t_fast_slow",
    "pedosphere.decomposition.t_fast_passive",
    "pedosphere.decomposition.t_slow_fast",
    "pedosphere.decomposition.t_slow_passive",
    "pedosphere.carbon_pools.turnover_rate_metabolic",
    "pedosphere.carbon_pools.turnover_rate_structural",
    "pedosphere.carbon_pools.turnover_rate_fast_som",
    "pedosphere.carbon_pools.turnover_rate_slow_som",
    "pedosphere.carbon_pools.turnover_rate_passive_som",
    "pedosphere.numerical.default_environmental_modifier",
}
DIAGNOSTIC_UNITS = {
    "diagnostics.pools_matrix.total_respiration_step": "g C m-2",
    "diagnostics.pools_matrix.total_transfer_step": "g C m-2",
    "diagnostics.pools_matrix.net_carbon_change_step": "g C m-2",
    "diagnostics.pools_matrix.total_carbon_stock": "g C m-2",
    "diagnostics.pools_matrix.total_litter_carbon_stock": "g C m-2",
    "diagnostics.pools_matrix.total_som_carbon_stock": "g C m-2",
    "diagnostics.pools_matrix.carbon_balance_residual": "g C m-2",
    "diagnostics.pools_matrix.carbon_balance_relative_residual": "dimensionless",
    "diagnostics.pools_matrix.carbon_balance_ok": "dimensionless",
}
REMOVED_OUTPUTS = {
    "diagnostics.total_respiration",
    "diagnostics.total_transfers",
    "diagnostics.net_carbon_change",
    "diagnostics.total_carbon",
    "diagnostics.total_litter_c",
    "diagnostics.total_som_c",
    "diagnostics.mass_balance_error",
    "diagnostics.mass_balance_relative_error",
    "diagnostics.mass_balance_ok",
}

@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _normalize_param_value(value):
    if isinstance(value, torch.Tensor):
        return _safe_float(value.detach().cpu().item())
    return value

def _set_param_value(param_manager, key: str, value) -> None:
    spec = param_manager.get_parameter(key)
    units = spec.units if spec is not None else None
    param_manager.set_value(key, value, units=units, validate=True, notify=False)

def _safe_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return builtin_float(value.detach().item())
    return builtin_float(value)


def _value(result: Dict[str, UpdateItem], key: str) -> torch.Tensor:
    """Return the tensor payload from an explicit update item."""
    item = result[key]
    assert isinstance(item, UpdateItem), f"{key} must use explicit UpdateItem semantics"
    assert isinstance(item.value, torch.Tensor), f"{key} payload must be a Tensor"
    return item.value


class TestPoolsMatrixL1:
    """L1 Unit Tests for PoolsMatrix functor."""

    @pytest.fixture(autouse=True)
    def _default_param_overrides(self, default_config, standard_rates):
        param_manager = get_global_manager()
        overrides = {
            "pedosphere.decomposition.pool_names": default_config["pool_names"],
            "pedosphere.decomposition.t_met_fast": default_config["t_met_fast"],
            "pedosphere.decomposition.t_met_slow": default_config["t_met_slow"],
            "pedosphere.decomposition.t_str_fast": default_config["t_str_fast"],
            "pedosphere.decomposition.t_str_slow": default_config["t_str_slow"],
            "pedosphere.decomposition.t_fast_slow": default_config["t_fast_slow"],
            "pedosphere.decomposition.t_fast_passive": default_config["t_fast_passive"],
            "pedosphere.decomposition.t_slow_fast": default_config["t_slow_fast"],
            "pedosphere.decomposition.t_slow_passive": default_config["t_slow_passive"],
            "pedosphere.numerical.mass_balance_tolerance": default_config["mass_balance_tolerance"],
            "pedosphere.numerical.max_pool_size": default_config["max_pool_size"],
            "pedosphere.carbon_pools.turnover_rate_metabolic": _safe_float(standard_rates[0]),
            "pedosphere.carbon_pools.turnover_rate_structural": _safe_float(standard_rates[1]),
            "pedosphere.carbon_pools.turnover_rate_fast_som": _safe_float(standard_rates[2]),
            "pedosphere.carbon_pools.turnover_rate_slow_som": _safe_float(standard_rates[3]),
            "pedosphere.carbon_pools.turnover_rate_passive_som": _safe_float(standard_rates[4]),
        }
        original_values = {
            key: _normalize_param_value(param_manager.get(key)) for key in overrides
        }
        for key, value in overrides.items():
            _set_param_value(param_manager, key, value)
        yield
        for key, value in original_values.items():
            _set_param_value(param_manager, key, value)
    
    @pytest.fixture
    def default_config(self) -> Dict:
        """Default configuration for PoolsMatrix."""
        return {
            'pool_names': ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som'],
            't_met_fast': 0.45,
            't_met_slow': 0.45,
            't_str_fast': 0.40,
            't_str_slow': 0.40,
            't_fast_slow': 0.42,
            't_fast_passive': 0.03,
            't_slow_fast': 0.45,
            't_slow_passive': 0.03,
            'mass_balance_tolerance': 1e-6,
            'max_pool_size': 1e3
        }
    
    @pytest.fixture  
    def pools_matrix(self, default_config) -> PoolsMatrix:
        """Create PoolsMatrix instance."""
        return PoolsMatrix(name="test_pools", config={})
    
    @pytest.fixture
    def standard_pools(self) -> torch.Tensor:
        """Standard carbon pool vector for testing."""
        return torch.tensor([0.1, 0.3, 0.8, 2.0, 4.0], dtype=torch.float32)  # g C m-2
    @pytest.fixture
    def standard_rates(self) -> torch.Tensor:
        """Standard decomposition rates for testing."""
        # Rates in s-1 (converted from per-year values)
        seconds_per_year = 365.25 * 24 * 3600
        return torch.tensor([
            14.8 / seconds_per_year,  # metabolic litter
            4.9 / seconds_per_year,   # structural litter  
            6.0 / seconds_per_year,   # fast som
            0.2 / seconds_per_year,   # slow som
            0.0045 / seconds_per_year # passive som
        ], dtype=torch.float32)
    
    @pytest.fixture
    def standard_state(self) -> Dict:
        """Standard state for testing."""
        return {
            'metabolic_litter_c': torch.tensor(0.1),
            'structural_litter_c': torch.tensor(0.3),
            'fast_som_c': torch.tensor(0.8),
            'slow_som_c': torch.tensor(2.0),
            'passive_som_c': torch.tensor(4.0),
            'environmental_modifier': torch.tensor(1.0),
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
            'fast_som_input': torch.tensor(0.0),
            'slow_som_input': torch.tensor(0.0),
            'passive_som_input': torch.tensor(0.0),
        }
        
    # =============================================================================
    # L1.1: Basic Instantiation and Configuration Tests
    # =============================================================================
    
    def test_instantiation_basic(self, default_config):
        """Test basic instantiation with default configuration."""
        pools = PoolsMatrix(name="test_pools", config={})
        
        assert pools.name == "test_pools"
        assert hasattr(pools, 'metadata')
        assert hasattr(pools, 'transfer_matrix')
        assert hasattr(pools, 'pool_names')
        assert hasattr(pools, 'n_pools')

    def test_canonical_default_runtime_name(self):
        """The default name must match the SCM candidate-writer contract."""
        pools = PoolsMatrix()

        assert pools.name == "pools_matrix"
        
    def test_instantiation_metadata(self, pools_matrix):
        """Metadata must declare the exact canonical matrix-closure interface."""
        metadata = pools_matrix.metadata
        
        # Check required metadata fields
        assert metadata.name is not None
        assert metadata.version is not None
        assert metadata.description is not None
        assert metadata.mathematical_expression is not None
        
        # Check scientific references
        assert len(metadata.references) >= 3
        assert any("Sierra" in ref for ref in metadata.references)
        assert any("Coleman" in ref or "RothC" in ref for ref in metadata.references)
        
        # Check parameters documentation
        assert set(metadata.parameters) == EXPECTED_PARAMETER_KEYS
        assert set(metadata.input_variables) == EXPECTED_INPUTS
        assert set(metadata.output_variables) == EXPECTED_OUTPUTS
        assert REMOVED_OUTPUTS.isdisjoint(metadata.output_variables)
        
    def test_pool_configuration(self, pools_matrix):
        """Test carbon pool configuration."""
        assert pools_matrix.n_pools == 5
        assert len(pools_matrix.pool_names) == 5
        
        expected_pools = ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']
        assert pools_matrix.pool_names == expected_pools
        
        # Check pool indices
        for i, pool in enumerate(expected_pools):
            assert pools_matrix.pool_indices[pool] == i
    
    def test_no_hidden_parameter_backed_state_initialization(self, pools_matrix):
        """The closure must consume StateManager pools instead of owning defaults."""
        assert not hasattr(pools_matrix, "initial_pools")

    def test_forward_is_the_one_rate_entry(self, pools_matrix):
        """No secondary _compute_rate_core entry evaluates the fluxes outside forward."""
        assert not hasattr(pools_matrix, "_compute_rate_core")
    
    def test_dependencies(self, pools_matrix):
        """The matrix variant must not depend on alternative decomposers."""
        deps = pools_matrix.get_dependencies()

        assert isinstance(deps, list)
        assert deps == []

    def test_forward_exact_output_and_update_contract(
        self,
        pools_matrix,
        standard_state,
        standard_rates,
    ):
        """Forward must expose one routed carbon ledger with explicit update semantics."""
        result = pools_matrix.forward(
            standard_state,
            time=0.0,
            dt=3600.0,
        )

        assert set(result) == EXPECTED_OUTPUTS
        assert REMOVED_OUTPUTS.isdisjoint(result)
        for key, item in result.items():
            assert isinstance(item, UpdateItem), f"{key} must be an UpdateItem"
            assert isinstance(item.value, torch.Tensor)
            assert item.source_functor == "test_pools"

        for key in POOL_OUTPUTS:
            item = result[key]
            assert item.mode == UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == "g C m-2"

        for key in FORMAL_FLUX_OUTPUTS + POOLS_MATRIX_DIAGNOSTICS:
            assert result[key].mode == UpdateMode.REPLACE
            assert result[key].per_timestep is False

        for key in FORMAL_FLUX_OUTPUTS:
            assert result[key].unit == "g C m-2 s-1"

        for key, unit in DIAGNOSTIC_UNITS.items():
            assert result[key].unit == unit

    def test_forward_fixed_numerics_and_open_carbon_ledger(
        self,
        pools_matrix,
        standard_state,
        standard_rates,
    ):
        """The compartment matrix must close pool, input, and respiration carbon."""
        dt = 3600.0
        input_rates = {
            "metabolic_litter_input": torch.tensor(1.0e-6),
            "structural_litter_input": torch.tensor(2.0e-6),
            "fast_som_input": torch.tensor(0.0),
            "slow_som_input": torch.tensor(0.0),
            "passive_som_input": torch.tensor(0.0),
        }
        open_state = {**standard_state, **input_rates}
        result = pools_matrix.forward(
            open_state,
            time=0.0,
            dt=dt,
        )

        expected_delta = torch.tensor(
            [
                3.431165823712945e-3,
                7.032306399196386e-3,
                -3.839835699182004e-4,
                3.274013288319111e-4,
                1.574264206283260e-5,
            ]
        )
        actual_delta = torch.stack([_value(result, key) for key in POOL_OUTPUTS])
        respiration_rate = _value(result, "heterotrophic_respiration")
        gross_rate = _value(result, "carbon_decomposition_rate")
        total_input_rate = torch.stack(tuple(input_rates.values())).sum()
        residual = actual_delta.sum() + respiration_rate * dt - total_input_rate * dt

        assert torch.allclose(actual_delta, expected_delta, rtol=1.0e-6, atol=1.0e-10)
        assert torch.allclose(gross_rate, torch.tensor(2.5882832233037334e-7), rtol=1.0e-6)
        assert torch.allclose(respiration_rate, torch.tensor(1.0482420265134351e-7), rtol=1.0e-6)
        assert abs(_safe_float(residual)) <= 2.0e-9
        assert torch.allclose(
            _value(result, "diagnostics.pools_matrix.carbon_balance_residual"),
            residual,
            atol=1.0e-12,
        )

    def test_integrated_pool_updates_scale_once_with_dt(
        self,
        pools_matrix,
        standard_state,
        standard_rates,
    ):
        """Per-step pool increments must contain exactly one timestep integration."""
        one = pools_matrix.forward(
            standard_state,
            time=0.0,
            dt=1800.0,
        )
        two = pools_matrix.forward(
            standard_state,
            time=0.0,
            dt=3600.0,
        )

        delta_one = torch.stack([_value(one, key) for key in POOL_OUTPUTS])
        delta_two = torch.stack([_value(two, key) for key in POOL_OUTPUTS])
        assert torch.allclose(delta_two, 2.0 * delta_one, rtol=1.0e-6, atol=1.0e-12)
        assert torch.allclose(
            _value(two, "heterotrophic_respiration"),
            _value(one, "heterotrophic_respiration"),
        )

    @pytest.mark.parametrize("missing_key", sorted(EXPECTED_INPUTS))
    def test_forward_rejects_missing_required_input(
        self,
        pools_matrix,
        standard_state,
        missing_key,
    ):
        """Runtime state must not be completed from ParameterManager defaults."""
        incomplete = dict(standard_state)
        incomplete.pop(missing_key)

        with pytest.raises(KeyError, match=missing_key):
            pools_matrix.forward(incomplete, time=0.0, dt=3600.0)

    @pytest.mark.parametrize(
        ("key", "bad_value"),
        [
            ("metabolic_litter_c", torch.tensor(float("nan"))),
            ("environmental_modifier", torch.tensor(float("inf"))),
            ("structural_litter_input", torch.tensor(float("nan"))),
        ],
    )
    def test_forward_rejects_nonfinite_input(
        self,
        pools_matrix,
        standard_state,
        key,
        bad_value,
    ):
        """All state and boundary inputs must be finite before calculation."""
        invalid = {**standard_state, key: bad_value}

        with pytest.raises(ValueError, match="finite"):
            pools_matrix.forward(invalid, time=0.0, dt=3600.0)

    @pytest.mark.parametrize(
        "key",
        ["metabolic_litter_c", "environmental_modifier", "fast_som_input"],
    )
    def test_forward_rejects_negative_input(
        self,
        pools_matrix,
        standard_state,
        key,
    ):
        """Pools, modifiers, and one-way boundary inputs must be non-negative."""
        invalid = {**standard_state, key: torch.tensor(-1.0)}

        with pytest.raises(ValueError, match="non-negative"):
            pools_matrix.forward(invalid, time=0.0, dt=3600.0)

    @pytest.mark.parametrize(
        "key",
        ["metabolic_litter_c", "environmental_modifier", "passive_som_input"],
    )
    def test_forward_rejects_non_tensor_input(
        self,
        pools_matrix,
        standard_state,
        key,
    ):
        """Scientific runtime inputs must remain differentiable tensors."""
        invalid = {**standard_state, key: 1.0}

        with pytest.raises(TypeError, match=key):
            pools_matrix.forward(invalid, time=0.0, dt=3600.0)

    def test_forward_rejects_runtime_parameter_and_boundary_kwargs(
        self,
        pools_matrix,
        standard_state,
    ):
        """The explicit signature must reject a second runtime parameter path."""
        with pytest.raises(
            TypeError,
            match="unexpected keyword argument 'k_fast_som'",
        ):
            pools_matrix.forward(
                standard_state,
                time=0.0,
                dt=3600.0,
                k_fast_som=torch.tensor(0.0),
            )

    @pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
    def test_forward_rejects_invalid_dt(self, pools_matrix, standard_state, dt):
        """The one-step reaction advance requires a finite positive scalar dt."""
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            pools_matrix.forward(standard_state, time=0.0, dt=dt)

    def test_forward_rejects_timestep_that_would_overdraw_a_pool(
        self,
        pools_matrix,
        standard_state,
    ):
        """An unstable explicit step must fail instead of silently clipping decay."""
        with pytest.raises(ValueError, match="decompose more carbon than available"):
            pools_matrix.forward(standard_state, time=0.0, dt=1.0e9)
        
    # =============================================================================
    # L1.2: Transfer Matrix Validation and Properties Tests
    # =============================================================================
    
    def test_transfer_matrix_dimensions(self, pools_matrix):
        """Test transfer matrix has correct dimensions."""
        transfer_matrix = pools_matrix.transfer_matrix
        
        assert isinstance(transfer_matrix, torch.Tensor)
        assert transfer_matrix.shape == (5, 5), "Transfer matrix should be 5x5"
        assert transfer_matrix.dtype == torch.float32
    
    def test_transfer_matrix_properties(self, pools_matrix):
        """Test transfer matrix mathematical properties."""
        T = pools_matrix.transfer_matrix
        
        # Non-negative elements
        assert torch.all(T >= 0), "All transfer efficiencies should be non-negative"
        
        # Diagonal should be zero (no self-transfers)
        diagonal = torch.diag(T)
        assert torch.all(diagonal <= 1e-10), "Diagonal elements should be zero (no self-transfers)"
        
        # Column sums should be <= 1 (conservation constraint)
        column_sums = torch.sum(T, dim=0)
        assert torch.all(column_sums <= 1.0 + 1e-6), "Column sums should not exceed 1.0"
    
    def test_transfer_matrix_scientific_structure(self, pools_matrix):
        """Test transfer matrix follows expected scientific structure."""
        T = pools_matrix.transfer_matrix
        
        # Get pool indices
        met_idx = pools_matrix.pool_indices['metabolic_litter']
        str_idx = pools_matrix.pool_indices['structural_litter'] 
        fast_idx = pools_matrix.pool_indices['fast_som']
        slow_idx = pools_matrix.pool_indices['slow_som']
        passive_idx = pools_matrix.pool_indices['passive_som']
        
        # Check key transfers exist and are reasonable
        # Metabolic litter should transfer to fast and slow SOM
        assert T[fast_idx, met_idx] > 0, "Metabolic litter should transfer to fast SOM"
        assert T[slow_idx, met_idx] > 0, "Metabolic litter should transfer to slow SOM"
        
        # Structural litter should also transfer to SOM
        assert T[fast_idx, str_idx] > 0, "Structural litter should transfer to fast SOM"
        
        # Fast SOM should transfer to slow SOM and passive SOM
        assert T[slow_idx, fast_idx] > 0, "Fast SOM should transfer to slow SOM"
        assert T[passive_idx, fast_idx] > 0, "Fast SOM should transfer to passive SOM"
        
        # Passive SOM should not transfer to other pools (terminal pool)
        passive_out = torch.sum(T[:, passive_idx])
        assert _safe_float(passive_out.detach().item()) < 1e-10, "Passive SOM should not transfer to other pools"
    
    def test_transfer_efficiency_ranges(self, pools_matrix):
        """Test transfer efficiencies are in realistic ranges."""
        T = pools_matrix.transfer_matrix
        
        # Non-zero elements should be in reasonable range
        nonzero_elements = T[T > 1e-10]
        
        assert torch.all(nonzero_elements >= 0.01), "Transfer efficiencies should be >= 1%"
        assert torch.all(nonzero_elements <= 0.8), "Transfer efficiencies should be <= 80%"
        
        # Check specific transfer ranges match literature
        met_idx = pools_matrix.pool_indices['metabolic_litter']
        fast_idx = pools_matrix.pool_indices['fast_som']
        
        met_to_fast = T[fast_idx, met_idx]
        met_to_fast_value = _safe_float(met_to_fast.detach().item())
        assert 0.3 <= met_to_fast_value <= 0.6, f"Metabolic to fast transfer should be 30-60%: {met_to_fast_value:.3f}"
    
    def test_respiration_fractions(self, pools_matrix):
        """Test respiration fractions (1 - column sums) are reasonable."""
        T = pools_matrix.transfer_matrix
        column_sums = torch.sum(T, dim=0)
        respiration_fractions = 1.0 - column_sums
        
        # All respiration fractions should be positive  
        assert torch.all(respiration_fractions >= 0), "Respiration fractions should be non-negative"
        
        # Should be reasonable values (not too high or low)
        assert torch.all(respiration_fractions >= 0.1), "Respiration fractions should be >= 10%"
        # Allow passive SOM to have high respiration (up to 100%)
        passive_idx = pools_matrix.pool_indices['passive_som']
        non_passive_mask = torch.ones_like(respiration_fractions, dtype=torch.bool)
        non_passive_mask[passive_idx] = False
        assert torch.all(respiration_fractions[non_passive_mask] <= 0.9), "Non-passive pool respiration fractions should be <= 90%"
        assert respiration_fractions[passive_idx] <= 1.0, "Passive SOM respiration fraction should be <= 100%"
        
        passive_resp = _safe_float(respiration_fractions[passive_idx].detach().item())
        
        # Passive SOM should have higher respiration fraction (all decomposition goes to CO2)
        assert passive_resp > 0.9, f"Passive SOM respiration fraction should be >90%: {passive_resp:.3f}"
    
    # =============================================================================
    # L1.3: Mass Balance and Conservation Laws Tests
    # =============================================================================
    
    def test_closed_system_mass_balance(self, pools_matrix, standard_state):
        """The canonical forward path must close the zero-input carbon ledger."""
        dt = 3600.0
        result = pools_matrix.forward(standard_state, time=0.0, dt=dt)

        pool_delta = torch.stack([_value(result, key) for key in POOL_OUTPUTS]).sum()
        respiration_step = _value(result, "heterotrophic_respiration") * dt
        residual = pool_delta + respiration_step

        assert abs(_safe_float(residual)) <= 2.0e-9
        assert torch.allclose(
            _value(result, "diagnostics.pools_matrix.carbon_balance_residual"),
            residual,
            atol=1.0e-12,
        )
    
    def test_conservation_with_external_inputs(self, pools_matrix, standard_state):
        """Test conservation laws with external inputs."""
        # Add external inputs to each pool
        external_inputs = {
            'metabolic_litter_input': torch.tensor(1e-6),
            'structural_litter_input': torch.tensor(2e-6),
            'fast_som_input': torch.tensor(0.0),
            'slow_som_input': torch.tensor(0.0),
            'passive_som_input': torch.tensor(0.0)
        }
        
        # Get initial carbon
        initial_carbon = sum(_safe_float(standard_state[key]) for key in standard_state.keys() 
                           if key.endswith('_c'))
        
        # Apply forward pass with inputs
        open_state = {**standard_state, **external_inputs}
        result = pools_matrix.forward(open_state, time=0.0, dt=3600.0)
        
        # Calculate final carbon (initial + per-step changes)
        final_carbon = initial_carbon
        for pool in ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']:
            pool_key = f'{pool}_c'
            if pool_key in result:
                final_carbon += _safe_float(_value(result, pool_key))
        
        # Calculate total inputs
        total_inputs = sum(_safe_float(external_inputs[key]) for key in external_inputs.keys()) * 3600.0
        
        # Get total respiration  
        total_respiration = _safe_float(
            _value(result, 'diagnostics.pools_matrix.total_respiration_step')
        )
        
        # Conservation: final_carbon + respiration = initial_carbon + inputs
        expected_final = initial_carbon + total_inputs - total_respiration
        conservation_error = abs(final_carbon - expected_final)
        relative_error = conservation_error / (initial_carbon + total_inputs)
        
        assert relative_error < 1e-5, \
            f"Conservation error with inputs: {conservation_error:.2e} (relative: {relative_error:.2e})"
    
    def test_mass_balance_over_time(self, pools_matrix, standard_state):
        """Test mass balance conservation over multiple time steps."""
        current_state = standard_state.copy()
        cumulative_respiration = 0.0
        dt = 3600.0
        
        initial_total = sum(_safe_float(standard_state[key]) for key in standard_state.keys() 
                           if key.endswith('_c'))
        
        # Run for several time steps
        for step in range(5):
            result = pools_matrix.forward(current_state, time=_safe_float(step*dt), dt=dt)
            
            # Update state
            for pool in ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']:
                pool_key = f'{pool}_c'
                if pool_key in current_state and pool_key in result:
                    current_state[pool_key] = current_state[pool_key] + _value(result, pool_key)
            
            # Accumulate respiration
            resp = _safe_float(
                _value(result, 'diagnostics.pools_matrix.total_respiration_step')
            )
            cumulative_respiration += resp
        
        # Final carbon in pools
        final_total = sum(_safe_float(current_state[key].detach().item()) for key in current_state.keys() 
                         if key.endswith('_c'))
        
        # Mass balance: initial = final + respired (no inputs)
        mass_balance_error = abs(initial_total - (final_total + cumulative_respiration))
        relative_error = mass_balance_error / initial_total
        
        assert relative_error < 0.01, \
            f"Mass balance error over time: {mass_balance_error:.2e} (relative: {relative_error:.4f})"
    
    # =============================================================================
    # L1.4: Matrix Operations and Numerical Accuracy Tests
    # =============================================================================
    
    def test_matrix_multiplication_accuracy(
        self,
        pools_matrix,
        standard_rates,
        standard_state,
    ):
        """The canonical forward delta must equal the compartment matrix product."""
        current_pools = torch.stack([standard_state[key] for key in POOL_OUTPUTS])
        T = pools_matrix.transfer_matrix.to(
            dtype=current_pools.dtype,
            device=current_pools.device,
        )
        identity = torch.eye(5, dtype=T.dtype, device=T.device)
        rates = standard_rates.to(dtype=current_pools.dtype, device=current_pools.device)
        expected_delta = (T - identity) @ (rates * current_pools)
        result = pools_matrix.forward(standard_state, time=0.0, dt=1.0)
        actual_delta = torch.stack([_value(result, key) for key in POOL_OUTPUTS])

        assert torch.allclose(actual_delta, expected_delta, rtol=1.0e-6, atol=1.0e-12)
    
    def test_numerical_precision_small_values(self, pools_matrix, standard_state):
        """Test numerical precision with very small pool values."""
        reference = standard_state["metabolic_litter_c"]
        small_pools = torch.tensor(
            [1e-12, 1e-11, 1e-10, 1e-9, 1e-8],
            dtype=reference.dtype,
            device=reference.device,
        )
        state = dict(standard_state)
        for key, value in zip(POOL_OUTPUTS, small_pools):
            state[key] = value

        result = pools_matrix.forward(state, time=0.0, dt=3600.0)
        updated_pools = small_pools + torch.stack(
            [_value(result, key) for key in POOL_OUTPUTS]
        )
        total_resp = _value(result, "heterotrophic_respiration")

        assert torch.isfinite(updated_pools).all(), "Updated pools should be finite"
        assert torch.isfinite(total_resp), "Total respiration should be finite"
        assert _safe_float(total_resp.detach().item()) < 1e-8, "Respiration from small pools should be small"
    
    def test_numerical_stability_large_rates(self, standard_state):
        """Large valid ParameterManager rates must remain conservative."""
        large_rates = torch.tensor([9e-6, 9e-7, 9e-7, 9e-8, 9e-9], dtype=torch.float64)
        manager = get_global_manager()
        for pool_name, value in zip(
            ("metabolic", "structural", "fast_som", "slow_som", "passive_som"),
            large_rates,
        ):
            _set_param_value(
                manager,
                f"pedosphere.carbon_pools.turnover_rate_{pool_name}",
                _safe_float(value),
            )
        pools_matrix = PoolsMatrix(name="large_rate_matrix", config={})
        dt = 3600.0
        result = pools_matrix.forward(standard_state, time=0.0, dt=dt)
        initial = torch.stack([standard_state[key] for key in POOL_OUTPUTS])
        updated_pools = initial + torch.stack([_value(result, key) for key in POOL_OUTPUTS])

        assert torch.all(updated_pools >= 0), "Pools should remain non-negative"
        max_decomp_fraction = torch.max(large_rates * dt)
        assert _safe_float(max_decomp_fraction) <= 1.0, "Should not decompose more than 100% per step"
    
    def test_matrix_inversion_conditioning(self, pools_matrix):
        """Test transfer matrix conditioning for numerical stability."""
        T = pools_matrix.transfer_matrix
        
        # Check matrix conditioning
        # For a 5x5 matrix, compute condition number if possible
        if T.numel() > 0:
            # Check for near-singular conditions
            column_sums = torch.sum(T, dim=0)
            
            # Passive SOM columns can be all zeros (only respiration, no transfers)
            # Check that at least some pools have transfers
            assert torch.sum(column_sums > 1e-10) >= 3, "At least 3 pools should have transfers"
            
            # Check for linear dependence (basic test)
            rank_approximation = torch.linalg.matrix_rank(T.double())  # Use double precision
            assert rank_approximation >= 2, "Matrix should have reasonable rank"
    
    # =============================================================================  
    # L1.5: Boundary Condition Handling Tests
    # =============================================================================
    
    def test_zero_pools_handling(self, pools_matrix, standard_state):
        """Test handling of zero carbon pools."""
        state = {**standard_state, **{key: torch.tensor(0.0) for key in POOL_OUTPUTS}}
        result = pools_matrix.forward(state, time=0.0, dt=3600.0)
        delta = torch.stack([_value(result, key) for key in POOL_OUTPUTS])
        total_resp = _value(result, "heterotrophic_respiration")

        assert torch.equal(delta, torch.zeros_like(delta))
        assert _safe_float(total_resp) < 1e-12, "Zero pools should produce zero respiration"
        assert bool(_value(result, "diagnostics.pools_matrix.carbon_balance_ok").item())
    
    def test_single_nonzero_pool(self, pools_matrix, standard_state):
        """Test behavior with only one non-zero pool."""
        state = {**standard_state, **{key: torch.tensor(0.0) for key in POOL_OUTPUTS}}
        state["fast_som_c"] = torch.tensor(1.0)
        result = pools_matrix.forward(state, time=0.0, dt=3600.0)

        assert _safe_float(_value(result, "fast_som_c")) < 0.0
        assert _safe_float(_value(result, "slow_som_c")) > 0.0
        assert _safe_float(_value(result, "passive_som_c")) > 0.0
        assert _safe_float(_value(result, "heterotrophic_respiration")) > 0.0

    def test_negative_pool_handling(self, pools_matrix, standard_state):
        """Negative carbon must fail instead of being clipped to a floor."""
        invalid = {**standard_state, "structural_litter_c": torch.tensor(-0.1)}

        with pytest.raises(ValueError, match="non-negative"):
            pools_matrix.forward(invalid, time=0.0, dt=3600.0)
    
    def test_extreme_environmental_modifier(self, pools_matrix, standard_state):
        """Test handling of extreme environmental modifiers."""
        extreme_modifiers = [0.0, 1e-10, 10.0, 100.0]
        
        for modifier in extreme_modifiers:
            state = {
                **standard_state,
                "environmental_modifier": torch.tensor(modifier),
            }
            result = pools_matrix.forward(state, time=0.0, dt=3600.0)
            initial = torch.stack([state[key] for key in POOL_OUTPUTS])
            updated_pools = initial + torch.stack(
                [_value(result, key) for key in POOL_OUTPUTS]
            )
            total_resp = _value(result, "heterotrophic_respiration")

            assert torch.isfinite(updated_pools).all(), f"Results should be finite for modifier {modifier}"
            assert torch.isfinite(total_resp), f"Respiration should be finite for modifier {modifier}"
            if modifier > 1e-8:
                assert _safe_float(total_resp) >= 0, "Respiration should be non-negative"
    
    # =============================================================================
    # L1.6: Parameter Sensitivity Analysis Tests
    # =============================================================================
    
    def test_transfer_efficiency_sensitivity(self, standard_state):
        """Test sensitivity to transfer efficiency parameters."""
        # Test different transfer efficiencies
        efficiencies = [0.015, 0.03, 0.05, 0.08]
        passive_changes = []
        param_manager = get_global_manager()
        transfer_key = "pedosphere.decomposition.t_fast_passive"
        
        for eff in efficiencies:
            _set_param_value(param_manager, transfer_key, eff)
            pools = PoolsMatrix(name="test", config={})
            result = pools.forward(standard_state, time=0.0, dt=3600.0)
            
            delta_passive = _safe_float(_value(result, 'passive_som_c'))
            passive_changes.append(delta_passive)
        
        # Higher transfer efficiency should generally increase input to passive pool
        # (though system complexity may cause non-monotonic behavior)
        assert passive_changes[-1] >= passive_changes[0], \
            "Higher transfer efficiency should increase passive pool input"
    
    def test_mass_balance_tolerance_sensitivity(self, standard_state):
        """Test sensitivity to mass balance tolerance parameter."""
        tolerances = [1e-6, 1e-4, 1e-2]  # Removed 1e-8 as it's too strict for numerical precision
        param_manager = get_global_manager()
        tolerance_key = "pedosphere.numerical.mass_balance_tolerance"

        for tol in tolerances:
            _set_param_value(param_manager, tolerance_key, tol)
            pools = PoolsMatrix(name="test", config={})
            
            # Should not affect calculations, only validation
            result = pools.forward(standard_state, time=0.0, dt=3600.0)
            
            # Mass balance check should use appropriate tolerance
            if 'diagnostics.pools_matrix.carbon_balance_ok' in result:
                # All should pass with reasonable inputs
                assert bool(
                    _value(result, 'diagnostics.pools_matrix.carbon_balance_ok').item()
                ), f"Mass balance should pass with tolerance {tol}"
    
    def test_max_pool_size_validation(self, standard_state):
        """The declared upper stock guard must reject an invalid advance."""
        param_manager = get_global_manager()
        max_key = "pedosphere.numerical.max_pool_size"
        _set_param_value(param_manager, max_key, 100.0)
        pools = PoolsMatrix(name="test", config={})
        invalid = {**standard_state, "slow_som_c": torch.tensor(200.0)}

        with pytest.raises(ValueError, match="max_pool_size"):
            pools.forward(invalid, time=0.0, dt=3600.0)
    
    # =============================================================================
    # L1.7: Performance and Stability Tests
    # =============================================================================
    
    def test_performance_statistics(self, pools_matrix, standard_state):
        """Test performance statistics tracking and calculation."""
        # Run several forward passes
        for i in range(3):
            pools_matrix.forward(standard_state, time=_safe_float(i*3600), dt=3600.0)
        
        # Get transfer statistics
        stats = pools_matrix.get_transfer_statistics()
        
        # Should have basic statistics
        assert 'total_inputs_processed' in stats
        assert 'total_transfers_processed' in stats  
        assert 'execution_count' in stats
        assert stats['execution_count'] == 3, "Should track execution count"
        
        # Transfer matrix statistics
        assert 'total_transfer_efficiency' in stats
        assert 'max_transfer_efficiency' in stats
        assert 'mean_respiration_fraction' in stats
        
        # Turnover times
        turnover_keys = [f'{pool}_turnover_years' for pool in pools_matrix.pool_names]
        for key in turnover_keys:
            assert key in stats, f"Should include {key}"
            assert stats[key] > 0, f"{key} should be positive"

    def test_reset_performance_tracking_resets_all_counters(
        self,
        pools_matrix,
        standard_state,
    ):
        """Reset should include the public execution counter."""
        pools_matrix.forward(standard_state, time=0.0, dt=3600.0)
        assert pools_matrix.get_transfer_statistics()["execution_count"] == 1

        pools_matrix.reset_performance_tracking()
        stats = pools_matrix.get_transfer_statistics()

        assert stats["execution_count"] == 0
        assert stats["total_inputs_processed"] == 0.0
        assert stats["total_transfers_processed"] == 0.0
        assert stats["total_outputs_generated"] == 0.0
        assert stats["cumulative_mass_error"] == 0.0
        assert stats["max_mass_error_observed"] == 0.0
    
    def test_repeated_execution_consistency(self, pools_matrix, standard_state):
        """Test consistency of repeated executions."""
        results = []
        
        # Run same conditions multiple times
        for _ in range(3):
            result = pools_matrix.forward(standard_state.copy(), time=0.0, dt=3600.0)
            total_resp = _safe_float(_value(result, 'heterotrophic_respiration'))
            results.append(total_resp)
        
        # Should be identical (deterministic)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-12, \
                f"Results should be deterministic: {results[0]:.2e} vs {results[i]:.2e}"
    
    def test_memory_efficiency_large_timesteps(self, pools_matrix, standard_state):
        """Test memory efficiency with large number of timesteps."""
        # Run many small timesteps
        dt = 60.0  # 1 minute
        n_steps = 100
        
        for i in range(n_steps):
            result = pools_matrix.forward(standard_state, time=_safe_float(i*dt), dt=dt)
            
            # Should not accumulate memory or degrade performance
            # Just check it doesn't crash and gives reasonable results
            assert 'diagnostics.pools_matrix.total_carbon_stock' in result
            total_c = _safe_float(
                _value(result, 'diagnostics.pools_matrix.total_carbon_stock')
            )
            assert 0.0 <= total_c <= 20.0, f"Total carbon should be reasonable: {total_c}"
    
    def test_gradient_computation_support(self, pools_matrix):
        """Test support for gradient computation (PyTorch autograd)."""
        # Create state with gradients enabled
        grad_state = {
            'metabolic_litter_c': torch.tensor(0.1, requires_grad=True),
            'structural_litter_c': torch.tensor(0.3, requires_grad=True),  
            'fast_som_c': torch.tensor(0.8, requires_grad=True),
            'slow_som_c': torch.tensor(2.0, requires_grad=True),
            'passive_som_c': torch.tensor(4.0, requires_grad=True),
            'environmental_modifier': torch.tensor(1.0, requires_grad=True),
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
            'fast_som_input': torch.tensor(0.0),
            'slow_som_input': torch.tensor(0.0),
            'passive_som_input': torch.tensor(0.0),
        }
        
        # Forward pass
        result = pools_matrix.forward(grad_state, time=0.0, dt=3600.0)
        
        total_resp = _value(result, 'heterotrophic_respiration')
        assert total_resp.requires_grad
        total_resp.backward()

        differentiable_inputs = [
            tensor
            for tensor in grad_state.values()
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad
        ]
        assert all(tensor.grad is not None for tensor in differentiable_inputs)
        assert any(torch.any(torch.abs(tensor.grad) > 0.0) for tensor in differentiable_inputs)
    
    def test_concurrent_execution_safety(self, pools_matrix, standard_state):
        """Test thread safety for concurrent execution."""
        import concurrent.futures
        
        def worker():
            result = pools_matrix.forward(standard_state.copy(), time=0.0, dt=3600.0)
            return _safe_float(_value(result, 'heterotrophic_respiration'))
        
        # Run multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(8)]
            results = [future.result() for future in futures]

        assert len(results) == 8, "Should have 8 results"
        
        # All results should be identical (deterministic, no shared state)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-12, \
                f"Concurrent results should be identical: {results[0]:.2e} vs {results[i]:.2e}"

