"""
StateManager L1 Unit Tests

Tests for the centralized state manager that handles transactional updates,
conservation checking, and schema validation.

Mathematical Expressions:
- State update: s_i(t+dt) = s_i(t) + u_i, where u_i are incremental updates
- Conservation check: |危 u_j| 鈮?蔚 for conserved quantities
- Bounds validation: lo 鈮?s_i 鈮?hi for all state variables

References:
- RELIC Architecture Documentation
- Transactional state management patterns
- Conservation laws in environmental modeling
"""

import pytest
import torch
import relic.core.state as state_package
import relic.core.state.manager as state_manager_module
import relic.contracts.contract_loader as contract_loader_module

# Mock imports to avoid complex dependencies
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from relic.core.state import (
        StateManager, StateVariableSpec, StateVariableType,
        ConservationSummary, ElementConservationReport
    )
    STATE_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Cannot import StateManager: {e}")
    STATE_MANAGER_AVAILABLE = False


def _expected_like(actual: torch.Tensor, value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=actual.dtype, device=actual.device)


def test_state_variable_units_have_no_default():
    """Units have no universal value: a StateVariableSpec declares them."""
    import dataclasses

    fields = {field.name: field for field in dataclasses.fields(StateVariableSpec)}
    assert fields["units"].default is dataclasses.MISSING
    with pytest.raises(TypeError, match="units"):
        StateVariableSpec(name="undeclared")


class TestStateManagerL1:
    """L1 unit tests for StateManager."""

    def test_state_package_has_no_global_state_manager_facade(self):
        for name in ("get_global_state_manager", "set_global_state_manager"):
            assert name not in state_package.__all__
            assert not hasattr(state_package, name)
            assert not hasattr(state_manager_module, name)
        assert not hasattr(state_manager_module, "_GLOBAL_STATE_MANAGER")

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_initialization_default(self):
        """Test StateManager initialization with default schema."""
        manager = StateManager()
        
        assert manager.device.type in ("cpu", "cuda")
        assert len(manager._schema) > 0
        assert isinstance(manager._state_variables, dict)
        assert isinstance(manager._history, list)
        assert isinstance(manager._conservation_history, list)
        assert manager._abs_conservation_tol == 1e-6
        assert manager._enable_auto_report is True

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_initialization_with_custom_schema(self):
        """Test StateManager initialization with custom schema."""
        custom_schema = {
            'test_var': StateVariableSpec(
                name='test_var',
                units='test_units',
                bounds=(0.0, 10.0),
                conserved=False,
                spatial_dims=0,
                required=True,
                default_value=5.0
            )
        }
        
        manager = StateManager(schema=custom_schema)
        
        assert len(manager._schema) == 1
        assert 'test_var' in manager._schema
        assert manager._schema['test_var'].units == 'test_units'
        assert manager._schema['test_var'].bounds == (0.0, 10.0)

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_register_variable(self):
        """Test variable registration."""
        manager = StateManager()
        
        spec = StateVariableSpec(
            name='new_var',
            units='kg m-2',
            bounds=(0.0, 100.0),
            conserved=True,
            spatial_dims=1,
            required=True,
            default_value=10.0
        )
        
        manager.register_variable(spec)
        
        assert 'new_var' in manager._schema
        assert manager._schema['new_var'] == spec
        assert 'new_var' in manager._state_variables
        assert manager._state_variables['new_var'].item() == 10.0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_register_variable_with_metadata(self):
        """Test variable registration with metadata."""
        manager = StateManager()

        spec = StateVariableSpec(
            name='managed_var',
            units='g C m-2',
            bounds=(0.0, 50.0),
            conserved=True,
            spatial_dims=0,
            required=False,
            default_value=15.0,
            variable_type=StateVariableType.SCALAR,
        )

        manager.register_variable(spec)

        assert 'managed_var' in manager._schema
        assert manager._schema['managed_var'].units == 'g C m-2'
        assert manager._schema['managed_var'].conserved is True
        assert manager._state_variables['managed_var'].item() == 15.0

    def test_state_variable_type_lists_storage_roles_only(self):
        """StateVariableType is a storage role only.

        The quantity basis (extensive/intensive) is owned by
        SpatialQuantityBasis and the prognostic/diagnostic distinction by the
        contract kind (StateVariableSpec.kind); neither is a storage role.
        """
        assert {member.name for member in StateVariableType} == {
            "SCALAR",
            "GRID",
            "CONCENTRATION",
            "PARAMETER",
        }

    def test_contract_autoload_malformed_contract_raises_naming_file(self, tmp_path, monkeypatch):
        """A malformed SCM contract must stop construction, not disable governance."""
        contracts_dir = tmp_path / "variables"
        contracts_dir.mkdir()
        (contracts_dir / "good_state.yaml").write_text(
            "variable: good_state\nkind: state\ndimension:\n  udunits: K\n",
            encoding="utf-8",
        )
        (contracts_dir / "broken_layer.yaml").write_text(
            "variable: broken_layer\nkind: state\ndimension:\n  udunits: m3 m-3\n"
            "layer_thickness_variable: 3\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("RELIC_AUTOLOAD_CONTRACT_REGISTRY", raising=False)
        monkeypatch.setattr(contract_loader_module, "PACKAGED_CONTRACTS_DIR", contracts_dir)
        with pytest.raises(ValueError, match="broken_layer.yaml"):
            StateManager()

    def test_contract_autoload_missing_directory_raises(self, tmp_path, monkeypatch):
        """A missing contracts directory must not yield a silent empty registry."""
        monkeypatch.delenv("RELIC_AUTOLOAD_CONTRACT_REGISTRY", raising=False)
        monkeypatch.setattr(
            contract_loader_module, "PACKAGED_CONTRACTS_DIR", tmp_path / "absent"
        )
        with pytest.raises(FileNotFoundError, match="absent"):
            StateManager()

    def test_contract_autoload_opt_out_is_explicit_and_logged(self, tmp_path, monkeypatch, caplog):
        """Opting out is the only way to run without contracts, and it is visible."""
        monkeypatch.setattr(
            contract_loader_module, "PACKAGED_CONTRACTS_DIR", tmp_path / "absent"
        )
        monkeypatch.delenv("RELIC_AUTOLOAD_CONTRACT_REGISTRY", raising=False)
        with caplog.at_level("INFO", logger="relic.core.state.manager"):
            manager = StateManager(autoload_contract_registry=False)
        assert manager._contract_registry is None
        assert "autoload_contract_registry=False" in caplog.text

        caplog.clear()
        monkeypatch.setenv("RELIC_AUTOLOAD_CONTRACT_REGISTRY", "0")
        with caplog.at_level("INFO", logger="relic.core.state.manager"):
            manager = StateManager()
        assert manager._contract_registry is None
        assert "RELIC_AUTOLOAD_CONTRACT_REGISTRY" in caplog.text

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_parameter_conservation_ledger_auto_loaded(self):
        """StateManager should auto-ingest ledger definitions from ParameterManager."""
        manager = StateManager()
        ledger = manager.get_conservation_ledger()
        assert "carbon" in ledger
        assert any("leaf_carbon" in pool for pool in ledger["carbon"])

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_initialize_with_state(self):
        """Test initialization with provided state."""
        manager = StateManager()
        
        initial_state = {
            'soil_moisture': 0.25,
            'temperature': 295.0,
            'carbon_flux': torch.tensor([1.0, 2.0, 3.0])
        }
        
        manager.initialize(initial_state)
        
        assert len(manager._state_variables) >= 3
        assert manager._state_variables['soil_moisture'].item() == 0.25
        assert manager._state_variables['temperature'].item() == 295.0
        actual = manager._state_variables['carbon_flux']
        assert torch.equal(actual, _expected_like(actual, [1.0, 2.0, 3.0]))

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_initialize_without_state(self):
        """Test initialization without provided state (use defaults)."""
        manager = StateManager()

        manager.initialize()

        # Should have no state variables since all default schema variables are not required
        assert len(manager._state_variables) == 0

        # Test that defaults are optional and bounded when provided by contracts
        assert 'soil_moisture' in manager._schema
        default_value = manager._schema['soil_moisture'].default_value
        if default_value is not None:
            assert 0.0 <= float(default_value) <= 1.0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_set_state(self):
        """Test setting full state."""
        manager = StateManager()
        
        state = {
            'var1': torch.tensor([1.0, 2.0]),
            'var2': 3.5,
            'var3': torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        }
        
        manager.set_state(state)
        
        assert len(manager._state_variables) == 3
        actual_var1 = manager._state_variables['var1']
        assert torch.equal(actual_var1, _expected_like(actual_var1, [1.0, 2.0]))
        assert manager._state_variables['var2'].item() == 3.5
        actual_var3 = manager._state_variables['var3']
        assert torch.equal(actual_var3, _expected_like(actual_var3, [[1.0, 2.0], [3.0, 4.0]]))

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_get_state(self):
        """Test getting state (all and specific variables)."""
        manager = StateManager()
        
        manager.set_state({
            'var1': 1.0,
            'var2': 2.0,
            'var3': 3.0
        })
        
        # Get all state
        all_state = manager.get_state()
        assert len(all_state) == 3
        assert all_state['var1'].item() == 1.0
        
        # Get specific variables
        partial_state = manager.get_state(['var1', 'var3'])
        assert len(partial_state) == 2
        assert partial_state['var1'].item() == 1.0
        assert partial_state['var3'].item() == 3.0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_apply_updates_simple(self):
        """Test simple state updates."""
        manager = StateManager()
        
        manager.set_state({
            'soil_moisture': 0.2,
            'temperature': 295.0
        })
        
        updates = {
            'soil_moisture': 0.05,  # +0.05
            'temperature': -2.0      # -2.0
        }
        
        manager.apply_updates(updates, dt=1800.0)
        
        assert manager._state_variables['soil_moisture'].item() == 0.25
        assert manager._state_variables['temperature'].item() == 293.0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_apply_updates_tensor(self):
        """Test tensor state updates."""
        manager = StateManager()
        
        manager.set_state({
            'grid_var': torch.tensor([1.0, 2.0, 3.0])
        })
        
        updates = {
            'grid_var': torch.tensor([0.1, -0.2, 0.3])
        }
        
        manager.apply_updates(updates)
        
        actual = manager._state_variables['grid_var']
        expected = _expected_like(actual, [1.1, 1.8, 3.3])
        assert torch.allclose(actual, expected)

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_apply_updates_bounds_violation(self):
        """Test bounds violation during updates is clipped with diagnostics."""
        manager = StateManager()

        # Register variable with bounds
        manager.register_variable(
            StateVariableSpec(
                name='bounded_var',
                units='test',
                bounds=(0.0, 10.0),
                conserved=False
            )
        )

        manager.set_state({'bounded_var': 5.0})

        # Update that would exceed bounds
        updates = {'bounded_var': 10.0}  # Would result in 15.0 > 10.0

        # Should clip to upper bound and record clip loss
        result = manager.apply_updates(updates)
        expected = torch.tensor(
            10.0,
            dtype=result['bounded_var'].dtype,
            device=result['bounded_var'].device,
        )
        assert torch.isclose(result['bounded_var'], expected)
        diagnostics = manager.get_diagnostics()
        assert 'bounded_var' in diagnostics.get('clip_loss', {})

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_apply_updates_conservation_check(self):
        """Test conservation checking during updates."""
        manager = StateManager()
        manager.conservation_tolerance = 1e-6

        for name in ("carbon_pool", "carbon_source_pool", "carbon_sink_pool"):
            manager.register_variable(
                StateVariableSpec(
                    name=name,
                    units="g C m-2",
                    conserved=True,
                    required=False,
                )
            )
        manager.set_state({
            'carbon_pool': 100.0,
            'carbon_source_pool': 0.0,
            'carbon_sink_pool': 0.0
        })

        # Conservation violation: net gain
        updates = {
            'carbon_pool': 5.0,
            'carbon_source_pool': 10.0,
            'carbon_sink_pool': -2.0,
        }

        # Should raise conservation violation
        with pytest.raises(ValueError, match="Conservation violation"):
            manager.apply_updates(updates, check_conservation=True)

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_apply_updates_rollback_on_failure(self):
        """Test rollback on update failure."""
        manager = StateManager()

        # Register variable with bounds to create violation
        manager.register_variable(
            StateVariableSpec(
                name='test_var',
                units='test',
                bounds=(0.0, 10.0),
                conserved=False
            )
        )
        manager.set_state({'test_var': 5.0})

        # Create violation and enable rollback
        updates = {'test_var': 10.0}  # Would result in 15.0 > 10.0

        result = manager.apply_updates(
            updates,
            rollback_on_failure=True
        )

        # Bounds clipping should occur; rollback is only for hard failures
        expected = torch.tensor(
            10.0,
            dtype=result['test_var'].dtype,
            device=result['test_var'].device,
        )
        assert torch.isclose(result['test_var'], expected)

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_validate_state_success(self):
        """Test successful state validation."""
        manager = StateManager()
        
        valid_state = {
            'soil_moisture': 0.3,  # Within (0.0, 1.0)
            'temperature': 295.0    # Within (0.0, 400.0)
        }
        
        errors = manager.validate_state(valid_state)
        assert len(errors) == 0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_validate_state_bounds_violation(self):
        """Test state validation with bounds violations."""
        manager = StateManager()

        invalid_state = {
            'soil_moisture': 1.5,   # Exceeds upper bound 1.0
            'air_temperature': -10.0    # Below lower bound 0.0
        }

        errors = manager.validate_state(invalid_state)
        assert any("air_temperature" in error for error in errors)

        soil_schema = manager._schema.get("soil_moisture")
        soil_bounds = None
        if soil_schema is not None:
            if isinstance(soil_schema, dict):
                soil_bounds = soil_schema.get("bounds")
            else:
                soil_bounds = getattr(soil_schema, "bounds", None)
        if soil_bounds is not None:
            assert any("soil_moisture" in error for error in errors)

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_generate_conservation_report(self):
        """Test conservation report generation."""
        manager = StateManager()

        specs = (
            StateVariableSpec(
                name="carbon_storage",
                units="g C m-2",
                conserved=True,
                required=False,
            ),
            StateVariableSpec(
                name="nitrogen_storage",
                units="g N m-2",
                conserved=True,
                required=False,
            ),
            StateVariableSpec(
                name="water_storage",
                units="kg m-2",
                conserved=True,
                required=False,
            ),
        )
        for spec in specs:
            manager.register_variable(spec)

        updates = {
            'carbon_storage': 1.0,
            'nitrogen_storage': -0.1,
            'water_storage': 0.5,
        }

        report = manager.generate_conservation_report(updates, current_time=1000.0)
        
        assert isinstance(report, ConservationSummary)
        assert report.timestamp == 1000.0
        assert hasattr(report, 'carbon_report')
        assert hasattr(report, 'nitrogen_report')
        assert hasattr(report, 'phosphorus_report')
        assert isinstance(report.balance_error, dict)

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_set_conservation_ledger(self):
        """Test setting conservation ledger."""
        manager = StateManager()
        
        ledger = {
            'carbon': ['c_pool_fast', 'c_pool_slow'],
            'nitrogen': ['n_pool_mineral', 'n_pool_organic'],
            'water': ['soil_water', 'surface_water']
        }
        
        manager.set_conservation_ledger(ledger)
        
        assert manager._conservation_ledger == ledger
        assert manager.get_conservation_ledger() == ledger

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_set_conservation_ledger_validation(self):
        """Test conservation ledger validation."""
        manager = StateManager()
        
        # Invalid ledger (non-dict)
        with pytest.raises(ValueError, match="must be a dictionary"):
            manager.set_conservation_ledger(['not', 'a', 'dict'])
        
        # Invalid ledger (non-string keys)
        with pytest.raises(ValueError, match="must map strings"):
            manager.set_conservation_ledger({123: ['var1', 'var2']})
        
        # Invalid ledger (non-list values)
        with pytest.raises(ValueError, match="must map strings to list"):
            manager.set_conservation_ledger({'carbon': 'not_a_list'})

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_ingest_module_diagnostics(self):
        """Test ingestion of module diagnostics."""
        manager = StateManager()
        
        diagnostics = {
            'photosynthesis_rate': 2.5,
            'stress_factor': torch.tensor(0.8),
            'water_limitation': torch.tensor([0.6, 0.7, 0.8])
        }

        manager.ingest_module_diagnostics('photosynthesis_module', diagnostics)

        # Check that numeric diagnostics were stored
        assert 'photosynthesis_module.photosynthesis_rate' in manager._current_diagnostics
        assert 'photosynthesis_module.stress_factor' in manager._current_diagnostics
        assert 'photosynthesis_module.water_limitation' in manager._current_diagnostics

    @pytest.mark.parametrize("value", ["active", None, [1.0, 2.0], {"nested": 1.0}])
    def test_non_numeric_module_diagnostics_are_rejected(self, value):
        """Stored diagnostics are real numbers: anything else raises, nothing is dropped."""
        manager = StateManager()
        with pytest.raises(TypeError, match="probe.status"):
            manager.ingest_module_diagnostics("probe", {"rate": 1.0, "status": value})
        assert "probe.rate" not in manager._current_diagnostics

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_get_spatial_variables(self):
        """Test getting list of spatial variables."""
        manager = StateManager()
        
        # Register variables with different spatial dimensions
        manager.register_variable(
            StateVariableSpec(name='scalar_var', spatial_dims=0, units='1')
        )
        manager.register_variable(
            StateVariableSpec(name='spatial_var_1d', spatial_dims=1, units='m')
        )
        manager.register_variable(
            StateVariableSpec(name='spatial_var_2d', spatial_dims=2, units='m')
        )
        
        spatial_vars = manager.get_spatial_variables()
        
        assert 'scalar_var' not in spatial_vars
        assert 'spatial_var_1d' in spatial_vars
        assert 'spatial_var_2d' in spatial_vars

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_get_variable_spatial_info(self):
        """Test getting spatial information for specific variable."""
        manager = StateManager()
        
        manager.register_variable(
            StateVariableSpec(
                name='test_grid',
                units='m',
                spatial_dims=2
            )
        )
        
        manager.set_state({
            'test_grid': torch.ones((10, 20))
        })
        
        info = manager.get_variable_spatial_info('test_grid')
        
        assert info['name'] == 'test_grid'
        assert info['shape'] == [10, 20]
        assert info['size'] == 200
        assert info['dimensions'] == 2
        assert info['spatial_dims'] == 2
        assert info['scale'] == 'grid_2d'

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_update_environment(self):
        """Test updating environment variables."""
        manager = StateManager()
        
        environment = {
            'air_temperature': 295.0,
            'solar_radiation': torch.tensor([800.0, 900.0, 1000.0]),
            'wind_speed': 3.5
        }
        
        manager.update_environment(environment)
        
        assert manager._state_variables['air_temperature'].item() == 295.0
        actual = manager._state_variables['solar_radiation']
        assert torch.equal(actual, _expected_like(actual, [800.0, 900.0, 1000.0]))
        assert manager._state_variables['wind_speed'].item() == 3.5

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_conservation_tolerance(self):
        """Test conservation tolerance property."""
        manager = StateManager()
        
        # Test default
        assert manager.conservation_tolerance == 1e-6
        
        # Test setting
        manager.conservation_tolerance = 1e-8
        assert manager.conservation_tolerance == 1e-8
        assert manager._abs_conservation_tol == 1e-8

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")

    def test_reset_functionality(self):
        """Test reset functionality."""
        manager = StateManager()
        
        # Add state and history
        manager.set_state({'test_var': 5.0})
        manager._conservation_history.append(ConservationSummary(total_violations=0, tolerance_value=1e-6))
        manager._last_summary = ConservationSummary(total_violations=0, tolerance_value=1e-6)
        
        # Reset
        manager.reset()
        
        assert len(manager._state_variables) == 0
        assert len(manager._history) == 0
        assert len(manager._conservation_history) == 0
        assert manager._last_summary is None
        assert len(manager._current_diagnostics) == 0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_state_accessors(self):
        """Test state accessor properties."""
        manager = StateManager()
        
        manager.set_state({'test_var': 5.0})
        
        # Test state_vars property
        assert 'test_var' in manager.state_vars
        assert manager.state_vars['test_var'].item() == 5.0
        
        # Test schema property
        assert isinstance(manager.schema, dict)
        assert len(manager.schema) > 0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_device_consistency(self):
        """Test device consistency."""
        device = torch.device("cpu")
        manager = StateManager(device=device)
        
        manager.set_state({
            'test_var': torch.tensor([1.0, 2.0, 3.0], device=device)
        })
        
        updates = {'test_var': torch.tensor([0.1, 0.2, 0.3], device=device)}
        
        manager.apply_updates(updates)
        
        expected = torch.tensor([1.1, 2.2, 3.3], device=device)
        assert torch.allclose(manager._state_variables['test_var'], expected)

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_tensor_conversion_handling(self):
        """Test handling of tensor vs scalar conversion."""
        manager = StateManager()
        
        # Set state with mixed types
        manager.set_state({
            'scalar_var': 5.0,
            'tensor_var': torch.tensor([1.0, 2.0])
        })
        
        # All should be stored as tensors
        assert isinstance(manager._state_variables['scalar_var'], torch.Tensor)
        assert isinstance(manager._state_variables['tensor_var'], torch.Tensor)
        
        # Updates should also handle mixed types
        manager.apply_updates({
            'scalar_var': 2.0,  # scalar
            'tensor_var': torch.tensor([0.5, -0.5])  # tensor
        })
        
        assert manager._state_variables['scalar_var'].item() == 7.0
        actual = manager._state_variables['tensor_var']
        assert torch.equal(actual, _expected_like(actual, [1.5, 1.5]))

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_conservation_check_categories(self):
        """Test conservation checking for different element categories."""
        manager = StateManager()

        manager.register_variable(
            StateVariableSpec(
                name="carbon_storage",
                units="g C m-2",
                conserved=True,
                required=False,
            )
        )
        manager.register_variable(
            StateVariableSpec(
                name="water_storage",
                units="kg m-2",
                conserved=True,
                required=False,
            )
        )

        # Use explicit storage units for tracked conservation categories.
        updates = {
            'carbon_storage': 2.0,
            'test_nonconserved_var': -0.5,
            'water_storage': 10.0,
            'other_signal': 1.0,
        }

        violations = manager._check_conservation(updates)

        # Only carbon and water should be detected
        assert 'carbon' in violations
        assert 'water' in violations
        assert violations['carbon'] == 2.0
        assert violations['water'] == 10.0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_explicit_state_manager_instances_are_isolated(self):
        """Explicit StateManager owners must not share prognostic state."""
        manager1 = StateManager()
        manager2 = StateManager()
        manager1.initialize_state({"temperature": torch.tensor(280.0)})
        manager2.initialize_state({"temperature": torch.tensor(290.0)})

        manager1.set_state("temperature", torch.tensor(281.0))

        assert manager1 is not manager2
        state1 = manager1.get_state("temperature")
        state2 = manager2.get_state("temperature")
        assert torch.equal(state1, _expected_like(state1, 281.0))
        assert torch.equal(state2, _expected_like(state2, 290.0))


class TestStateManagerElementConservation:
    """Test element conservation functionality."""

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_element_conservation_report(self):
        """Test ElementConservationReport creation."""
        report = ElementConservationReport(
            balance_error=1e-6,
            relative_error=1e-8
        )
        
        assert report.balance_error == 1e-6
        assert report.relative_error == 1e-8

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_conservation_summary(self):
        """Test ConservationSummary creation."""
        summary = ConservationSummary(
            total_violations=2,
            warnings=['Warning 1', 'Warning 2'],
            custom_checks={'test_check': 1.0}, tolerance_value=1e-6
        )
        
        assert summary.total_violations == 2
        assert len(summary.warnings) == 2
        assert summary.custom_checks['test_check'] == 1.0


class TestStateManagerErrorPropagation:
    """StateManager paths raise instead of substituting modes, values or grid metadata."""

    def test_state_manager_does_not_guess_grid_resolution(self):
        """Grid geometry is declared by the scale owner, never guessed from variable names."""
        assert not hasattr(StateManager, "get_grid_resolution")
        assert not hasattr(StateManager, "set_grid_resolution")

    def test_run_spec_failures_are_not_replaced_by_default_modes(self):
        from types import SimpleNamespace

        from relic.core.state import UpdateItem, UpdateMode

        manager = StateManager()

        def _broken(name, mode):
            raise KeyError(f"no run spec for {name}")

        manager._run_specs = SimpleNamespace(
            resolve_update_mode=_broken, resolve_semantic_mode=_broken, strict=False,
        )
        item = UpdateItem(value=torch.tensor(1.0), mode=UpdateMode.ACCUMULATE)
        with pytest.raises(KeyError, match="no run spec"):
            manager._apply_run_specs("x", item)
        with pytest.raises(KeyError, match="no run spec"):
            manager._resolve_write_mode("x", item)

        manager._run_specs = SimpleNamespace(resolve_update_mode=lambda name, mode: "overwrite", strict=False)
        with pytest.raises(ValueError, match="overwrite"):
            manager._apply_run_specs("x", item)

    @pytest.mark.parametrize("value", ["bogus", 3])
    def test_unknown_write_mode_is_rejected(self, value):
        with pytest.raises((ValueError, TypeError)):
            StateManager._coerce_write_mode(value)

    def test_update_tuple_with_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="bogus"):
            StateManager._coerce_update_item((1.0, "bogus"))
        with pytest.raises(ValueError, match="bogus"):
            StateManager._coerce_update_item({"value": 1.0, "mode": "bogus"})
        assert StateManager._coerce_update_item((1.0, "derived")).write_mode.value == "derived"

    def test_accumulate_onto_a_non_numeric_stored_value_is_rejected(self):
        """A stored value that is not numeric is not silently replaced by the increment.

        set_state() rejects such a value, so it is planted in the store directly.
        """
        manager = StateManager()
        manager.set_state({"label": 0.0}, validate=False)
        manager._state_variables["label"] = "not a number"
        with pytest.raises((TypeError, ValueError)):
            manager.apply_updates({"label": torch.tensor(1.0)})
        assert manager._state_variables["label"] == "not a number"

    def test_module_diagnostics_are_stored_or_rejected_never_dropped(self):
        manager = StateManager()
        manager.ingest_module_diagnostics("probe", {"mask": torch.tensor([True, False, True, True])})
        assert manager._current_diagnostics["probe.mask"] == pytest.approx(0.75)
        with pytest.raises(TypeError, match="complex"):
            manager.ingest_module_diagnostics("probe", {"phase": torch.tensor([1 + 1j, 2 + 0j])})


class TestStateManagerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_empty_updates(self):
        """Test applying empty updates."""
        manager = StateManager()
        manager.set_state({'test_var': 5.0})
        
        manager.apply_updates({})
        assert manager._state_variables['test_var'].item() == 5.0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_invalid_update_types(self):
        """Test handling invalid update types."""
        manager = StateManager()
        manager.set_state({'test_var': 5.0})
        
        counter_before = manager._update_counter
        for invalid in ('invalid_string', {'invalid': 'dict'}, [1, 2, 3]):
            with pytest.raises(TypeError, match="state payload"):
                manager.apply_updates({'test_var': invalid})
        
        # State should be unchanged
        assert manager._state_variables['test_var'].item() == 5.0
        assert manager._update_counter == counter_before

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_nonexistent_variable_update(self):
        """Test updating nonexistent variables."""
        manager = StateManager()

        # Use variable names that won't trigger conservation checks
        manager.apply_updates({
            'test_variable': 5.0,
            'dummy_var': 3.0
        }, check_conservation=False)  # Disable conservation check for this test

        # Should create new variables
        assert 'test_variable' in manager._state_variables
        assert manager._state_variables['test_variable'].item() == 5.0
        assert 'dummy_var' in manager._state_variables
        assert manager._state_variables['dummy_var'].item() == 3.0

    @pytest.mark.skipif(not STATE_MANAGER_AVAILABLE, reason="StateManager not available")
    def test_large_tensor_updates(self):
        """Test handling large tensor updates."""
        manager = StateManager()
        
        # Create large tensor
        large_tensor = torch.randn(1000, 1000)
        manager.set_state({'large_var': large_tensor})
        
        # Apply update
        update_tensor = torch.ones_like(large_tensor) * 0.001
        manager.apply_updates({'large_var': update_tensor})
        
        # Check update was applied correctly
        actual = manager._state_variables['large_var']
        expected = (large_tensor + update_tensor).to(device=actual.device, dtype=actual.dtype)
        assert torch.allclose(actual, expected, atol=1e-6)


def run_state_manager_l1_tests():
    """Run all StateManager L1 unit tests."""
    print("Running StateManager L1 Unit Tests...")
    print("=" * 60)
    
    if not STATE_MANAGER_AVAILABLE:
        print("WARNING: StateManager not available. Install dependencies to run tests.")
        return False
    
    test_instance = TestStateManagerL1()
    test_conservation = TestStateManagerElementConservation()
    test_edge_cases = TestStateManagerEdgeCases()
    
    all_test_methods = []
    
    # Collect all test methods
    for test_class in [test_instance, test_conservation, test_edge_cases]:
        methods = [method for method in dir(test_class) if method.startswith('test_')]
        all_test_methods.extend([getattr(test_class, method) for method in methods])
    
    passed = 0
    failed = 0
    
    for test_method in all_test_methods:
        try:
            print(f"Running {test_method.__name__}... ", end="")
            test_method()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"StateManager L1 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All StateManager L1 unit tests PASSED!")
        return True
    else:
        print("Some StateManager L1 tests FAILED. Check the implementation.")
        return False


if __name__ == "__main__":
    run_state_manager_l1_tests()

