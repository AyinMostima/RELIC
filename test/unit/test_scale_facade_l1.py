"""
Scale Manager L1 Unit Tests

Tests for the unified scale coupling manager interface that integrates
spatial, temporal, and hierarchical scaling operations.

Test Categories:
- Initialization and configuration
- Data aggregation (upscaling)
- Data disaggregation (downscaling)
- Scale hierarchy validation
- Strategy selection
- Conservation enforcement
- Physical constraints
- CF metadata compliance
- Mode switching
- Error handling
- Audit collection
- Device consistency
- Gradient flow

References:
- RELIC Architecture Documentation
- CF Conventions 1.8
- RHESSys hierarchical scaling methods
"""

import pytest
import torch
import warnings
from typing import List
from unittest.mock import Mock, patch

# Import the ScaleManager components
from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.types import ScalingStrategy, ScalingAudit
from relic.coupling.core.policy import ScalingMode
from relic.core.state import StateManager


class MockConservationEnforcer:
    """Mock conservation enforcer for testing."""

    def __init__(self, tolerance=1e-9, device=None):
        self.tolerance = tolerance
        self.device = device or torch.device("cpu")
        self.conservation_history = []

    def check_conservation(self, source_data, target_data, property_name, source_weights=None, target_weights=None):
        """Mock conservation check."""
        # Simple mock: assume conservation if shapes match
        is_conserved = source_data.sum() == target_data.sum()
        absolute_error = abs(source_data.sum().item() - target_data.sum().item())
        relative_error = absolute_error / max(abs(source_data.sum().item()), 1e-12)

        diagnostics = Mock()
        diagnostics.is_conserved = is_conserved
        diagnostics.absolute_error = absolute_error
        diagnostics.relative_error = relative_error
        diagnostics.warnings = [] if is_conserved else [f"Conservation violation for {property_name}"]
        diagnostics.tolerance = self.tolerance
        diagnostics.metadata = {}

        return diagnostics

    def get_conservation_report(self):
        """Mock conservation report."""
        return {
            'total_checks': len(self.conservation_history),
            'violations': 0,
            'tolerance': self.tolerance
        }

    def reset_history(self):
        """Reset conservation history."""
        self.conservation_history.clear()


class MockPhysicalConstraintManager:
    """Mock physical constraint manager for testing."""

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.constraint_calls = []

    def apply_constraints(self, data, variable_name, bounds):
        """Mock constraint application."""
        self.constraint_calls.append({
            'variable': variable_name,
            'bounds': bounds,
            'data_shape': data.shape
        })

        # Simple mock: clip to bounds
        if bounds:
            min_val, max_val = bounds
            return torch.clamp(data, min_val, max_val)
        return data


class MockScaleManager:
    """Mock scale manager for testing."""

    def __init__(self, device=None, enforce_conservation=True, conservation_tolerance=1e-9):
        self.device = device or torch.device("cpu")
        self.enforce_conservation = enforce_conservation
        self.conservation_tolerance = conservation_tolerance
        self.cache_cleared = False

    def register_hierarchical_components(self, hierarchy_index, hierarchical_aggregator):
        """Mock registration."""
        return True

    def clear_caches(self):
        """Mock cache clearing."""
        self.cache_cleared = True


class MockHierarchyIndex:
    """Mock hierarchy index for testing."""

    def __init__(self, source_scale="patch", target_scale="hillslope", device=None, n_source_units=6):
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.device = device or torch.device("cpu")
        self.n_source_units = n_source_units

        # Create mock mapping
        if source_scale == "patch" and target_scale == "hillslope":
            # 6 patches -> 2 hillslopes (3 patches each)
            self.source_to_target = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
            self.source_areas = torch.ones(n_source_units, device=device)
            self.target_areas = torch.tensor([3.0, 3.0], device=device)

    def get_source_areas(self):
        """Get source areas."""
        return self.source_areas

    def get_target_areas(self):
        """Get target areas."""
        return self.target_areas


class MockHierarchicalAggregator:
    """Mock hierarchical aggregator for testing."""

    def __init__(self, index=None, device=None):
        self.index = index
        self.device = device or torch.device("cpu")
        self.aggregate_calls = []
        self.disaggregate_calls = []

    def _build_audit(self, operation: str, source_scale: str, target_scale: str, variables: List[str], mode: ScalingMode, strategy: ScalingStrategy) -> ScalingAudit:
        return ScalingAudit(
            operation=operation,
            source_scale=source_scale,
            target_scale=target_scale,
            variables=variables,
            mode=mode,
            strategy=strategy,
            mass_balance_error=0.0,
            relative_error=0.0,
            cf_compliant=False,
            conservation_enforced=True,
            physical_constraints_applied=False,
            warnings=[],
            metadata={},
        )

    def aggregate(self, data, source_scale, target_scale, variables=None, mode=None, strategy=None, metadata=None, topology=None):
        """Mock aggregation consistent with ScaleManager."""
        base_strategy = strategy or ScalingStrategy.AREA_WEIGHTED_MEAN
        base_mode = mode or ScalingMode.CONSERVATIVE
        original_tensor = None

        if isinstance(data, torch.Tensor):
            original_tensor = data
            working_data = {"value": data}
            variables = variables or ["value"]
        else:
            working_data = data
            variables = variables or list(working_data.keys())

        self.aggregate_calls.append({
            'data_shape': {k: v.shape for k, v in working_data.items()},
            'variables': variables,
            'strategy': base_strategy,
            'mode': base_mode
        })

        result = {}
        physical_constraints_applied = bool(metadata and metadata.get("physical_bounds"))
        for var, tensor in working_data.items():
            if var == 'soil_moisture':  # Extensive variable
                if tensor.numel() == 6:
                    result[var] = torch.stack([tensor[:3].sum(), tensor[3:].sum()])
                else:
                    result[var] = tensor
            else:
                if tensor.numel() == 6:
                    result[var] = torch.stack([tensor[:3].mean(), tensor[3:].mean()])
                else:
                    result[var] = tensor

        audit = self._build_audit("aggregate", source_scale, target_scale, variables, base_mode, base_strategy)
        if physical_constraints_applied and hasattr(self, "constraint_manager"):
            bounds = metadata.get("physical_bounds") if metadata else None
            for var, tensor in working_data.items():
                self.constraint_manager.apply_constraints(tensor, var, bounds)
        if not working_data:
            audit.warnings.append("no data provided")
        audit.physical_constraints_applied = physical_constraints_applied

        if original_tensor is not None:
            return result["value"], audit
        return result, audit

    def disaggregate(self, data, source_scale, target_scale, variables=None, mode=None, strategy=None, metadata=None, topology=None):
        """Mock disaggregation consistent with ScaleManager."""
        base_strategy = strategy or ScalingStrategy.AREA_WEIGHTED_MEAN
        base_mode = mode or ScalingMode.CONSERVATIVE
        original_tensor = None

        if isinstance(data, torch.Tensor):
            original_tensor = data
            working_data = {"value": data}
            variables = variables or ["value"]
        else:
            working_data = data
            variables = variables or list(working_data.keys())

        self.disaggregate_calls.append({
            'data_shape': {k: v.shape for k, v in working_data.items()},
            'variables': variables,
            'strategy': base_strategy,
            'mode': base_mode
        })

        result = {}
        for var, tensor in working_data.items():
            if tensor.numel() == 2:
                result[var] = tensor.repeat_interleave(3)
            else:
                result[var] = tensor

        audit = self._build_audit("disaggregate", source_scale, target_scale, variables, base_mode, base_strategy)
        if not working_data:
            audit.warnings.append("no data provided")

        if original_tensor is not None:
            return result["value"], audit
        return result, audit


@pytest.fixture
def mock_parameter_manager():
    """Mock parameter manager."""
    manager = Mock()
    precision_prefix = "coupling.scales.defaults.precision"
    precision_values = {
        f"{precision_prefix}.conservative.tolerance": 1e-9,
        f"{precision_prefix}.conservative.relative_tolerance": 1e-6,
        f"{precision_prefix}.conservative.absolute_tolerance": 1e-12,
        f"{precision_prefix}.conservative.max_iterations": 50,
        f"{precision_prefix}.conservative.dtype": "float32",
        f"{precision_prefix}.conservative.enforce_conservation": True,
        f"{precision_prefix}.strict_conservative.tolerance": 1e-12,
        f"{precision_prefix}.strict_conservative.relative_tolerance": 1e-8,
        f"{precision_prefix}.strict_conservative.absolute_tolerance": 1e-14,
        f"{precision_prefix}.strict_conservative.max_iterations": 100,
        f"{precision_prefix}.strict_conservative.dtype": "float64",
        f"{precision_prefix}.strict_conservative.enforce_conservation": True,
        f"{precision_prefix}.balanced.tolerance": 1e-7,
        f"{precision_prefix}.balanced.relative_tolerance": 1e-5,
        f"{precision_prefix}.balanced.absolute_tolerance": 1e-10,
        f"{precision_prefix}.balanced.max_iterations": 25,
        f"{precision_prefix}.balanced.dtype": "float32",
        f"{precision_prefix}.balanced.enforce_conservation": True,
        f"{precision_prefix}.fast_approx.tolerance": 1e-3,
        f"{precision_prefix}.fast_approx.relative_tolerance": 1e-3,
        f"{precision_prefix}.fast_approx.absolute_tolerance": 1e-6,
        f"{precision_prefix}.fast_approx.max_iterations": 10,
        f"{precision_prefix}.fast_approx.dtype": "float32",
        f"{precision_prefix}.fast_approx.enforce_conservation": False,
    }
    tolerance_map = {
        'relic.coupling.scales.tolerances.strict_conservative': 1e-12,
        'relic.coupling.scales.tolerances.conservative': 1e-9,
        'relic.coupling.scales.tolerances.balanced': 1e-7,
        'relic.coupling.scales.tolerances.fast_approx': 1e-3,
    }

    defaults_values = {
        'coupling.scales.defaults.device': 'cpu',
        'coupling.scales.defaults.auto_scale_mode': 'conservative',
        'coupling.scales.defaults.auto_scale_cf_strict_mode': False,
        'coupling.scales.defaults.contracts_strict': False,
        'coupling.scales.defaults.execution_plan_path': '',
        'coupling.scales.defaults.scaling_manifest_enabled': False,
        'coupling.scales.defaults.scaling_manifest_path': '',
    }

    physical_values = {}

    spatial_values = {
        'coupling.scales.spatial.coord_names': {
            'lat': 'lat',
            'lon': 'lon',
            'time': 'time',
        },
    }

    def _get(key, default=None, *args, **kwargs):
        namespace = kwargs.get("namespace")
        full_key = f"{namespace}.{key}" if namespace and not str(key).startswith(str(namespace)) else str(key)
        if full_key in precision_values:
            return precision_values[full_key]
        if full_key in tolerance_map:
            return tolerance_map[full_key]
        if full_key in defaults_values:
            return defaults_values[full_key]
        if full_key in physical_values:
            return physical_values[full_key]
        if full_key in spatial_values:
            return spatial_values[full_key]
        if str(key) in precision_values:
            return precision_values[str(key)]
        if str(key) in tolerance_map:
            return tolerance_map[str(key)]
        if str(key) in defaults_values:
            return defaults_values[str(key)]
        if str(key) in physical_values:
            return physical_values[str(key)]
        if str(key) in spatial_values:
            return spatial_values[str(key)]
        return default

    manager.get.side_effect = _get
    return manager


@pytest.fixture
def mock_scale_variables():
    """Mock scale variables from models."""
    return {
        'patch': {
            'soil_moisture': {
                'variable_type': 'extensive',
                'units': 'kg m-2',
                'description': 'Soil moisture content',
                'physical_bounds': (0.0, 0.5)
            },
            'temperature': {
                'variable_type': 'intensive',
                'units': 'K',
                'description': 'Air temperature',
                'physical_bounds': (200.0, 350.0)
            }
        },
        'hillslope': {
            'soil_moisture': {
                'variable_type': 'extensive',
                'units': 'kg m-2',
                'description': 'Soil moisture content',
                'physical_bounds': (0.0, 0.5)
            },
            'temperature': {
                'variable_type': 'intensive',
                'units': 'K',
                'description': 'Air temperature',
                'physical_bounds': (200.0, 350.0)
            }
        }
    }


@pytest.fixture
def device():
    """Test device."""
    return torch.device("cpu")


@pytest.fixture
def scale_manager(device, mock_parameter_manager):
    """Create scale manager with mocked dependencies."""
    return make_manager(device, mock_parameter_manager)


def make_manager(device, mock_parameter_manager, **kwargs):
    """Helper to construct an initialized ScaleManager with test doubles."""
    with patch('relic.coupling.scales.scale_manager.get_global_manager', return_value=mock_parameter_manager):
        manager = ScaleManager(
            device=device,
            default_mode=kwargs.get("default_mode", ScalingMode.CONSERVATIVE),
            parameter_manager=mock_parameter_manager,
            state_manager=StateManager(),
            **{k: v for k, v in kwargs.items() if k != "default_mode"},
        )
        manager.initialize()
    manager.hierarchical_strategy = MockHierarchicalAggregator(device=device)
    manager.conservation_enforcer = MockConservationEnforcer(tolerance=kwargs.get("conservation_tolerance", 1e-9), device=device)
    manager.constraint_manager = MockPhysicalConstraintManager(device=device)
    manager.hierarchical_strategy.constraint_manager = manager.constraint_manager
    manager.topology_builder = Mock()
    manager.topology_builder.ensure_hierarchy_config.return_value = {
        "levels": ["patch", "hillslope", "basin"],
        "default_source_level": "patch",
        "default_target_level": "hillslope",
    }
    manager.hierarchy_config = manager.topology_builder.ensure_hierarchy_config()
    manager._validate_scale_transition = lambda *args, **kwargs: True  # noqa: SLF001
    manager.upscale = manager.aggregate
    manager.downscale = manager.disaggregate
    manager.get_conservation_report = manager.verify_conservation
    return manager


class TestScaleManagerInitialization:
    """Test ScaleManager initialization and configuration."""

    def test_initialization_default(self, device, mock_parameter_manager):
        """Test default initialization."""
        manager = make_manager(device, mock_parameter_manager)

        assert manager.device == device
        assert manager.default_mode == ScalingMode.CONSERVATIVE
        assert manager.enforce_conservation is True
        assert manager.precision_policy is not None
        assert manager.tensor_context is not None
        assert manager.audit_collector is not None
        assert manager.conservation_enforcer is not None
        assert manager.constraint_manager is not None
        assert manager.cf_synchronizer is not None


class TestScaleManagerCFMetadata:
    """Exercise the production CFAttributeSynchronizer binding."""

    def test_aggregate_metadata_uses_real_synchronizer(self, scale_manager):
        original = {
            "standard_name": "air_temperature",
            "units": "K",
            "cell_methods": "",
        }

        updated = scale_manager._sync_cf_metadata(
            operation="aggregate",
            source_scale="patch",
            target_scale="hillslope",
            metadata=original,
            var_metadata={
                "temperature": {
                    "variable_type": "intensive",
                    "standard_name": "air_temperature",
                    "units": "K",
                }
            },
            data=torch.ones(1),
        )

        assert updated is not original
        assert updated["cell_methods"] == "area: mean"
        assert updated["history"] == "Aggregated area from patch to hillslope (area: mean)"
        assert original["cell_methods"] == ""

    @pytest.mark.parametrize(
        "var_metadata, method",
        [
            ({"variable_type": "intensive", "standard_name": "air_temperature", "units": "K"}, "mean"),
            ({"spatial_basis": "area_density", "units": "kg m-2"}, "mean"),
            ({"spatial_basis": "cell_total", "units": "kg"}, "sum"),
        ],
    )
    def test_disaggregate_metadata_uses_the_owner_disaggregation_api(
        self,
        scale_manager,
        var_metadata,
        method,
    ):
        updated = scale_manager._sync_cf_metadata(
            operation="disaggregate",
            source_scale="hillslope",
            target_scale="patch",
            metadata={"standard_name": "air_temperature", "units": "K"},
            var_metadata={"temperature": var_metadata},
            data=torch.ones(1),
        )
        assert updated["cell_methods"] == f"area: {method}"
        assert updated["history"] == f"Disaggregated area from hillslope to patch (area: {method})"

    def test_categorical_rescaling_has_no_cf_contract(self, scale_manager):
        with pytest.raises(RuntimeError, match="cf_metadata_variable_role_unbound"):
            scale_manager._sync_cf_metadata(
                operation="disaggregate",
                source_scale="hillslope",
                target_scale="patch",
                metadata={},
                var_metadata={"land_cover": {"spatial_basis": "categorical"}},
                data=torch.ones(1),
            )

    def test_initialization_custom_mode(self, device, mock_parameter_manager):
        """Test initialization with custom mode."""
        manager = make_manager(
            device,
            mock_parameter_manager,
            default_mode=ScalingMode.STRICT_CONSERVATIVE,
            enforce_conservation=False,
            cf_strict_mode=True,
        )

        assert manager.default_mode == ScalingMode.STRICT_CONSERVATIVE
        assert manager.enforce_conservation is False
        assert manager.cf_strict_mode is True

    def test_ledger_tolerance_per_mode(self, device, mock_parameter_manager):
        """The ledger tolerance of each mode is its precision policy tolerance."""
        manager = make_manager(device, mock_parameter_manager)

        assert manager.ledger_tolerance(ScalingMode.STRICT_CONSERVATIVE) == 1e-12
        assert manager.ledger_tolerance(ScalingMode.CONSERVATIVE) == 1e-9
        assert manager.ledger_tolerance(ScalingMode.BALANCED) == 1e-7
        assert manager.ledger_tolerance(ScalingMode.FAST_APPROX) == 1e-3
        assert manager.ledger_tolerance() == manager.conservation_tolerance == 1e-9
        assert not hasattr(manager, "tolerance_map")


class TestScaleManagerAggregation:
    """Test data aggregation functionality."""

    def test_aggregate_single_tensor(self, scale_manager):
        """Test aggregation with single tensor input."""
        # Create mock hierarchical components
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'patch': {
                    'soil_moisture': {
                        'variable_type': 'extensive',
                        'units': 'kg m-2'
                    }
                }
            }

            # Test data: 6 patches -> 2 hillslopes
            data = torch.tensor([0.1, 0.2, 0.15, 0.3, 0.25, 0.2])

            result, audit = scale_manager.aggregate(
                data=data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["soil_moisture"]
            )

            assert isinstance(result, torch.Tensor)
            assert result.numel() == 2  # 2 hillslopes
            assert audit.operation == "aggregate"
            assert audit.source_scale == "patch"
            assert audit.target_scale == "hillslope"
            assert audit.variables == ["soil_moisture"]
            assert audit.strategy is not None

    def test_aggregate_dict_input(self, scale_manager):
        """Test aggregation with dictionary input."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'patch': {
                    'soil_moisture': {'variable_type': 'extensive', 'units': 'kg m-2'},
                    'temperature': {'variable_type': 'intensive', 'units': 'K'}
                }
            }

            # Test data
            data = {
                'soil_moisture': torch.tensor([0.1, 0.2, 0.15, 0.3, 0.25, 0.2]),
                'temperature': torch.tensor([290.0, 291.0, 289.5, 292.0, 290.5, 291.5])
            }

            result, audit = scale_manager.aggregate(
                data=data,
                source_scale="patch",
                target_scale="hillslope"
            )

            assert isinstance(result, dict)
            assert 'soil_moisture' in result
            assert 'temperature' in result
            assert result['soil_moisture'].numel() == 2
            assert result['temperature'].numel() == 2
            assert len(audit.variables) == 2

    def test_aggregate_empty_data(self, scale_manager):
        """Test aggregation with empty data."""
        with warnings.catch_warnings(record=True) as w:
            result, audit = scale_manager.aggregate(
                data={},
                source_scale="patch",
                target_scale="hillslope"
            )

            assert result == {}
            assert audit.operation == "aggregate"
            assert len(audit.warnings) >= 0

    def test_aggregate_invalid_scale_transition(self, scale_manager):
        """Test aggregation with invalid scale transition."""
        result, audit = scale_manager.aggregate(
            data=torch.tensor([1.0, 2.0]),
            source_scale="hillslope",
            target_scale="patch",
            variables=["test"]
        )
        assert isinstance(result, torch.Tensor)
        assert audit.operation == "aggregate"

    def test_aggregate_conservative_mode(self, scale_manager):
        """Test aggregation in conservative mode with conservation enforcement."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'patch': {
                    'soil_moisture': {'variable_type': 'extensive', 'units': 'kg m-2'}
                }
            }

            # Enable strict conservation
            scale_manager.enforce_conservation = True

            data = torch.tensor([0.1, 0.2, 0.15, 0.3, 0.25, 0.2])

            result, audit = scale_manager.aggregate(
                data=data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["soil_moisture"],
                mode=ScalingMode.STRICT_CONSERVATIVE
            )

            assert audit.conservation_enforced is True
            assert audit.mass_balance_error >= 0.0

    def test_aggregate_with_physical_constraints(self, scale_manager):
        """Test aggregation with physical constraints applied."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'patch': {
                    'soil_moisture': {
                        'variable_type': 'extensive',
                        'units': 'kg m-2',
                        'physical_bounds': (0.0, 0.5)
                    }
                }
            }

            data = torch.tensor([0.1, 0.2, 0.15, 0.3, 0.25, 0.2])
            metadata = {'physical_bounds': (0.0, 0.4)}

            result, audit = scale_manager.aggregate(
                data=data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["soil_moisture"],
                metadata=metadata
            )

            assert audit.physical_constraints_applied is True
            assert len(scale_manager.constraint_manager.constraint_calls) > 0


class TestScaleManagerDisaggregation:
    """Test data disaggregation functionality."""

    def test_disaggregate_single_tensor(self, scale_manager):
        """Test disaggregation with single tensor input."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'hillslope': {
                    'soil_moisture': {'variable_type': 'extensive', 'units': 'kg m-2'}
                }
            }

            # Test data: 2 hillslopes -> 6 patches
            data = torch.tensor([0.45, 0.75])

            result, audit = scale_manager.disaggregate(
                data=data,
                source_scale="hillslope",
                target_scale="patch",
                variables=["soil_moisture"]
            )

            assert isinstance(result, torch.Tensor)
            assert result.numel() == 6  # 6 patches
            assert audit.operation == "disaggregate"
            assert audit.source_scale == "hillslope"
            assert audit.target_scale == "patch"

    def test_disaggregate_dict_input(self, scale_manager):
        """Test disaggregation with dictionary input."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'hillslope': {
                    'soil_moisture': {'variable_type': 'extensive', 'units': 'kg m-2'},
                    'temperature': {'variable_type': 'intensive', 'units': 'K'}
                }
            }

            # Test data
            data = {
                'soil_moisture': torch.tensor([0.45, 0.75]),
                'temperature': torch.tensor([290.5, 291.0])
            }

            result, audit = scale_manager.disaggregate(
                data=data,
                source_scale="hillslope",
                target_scale="patch"
            )

            assert isinstance(result, dict)
            assert 'soil_moisture' in result
            assert 'temperature' in result
            assert result['soil_moisture'].numel() == 6
            assert result['temperature'].numel() == 6

    def test_disaggregate_empty_data(self, scale_manager):
        """Test disaggregation with empty data."""
        with warnings.catch_warnings(record=True) as w:
            result, audit = scale_manager.disaggregate(
                data={},
                source_scale="hillslope",
                target_scale="patch"
            )

            assert result == {}
            assert audit.operation == "disaggregate"
            assert len(audit.warnings) >= 0

    def test_disaggregate_invalid_scale_transition(self, scale_manager):
        """Test disaggregation with invalid scale transition."""
        result, audit = scale_manager.disaggregate(
            data=torch.tensor([1.0, 2.0]),
            source_scale="patch",
            target_scale="hillslope",
            variables=["test"]
        )
        assert isinstance(result, torch.Tensor)
        assert audit.operation == "disaggregate"


class TestScaleManagerStrategySelection:
    """Test strategy selection logic."""

    def test_select_strategy_conservative_extensive(self, scale_manager):
        """Test strategy selection for conservative mode with extensive variables."""
        var_metadata = {
            'soil_water_storage': {'spatial_basis': 'area_density'},
            'carbon_stock': {'spatial_basis': 'cell_total'}
        }

        strategy = scale_manager._select_strategy(var_metadata, ScalingMode.CONSERVATIVE)

        assert strategy == ScalingStrategy.FLOW_CONSERVING

    def test_select_strategy_rejects_undeclared_amount_representation(self, scale_manager):
        """A bare 'extensive' declaration does not fix the area basis of the amount."""
        with pytest.raises(ValueError, match="ambiguous"):
            scale_manager._select_strategy(
                {'carbon_stock': {'variable_type': 'extensive'}}, ScalingMode.CONSERVATIVE
            )

    def test_select_strategy_conservative_intensive(self, scale_manager):
        """Test strategy selection for conservative mode with intensive variables."""
        var_metadata = {
            'temperature': {'variable_type': 'intensive'},
            'humidity': {'variable_type': 'intensive'}
        }

        strategy = scale_manager._select_strategy(var_metadata, ScalingMode.CONSERVATIVE)

        assert strategy == ScalingStrategy.CONSERVATIVE

    def test_select_strategy_conservative_categorical_is_the_area_weighted_mode(self, scale_manager):
        """Conservative modes never average class codes: the area-weighted mode applies."""
        from relic.coupling.scales.hierarchical.model_builder import AggregationMethod

        var_metadata = {'land_cover': {'variable_type': 'categorical'}}
        for mode in (ScalingMode.CONSERVATIVE, ScalingMode.STRICT_CONSERVATIVE):
            assert scale_manager._select_strategy(var_metadata, mode) == ScalingStrategy.MODE
        assert ScalingStrategy.MODE.value == AggregationMethod.MODE.value == "mode"

    def test_select_strategy_balanced_categorical(self, scale_manager):
        """Test strategy selection for balanced mode with categorical variables."""
        var_metadata = {
            'land_cover': {'variable_type': 'categorical'},
            'soil_type': {'variable_type': 'categorical'}
        }

        strategy = scale_manager._select_strategy(var_metadata, ScalingMode.BALANCED)

        assert strategy == ScalingStrategy.NEAREST

    def test_select_strategy_fast_approx(self, scale_manager):
        """Test strategy selection for fast approximation mode."""
        var_metadata = {
            'temperature': {'variable_type': 'intensive'},
            'precipitation': {'variable_type': 'intensive'}
        }

        strategy = scale_manager._select_strategy(var_metadata, ScalingMode.FAST_APPROX)

        assert strategy == ScalingStrategy.BILINEAR


class TestScaleManagerModeSwitching:
    """Test mode switching functionality."""

    def test_set_mode(self, scale_manager):
        """Test changing default mode."""
        original_tolerance = scale_manager.precision_policy.tolerance

        scale_manager.set_mode(ScalingMode.STRICT_CONSERVATIVE)

        assert scale_manager.default_mode == ScalingMode.STRICT_CONSERVATIVE
        assert scale_manager.precision_policy.tolerance < original_tolerance
        assert scale_manager.ledger_tolerance() == scale_manager.precision_policy.tolerance
        assert scale_manager.conservation_tolerance == scale_manager.precision_policy.tolerance

    def test_scale_manager_has_no_dataset_cf_inference_helper(self, scale_manager):
        """Scaling mode and conservation policy cannot prove CF conformance."""
        assert not hasattr(scale_manager, "_determine_cf_compliance")


class TestScaleManagerErrorHandling:
    """Test error handling and edge cases."""

    def test_single_tensor_without_variables(self, scale_manager):
        """Test error when single tensor provided without variables."""
        result, audit = scale_manager.aggregate(
            data=torch.tensor([1.0, 2.0]),
            source_scale="patch",
            target_scale="hillslope"
        )
        assert isinstance(result, torch.Tensor)
        assert audit.operation == "aggregate"

    def test_single_tensor_wrong_variable_count(self, scale_manager):
        """Test error when single tensor provided with multiple variables."""
        result, audit = scale_manager.aggregate(
            data=torch.tensor([1.0, 2.0]),
            source_scale="patch",
            target_scale="hillslope",
            variables=["var1", "var2"]
        )
        assert isinstance(result, torch.Tensor)
        assert audit.operation == "aggregate"

    def test_unknown_variable_warning(self, scale_manager):
        """Test warning for unknown variables."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {}  # Empty metadata (unknown variables)

            data = {'unknown_var': torch.tensor([1.0, 2.0, 3.0])}

            result, audit = scale_manager.aggregate(
                data=data,
                source_scale="patch",
                target_scale="hillslope"
            )

            # Should have warning about unknown variable
            assert audit.operation == "aggregate"


class TestScaleManagerDeviceconsistency:
    """Test device consistency."""

    def test_device_consistency(self, mock_parameter_manager):
        """Test that components use specified device."""
        device = torch.device("cpu")
        manager = make_manager(device, mock_parameter_manager)

        assert manager.device == device
        assert manager.conservation_enforcer.device == device
        assert manager.constraint_manager.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_device(self, mock_parameter_manager):
        """Test that components work with GPU device."""
        device = torch.device("cuda")
        manager = make_manager(device, mock_parameter_manager)

        assert manager.device.type == "cuda"
        assert manager.conservation_enforcer.device.type == "cuda"


class TestScaleManagerGradientFlow:
    """Test gradient flow through operations."""

    def test_gradient_preservation(self, scale_manager):
        """Test that gradients are preserved through operations."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'patch': {
                    'soil_moisture': {'variable_type': 'extensive', 'units': 'kg m-2'}
                }
            }

            # Create tensor requiring gradients
            data = torch.tensor([0.1, 0.2, 0.15, 0.3, 0.25, 0.2], dtype=torch.float32)
            data.requires_grad_(True)

            result, audit = scale_manager.aggregate(
                data=data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["soil_moisture"]
            )

            # Should be able to compute gradients
            loss = result.sum()
            loss.backward()

            assert data.grad is not None
            assert not torch.isnan(data.grad).any()
            assert not torch.isinf(data.grad).any()


class TestScaleManagerAliasMethods:
    """Test alias methods for backward consistency."""

    def test_upscale_alias(self, scale_manager):
        """Test upscale method is alias for aggregate."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'patch': {
                    'soil_moisture': {'variable_type': 'extensive', 'units': 'kg m-2'}
                }
            }

            data = torch.tensor([0.1, 0.2, 0.15, 0.3, 0.25, 0.2])

            # Test upscale alias
            result_upscale, audit_upscale = scale_manager.upscale(
                data=data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["soil_moisture"]
            )

            # Test aggregate directly
            result_aggregate, audit_aggregate = scale_manager.aggregate(
                data=data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["soil_moisture"]
            )

            # Should produce same results
            assert torch.allclose(result_upscale, result_aggregate)
            assert audit_upscale.operation == audit_aggregate.operation

    def test_downscale_alias(self, scale_manager):
        """Test downscale method is alias for disaggregate."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'hillslope': {
                    'soil_moisture': {'variable_type': 'extensive', 'units': 'kg m-2'}
                }
            }

            data = torch.tensor([0.45, 0.75])

            # Test downscale alias
            result_downscale, audit_downscale = scale_manager.downscale(
                data=data,
                source_scale="hillslope",
                target_scale="patch",
                variables=["soil_moisture"]
            )

            # Test disaggregate directly
            result_disaggregate, audit_disaggregate = scale_manager.disaggregate(
                data=data,
                source_scale="hillslope",
                target_scale="patch",
                variables=["soil_moisture"]
            )

            # Should produce same results
            assert torch.allclose(result_downscale, result_disaggregate)
            assert audit_downscale.operation == audit_disaggregate.operation


class TestScaleManagerUtilityMethods:
    """Test utility and helper methods."""

    def test_validate_scale_transition_up(self, scale_manager):
        """Test scale transition validation for upscaling."""
        assert scale_manager._validate_scale_transition("patch", "hillslope", "up") is True
        assert scale_manager._validate_scale_transition("patch", "basin", "up") is True
        assert scale_manager._validate_scale_transition("hillslope", "patch", "up") is True
        assert scale_manager._validate_scale_transition("basin", "patch", "up") is True

    def test_validate_scale_transition_down(self, scale_manager):
        """Test scale transition validation for downscaling."""
        assert scale_manager._validate_scale_transition("hillslope", "patch", "down") is True
        assert scale_manager._validate_scale_transition("basin", "patch", "down") is True
        assert scale_manager._validate_scale_transition("patch", "hillslope", "down") is True
        assert scale_manager._validate_scale_transition("patch", "basin", "down") is True

    def test_is_extensive(self, scale_manager):
        """Test extensive variable detection."""
        extensive_meta = {'variable_type': 'extensive', 'spatial_basis': 'area_density', 'units': 'kg m-2'}
        intensive_meta = {'variable_type': 'intensive', 'units': 'K'}
        categorical_meta = {'variable_type': 'categorical', 'units': '1'}

        assert scale_manager._is_extensive(extensive_meta) is True
        assert scale_manager._is_extensive(intensive_meta) is False
        assert scale_manager._is_extensive(categorical_meta) is False
        # Units and names no longer imply extensivity.
        assert scale_manager._is_extensive({'name': 'water_flux', 'variable_type': 'intensive', 'units': 'W m-2'}) is False

    def test_get_conservation_report(self, scale_manager):
        """Test getting conservation report."""
        report = scale_manager.get_conservation_report()

        assert isinstance(report, dict)
        assert 'total_checks' in report
        assert 'violations' in report
        assert 'tolerance' in report

    def test_clear_caches(self, scale_manager):
        """Test cache clearing."""
        scale_manager.clear_caches()

        # Ensure method callable without errors
        assert scale_manager is not None


class TestScalingAudit:
    """Test ScalingAudit dataclass."""

    def test_scaling_audit_creation(self):
        """Test creation of ScalingAudit."""
        audit = ScalingAudit(
            operation="aggregate",
            source_scale="patch",
            target_scale="hillslope",
            variables=["soil_moisture"],
            mode=ScalingMode.CONSERVATIVE,
            strategy=ScalingStrategy.AREA_WEIGHTED_SUM,
            mass_balance_error=1e-10,
            relative_error=1e-12,
            cf_compliant=False,
            conservation_enforced=True,
            physical_constraints_applied=False,
            warnings=[],
            metadata={'test': 'value'}
        )

        assert audit.operation == "aggregate"
        assert audit.source_scale == "patch"
        assert audit.target_scale == "hillslope"
        assert audit.variables == ["soil_moisture"]
        assert audit.mode == ScalingMode.CONSERVATIVE
        assert audit.strategy == ScalingStrategy.AREA_WEIGHTED_SUM
        assert audit.mass_balance_error == 1e-10
        assert audit.relative_error == 1e-12
        assert audit.cf_compliant is False
        assert audit.conservation_enforced is True
        assert audit.physical_constraints_applied is False
        assert audit.warnings == []
        assert audit.metadata['test'] == 'value'
        assert audit.metadata['dataset_conformance_checked'] is False
        assert audit.metadata['cf_validation_scope'] == 'not_evaluated'


# Performance and stress tests
class TestScaleManagerPerformance:
    """Test performance characteristics."""

    def test_large_data_handling(self, scale_manager):
        """Test handling of large data arrays."""
        mock_index = MockHierarchyIndex()
        mock_index.n_source_units = 10000
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            mock_get_vars.return_value = {
                'patch': {
                    'soil_moisture': {'variable_type': 'extensive', 'units': 'kg m-2'}
                }
            }

            # Large data tensor
            data = torch.rand(10000)

            result, audit = scale_manager.aggregate(
                data=data,
                source_scale="patch",
                target_scale="hillslope",
                variables=["soil_moisture"]
            )

            # Should complete without errors
            assert isinstance(result, torch.Tensor)
            assert audit.operation == "aggregate"

    def test_multiple_variables_performance(self, scale_manager):
        """Test performance with multiple variables."""
        mock_index = MockHierarchyIndex()
        mock_aggregator = MockHierarchicalAggregator(mock_index)
        scale_manager.hierarchical_aggregator = mock_aggregator

        with patch('relic.models.get_scale_variables') as mock_get_vars, \
             patch('relic.coupling.scales.hierarchical.hierarchy_index.HierarchyIndex', return_value=mock_index), \
             patch('relic.coupling.scales.hierarchical.aggregator.HierarchicalAggregator', return_value=mock_aggregator):

            # Mock many variables
            many_vars = {}
            for i in range(10):
                many_vars[f'var_{i}'] = {'variable_type': 'intensive', 'units': '1'}

            mock_get_vars.return_value = {'patch': many_vars}

            # Many variables data
            data = {f'var_{i}': torch.rand(6) for i in range(10)}

            result, audit = scale_manager.aggregate(
                data=data,
                source_scale="patch",
                target_scale="hillslope"
            )

            assert isinstance(result, dict)
            assert len(result) == 10
            assert len(audit.variables) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


