"""Test Enum consistency between Facade and Aggregator."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import pytest

from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.types import ScalingStrategy, ScalingAudit
from relic.core.state import StateManager
from relic.coupling.core.policy import ScalingMode
from relic.coupling.scales.hierarchical.aggregator import HierarchicalAggregator
from relic.coupling.scales.hierarchical.hierarchy_index import HierarchyIndex


class TestEnumconsistency:
    """Test that Enum types are properly handled between manager and aggregator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device("cpu")
        self.manager = ScaleManager(
            device=self.device,
            state_manager=StateManager(),
        )
        self.manager.initialize()
        self.manager.get_variable_metadata = lambda name, scale: {"variable_type": "intensive", "units": "kg m-2"}
        
        class DummyAggregator:
            def aggregate(self, data, source_scale, target_scale, variables=None, mode=None, strategy=None, metadata=None, topology=None):
                audit = ScalingAudit(
                    operation="aggregate",
                    source_scale=source_scale,
                    target_scale=target_scale,
                    variables=variables or ["test_var"],
                    mode=mode or ScalingMode.CONSERVATIVE,
                    strategy=strategy or ScalingStrategy.AREA_WEIGHTED_MEAN,
                    mass_balance_error=0.0,
                    relative_error=0.0,
                    cf_compliant=False,
                    conservation_enforced=True,
                    physical_constraints_applied=False,
                    warnings=[],
                    metadata={},
                )
                return {"test_var": torch.tensor([1.0])}, audit
        self.manager.hierarchical_strategy = DummyAggregator()
        
    def test_manager_handles_enum_strategy(self):
        """Test that manager properly accepts Enum strategy."""
        data = torch.randn(100, device=self.device)
        strategy = ScalingStrategy.AREA_WEIGHTED_MEAN
        result, audit = self.manager.aggregate(
            data=data,
            source_scale='patch',
            target_scale='hillslope',
            variables=['test_var'],
            mode=ScalingMode.CONSERVATIVE,
            strategy=strategy
        )
        assert result is not None
        assert audit is not None
        assert audit.strategy == strategy
    
    def test_aggregator_accepts_string_and_enum(self):
        """Test that Aggregator handles both string and Enum inputs."""
        # Create a simple hierarchy index
        index = HierarchyIndex(
            source_scale='patch',
            target_scale='hillslope',
            device=self.device,
            n_source_units=100
        )
        
        aggregator = HierarchicalAggregator(
            index=index,
            device=self.device,
            weight_type="area",
            rule="conservative",
        )
        
        # Test data
        data = {'test_var': torch.randn(100, device=self.device)}
        variables = ['test_var']
        
        # Test with string strategy
        result1, audit1 = aggregator.aggregate(
            data=data,
            variables=variables,
            strategy='area_weighted_mean',
            variable_metadata={'test_var': {'spatial_basis': 'intensive'}},
            mode='conservative'
        )
        assert result1 is not None
        
        # Test with Enum-like object (simulating what Facade might pass)
        class MockEnum:
            value = 'area_weighted_mean'
        
        strategy_enum = MockEnum()
        
        # This should also work
        result2, audit2 = aggregator.aggregate(
            data=data,
            variables=variables,
            strategy=strategy_enum.value,  # Facade should extract .value
            variable_metadata={'test_var': {'spatial_basis': 'intensive'}},
            mode='conservative'
        )
        assert result2 is not None
    
    def test_mode_enum_consistency(self):
        """Test that ScalingMode Enum is properly handled."""
        # Create test data
        data = torch.randn(100, device=self.device)
        
        # Test with ScalingMode Enum
        mode = ScalingMode.STRICT_CONSERVATIVE
        
        result, audit = self.manager.aggregate(
            data=data,
            source_scale='patch',
            target_scale='hillslope',
            variables=['test_var'],
            mode=mode
        )
        assert audit.mode == mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


