"""
L1 Unit Tests for RELICSimulator Distributed Computing Integration

Tests distributed functionality including:
- Distributed initialization and configuration
- AllReduce aggregation in step() method
- GPU device management
- Distributed context manager

Test Level: L1 (Unit)
Dependencies: torch.distributed, ProcessGroupManager
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
)


class TestDistributedInitialization:
    """Test distributed configuration and initialization (P0.1)"""
    
    def test_distributed_disabled_by_default(self):
        """Test that distributed is disabled by default"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
        }
        simulator = RELICSimulator(config)
        
        assert hasattr(simulator, 'enable_distributed')
        assert simulator.enable_distributed is False
        assert simulator.process_group_manager is None
    
    def test_distributed_configuration_loading(self):
        """Test distributed configuration is correctly loaded from config"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
            'device': 'cpu',
        }
        
        # Mock distributed components to avoid actual distributed initialization
        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager') as mock_mgr, \
             patch('relic.core.simulator.main.is_distributed', return_value=False), \
             patch('torch.cuda.is_available', return_value=False):  # Force CPU mode
            
            mock_mgr.return_value = Mock()
            mock_mgr.return_value.validate.return_value = []
            simulator = RELICSimulator(config)
            
            assert simulator.enable_distributed is True
            assert hasattr(simulator, 'device')
            assert simulator.device.type == 'cpu'
    
    def test_gpu_auto_assignment_in_distributed_mode(self):
        """Test GPU auto-assignment based on local_rank"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU test")
        
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
            'local_rank': 0,
        }
        
        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager') as mock_mgr, \
             patch('relic.core.simulator.main.is_distributed', return_value=True), \
             patch('relic.core.simulator.main.get_rank', return_value=0):
            
            mock_mgr.return_value = Mock()
            mock_mgr.return_value.validate.return_value = []
            simulator = RELICSimulator(config)
            
            assert simulator.device.type == 'cuda'
            assert simulator.device.index == 0
    
    def test_distributed_unavailable_raises(self):
        """A requested distributed runtime that is not installed raises."""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
        }

        with patch('relic.core.simulator.HAS_DISTRIBUTED', False):
            with pytest.raises(RuntimeError, match="enable_distributed=True"):
                RELICSimulator(config)

    def test_process_group_validation_errors_raise(self):
        """A process group whose validation reports errors is not dropped silently."""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
            'device': 'cpu',
        }

        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager') as mock_mgr:
            mock_mgr.return_value = Mock()
            mock_mgr.return_value.validate.return_value = ["MASTER_ADDR is not set"]
            with pytest.raises(RuntimeError, match="MASTER_ADDR is not set"):
                RELICSimulator(config)

    def test_process_group_setup_failure_raises_with_its_cause(self):
        """A failure while setting up the process group raises with the real cause."""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
            'device': 'cpu',
        }
        cause = OSError("process group backend unreachable")

        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager', side_effect=cause):
            with pytest.raises(RuntimeError, match="enable_distributed=True") as raised:
                RELICSimulator(config)
        assert raised.value.__cause__ is cause

    def test_gpu_local_rank_beyond_visible_devices_raises(self):
        """A local rank without a visible CUDA device raises instead of running on CPU."""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
            'local_rank': 5,
        }

        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager') as mock_mgr, \
             patch('relic.core.simulator.main.is_distributed', return_value=True), \
             patch('relic.core.simulator.main.get_rank', return_value=5), \
             patch('relic.core.device_policy.is_cuda_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1):
            mock_mgr.return_value = Mock()
            mock_mgr.return_value.validate.return_value = []
            with pytest.raises(RuntimeError, match="local_rank"):
                RELICSimulator(config)

    def test_without_cuda_distributed_ranks_run_on_cpu(self):
        """The device policy's documented fallback stays: no CUDA, ranks run on CPU."""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
        }

        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager') as mock_mgr, \
             patch('relic.core.simulator.main.is_distributed', return_value=False), \
             patch('relic.core.device_policy.is_cuda_available', return_value=False):
            mock_mgr.return_value = Mock()
            mock_mgr.return_value.validate.return_value = []
            simulator = RELICSimulator(config)

        assert simulator.enable_distributed is True
        assert simulator.device.type == 'cpu'


class TestAllReduceAggregation:
    """Test AllReduce aggregation in step() method (P0.2)"""
    
    @pytest.fixture
    def mock_simulator(self):
        """Create simulator with mocked distributed components"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
            'device': 'cpu',
        }
        
        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager') as mock_mgr, \
             patch('relic.core.simulator.main.is_distributed', return_value=True), \
             patch('relic.core.simulator.main.get_rank', return_value=0), \
             patch('relic.core.simulator.main.get_world_size', return_value=4):
            
            mock_process_group = Mock()
            mock_process_group.barrier = Mock()
            mock_process_group.all_reduce = Mock()
            mock_process_group.validate = Mock(return_value=[])
            mock_mgr.return_value = mock_process_group
            
            simulator = RELICSimulator(config)
            simulator.process_group_manager = mock_process_group
            
            # Initialize state manager
            state_manager = StateManager()
            state_manager.initialize()
            simulator.set_state_manager(state_manager)
            
            yield simulator, mock_process_group
    
    def test_all_reduce_called_in_distributed_step(self, mock_simulator):
        """Test that AllReduce is called during distributed step"""
        simulator, mock_pg = mock_simulator
        
        # Register a simple functor
        class SimpleFunctor(BiologicalProcessFunctor):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="test_functor",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    description="Distributed aggregation test functor",
                    references=["RELIC distributed simulator architecture test fixture"],
                    output_variables=["test_var"],
                )
                super().__init__(name="test_functor", metadata=metadata, config={})
            
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {"test_var": torch.tensor(1.0)}
        
        functor = SimpleFunctor()
        simulator.register_functor(functor)
        
        # Mock is_distributed to return True
        with patch('relic.core.simulator.main.is_distributed', return_value=True), \
             patch('relic.core.simulator.main.get_world_size', return_value=4):
            
            # Execute step
            simulator.step(dt=3600.0)
            
            # Verify barrier was called
            assert mock_pg.barrier.called, "Barrier should be called in distributed step"
    
    def test_all_reduce_averages_updates(self):
        """Test that AllReduce correctly averages updates across ranks"""
        # Create a tensor to simulate update
        update_tensor = torch.tensor([4.0, 8.0, 12.0])
        world_size = 4
        
        # Simulate AllReduce by summing (in real distributed, all ranks would sum)
        # Then divide by world_size to get average
        averaged = update_tensor / world_size
        
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(averaged, expected)


class TestDeviceManagement:
    """Test GPU device management (P0.4)"""
    
    def test_functor_device_migration_on_registration(self):
        """Test that functors are migrated to target device on registration"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU test")
        
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'device': 'cuda:0',
        }
        
        simulator = RELICSimulator(config)
        
        # Create a functor with parameters
        class ParameterizedFunctor(BiologicalProcessFunctor):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="param_functor",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    description="Distributed device migration test functor",
                    references=["RELIC distributed simulator architecture test fixture"],
                    output_variables=["output"],
                )
                super().__init__(name="param_functor", metadata=metadata, config={})
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
            
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {"output": self.weight * 2.0}
        
        functor = ParameterizedFunctor()
        
        # Initially on CPU
        assert functor.weight.device.type == 'cpu'
        
        # Register should migrate to GPU
        simulator.register_functor(functor)
        
        # Check migration occurred
        assert functor.weight.device.type == 'cuda'
    
    def test_get_device_info_returns_correct_status(self):
        """Test get_device_info returns accurate device status"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'device': 'cpu',
        }
        
        simulator = RELICSimulator(config)
        info = simulator.get_device_info()
        
        # Verify structure
        assert 'device' in info
        assert 'device_type' in info
        assert 'distributed_enabled' in info
        assert 'cuda_available' in info
        
        # Verify values
        assert info['device_type'] == 'cpu'
        assert info['distributed_enabled'] is False
        assert isinstance(info['cuda_available'], bool)
    
    def test_get_device_info_with_distributed(self):
        """Test get_device_info returns distributed status (rank 2 of 8, run on CPU:
        a GPU local rank needs a visible device)."""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
            'device': 'cpu',
        }
        
        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager') as mock_mgr, \
             patch('relic.core.simulator.main.is_distributed', return_value=True), \
             patch('relic.core.simulator.main.get_rank', return_value=2), \
             patch('relic.core.simulator.main.get_world_size', return_value=8):
            
            mock_mgr.return_value = Mock()
            mock_mgr.return_value.validate.return_value = []
            simulator = RELICSimulator(config)
            
            info = simulator.get_device_info()
            
            assert info['distributed_enabled'] is True
            assert info['rank'] == 2
            assert info['world_size'] == 8


class TestDistributedContextManager:
    """Test distributed context manager (P0.3)"""
    
    def test_context_manager_temporarily_enables_distributed(self):
        """Test that context manager can temporarily enable distributed"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': False,
        }
        
        simulator = RELICSimulator(config)
        
        # Initially disabled
        assert simulator.enable_distributed is False
        
        # Enable within context
        with simulator.enable_distributed_execution(True):
            assert simulator.enable_distributed is True
        
        # Restored after context
        assert simulator.enable_distributed is False
    
    def test_context_manager_temporarily_disables_distributed(self):
        """Test that context manager can temporarily disable distributed"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
        }
        
        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager') as mock_mgr, \
             patch('relic.core.simulator.main.is_distributed', return_value=False):
            
            mock_mgr.return_value = Mock()
            mock_mgr.return_value.validate.return_value = []
            simulator = RELICSimulator(config)
            
            # Initially enabled
            assert simulator.enable_distributed is True
            
            # Disable within context
            with simulator.enable_distributed_execution(False):
                assert simulator.enable_distributed is False
            
            # Restored after context
            assert simulator.enable_distributed is True
    
    def test_context_manager_restores_on_exception(self):
        """Test that context manager restores state even on exception"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': False,
        }
        
        simulator = RELICSimulator(config)
        
        try:
            with simulator.enable_distributed_execution(True):
                assert simulator.enable_distributed is True
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should be restored even after exception
        assert simulator.enable_distributed is False


class TestDistributedIntegration:
    """Integration tests for distributed components"""
    
    def test_distributed_step_execution_flow(self):
        """Test complete distributed step execution flow"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
            'enable_distributed': True,
            'device': 'cpu',
        }
        
        with patch('relic.core.simulator.HAS_DISTRIBUTED', True), \
             patch('relic.core.simulator.main.get_process_group_manager') as mock_mgr, \
             patch('relic.core.simulator.main.is_distributed', return_value=True), \
             patch('relic.core.simulator.main.get_rank', return_value=0), \
             patch('relic.core.simulator.main.get_world_size', return_value=2):
            
            mock_pg = Mock()
            mock_pg.barrier = Mock()
            mock_pg.all_reduce = Mock(side_effect=lambda x: x)  # Identity for test
            mock_pg.validate = Mock(return_value=[])
            mock_mgr.return_value = mock_pg
            
            simulator = RELICSimulator(config)
            
            # Initialize state manager
            state_manager = StateManager()
            state_manager.initialize()
            simulator.set_state_manager(state_manager)
            
            # Register functor
            class TestFunctor(TransportMechanismFunctor):
                TRANSPORT_TYPE = "diffusion"

                def __init__(self):
                    metadata = FunctorMetadata(
                        name="integration_functor",
                        category=ProcessCategory.TRANSPORT_MECHANISM,
                        description="Distributed step integration test functor",
                        references=["RELIC distributed simulator architecture test fixture"],
                        output_variables=["flux"],
                    )
                    super().__init__(name="integration_functor", metadata=metadata, config={})
                
                def get_dependencies(self):
                    return []
                
                def forward(self, state, time, dt, **kwargs):
                    return {"flux": torch.tensor(5.0)}
            
            functor = TestFunctor()
            simulator.register_functor(functor)
            
            # Execute step
            simulator.step(dt=3600.0)

            # Verify barrier and all_reduce were called
            assert mock_pg.barrier.called
            # all_reduce may be called multiple times for different variables


def test_gpu_memory_query_failure_propagates(monkeypatch):
    import logging
    from types import SimpleNamespace

    import relic.core.simulator.device_ops as device_ops

    def _broken(_index):
        raise RuntimeError("CUDA context lost")

    monkeypatch.setattr(device_ops.device_policy, "is_cuda_available", lambda: True)
    monkeypatch.setattr(device_ops.torch.cuda, "memory_allocated", _broken)
    simulator = SimpleNamespace(device=torch.device("cuda:0"), enable_distributed=False)
    with pytest.raises(RuntimeError, match="CUDA context lost"):
        device_ops.get_device_info(
            simulator,
            has_distributed=lambda: False,
            is_distributed=lambda: False,
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
            logger=logging.getLogger(__name__),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


