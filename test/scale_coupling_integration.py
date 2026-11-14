"""
Test Multi-Scale Coupling Integration

This test validates that the ScaleManager integration in RELICSimulator
works correctly for automatic scale detection and regridding operations.

References:
    - RELIC Multi-scale Coupling Framework
    - Conservative regridding algorithms
    - PyTorch tensor operations
"""

import pytest
import torch
import tempfile
import os
import shutil
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from relic.core.simulator import RELICSimulator
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Skipping multi-scale coupling tests due to missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'basin_runoff': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class TestMultiScaleCouplingIntegration:
    """Test multi-scale coupling integration in RELICSimulator."""

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Required dependencies not available")
    def test_simulator_auto_scale_initialization(self):
        """Test simulator initialization with auto-scale enabled."""
        config = {
            'time_step': 60.0,
            'max_time': 3600.0,
            'enable_auto_scale': True
        }

        simulator = _declared_simulator(config)

        # Verify auto-scale is enabled
        assert simulator.enable_auto_scale is True
        assert getattr(simulator, "scale_manager", None) is not None
        assert len(simulator._scale_check_history) == 0

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Required dependencies not available")
    def test_simulator_auto_scale_disabled(self):
        """Test simulator initialization with auto-scale disabled."""
        config = {
            'time_step': 60.0,
            'max_time': 3600.0,
            'enable_auto_scale': False
        }

        simulator = _declared_simulator(config)

        # Verify auto-scale is disabled
        assert simulator.enable_auto_scale is False
        assert getattr(simulator, "scale_manager", None) is None

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Required dependencies not available")
    def test_scale_consistency_check(self):
        """Test scale consistency checking functionality."""
        config = {
            'time_step': 60.0,
            'max_time': 3600.0,
            'enable_auto_scale': True
        }

        simulator = _declared_simulator(config)

        # Test with single scale (no regridding needed)
        single_scale_state = {
            'soil_moisture': torch.tensor([0.3, 0.4, 0.2]),  # 3 patches
            'temperature': torch.tensor([293.0, 294.0, 295.0]),  # 3 patches
        }

        scale_check = simulator._check_scale_consistency(single_scale_state)
        assert scale_check['needs_regrid'] is False
        assert scale_check['reason'] is None

        # Test with multiple scales (regridding needed)
        mixed_scale_state = {
            'soil_moisture': torch.tensor([0.3, 0.4, 0.2]),  # 3 patches
            'basin_flux': torch.tensor([1.5, 2.0]),  # 2 hillslopes
        }

        scale_check = simulator._check_scale_consistency(mixed_scale_state)
        assert scale_check['needs_regrid'] is True
        assert scale_check['reason'] == 'multiple scales detected'
        assert scale_check['target_scale'] in ['patch', 'hillslope']

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Required dependencies not available")
    def test_regridding_operations(self):
        """Test regridding operations between scales."""
        config = {
            'time_step': 60.0,
            'max_time': 3600.0,
            'enable_auto_scale': True,
            # The CPU state tensors below are regridded on explicit grids built on
            # the simulator device, so the simulator runs on the CPU as well.
            'device': 'cpu',
            'auto_scale': {
                'source_grid': {
                    'grid_type': 'regular',
                    'coord_system': 'cartesian',
                    'shape': (6, 1),
                    'spacing': (1.0, 1.0),
                    'origin': (0.0, 0.0),
                },
                'target_grid': {
                    'grid_type': 'regular',
                    'coord_system': 'cartesian',
                    'shape': (2, 1),
                    'spacing': (3.0, 1.0),
                    'origin': (0.0, 0.0),
                    # CF grid metadata required by the attached CF synchronizer.
                    'grid_mapping': 'crs',
                    'lat_name': 'lat',
                    'lon_name': 'lon',
                },
            },
        }

        simulator = _declared_simulator(config)

        # Test patch to hillslope upscaling
        patch_state = {
            'soil_moisture': torch.tensor([[0.3, 0.4, 0.2, 0.5, 0.1, 0.3]]),
            'carbon': torch.tensor([[100.0, 120.0, 80.0, 150.0, 60.0, 90.0]])
        }
        # The ScaleManager requires the simulator's StateManager as its explicit owner, so the
        # simulator state is initialized before the auto-regrid path is used.
        simulator.initialize_state(dict(patch_state))

        scale_check = {
            'needs_regrid': True,
            'target_scale': 'hillslope',
            'detected_scales': {'soil_moisture': 'patch', 'carbon': 'patch'},
        }

        regridded_state = simulator._perform_regridding(patch_state, scale_check)

        # Verify regridding occurred
        assert 'soil_moisture' in regridded_state
        assert 'carbon' in regridded_state

        # Check shape changes (6 patches -> 2 hillslopes)
        expected_shape = (1, 2)
        assert regridded_state['soil_moisture'].shape == expected_shape
        assert regridded_state['carbon'].shape == expected_shape

        # Verify the area-integrated carbon stock on the explicit grids.
        original_carbon_sum = (patch_state['carbon'] * 1.0).sum()
        regridded_carbon_sum = (regridded_state['carbon'] * 3.0).sum()
        relative_error = abs(regridded_carbon_sum - original_carbon_sum) / original_carbon_sum
        assert relative_error < 1e-6, f"Conservation violation: {relative_error}"

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Required dependencies not available")
    def test_scale_integration_in_step_execution(self):
        """Test that scale checking is integrated into step execution."""
        config = {
            'time_step': 60.0,
            'max_time': 3600.0,
            'enable_auto_scale': True,
            'enable_bootstrap_modules': True,
            'conservation_checks': False,
        }

        simulator = _declared_simulator(config)

        # Initialize simulator with mixed scale state (without farquhar to avoid complexity)
        initial_state = {
            'soil_moisture': torch.tensor([0.3, 0.4, 0.2, 0.5, 0.1, 0.3]),  # 6 patches
            'basin_runoff': torch.tensor([1.5, 2.0]),  # 2 hillslopes
            'temperature': torch.tensor(293.0),  # scalar
            # Add required state variables for bucket_soil_moisture
            'precipitation_flux': torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),  # 6 patches
            'evapotranspiration_flux': torch.tensor([0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005])  # 6 patches
        }

        simulator.initialize_state(initial_state)

        # Execute a step
        updates = simulator.step(dt=60.0)

        # Verify step completed successfully
        assert isinstance(updates, dict)
        assert simulator.step_count == 1

        # Verify scale check history was populated when auto-scale is active and ready
        has_scale_checks = (
            simulator.enable_auto_scale
            and simulator.scale_manager is not None
            and bool(simulator.execution_order)
        )
        if has_scale_checks:
            assert len(simulator._scale_check_history) > 0
        else:
            assert len(simulator._scale_check_history) == 0

        # Verify scale statistics are available
        stats = simulator.get_coupling_scales_statistics()
        assert stats['auto_scale_enabled'] is True
        if has_scale_checks:
            assert stats['total_scale_checks'] > 0
        else:
            assert stats['total_scale_checks'] == 0

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Required dependencies not available")
    def test_simulation_summary_includes_scale_info(self):
        """Test that simulation summary includes scale coupling information."""
        config = {
            'time_step': 60.0,
            'max_time': 3600.0,
            'enable_auto_scale': True,
            'enable_bootstrap_modules': True,
            'conservation_checks': False,
        }

        simulator = _declared_simulator(config)

        # Initialize and run a few steps
        initial_state = {
            'soil_moisture': torch.tensor([0.3, 0.4, 0.2]),
            'temperature': torch.tensor(293.0),
            # Add required state variables for bucket_soil_moisture
            'precipitation_flux': torch.tensor([0.001, 0.001, 0.001]),
            'evapotranspiration_flux': torch.tensor([0.0005, 0.0005, 0.0005])
        }

        simulator.initialize_state(initial_state)

        # Execute a few steps to generate scale check history
        for _ in range(3):
            simulator.step(dt=60.0)

        # Get simulation summary
        summary = simulator.get_simulation_summary()

        # Verify scale coupling information is included
        assert 'coupling.scales' in summary
        scale_info = summary['coupling.scales']
        assert scale_info['auto_scale_enabled'] is True
        has_scale_checks = (
            simulator.enable_auto_scale
            and simulator.scale_manager is not None
            and bool(simulator.execution_order)
        )
        if has_scale_checks:
            assert scale_info['total_scale_checks'] > 0
        else:
            assert scale_info['total_scale_checks'] == 0

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Required dependencies not available")
    def test_checkpoint_restore_with_scale_state(self):
        """Test checkpoint creation and restoration with scale coupling state."""
        config = {
            'time_step': 60.0,
            'max_time': 3600.0,
            'enable_auto_scale': True,
            'enable_bootstrap_modules': True,
            'conservation_checks': False,
        }

        simulator = _declared_simulator(config)

        # Initialize and run some steps
        initial_state = {
            'soil_moisture': torch.tensor([0.3, 0.4, 0.2, 0.5, 0.1, 0.3]),
            'temperature': torch.tensor(293.0),
            # Add required state variables for bucket_soil_moisture
            'precipitation_flux': torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
            'evapotranspiration_flux': torch.tensor([0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005])
        }

        simulator.initialize_state(initial_state)

        # Execute steps to generate scale check history
        for _ in range(5):
            simulator.step(dt=60.0)

        # Record state before checkpoint
        original_time = simulator.current_time
        original_step_count = simulator.step_count
        original_scale_history_len = len(simulator._scale_check_history)

        # Create checkpoint at an explicit, not yet existing path (exclusive-create API)
        checkpoint_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(checkpoint_dir, 'scale_state_checkpoint.pt')

        try:
            simulator.create_checkpoint(checkpoint_path)
            assert os.path.exists(checkpoint_path)

            # Reset simulator
            simulator.reset()

            # Verify reset
            assert simulator.current_time == 0.0
            assert simulator.step_count == 0
            assert len(simulator._scale_check_history) == 0

            # Restore from checkpoint
            simulator.restore_checkpoint(checkpoint_path)

            # Verify restoration
            assert simulator.current_time == original_time
            assert simulator.step_count == original_step_count
            assert simulator.enable_auto_scale is True

        finally:
            # Clean up
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def run_integration_tests():
    """Run the multi-scale coupling integration tests."""
    print("Running Multi-Scale Coupling Integration Tests...")
    print("=" * 60)

    if not DEPENDENCIES_AVAILABLE:
        print("WARNING: Required dependencies not available. Install RELIC dependencies to run tests.")
        return False

    test_instance = TestMultiScaleCouplingIntegration()

    test_methods = [
        test_instance.test_simulator_auto_scale_initialization,
        test_instance.test_simulator_auto_scale_disabled,
        test_instance.test_scale_consistency_check,
        test_instance.test_regridding_operations,
        test_instance.test_scale_integration_in_step_execution,
        test_instance.test_simulation_summary_includes_scale_info,
        test_instance.test_checkpoint_restore_with_scale_state
    ]

    passed = 0
    failed = 0

    for test_method in test_methods:
        try:
            print(f"Running {test_method.__name__}... ", end="")
            test_method()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All multi-scale coupling integration tests PASSED!")
        return True
    else:
        print("Some tests FAILED. Check the implementation.")
        return False


if __name__ == "__main__":
    run_integration_tests()



