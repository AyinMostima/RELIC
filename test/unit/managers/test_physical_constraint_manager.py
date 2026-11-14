"""
Unit tests for PhysicalConstraintManager v2 migration.

Tests the PhysicalConstraintManager's integration with BaseManager architecture
while ensuring all constraint operations remain intact.
"""

import pytest
import torch
import sys
import os
import relic.coupling.scales.physical_constraint_manager as constraint_module

# Add relic to path
sys.path.insert(0, os.path.abspath('.'))

from relic.coupling.scales.physical_constraint_manager import (
    PhysicalConstraintManager,
    ConstraintType,
    ConstraintSpec,
)
from relic.core.managers.base import ManagerState


class TestPhysicalConstraintManagerV2Migration:
    """Test PhysicalConstraintManager v2 migration and BaseManager integration."""
    
    def test_inheritance_from_base_manager(self):
        """Test that PhysicalConstraintManager properly inherits from BaseManager."""
        manager = PhysicalConstraintManager()
        
        # Check inheritance
        from relic.core.managers.base import BaseManager
        assert isinstance(manager, BaseManager)
        
        # Check required methods are present
        assert hasattr(manager, '_do_initialize')
        assert hasattr(manager, '_do_configure')
        assert hasattr(manager, '_do_validate')
        assert hasattr(manager, '_do_cleanup')
        
        # Check metadata creation
        assert hasattr(manager, '_create_metadata')
        metadata = manager._create_metadata()
        assert metadata.name == "PhysicalConstraintManager"
        assert metadata.version == "2.0.0"
        
        print("[OK] PhysicalConstraintManager properly inherits from BaseManager")
    
    def test_lifecycle_management(self):
        """Test PhysicalConstraintManager lifecycle through BaseManager."""
        manager = PhysicalConstraintManager()
        
        # Initial state should be UNINITIALIZED
        assert manager.state == ManagerState.UNINITIALIZED
        
        # Initialize
        manager.initialize()
        assert manager.state == ManagerState.INITIALIZING
        
        # Configure
        config = {
            'method': 'clip',
            'preserve_gradients': True
        }
        manager.configure(config)
        assert manager.method == 'clip'
        assert manager.preserve_gradients == True
        
        # Validate to transition to READY
        manager.validate()
        assert manager.state == ManagerState.READY
        assert hasattr(manager, 'constraints')
        assert hasattr(manager, 'violation_history')
        
        # Cleanup
        manager.cleanup()
        # After cleanup, state returns to UNINITIALIZED for potential reuse
        assert manager.state == ManagerState.UNINITIALIZED
        
        print("[OK] Lifecycle management working correctly")
    
    def test_constraint_application(self):
        """Test core constraint application functionality remains intact."""
        manager = PhysicalConstraintManager()
        manager.initialize()
        manager.validate()
        
        # Add non-negative constraint
        constraint = ConstraintSpec(
            name="non_negative",
            type=ConstraintType.NON_NEGATIVE
        )
        manager.add_constraint(constraint)
        
        # Test data with negative values
        data = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        
        # Apply constraints
        constrained = manager.apply_constraints(data)
        
        # Check all values are non-negative
        assert torch.all(constrained >= 0)
        assert constrained[0, 0] == 0.0
        assert constrained[0, 1] == 2.0
        assert constrained[1, 0] == 3.0
        assert constrained[1, 1] == 0.0
        
        print("[OK] Constraint application functionality intact")
    
    def test_bounded_constraint(self):
        """Test bounded constraint application."""
        manager = PhysicalConstraintManager(method='clip')
        manager.initialize()
        manager.validate()
        
        # Add bounded constraint
        constraint = ConstraintSpec(
            name="bounded",
            type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=10.0
        )
        manager.add_constraint(constraint)
        
        # Test data with values outside bounds
        data = torch.tensor([[-5.0, 5.0], [15.0, 7.5]])
        
        # Apply constraints
        constrained = manager.apply_constraints(data)
        
        # Check all values are within bounds
        assert torch.all(constrained >= 0.0)
        assert torch.all(constrained <= 10.0)
        assert constrained[0, 0] == 0.0
        assert constrained[0, 1] == 5.0
        assert constrained[1, 0] == 10.0
        assert constrained[1, 1] == 7.5
        
        print("[OK] Bounded constraint working")
    
    def test_violation_tracking(self):
        """Test violation tracking functionality."""
        manager = PhysicalConstraintManager()
        manager.initialize()
        manager.validate()
        
        # Add constraint
        constraint = ConstraintSpec(
            name="positive",
            type=ConstraintType.NON_NEGATIVE
        )
        manager.add_constraint(constraint)
        
        # Apply to data with violations
        data = torch.tensor([[-1.0, -2.0], [3.0, -4.0]])
        constrained, mask = manager.apply_constraints(data, return_mask=True)
        
        # Check violation mask
        assert mask[0, 0] == True  # -1.0 violates
        assert mask[0, 1] == True  # -2.0 violates
        assert mask[1, 0] == False  # 3.0 ok
        assert mask[1, 1] == True  # -4.0 violates
        
        # Check violation report
        report = manager.get_violation_report()
        assert report['total_violations'] == 3
        assert report['violation_rate'] == 0.75
        
        print("[OK] Violation tracking working")
    
    def test_device_management(self):
        """Test device-aware functionality."""
        # Test with CPU (always available)
        device = torch.device("cpu")
        manager = PhysicalConstraintManager(device=device)
        manager.initialize()
        manager.validate()
        
        # Check device is set correctly
        assert manager.device == device
        
        # Create data on same device
        data = torch.tensor([[1.0, 2.0]], device=device)
        
        # Apply constraint
        constraint = ConstraintSpec(
            name="test",
            type=ConstraintType.NON_NEGATIVE
        )
        manager.add_constraint(constraint)
        result = manager.apply_constraints(data)
        
        # Check result is on correct device
        assert result.device == device
        
        print("[OK] Device management working")

    def test_parameter_bound_resolution(self):
        """Bounds are the catalog bounds of the variable when the call passes none."""
        manager = PhysicalConstraintManager()
        manager.initialize()
        manager.validate()

        data = torch.tensor([150.0, 340.0])
        constrained = manager.apply_constraints(data, variable_name="air_temperature")

        torch.testing.assert_close(constrained, torch.tensor([180.0, 330.0]))
        assert manager._last_bounds_source == "coupling.scales.defaults.variables.air_temperature.bounds"

    def test_bounds_come_only_from_the_scale_catalog(self):
        """The catalog entry (coupling.scales.<scale>.variables, then coupling.scales.defaults.variables)
        is the one source: a variable outside the catalog or an entry without bounds raises, bounds
        declared null mean no bound, and no other namespace or clamp key is consulted."""
        manager = PhysicalConstraintManager()
        manager.initialize()
        data = torch.tensor([-5.0, 5.0])
        # hydrosphere.numerical.temperature_bounds exists, but "temperature" is no catalog variable.
        with pytest.raises(KeyError, match="temperature"):
            manager.apply_constraints(data, variable_name="temperature")
        # leaf_temperature is catalogued without bounds.
        with pytest.raises(KeyError, match="bounds"):
            manager.apply_constraints(data, variable_name="leaf_temperature")
        # soil_matric_potential declares bounds: null.
        torch.testing.assert_close(manager.apply_constraints(data, variable_name="soil_matric_potential"), data)
        assert manager._last_bounds_source is None

    def test_parameter_clamp_min_resolution(self):
        """The catalog bounds of evapotranspiration_flux apply: the signed valid range
        [-1.0e-4, 6.0e-4] kg m-2 s-1 (condensation negative) of coupling.scales.defaults.variables."""
        manager = PhysicalConstraintManager()
        manager.initialize()
        manager.validate()

        data = torch.tensor([-0.5, 0.1, -5.0e-5])
        constrained = manager.apply_constraints(data, variable_name="evapotranspiration_flux")

        torch.testing.assert_close(constrained, torch.tensor([-1.0e-4, 6.0e-4, -5.0e-5]))
    
    def test_gradient_preservation(self):
        """Test gradient preservation through constraints."""
        manager = PhysicalConstraintManager(preserve_gradients=True)
        manager.initialize()
        manager.validate()
        
        # Add constraint
        constraint = ConstraintSpec(
            name="bounded",
            type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0
        )
        manager.add_constraint(constraint)
        
        # Create data that requires grad
        data = torch.tensor([[0.5, 1.5]], requires_grad=True)
        
        # Apply constraints
        constrained = manager.apply_constraints(data)
        
        # Check gradients can flow
        loss = constrained.sum()
        loss.backward()
        
        assert data.grad is not None
        
        print("[OK] Gradient preservation working")
    
    def test_validation_system(self):
        """Test validation system."""
        manager = PhysicalConstraintManager()
        manager.initialize()
        
        # Validation should pass for valid configuration
        errors = manager._do_validate()
        assert len(errors) == 0
        
        # Test with invalid method
        manager.method = "invalid_method"
        errors = manager._do_validate()
        assert len(errors) > 0
        assert any('method' in e for e in errors)
        
        # Test with invalid constraint
        manager.method = "clip"
        manager.constraints = [
            ConstraintSpec(
                name="invalid",
                type=ConstraintType.BOUNDED,
                min_value=10.0,
                max_value=5.0  # min > max
            )
        ]
        errors = manager._do_validate()
        assert len(errors) > 0
        assert any('min_value' in e and 'max_value' in e for e in errors)
        
        # Reset to valid state
        manager.constraints = []
        manager.validate()
        assert manager.state == ManagerState.READY
        
        print("[OK] Validation system working")
    
    @staticmethod
    def _allowed_methods():
        from relic.parameters import get_global_manager

        spec = get_global_manager().get_parameter("coupling.scales.defaults.physical_constraint_method")
        return list(spec.allowed_values)

    def test_an_invalid_constructor_method_is_rejected(self):
        """The constructor rejects a method outside the allowed_values of
        coupling.scales.defaults.physical_constraint_method and names them."""
        with pytest.raises(ValueError, match="bogus") as excinfo:
            PhysicalConstraintManager(method="bogus")
        for method in self._allowed_methods():
            assert method in str(excinfo.value)

    def test_an_invalid_configured_method_is_rejected_and_changes_nothing(self):
        """configure rejects an invalid method with a ValueError naming the allowed methods before
        any setting of the call is applied; the next valid configure proceeds."""
        manager = PhysicalConstraintManager()
        manager.initialize()
        manager.configure({"method": "scale", "preserve_gradients": True})
        before = (manager.state, manager.error_info, dict(manager.config), manager.method,
                  manager.preserve_gradients, list(manager.constraints))

        with pytest.raises(ValueError, match="invalid_method") as excinfo:
            manager.configure({"method": "invalid_method", "preserve_gradients": False})

        for method in self._allowed_methods():
            assert method in str(excinfo.value)
        assert (manager.state, manager.error_info, dict(manager.config), manager.method,
                manager.preserve_gradients, list(manager.constraints)) == before
        manager.configure({"method": "clip"})
        assert manager.method == "clip"
        manager.validate()
        assert manager.state == ManagerState.READY

    @pytest.mark.parametrize(
        "rejected, match",
        [
            ({"method": "clip", "presrve_gradients": True}, "presrve_gradients"),
            ({"method": "clip", "preserve_gradients": "yes"}, "preserve_gradients"),
            ({"method": "clip", "preserve_gradients": 1}, "preserve_gradients"),
            ({"method": "clip", "constraints": ["not a mapping"]}, "constraints"),
            ({"method": "clip", "constraints": [{"name": "c", "type": "bogus_type"}]}, "bogus_type"),
            ({"method": "clip", "constraints": [{"name": "c", "type": "bounded", "colour": "red"}]}, "colour"),
            ({"method": "clip", "device": "not_a_device"}, "device"),
        ],
    )
    def test_an_invalid_configure_setting_is_rejected_and_changes_nothing(self, rejected, match):
        """Unknown keys, non-bool switches, non-mapping constraint entries and unknown constraint
        types raise ValueError before the call applies anything (its valid method included)."""
        manager = PhysicalConstraintManager()
        manager.initialize()
        manager.configure({"method": "scale", "preserve_gradients": True})
        snapshot = lambda: (manager.state, manager.error_info, dict(manager.config), manager.method,
                            manager.preserve_gradients, manager.device, list(manager.constraints))
        before = snapshot()

        with pytest.raises(ValueError, match=match):
            manager.configure(rejected)

        assert snapshot() == before
        manager.configure({"preserve_gradients": False, "constraints": [{"name": "c", "type": "non_negative"}]})
        assert manager.preserve_gradients is False
        assert manager.constraints[-1].type is ConstraintType.NON_NEGATIVE
        manager.validate()
        assert manager.state == ManagerState.READY

    def test_a_constraint_of_unknown_type_is_rejected_not_skipped(self):
        """ConstraintSpec parses its type; an unknown type raises at construction and, if set later,
        at application instead of being skipped with a warning."""
        with pytest.raises(ValueError, match="bogus_type"):
            ConstraintSpec(name="c", type="bogus_type")
        spec = ConstraintSpec(name="c", type="non_negative")
        assert spec.type is ConstraintType.NON_NEGATIVE
        manager = PhysicalConstraintManager()
        manager.initialize()
        with pytest.raises(ValueError, match="ConstraintSpec"):
            manager.add_constraint({"name": "c", "type": "non_negative"})
        spec.type = "bogus_type"
        with pytest.raises(ValueError, match="bogus_type"):
            manager.add_constraint(spec)
        # A type changed after the constraint was added is not skipped at application either.
        added = ConstraintSpec(name="d", type="non_negative")
        manager.add_constraint(added)
        added.type = "bogus_type"
        with pytest.raises(ValueError, match="bogus_type"):
            manager.apply_constraints(torch.tensor([1.0]))

    def test_invalid_constructor_settings_are_rejected(self):
        """The constructor takes ConstraintSpec entries and a bool preserve_gradients; it has no
        config argument that would be ignored."""
        import inspect

        with pytest.raises(ValueError, match="preserve_gradients"):
            PhysicalConstraintManager(preserve_gradients="yes")
        with pytest.raises(ValueError, match="ConstraintSpec"):
            PhysicalConstraintManager(constraints=[{"name": "c", "type": "non_negative"}])
        assert "config" not in inspect.signature(PhysicalConstraintManager.__init__).parameters

    def test_module_exposes_no_global_lifecycle_bypass(self):
        """StateManager ownership must not be bypassed by module globals."""
        assert not hasattr(constraint_module, "_global_physical_constraint_manager")
        assert not hasattr(constraint_module, "get_global_physical_constraint_manager")
        assert not hasattr(constraint_module, "set_global_physical_constraint_manager")
    
    def test_backward_consistency(self):
        """Test backward consistency with existing code."""
        # Old-style initialization should still work
        manager = PhysicalConstraintManager(
            constraints=[],
            method='clip',
            preserve_gradients=True,
            device=torch.device('cpu')
        )
        
        # Check old methods still exist
        assert hasattr(manager, 'add_constraint')
        assert hasattr(manager, 'apply_constraints')
        assert hasattr(manager, 'get_violation_report')
        assert hasattr(manager, 'reset_tracking')
        
        # Initialize and test basic operation
        manager.initialize()
        manager.validate()
        assert manager.state == ManagerState.READY
        
        # Test constraint application still works
        constraint = ConstraintSpec(
            name="test",
            type=ConstraintType.NON_NEGATIVE
        )
        manager.add_constraint(constraint)
        
        data = torch.tensor([[-1.0, 1.0]])
        result = manager.apply_constraints(data)
        assert torch.all(result >= 0)
        
        print("[OK] Backward consistency maintained")


def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("PhysicalConstraintManager V2 Migration Tests")
    print("="*60 + "\n")
    
    test_suite = TestPhysicalConstraintManagerV2Migration()
    
    tests = [
        test_suite.test_inheritance_from_base_manager,
        test_suite.test_lifecycle_management,
        test_suite.test_constraint_application,
        test_suite.test_bounded_constraint,
        test_suite.test_violation_tracking,
        test_suite.test_device_management,
        test_suite.test_parameter_bound_resolution,
        test_suite.test_parameter_clamp_min_resolution,
        test_suite.test_gradient_preservation,
        test_suite.test_validation_system,
        test_suite.test_module_exposes_no_global_lifecycle_bypass,
        test_suite.test_import_does_not_register_competing_manager_owner,
        test_suite.test_backward_consistency
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        test_name = test.__name__
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_name}: {e}")
            failed += 1
        except Exception as e:
            if "[SKIP]" in str(e):
                skipped += 1
            else:
                print(f"[ERROR] {test_name}: {e}")
                failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)


