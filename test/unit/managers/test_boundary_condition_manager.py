"""
Unit tests for BoundaryConditionManager v2 migration.

Tests the BoundaryConditionManager's integration with BaseManager architecture
while ensuring all boundary condition operations remain intact.
"""

import pytest
import torch
import sys
import os

# Add relic to path
sys.path.insert(0, os.path.abspath('.'))

from relic.scientific.transport import boundary_conditions as boundary_conditions_module
from relic.scientific.transport.boundary_conditions import (
    BoundaryConditionManager,
    BoundaryType,
    create_dirichlet_walls,
    create_periodic_boundaries,
    create_open_boundaries
)
from relic.core.managers.base import DeviceAwareManagerMixin, ManagerState


class TestBoundaryConditionManagerV2Migration:
    """Test BoundaryConditionManager v2 migration and BaseManager integration."""
    
    def test_inheritance_from_base_manager(self):
        """Test that BoundaryConditionManager properly inherits from BaseManager."""
        manager = BoundaryConditionManager()
        
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
        assert metadata.name == "BoundaryConditionManager"
        assert metadata.version == "2.0.0"
        
        print("[OK] BoundaryConditionManager properly inherits from BaseManager")
    
    def test_lifecycle_management(self):
        """Test BoundaryConditionManager lifecycle through BaseManager."""
        manager = BoundaryConditionManager(spatial_dims=2)
        
        # Initial state should be UNINITIALIZED
        assert manager.state == ManagerState.UNINITIALIZED
        
        # Initialize
        manager.initialize()
        assert manager.state == ManagerState.INITIALIZING
        
        # Configure
        config = {
            'spatial_dims': 3,
            'device': 'cpu'
        }
        manager.configure(config)
        assert manager.spatial_dims == 3
        
        # Validate to transition to READY
        manager.validate()
        assert manager.state == ManagerState.READY
        assert hasattr(manager, 'boundary_conditions')
        
        # Cleanup
        manager.cleanup()
        # After cleanup, state returns to UNINITIALIZED for potential reuse
        assert manager.state == ManagerState.UNINITIALIZED
        
        print("[OK] Lifecycle management working correctly")

    def test_constructor_config_is_not_a_second_lifecycle_path(self):
        """Runtime configuration must use BaseManager.configure explicitly."""
        with pytest.raises(TypeError):
            BoundaryConditionManager(config={"spatial_dims": 3})
    
    def test_boundary_condition_application(self):
        """Test core boundary condition application functionality remains intact."""
        manager = BoundaryConditionManager(spatial_dims=2)
        manager.initialize()
        manager.validate()
        
        # Set Dirichlet boundary conditions
        manager.set_boundary_condition("x_min", BoundaryType.DIRICHLET, 0.0)
        manager.set_boundary_condition("x_max", BoundaryType.DIRICHLET, 1.0)
        
        # Create a test field
        field = torch.ones((1, 10, 10))  # batch, ny, nx
        
        # Apply boundary conditions
        field_bc = manager.apply_boundary_conditions(field, grid_spacing=0.1)
        
        # Check boundary values
        assert torch.all(field_bc[:, :, 0] == 0.0)  # x_min boundary
        assert torch.all(field_bc[:, :, -1] == 1.0)  # x_max boundary
        
        print("[OK] Boundary condition application functionality intact")
    
    def test_neumann_boundary(self):
        """Test Neumann boundary condition application."""
        manager = BoundaryConditionManager(spatial_dims=1)
        manager.initialize()
        manager.validate()
        
        # Set Neumann boundary condition with flux
        flux_value = 2.0
        manager.set_boundary_condition("x_min", BoundaryType.NEUMANN, flux_value)
        
        # Create a test field
        field = torch.ones((1, 10))  # batch, nx
        dx = 0.1
        
        # Apply boundary conditions
        field_bc = manager.apply_boundary_conditions(field, grid_spacing=dx)
        
        # Check Neumann condition: 蠁[0] = 蠁[1] - flux路dx
        expected = field[0, 1] - flux_value * dx
        assert torch.allclose(field_bc[0, 0], expected)
        
        print("[OK] Neumann boundary working")
    
    def test_periodic_boundary(self):
        """Test periodic boundary condition application."""
        manager = BoundaryConditionManager(spatial_dims=1)
        manager.initialize()
        manager.validate()
        
        # Set periodic boundary conditions
        manager.set_boundary_condition("x_min", BoundaryType.PERIODIC)
        manager.set_boundary_condition("x_max", BoundaryType.PERIODIC)
        
        # Create a test field with different values at boundaries
        field = torch.arange(10, dtype=torch.float32).unsqueeze(0)  # batch, nx
        
        # Apply boundary conditions
        field_bc = manager.apply_boundary_conditions(field, grid_spacing=0.1)
        
        # Check periodic condition: 蠁[0] = 蠁[-1]
        assert field_bc[0, 0] == field_bc[0, -1]
        
        print("[OK] Periodic boundary working")
    
    def test_robin_boundary(self):
        """Test Robin boundary condition application."""
        manager = BoundaryConditionManager(spatial_dims=1)
        manager.initialize()
        manager.validate()
        
        # Set Robin boundary condition: 伪路蠁 + 尾路鈭傁?鈭俷 = 纬
        alpha, beta, gamma = 1.0, 1.0, 2.0
        manager.set_boundary_condition("x_min", BoundaryType.ROBIN, (alpha, beta, gamma))
        
        # Create a test field
        field = torch.ones((1, 10))  # batch, nx
        dx = 0.1
        
        # Apply boundary conditions
        field_bc = manager.apply_boundary_conditions(field, grid_spacing=dx)
        
        # Check Robin condition: (伪 + 尾/dx)路蠁[0] = 纬 + 尾路蠁[1]/dx
        expected = (gamma + beta * field[0, 1] / dx) / (alpha + beta / dx)
        assert torch.allclose(field_bc[0, 0], expected)
        
        print("[OK] Robin boundary working")
    
    def test_2d_boundary_conditions(self):
        """Test boundary conditions in 2D."""
        manager = BoundaryConditionManager(spatial_dims=2)
        manager.initialize()
        manager.validate()
        
        # Set different boundary conditions on different faces
        manager.set_boundary_condition("x_min", BoundaryType.DIRICHLET, 0.0)
        manager.set_boundary_condition("x_max", BoundaryType.DIRICHLET, 1.0)
        manager.set_boundary_condition("y_min", BoundaryType.NEUMANN, 0.0)
        manager.set_boundary_condition("y_max", BoundaryType.OPEN)
        
        # Create a test field
        field = torch.ones((1, 10, 10))  # batch, ny, nx
        
        # Apply boundary conditions
        field_bc = manager.apply_boundary_conditions(field, grid_spacing=0.1)
        
        # Check x boundaries (Dirichlet)
        assert torch.all(field_bc[:, :, 0] == 0.0)
        assert torch.all(field_bc[:, :, -1] == 1.0)
        
        # Check y boundaries (Neumann and Open)
        # Neumann with zero flux: 蠁[0] = 蠁[1]
        assert torch.allclose(field_bc[:, 0, :], field_bc[:, 1, :])
        # Open boundary: 蠁[-1] = 蠁[-2]
        assert torch.allclose(field_bc[:, -1, :], field_bc[:, -2, :])
        
        print("[OK] 2D boundary conditions working")
    
    def test_device_management(self):
        """Test device-aware functionality."""
        # Test with CPU (always available)
        device = torch.device("cpu")
        manager = BoundaryConditionManager(spatial_dims=1, device=device)
        manager.initialize()
        manager.validate()
        
        # Check device is set correctly
        assert manager.device == device
        
        # Create data on same device
        field = torch.ones((1, 10), device=device)
        
        # Apply boundary conditions
        manager.set_boundary_condition("x_min", BoundaryType.DIRICHLET, 0.0)
        field_bc = manager.apply_boundary_conditions(field, grid_spacing=0.1)
        
        # Check result is on correct device
        assert field_bc.device == device
        
        print("[OK] Device management working")

    def test_device_migration_reuses_mixin_and_moves_owned_tensors(self):
        """The manager device contract includes tensor-valued BC data."""
        manager = BoundaryConditionManager(spatial_dims=1)
        assert isinstance(manager, DeviceAwareManagerMixin)
        manager.initialize()
        manager.set_boundary_condition(
            "x_min",
            BoundaryType.DIRICHLET,
            torch.tensor(1.0),
        )
        manager.set_boundary_condition(
            "x_max",
            BoundaryType.ROBIN,
            (1.0, 1.0, torch.tensor(2.0)),
        )

        manager.to_device("meta")

        assert manager.device == torch.device("meta")
        assert manager.boundary_conditions["x_min"].value.device.type == "meta"
        assert manager.boundary_conditions["x_max"].gamma.device.type == "meta"

        manager.set_boundary_condition(
            "x_min",
            BoundaryType.DIRICHLET,
            torch.tensor(3.0),
        )
        assert manager.boundary_conditions["x_min"].value.device.type == "meta"
    
    def test_validation_system(self):
        """Test validation system."""
        manager = BoundaryConditionManager(spatial_dims=2)
        manager.initialize()
        
        # Validation should pass for valid configuration
        errors = manager._do_validate()
        assert len(errors) == 0
        
        # Test with invalid spatial_dims
        manager.spatial_dims = 4
        errors = manager._do_validate()
        assert len(errors) > 0
        assert any('spatial_dims' in e for e in errors)
        
        # Test with inconsistent periodic boundaries
        manager.spatial_dims = 2
        manager.set_boundary_condition("x_min", BoundaryType.PERIODIC)
        manager.set_boundary_condition("x_max", BoundaryType.DIRICHLET, 0.0)
        errors = manager._do_validate()
        assert len(errors) > 0
        assert any('periodic' in e.lower() for e in errors)
        
        # Reset to valid state
        manager.set_boundary_condition("x_max", BoundaryType.PERIODIC)
        manager.validate()
        assert manager.state == ManagerState.READY
        
        print("[OK] Validation system working")
    
    def test_global_singleton_api_removed(self):
        """Boundary managers must be owned by a transport process or helper."""
        assert not hasattr(
            boundary_conditions_module,
            "_global_boundary_condition_manager",
        )
        assert not hasattr(
            boundary_conditions_module,
            "get_global_boundary_condition_manager",
        )
        assert not hasattr(
            boundary_conditions_module,
            "set_global_boundary_condition_manager",
        )

        manager1 = BoundaryConditionManager()
        manager2 = BoundaryConditionManager()
        assert manager1 is not manager2
    
    def test_utility_functions(self):
        """Test utility functions for common boundary setups."""
        # Test Dirichlet walls
        bc_manager = create_dirichlet_walls(spatial_dims=2, wall_value=0.0)
        assert bc_manager.state == ManagerState.READY
        assert bc_manager.boundary_conditions["x_min"].bc_type == BoundaryType.DIRICHLET
        assert bc_manager.boundary_conditions["x_max"].bc_type == BoundaryType.DIRICHLET
        
        # Test periodic boundaries
        bc_manager = create_periodic_boundaries(spatial_dims=2)
        assert bc_manager.state == ManagerState.READY
        assert bc_manager.boundary_conditions["x_min"].bc_type == BoundaryType.PERIODIC
        assert bc_manager.boundary_conditions["x_max"].bc_type == BoundaryType.PERIODIC
        
        # Test open boundaries
        bc_manager = create_open_boundaries(spatial_dims=2)
        assert bc_manager.state == ManagerState.READY
        assert bc_manager.boundary_conditions["x_min"].bc_type == BoundaryType.OPEN
        assert bc_manager.boundary_conditions["x_max"].bc_type == BoundaryType.OPEN
        
        print("[OK] Utility functions working")

    def test_utility_functions_validate_final_boundary_types(self, monkeypatch):
        """Factory validation must inspect the final boundary configuration."""
        validation_snapshots = []
        original_validate = BoundaryConditionManager._do_validate

        def record_validation(manager):
            validation_snapshots.append(
                {
                    face: condition.bc_type
                    for face, condition in manager.boundary_conditions.items()
                }
            )
            return original_validate(manager)

        monkeypatch.setattr(
            BoundaryConditionManager,
            "_do_validate",
            record_validation,
        )

        cases = (
            (lambda: create_dirichlet_walls(2), BoundaryType.DIRICHLET),
            (lambda: create_periodic_boundaries(2), BoundaryType.PERIODIC),
            (lambda: create_open_boundaries(2), BoundaryType.OPEN),
        )
        for factory, expected_type in cases:
            validation_snapshots.clear()
            manager = factory()
            required_faces = set(manager.boundary_conditions)
            assert validation_snapshots
            assert set(validation_snapshots[-1]) == required_faces
            assert all(
                boundary_type == expected_type
                for boundary_type in validation_snapshots[-1].values()
            )
    
    def test_backward_consistency(self):
        """Test backward consistency with existing code."""
        # Old-style initialization should still work
        manager = BoundaryConditionManager(
            spatial_dims=2,
            device=torch.device('cpu')
        )
        
        # Check old methods still exist
        assert hasattr(manager, 'set_boundary_condition')
        assert hasattr(manager, 'apply_boundary_conditions')
        assert hasattr(manager, 'get_boundary_flux')
        assert hasattr(manager, 'validate_boundary_conditions')
        
        # Initialize and test basic operation
        manager.initialize()
        manager.validate()
        assert manager.state == ManagerState.READY
        
        # Test boundary condition application still works
        manager.set_boundary_condition("x_min", BoundaryType.DIRICHLET, 0.0)
        field = torch.ones((1, 10, 10))
        field_bc = manager.apply_boundary_conditions(field, grid_spacing=0.1)
        assert torch.all(field_bc[:, :, 0] == 0.0)
        
        print("[OK] Backward consistency maintained")


class TestBoundaryFluxOutwardOrientation:
    """get_boundary_flux must return an outward-positive diffusive flux on every face."""

    @staticmethod
    def _manager(spatial_dims):
        manager = BoundaryConditionManager(spatial_dims=spatial_dims, device=torch.device('cpu'))
        manager.initialize()
        manager.validate()
        return manager

    def test_one_sided_dirichlet_gradient_uses_outward_normal_on_both_faces(self):
        manager = self._manager(1)
        manager.set_boundary_condition("x_min", BoundaryType.DIRICHLET, 0.0)
        manager.set_boundary_condition("x_max", BoundaryType.DIRICHLET, 0.0)
        field = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
        # References: Patankar (1980); LeVeque (2007).
        # Mathematical expression: J_out = -D*dphi/dn; dphi/ds = 1 per metre here, so
        #   J_out(x_min) = -D*(-1) = +3 and J_out(x_max) = -D*(+1) = -3 for D = 3.
        # Parameter meanings: D diffusivity, s the x coordinate, n the outward normal.
        # Units: phi [state_unit]; dx [m]; D [m2 s-1]; J_out [state_unit m s-1].
        # Domain/constraints: one-sided first-order differences on cell-center spacing.
        # Sign convention: positive exports through the face; negative imports.
        flux_min = manager.get_boundary_flux(field, "x_min", 1.0, diffusivity=3.0)
        flux_max = manager.get_boundary_flux(field, "x_max", 1.0, diffusivity=3.0)
        assert flux_min.dtype == torch.float64
        torch.testing.assert_close(flux_min, torch.tensor([3.0], dtype=torch.float64), rtol=0.0, atol=0.0)
        torch.testing.assert_close(flux_max, torch.tensor([-3.0], dtype=torch.float64), rtol=0.0, atol=0.0)

    def test_neumann_prescribed_axis_gradient_is_converted_to_an_outward_flux(self):
        manager = self._manager(1)
        manager.set_boundary_condition("x_min", BoundaryType.NEUMANN, 2.0)
        manager.set_boundary_condition("x_max", BoundaryType.NEUMANN, 2.0)
        field = torch.zeros((1, 3), dtype=torch.float64)
        # References: BoundaryCondition contract, Neumann datum g_s = dphi/ds [state_unit m-1].
        # Mathematical expression: J_out(x_min) = -D*(-g_s) = +6; J_out(x_max) = -D*g_s = -6.
        # Parameter meanings: g_s = 2 per metre, D = 3.
        # Units: g_s [state_unit m-1]; D [m2 s-1]; J_out [state_unit m s-1].
        # Domain/constraints: the datum is never returned as a flux without D.
        # Sign convention: outward positive on both faces.
        flux_min = manager.get_boundary_flux(field, "x_min", 1.0, diffusivity=3.0)
        flux_max = manager.get_boundary_flux(field, "x_max", 1.0, diffusivity=3.0)
        assert flux_min.dtype == torch.float64
        torch.testing.assert_close(flux_min, torch.tensor(6.0, dtype=torch.float64), rtol=0.0, atol=0.0)
        torch.testing.assert_close(flux_max, torch.tensor(-6.0, dtype=torch.float64), rtol=0.0, atol=0.0)

    def test_y_faces_use_the_physical_y_spacing_and_outward_normal(self):
        manager = self._manager(2)
        for face in ("y_min", "y_max"):
            manager.set_boundary_condition(face, BoundaryType.DIRICHLET, 0.0)
        rows = torch.arange(3, dtype=torch.float64).reshape(1, 3, 1)
        field = rows.expand(1, 3, 4).clone()
        # References: Patankar (1980).
        # Mathematical expression: dphi/dy = 1/dy = 0.5 with dy = 2; J_out(y_min) = +0.5*D,
        #   J_out(y_max) = -0.5*D with D = 1; the x spacing (1.0) must not be used.
        # Parameter meanings: grid_spacing is physical (dx, dy) = (1.0, 2.0).
        # Units: phi [state_unit]; dy [m]; D [m2 s-1]; J_out [state_unit m s-1].
        # Domain/constraints: tensor storage is (batch, y, x).
        # Sign convention: outward positive.
        flux_min = manager.get_boundary_flux(field, "y_min", (1.0, 2.0), diffusivity=1.0)
        flux_max = manager.get_boundary_flux(field, "y_max", (1.0, 2.0), diffusivity=1.0)
        assert tuple(flux_min.shape) == (1, 4)
        torch.testing.assert_close(flux_min, torch.full((1, 4), 0.5, dtype=torch.float64), rtol=0.0, atol=0.0)
        torch.testing.assert_close(flux_max, torch.full((1, 4), -0.5, dtype=torch.float64), rtol=0.0, atol=0.0)

    def test_boundary_flux_rejects_missing_diffusivity_and_unusable_geometry(self):
        manager = self._manager(2)
        field = torch.zeros((1, 3, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="requires an explicit diffusivity"):
            manager.get_boundary_flux(field, "x_min", 1.0, diffusivity=None)
        with pytest.raises(ValueError, match="No boundary condition set"):
            manager.get_boundary_flux(field, "z_min", 1.0, diffusivity=1.0)
        manager.set_boundary_condition("x_min", BoundaryType.DIRICHLET, 0.0)
        with pytest.raises(ValueError, match="positive"):
            manager.get_boundary_flux(field, "x_min", (0.0, 1.0), diffusivity=1.0)
        with pytest.raises(ValueError, match="at least two cells"):
            manager.get_boundary_flux(torch.zeros((1, 3, 1), dtype=torch.float64), "x_min", 1.0, diffusivity=1.0)


def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("BoundaryConditionManager V2 Migration Tests")
    print("="*60 + "\n")
    
    test_suite = TestBoundaryConditionManagerV2Migration()
    
    tests = [
        test_suite.test_inheritance_from_base_manager,
        test_suite.test_lifecycle_management,
        test_suite.test_boundary_condition_application,
        test_suite.test_neumann_boundary,
        test_suite.test_periodic_boundary,
        test_suite.test_robin_boundary,
        test_suite.test_2d_boundary_conditions,
        test_suite.test_device_management,
        test_suite.test_validation_system,
        test_suite.test_global_singleton_api_removed,
        test_suite.test_import_does_not_register_competing_manager_owner,
        test_suite.test_utility_functions,
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
            if "[SKIP]" in str(e) or "[INFO]" in str(e):
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


