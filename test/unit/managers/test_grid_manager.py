"""
Unit tests for GridManager v2 migration.

Tests the GridManager's integration with BaseManager architecture
while ensuring all grid operations remain intact.
"""

import torch
import sys
import os

# Add relic to path
sys.path.insert(0, os.path.abspath('.'))

from relic.coupling.scales.spatial import grid_manager as grid_manager_module
from relic.coupling.scales.spatial.grid_manager import GridManager
from relic.core.managers.base import ManagerState


class TestGridManagerV2Migration:
    """Test GridManager v2 migration and BaseManager integration."""
    
    def test_inheritance_from_base_manager(self):
        """Test that GridManager properly inherits from BaseManager."""
        manager = GridManager()
        
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
        assert metadata.name == "GridManager"
        assert metadata.version == "2.0.0"
        
        print("[OK] GridManager properly inherits from BaseManager")
    
    def test_lifecycle_management(self):
        """Test GridManager lifecycle through BaseManager."""
        manager = GridManager()
        
        # Initial state should be UNINITIALIZED
        assert manager.state == ManagerState.UNINITIALIZED
        
        # Initialize
        manager.initialize()
        assert manager.state == ManagerState.INITIALIZING
        
        # Configure
        config = {
            'earth_radius': 6371000.0,
            'coord_names': {'x': 'longitude', 'y': 'latitude'}
        }
        manager.configure(config)
        assert manager.earth_radius == 6371000.0
        
        # Validate to transition to READY
        manager.validate()
        assert manager.state == ManagerState.READY
        assert hasattr(manager, '_grid_utils')
        assert hasattr(manager, '_area_cache')
        
        # Cleanup
        manager.cleanup()
        # After cleanup, state returns to UNINITIALIZED for potential reuse
        assert manager.state == ManagerState.UNINITIALIZED
        
        print("[OK] Lifecycle management working correctly")
    
    def test_grid_creation_functionality(self):
        """Test core grid creation functionality remains intact."""
        manager = GridManager()
        manager.initialize()
        manager.validate()
        
        # Test regular grid creation
        grid = manager.build_from_regular(
            lon_min=0.0,
            lon_max=10.0,
            lat_min=0.0,
            lat_max=10.0,
            nlon=10,
            nlat=10
        )
        
        assert grid is not None
        assert grid.grid_type == "regular"
        assert grid.coord_system == "latlon"
        assert grid.dims == (10, 10)
        assert grid.x_coords.shape == (10,)
        assert grid.y_coords.shape == (10,)
        
        # Check bounds were created
        assert grid.x_bounds is not None
        assert grid.y_bounds is not None
        assert grid.x_bounds.shape == (10, 2)
        assert grid.y_bounds.shape == (10, 2)
        
        # Check areas were computed
        assert grid.cell_areas is not None
        assert grid.cell_areas.shape == (10, 10)
        assert torch.all(grid.cell_areas > 0)
        
        print("[OK] Grid creation functionality intact")
    
    def test_grid_from_tensor(self):
        """Test creating grid from coordinate tensors."""
        manager = GridManager()
        manager.initialize()
        manager.validate()
        
        # Cell edges declare the bounds; the centres are the edge midpoints of this grid.
        lat_edges = torch.linspace(-90, 90, 19, dtype=torch.float64)
        lon_edges = torch.linspace(-180, 180, 37, dtype=torch.float64)
        lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_bounds = torch.stack([lat_edges[:-1], lat_edges[1:]], dim=-1)
        lon_bounds = torch.stack([lon_edges[:-1], lon_edges[1:]], dim=-1)

        # Create grid from tensors
        grid = manager.from_tensor(lat=lat, lon=lon, lat_bounds=lat_bounds, lon_bounds=lon_bounds)
        
        assert grid is not None
        assert grid.grid_type == "regular"
        assert grid.coord_system == "latlon"
        assert grid.dims == (18, 36)
        assert torch.allclose(grid.x_coords, lon)
        assert torch.allclose(grid.y_coords, lat)
        
        # Check areas were computed from the declared bounds: the cells tile the sphere.
        assert grid.cell_areas is not None
        radius = float(manager._grid_utils.earth_radius)
        total = float(grid.cell_areas.sum())
        assert abs(total - 4.0 * 3.141592653589793 * radius ** 2) <= 1e-4 * total

        print("[OK] Grid from tensor working")

    def test_cell_bounds_are_declared_not_inferred(self):
        """Bounds come from the caller; areas of a grid without bounds raise; the cache keys on bounds."""
        import pytest

        from relic.coupling.scales.spatial.grid_utils import GridDescriptor

        manager = GridManager()
        manager.initialize()
        # Centres 0.5, 1.5, 3.0: spacing-inferred bounds (first spacing 1.0) would be wrong.
        with pytest.raises(TypeError, match="lat_bounds"):
            manager.from_tensor(lat=torch.tensor([0.5, 2.0]), lon=torch.tensor([0.5, 1.5, 3.0]))
        bare = GridDescriptor(
            grid_type="regular",
            coord_system="cartesian",
            dims=(1, 3),
            x_coords=torch.tensor([0.5, 1.5, 3.0]),
            y_coords=torch.tensor([0.5]),
        )
        with pytest.raises(ValueError, match="bounds"):
            manager._grid_utils.compute_grid_areas(bare)

        def strip(x_edges):
            n = len(x_edges) - 1
            return GridDescriptor(
                grid_type="regular",
                coord_system="cartesian",
                dims=(1, n),
                x_coords=torch.tensor([0.5 * (x_edges[i] + x_edges[i + 1]) for i in range(n)]),
                y_coords=torch.tensor([0.5]),
                x_bounds=torch.tensor([[x_edges[i], x_edges[i + 1]] for i in range(n)]),
                y_bounds=torch.tensor([[0.0, 1.0]]),
            )

        # Same shape, different cells: the area cache must not hand the first areas to the second.
        first = manager.ensure_areas(strip([0.0, 1.0, 2.0]))
        second = manager.ensure_areas(strip([0.0, 1.0, 3.0]))
        torch.testing.assert_close(first.cell_areas.reshape(-1), torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(second.cell_areas.reshape(-1), torch.tensor([1.0, 2.0]))
    
    def test_device_management(self):
        """Test device-aware functionality."""
        # Test with CPU (always available)
        device = torch.device("cpu")
        manager = GridManager(device=device)
        manager.initialize()
        manager.validate()
        
        # Check device is set correctly
        assert manager.device == device
        
        # Create grid and check tensors are on correct device
        grid = manager.build_from_regular(
            lon_min=0.0,
            lon_max=10.0,
            lat_min=0.0,
            lat_max=10.0,
            nlon=5,
            nlat=5
        )
        
        assert grid.x_coords.device == device
        assert grid.y_coords.device == device
        if grid.cell_areas is not None:
            assert grid.cell_areas.device == device
        
        print("[OK] Device management working")
    
    def test_area_caching(self):
        """Test area caching functionality."""
        manager = GridManager()
        manager.initialize()
        manager.validate()
        
        # Create first grid
        grid1 = manager.build_from_regular(
            lon_min=0.0,
            lon_max=10.0,
            lat_min=0.0,
            lat_max=10.0,
            nlon=5,
            nlat=5
        )
        
        # Cache should have one entry
        assert len(manager._area_cache) == 1
        
        # Create identical grid - should use cached areas
        grid2 = manager.build_from_regular(
            lon_min=0.0,
            lon_max=10.0,
            lat_min=0.0,
            lat_max=10.0,
            nlon=5,
            nlat=5
        )
        
        # Cache should still have one entry
        assert len(manager._area_cache) == 1
        
        # Areas should be identical (same object)
        assert grid1.cell_areas is grid2.cell_areas
        
        # Clear cache
        manager.clear_cache()
        assert len(manager._area_cache) == 0
        
        print("[OK] Area caching working")
    
    def test_validation_system(self):
        """Test validation system."""
        manager = GridManager()
        manager.initialize()
        
        # Validation should pass for valid configuration
        errors = manager._do_validate()
        assert len(errors) == 0
        
        # Test with invalid earth radius
        manager.earth_radius = -100.0
        errors = manager._do_validate()
        assert len(errors) > 0
        assert any('radius' in e for e in errors)
        
        # Reset to valid state
        manager.earth_radius = 6371000.0
        manager.validate()
        assert manager.state == ManagerState.READY
        
        print("[OK] Validation system working")
    
    def test_global_singleton_api_removed(self):
        """GridManager must be owned explicitly by the ScaleManager lifecycle."""
        assert not hasattr(grid_manager_module, "_global_grid_manager")
        assert not hasattr(grid_manager_module, "get_global_grid_manager")
        assert not hasattr(grid_manager_module, "set_global_grid_manager")

        manager1 = GridManager()
        manager2 = GridManager()
        assert manager1 is not manager2

    def test_explicit_parameter_manager_owner_is_preserved(self):
        """Initialization must not replace an explicitly injected parameter owner."""
        explicit_parameter_manager = object()
        manager = GridManager(
            device=torch.device("cpu"),
            coord_names={"x": "lon", "y": "lat"},
            parameter_manager=explicit_parameter_manager,
        )

        manager.initialize()

        assert manager.parameter_manager is explicit_parameter_manager
    
    def test_backward_consistency(self):
        """Test backward consistency with existing code."""
        # Old-style initialization should still work
        manager = GridManager(
            device=None,
            coord_names={'x': 'lon', 'y': 'lat'},
            earth_radius=6371000.0
        )
        
        # Check old methods still exist
        assert hasattr(manager, 'from_tensor')
        assert hasattr(manager, 'build_from_regular')
        assert hasattr(manager, 'ensure_areas')
        assert hasattr(manager, 'move_to_device')
        assert hasattr(manager, 'validate_grid')
        assert hasattr(manager, 'clear_cache')
        
        # Initialize and test basic operation
        manager.initialize()
        manager.validate()
        assert manager.state == ManagerState.READY
        
        # Test grid creation still works
        grid = manager.build_from_regular(
            lon_min=-10.0,
            lon_max=10.0,
            lat_min=-10.0,
            lat_max=10.0,
            nlon=4,
            nlat=4
        )
        assert grid is not None
        assert grid.dims == (4, 4)
        
        print("[OK] Backward consistency maintained")


def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("GridManager V2 Migration Tests")
    print("="*60 + "\n")
    
    test_suite = TestGridManagerV2Migration()
    
    tests = [
        test_suite.test_inheritance_from_base_manager,
        test_suite.test_lifecycle_management,
        test_suite.test_grid_creation_functionality,
        test_suite.test_grid_from_tensor,
        test_suite.test_device_management,
        test_suite.test_area_caching,
        test_suite.test_validation_system,
        test_suite.test_global_singleton_api_removed,
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


