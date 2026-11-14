"""
L3 System Tests for Coupling Module

This module contains comprehensive system-level tests for the coupling module,
testing end-to-end functionality, multi-scale coupling workflows, conservation
law enforcement across complete system simulations, and long-term stability.

Test Coverage:
- End-to-end multi-scale coupling workflows
- Conservation law enforcement across full simulation cycles
- Long-term simulation stability and consistency
- Performance benchmarks and scalability tests
- Real-world scenario simulations
- System integration with RELIC simulator framework
- Comprehensive validation against expected system behavior

References:
- RELIC System Architecture Documentation
- Conservation Law Enforcement in Ecosystem Models
- Multi-scale Modeling Best Practices
- CF-1.8 Conventions for System-level Data Exchange
"""

import pytest
import torch
import numpy as np
import time
from typing import Any, Tuple
import logging
import os

# Import coupling module classes
from relic.coupling.scales.adapters import (
    SpatialAggregation,
    ConservativeRemap
)

# Import core system components.
from relic.core.simulator import RELICSimulator

PERF_TESTS_ENABLED = os.environ.get("RELIC_RUN_PERF_TESTS", "0").strip().lower() in {"1", "true", "yes", "on"}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Units of this test's own scenario variables, which no SCM contract or SSOT
# entry declares; they follow the units stated with the test data. The simulator
# infers no units from names (relic/core/simulator/state_init.py).
_TEST_STATE_UNITS = {
    'canopy_height': 'm',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


def _synchronize_module(module: Any) -> None:
    """Synchronize CUDA work when a benchmarked module runs on CUDA."""
    device = torch.device(getattr(module, "device", "cpu"))
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_runtime(module: Any, operation: Any, repeats: int = 3) -> Tuple[Any, float]:
    """Return one result and a synchronized median runtime."""
    result = operation()
    _synchronize_module(module)
    durations = []
    for _ in range(repeats):
        _synchronize_module(module)
        start_time = time.perf_counter()
        result = operation()
        _synchronize_module(module)
        durations.append(time.perf_counter() - start_time)
    return result, float(np.median(np.asarray(durations, dtype=float)))




def _whole_domain_geometry(n_source: int, variables, spatial_basis: str = "cell_total") -> dict:
    """One target cell covering n_source unit-area source cells (D64 explicit geometry).

    The overlap-weighted remap then reduces to the domain total for cell totals, and to
    the domain mean for intensive fields and area densities, which is what these consumer
    tests exercise.
    """
    return {
        "overlap_areas": torch.ones(n_source, 1, dtype=torch.float64),
        "source_areas": torch.ones(n_source, dtype=torch.float64),
        "target_areas": torch.tensor([float(n_source)], dtype=torch.float64),
        "variable_metadata": {name: {"spatial_basis": spatial_basis} for name in variables},
    }


# Explicit spatial bases of the synthetic ecosystem state (SpatialQuantityBasis values):
# per-area fluxes and stocks are area densities, drivers and states are intensive, and the
# cell area itself is a per-cell total.
_ECOSYSTEM_BASES = {
    'temperature': 'intensive',
    'precipitation': 'area_density',
    'solar_radiation': 'intensive',
    'wind_speed': 'intensive',
    'relative_humidity': 'intensive',
    'atmospheric_pressure': 'intensive',
    'gpp': 'area_density',
    'npp': 'area_density',
    'autotrophic_respiration': 'area_density',
    'heterotrophic_respiration': 'area_density',
    'litter_fall': 'area_density',
    'evapotranspiration': 'area_density',
    'surface_runoff': 'area_density',
    'groundwater_recharge': 'area_density',
    'soil_moisture': 'intensive',
    'soil_temperature': 'intensive',
    'latent_heat_flux': 'area_density',
    'sensible_heat_flux': 'area_density',
    'ground_heat_flux': 'area_density',
    'soil_carbon_stock': 'area_density',
    'vegetation_biomass': 'area_density',
    'leaf_area_index': 'intensive',
    'canopy_height': 'intensive',
    'cell_area': 'cell_total',
    'elevation': 'intensive',
    'slope': 'intensive',
    'aspect': 'intensive',
}


def _basis_metadata(names) -> dict:
    """Per-variable metadata declaring the explicit spatial basis."""
    return {name: {'spatial_basis': _ECOSYSTEM_BASES[name]} for name in names}

class TestEndToEndCouplingWorkflows:
    """Test suite for end-to-end coupling workflows."""
    
    @pytest.fixture
    def complete_coupling_system(self):
        """Create a coupling system with the remaining spatial functor types."""
        return {
            'spatial_aggregator': SpatialAggregation(
                aggregation_strategy="area_weighted",
                preserve_totals=True,
                weighting_type="area",
                source_scale="fine_grid",
                target_scale="coarse_grid",
            ),
            'conservative_remap': ConservativeRemap(
                preserve_mass=True,
                preserve_energy=True,
                preserve_positivity=True
            )
        }
    
    @pytest.fixture
    def ecosystem_simulation_state(self):
        """Create comprehensive ecosystem simulation state data."""
        # One year of daily data at 10x10 spatial resolution
        n_days = 365
        spatial_shape = (10, 10)
        # A fixed seed makes every draw reproducible; at this seed the aspect field has a
        # negative domain mean, so the remap workflow exercises the positivity clip.
        generator = torch.Generator().manual_seed(2)
        
        return {
            # Climate drivers (atmosphere)
            'temperature': torch.randn(n_days, *spatial_shape, generator=generator) * 10 + 293.15,  # K
            'precipitation': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 5e-8,  # m/s
            'solar_radiation': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 300 + 200,  # W/m2
            'wind_speed': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 3 + 1,  # m/s
            'relative_humidity': torch.clamp(torch.randn(n_days, *spatial_shape, generator=generator) * 0.3 + 0.6, 0, 1),
            'atmospheric_pressure': torch.ones(n_days, *spatial_shape) * 101325,  # Pa
            
            # Carbon cycle (biosphere)
            'gpp': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 8e-6,  # kg C/m2/s
            'npp': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 5e-6,  # kg C/m2/s
            'autotrophic_respiration': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 3e-6,
            'heterotrophic_respiration': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 2e-6,
            'litter_fall': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 1e-6,
            
            # Water cycle (hydrosphere + pedosphere)
            'evapotranspiration': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 3e-8,  # m/s
            'surface_runoff': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 2e-8,  # m/s
            'groundwater_recharge': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 1e-8,  # m/s
            'soil_moisture': torch.clamp(torch.randn(n_days, *spatial_shape, generator=generator) * 0.2 + 0.3, 0.05, 0.6),
            'soil_temperature': torch.randn(n_days, *spatial_shape, generator=generator) * 8 + 288.15,  # K
            
            # Energy fluxes
            'latent_heat_flux': torch.abs(torch.randn(n_days, *spatial_shape, generator=generator)) * 50 + 100,  # W/m2
            'sensible_heat_flux': torch.randn(n_days, *spatial_shape, generator=generator) * 30 + 50,  # W/m2
            'ground_heat_flux': torch.randn(n_days, *spatial_shape, generator=generator) * 10 + 5,  # W/m2
            
            # State variables
            'soil_carbon_stock': torch.abs(torch.randn(*spatial_shape, generator=generator)) * 15 + 5,  # kg C/m2
            'vegetation_biomass': torch.abs(torch.randn(*spatial_shape, generator=generator)) * 10 + 2,  # kg C/m2
            'leaf_area_index': torch.abs(torch.randn(*spatial_shape, generator=generator)) * 3 + 1,  # m2/m2
            'canopy_height': torch.abs(torch.randn(*spatial_shape, generator=generator)) * 15 + 5,  # m
            
            # Grid properties
            'cell_area': torch.ones(*spatial_shape) * 10000,  # m2 (100m x 100m cells)
            'elevation': torch.randn(*spatial_shape, generator=generator) * 100 + 200,  # m above sea level
            'slope': torch.abs(torch.randn(*spatial_shape, generator=generator)) * 0.2 + 0.01,  # fraction
            'aspect': torch.randn(*spatial_shape, generator=generator) * 180,  # degrees
        }
    
    def test_spatial_system_integration(
        self,
        complete_coupling_system,
        ecosystem_simulation_state,
    ):
        """Apply the spatial owner to one explicit ecosystem snapshot."""
        spatial_agg = complete_coupling_system['spatial_aggregator']
        spatial_state = {
            key: value[0] if value.dim() == 3 else value
            for key, value in ecosystem_simulation_state.items()
        }
        spatial_results = spatial_agg.forward(
            state=spatial_state,
            time=0.0,
            dt=1.0,
            weights=spatial_state['cell_area'],
            variable_metadata=_basis_metadata(spatial_state),
        )
        assert len(spatial_results) == len(spatial_state)
        assert all(isinstance(value, torch.Tensor) for value in spatial_results.values())
        assert all(torch.all(torch.isfinite(value)) for value in spatial_results.values())
        # An area density keeps its basis: its area-weighted mean over the 1e4 m2 cells.
        areas = spatial_state['cell_area']
        torch.testing.assert_close(
            spatial_results['gpp_aggregated'],
            ((spatial_state['gpp'] * areas).sum() / areas.sum()).to(spatial_results['gpp_aggregated']),
        )

    def test_conservative_remap_integration_workflow(self, complete_coupling_system, ecosystem_simulation_state):
        """Test integration of conservative remapping with other coupling operations."""
        spatial_agg = complete_coupling_system['spatial_aggregator']
        conservative_remap = complete_coupling_system['conservative_remap']
        
        # Step 1: Extract spatial data for a single day
        daily_spatial_data = {}
        for key, value in ecosystem_simulation_state.items():
            if value.dim() == 3 and value.shape[0] == 365:
                daily_spatial_data[key] = value[0]  # First day
            elif value.dim() == 2:
                daily_spatial_data[key] = value
        
        # Step 2: Spatial aggregation to coarse resolution
        coarse_results = spatial_agg.forward(
            state=daily_spatial_data,
            time=0.0,
            dt=1.0,
            weights=daily_spatial_data['cell_area'],
            variable_metadata=_basis_metadata(daily_spatial_data),
        )
        output_bases = {
            f"{name}_aggregated": basis
            for name, basis in spatial_agg.last_scaling_audit.metadata["output_basis"].items()
        }
        
        # Step 3: Conservative remapping to different grid
        # Create synthetic coarse grid data for remapping
        remap_state = {}
        for key, value in coarse_results.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                # Expand to 2D for remapping if scalar
                if value.dim() == 0:
                    remap_state[key] = value.expand(3, 3)  # 3x3 coarse grid
                else:
                    remap_state[key] = value
        
        # Apply conservative remapping on the 3x3 coarse cells with one covering target cell;
        # each aggregate carries the output basis reported by SpatialAggregation (every field
        # keeps its input basis: intensive means, area-density means and cell totals).
        remap_state = {key: value for key, value in remap_state.items() if value.numel() == 9}
        geometry = _whole_domain_geometry(9, list(remap_state.keys()))
        geometry["variable_metadata"] = {
            key: {"spatial_basis": output_bases[key]} for key in remap_state
        }
        remap_results = conservative_remap.forward(
            state=remap_state,
            time=0.0,
            dt=1.0,
            variables=list(remap_state.keys()),
            **geometry,
        )
        
        # Verify remapping results
        assert len(remap_results) > 0
        for value in remap_results.values():
            assert torch.all(torch.isfinite(value))

        # Positivity bounds apply to the remapped fields only: y_c = max(y, 0).
        remapped = {key: remap_results[f"{key}_remapped"] for key in remap_state}
        for value in remapped.values():
            assert torch.all(value >= 0)
        # One target covering the nine unit-area cells: the unclipped remap of every field is its
        # (uniform) coarse value s for an intensive field or an area density and 9 s for a cell
        # total; the ledger measure is the covered (= target) area 9 m2 or 1 for a cell total.
        # Documented sign (theory 2.3, StateManager clip_loss): P = sum (y - max(y, 0)) a <= 0,
        # the requested minus the clipped amount, negative by the amount the clip adds.
        expected_residual = 0.0
        for key, value in remap_state.items():
            coarse = float(value.to(torch.float64).reshape(-1)[0])
            cell_total = output_bases[key] == "cell_total"
            unclipped = 9.0 * coarse if cell_total else coarse
            measure = 1.0 if cell_total else 9.0
            expected_residual += (unclipped - max(unclipped, 0.0)) * measure
            assert float(remapped[key]) == pytest.approx(max(unclipped, 0.0), rel=1e-6, abs=1e-12)
            assert float(remap_results[f"{key}_uncovered_source_amount"]) == 0.0
        residual = float(remap_results["diagnostics.conservative_remap.positivity_clip_residual"])
        assert residual <= 0.0
        assert residual == pytest.approx(expected_residual, rel=1e-6, abs=1e-9)
        # The assertions above hold for either sign of the aggregates; at this seed the mean aspect
        # (degrees) is negative, so the clip adds 9 |s| and the residual is negative.
        assert float(remap_state["aspect_aggregated"].reshape(-1)[0]) < 0.0
        assert residual < 0.0
    
    def test_full_simulator_integration(
        self,
        complete_coupling_system,
        ecosystem_simulation_state,
    ):
        """Register the remaining spatial Functors and verify direct outputs."""
        simulator = _declared_simulator()
        for functor in complete_coupling_system.values():
            simulator.register_functor(functor)
        initial_state = {
            key: (
                ecosystem_simulation_state[key][0]
                if ecosystem_simulation_state[key].dim() == 3
                else ecosystem_simulation_state[key]
            )
            for key in (
                'soil_moisture',
                'soil_temperature',
                'leaf_area_index',
                'canopy_height',
            )
        }
        simulator.initialize_state(initial_state)
        summary = simulator.get_functor_summary()
        for functor in complete_coupling_system.values():
            assert functor.name in summary
        spatial_updates = complete_coupling_system['spatial_aggregator'].forward(
            state=initial_state,
            time=0.0,
            dt=1.0,
            variable_metadata=_basis_metadata(initial_state),
        )
        assert spatial_updates
        assert all(torch.all(torch.isfinite(value)) for value in spatial_updates.values())

class TestConservationLawEnforcement:
    """Test suite for conservation law enforcement across complete systems."""
    
    @pytest.fixture
    def conservation_test_data(self):
        """Create data for conservation law testing."""
        return {
            # Carbon cycle with known totals
            'carbon_input': torch.ones(365, 8, 8) * 1e-6,  # kg C/m2 day-1
            'carbon_output': torch.ones(365, 8, 8) * 0.8e-6,  # kg C/m2 day-1
            'carbon_storage': torch.ones(8, 8) * 100,  # kg C/m2
            
            # Water cycle with known totals
            'water_input': torch.ones(365, 8, 8) * 5e-8,  # m/s
            'water_output': torch.ones(365, 8, 8) * 4e-8,  # m/s
            'water_storage': torch.ones(8, 8) * 0.3,  # m3/m3
            
            # Energy cycle with known totals
            'energy_input': torch.ones(365, 8, 8) * 300,  # W/m2
            'energy_output': torch.ones(365, 8, 8) * 280,  # W/m2
            'energy_storage': torch.ones(8, 8) * 1000,  # J/m2
        }
    
    def test_carbon_conservation_across_spatial_remap(
        self,
        conservation_test_data,
    ):
        """Verify the carbon ledger of the spatial mean and a conservative remap of the density."""
        spatial = SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
        remapper = ConservativeRemap(
            preserve_mass=True,
            preserve_positivity=True,
        )
        # carbon_storage [kg C m-2] is an area density on unit-area cells.
        spatial_result = spatial.forward(
            state={'carbon_storage': conservation_test_data['carbon_storage']},
            time=0.0,
            dt=1.0,
            weights=torch.ones(8, 8),
            variable_metadata={'carbon_storage': {'spatial_basis': 'area_density'}},
        )
        aggregated = spatial_result['carbon_storage_aggregated']
        # The density stays a density (100 kg C m-2); the ledger carries the 64 m2 amount.
        torch.testing.assert_close(aggregated, torch.tensor(100.0).to(aggregated))
        ledger = spatial.last_scaling_audit.metadata['conservation_audit']['carbon_storage']
        assert ledger.source_total == pytest.approx(float(conservation_test_data['carbon_storage'].sum()))
        assert ledger.target_total == pytest.approx(ledger.source_total)
        remap_result = remapper.forward(
            state={'carbon_density': aggregated.expand(4, 4)},
            time=0.0,
            dt=1.0,
            variables=['carbon_density'],
            **_whole_domain_geometry(16, ['carbon_density'], spatial_basis='area_density'),
        )
        remapped = remap_result['carbon_density_remapped']
        assert torch.all(torch.isfinite(remapped))
        assert torch.all(remapped >= 0)
        torch.testing.assert_close(remapped, torch.full_like(remapped, 100.0))

    def test_water_storage_spatial_aggregation(self, conservation_test_data):
        """Verify spatial aggregation of bounded water storage."""
        spatial = SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
        # water_storage [m3 m-3] is an intensive volumetric fraction.
        result = spatial.forward(
            state={'water_storage': conservation_test_data['water_storage']},
            time=0.0,
            dt=1.0,
            variable_metadata={'water_storage': {'spatial_basis': 'intensive'}},
        )
        value = result['water_storage_aggregated']
        assert torch.all(torch.isfinite(value))
        torch.testing.assert_close(value, torch.tensor(0.3).to(value))

    def test_energy_conservative_remap(self, conservation_test_data):
        """Verify conservative remap keeps energy storage finite."""
        remapper = ConservativeRemap(
            preserve_mass=False,
            preserve_energy=True,
            preserve_positivity=True,
        )
        result = remapper.forward(
            state={'energy_storage': conservation_test_data['energy_storage']},
            time=0.0,
            dt=1.0,
            variables=['energy_storage'],
            # energy_storage [J m-2] is an area density: the single target keeps the mean density.
            **_whole_domain_geometry(64, ['energy_storage'], spatial_basis='area_density'),
        )
        value = result['energy_storage_remapped']
        assert torch.all(torch.isfinite(value))
        assert torch.all(value >= 0)

    def test_coupled_spatial_conservation_payloads(self, conservation_test_data):
        """Verify multiple conserved storage payloads share the spatial path."""
        spatial = SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
        result = spatial.forward(
            state={
                'carbon_storage': conservation_test_data['carbon_storage'],
                'water_storage': conservation_test_data['water_storage'],
                'energy_storage': conservation_test_data['energy_storage'],
            },
            time=0.0,
            dt=1.0,
            weights=torch.ones(8, 8),
            variable_metadata={
                'carbon_storage': {'spatial_basis': 'area_density'},
                'water_storage': {'spatial_basis': 'intensive'},
                'energy_storage': {'spatial_basis': 'area_density'},
            },
        )
        for name in ('carbon_storage', 'water_storage', 'energy_storage'):
            value = result[f'{name}_aggregated']
            assert torch.all(torch.isfinite(value))

class TestLongTermStability:
    """Test suite for long-term simulation stability and consistency."""
    
    def test_multi_year_spatial_stability(self):
        """Verify finite spatial summaries across five synthetic years."""
        spatial = SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
        annual_results = []
        for year in range(5):
            yearly_snapshot = {
                'gpp': torch.abs(torch.randn(6, 6) * 5e-6 * (1.0 + year * 0.1)),
                'temperature': torch.randn(6, 6) * 10 + 293.15 + year * 0.5,
                'soil_moisture': torch.clamp(
                    torch.randn(6, 6) * 0.2 + 0.3 - year * 0.01,
                    0.1,
                    0.6,
                ),
            }
            annual_results.append(
                spatial.forward(
                    state=yearly_snapshot,
                    time=float(year),
                    dt=1.0,
                    weights=torch.ones(6, 6),
                    variable_metadata=_basis_metadata(yearly_snapshot),
                )
            )
        assert len(annual_results) == 5
        for result in annual_results:
            assert result
            assert all(torch.all(torch.isfinite(value)) for value in result.values())

    def test_spatial_memory_stability(self):
        """Exercise repeated spatial summaries without retaining input tensors."""
        spatial = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
        for chunk in range(24):
            state = {
                'variable_1': torch.randn(8, 8),
                'variable_2': torch.randn(8, 8),
                'variable_3': torch.randn(8, 8),
            }
            result = spatial.forward(
                state=state,
                time=float(chunk),
                dt=1.0,
                variable_metadata={name: {'spatial_basis': 'intensive'} for name in state},
            )
            assert len(result) == 3
            assert all(torch.all(torch.isfinite(value)) for value in result.values())

class TestPerformanceBenchmarks:
    """Test suite for performance benchmarks and scalability."""
    pytestmark = [
        pytest.mark.perf,
        pytest.mark.skipif(not PERF_TESTS_ENABLED, reason="Set RELIC_RUN_PERF_TESTS=1 to enable performance benchmarks.")
    ]
    
    def test_spatial_aggregation_performance(self):
        """Test spatial aggregation performance with different grid sizes."""
        aggregator = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
        
        # Test different grid sizes
        grid_sizes = [
            (10, 10),    # Small grid
            (50, 50),    # Medium grid
            (100, 100),  # Large grid
            (200, 200),  # Very large grid
        ]
        
        performance_results = {}
        
        for height, width in grid_sizes:
            # Generate test data
            n_vars = 5
            test_data = {}
            
            for i in range(n_vars):
                test_data[f'var_{i}'] = torch.randn(height, width)
            
            result, execution_time = _median_runtime(
                aggregator,
                lambda: aggregator.forward(
                    state=test_data,
                    time=0.0,
                    dt=1.0,
                    variable_metadata={name: {'spatial_basis': 'intensive'} for name in test_data},
                ),
            )
            assert execution_time > 0.0

            # Store performance metrics
            total_elements = height * width * n_vars
            performance_results[f'{height}x{width}'] = {
                'execution_time': execution_time,
                'total_elements': total_elements,
                'throughput': total_elements / execution_time,
            }
            
            # Verify results
            assert len(result) == n_vars
            
            for key, value in result.items():
                assert value.numel() <= height * width  # Should be reduced dimensionality
        
        sizes = np.asarray(
            [metrics['total_elements'] for metrics in performance_results.values()],
            dtype=float,
        )
        runtimes = np.asarray(
            [metrics['execution_time'] for metrics in performance_results.values()],
            dtype=float,
        )
        scaling_exponent = np.log(runtimes[-1] / runtimes[0]) / np.log(sizes[-1] / sizes[0])
        # Fixed dispatch overhead and timer jitter can make the largest case
        # marginally faster. Positive runtimes are checked above; this gate
        # rejects only superlinear growth beyond the allowed tolerance.
        assert scaling_exponent <= 1.25
        
        # Large grids should still complete in reasonable time
        for case_name, metrics in performance_results.items():
            assert metrics['execution_time'] < 5.0, f"Spatial aggregation for {case_name} took too long: {metrics['execution_time']:.2f}s"
    
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
