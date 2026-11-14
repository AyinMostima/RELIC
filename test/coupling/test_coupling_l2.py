"""
L2 Integration Tests for Coupling Module

This module contains comprehensive integration tests for the coupling module,
testing interactions between coupling functors and other RELIC modules,
including transport, biosphere, atmosphere, hydrosphere, and pedosphere components.

Test Coverage:
- Coupling with transport module integration
- Cross-sphere parameter transfer and conversion
- Multi-scale coupling consistency
- End-to-end conservation across module boundaries
- Data format consistency and unit conversion
- Error propagation and handling across module boundaries

References:
- RELIC Architecture Documentation
- CF-1.8 Conventions for cross-module data exchange
- RELIC Transport Module Integration Guide
"""

import pytest
import torch
import logging

# Import coupling module classes
from relic.coupling.scales.adapters import (
    SpatialAggregation,
    ConservativeRemap
)

# Import canonical transport modules for integration testing.
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.diffusion import FickianDiffusion

# Import the canonical biosphere photosynthesis process for integration testing.
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




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

class TestTransportCouplingIntegration:
    """Test suite for coupling module integration with transport module."""
    
    @pytest.fixture
    def spatial_aggregator(self):
        """Create a SpatialAggregation instance."""
        return SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
    
    @pytest.fixture
    def conservative_remap(self):
        """Create a ConservativeRemap instance."""
        return ConservativeRemap(
            preserve_mass=True,
            preserve_energy=True,
            preserve_positivity=True
        )
    
    @pytest.fixture
    def sample_transport_state(self):
        """Create sample transport state data."""
        return {
            'tracer_concentration': torch.abs(torch.randn(24, 8, 8)) * 1e-6,
            'air_temperature': torch.randn(24, 8, 8) * 5 + 293.15,
            'wind_speed': torch.abs(torch.randn(24, 8, 8) * 3 + 2),  # m/s
            'pressure': torch.ones(24, 8, 8) * 101325.0,  # Pa
            'relative_humidity': torch.clamp(torch.randn(24, 8, 8) * 0.3 + 0.5, 0, 1),
        }

    def test_advection_updates_preserve_update_item_contract(
        self,
        sample_transport_state,
    ):
        """Verify hourly advection outputs without an obsolete temporal facade."""
        velocity_field = (
            torch.ones(8, 8) * 0.1,
            torch.zeros(8, 8),
        )
        advection = LinearAdvection(
            spatial_dims=2,
            velocity_field=velocity_field,
            grid_spacing=(1000.0, 1000.0),
        )

        for hour in range(24):
            outputs = advection.forward(
                state={
                    'tracer_concentration': sample_transport_state[
                        'tracer_concentration'
                    ][hour],
                    'velocity_field': velocity_field,
                },
                time=3600.0 * hour,
                dt=3600.0,
            )
            increment = outputs['tracer_concentration']
            assert isinstance(increment, UpdateItem)
            assert increment.mode is UpdateMode.ACCUMULATE
            assert increment.per_timestep is True

    def test_spatial_coupling_with_diffusion(self, spatial_aggregator, sample_transport_state):
        """Test spatial aggregation with diffusion transport."""
        # Create diffusion functor
        diffusion = FickianDiffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=(1.0, 1.0),
        )
        
        # Calculate diffusion on a single time slice
        concentration_field = sample_transport_state['tracer_concentration'][0]
        
        # Create a simple gradient field for diffusion
        gradient_field = torch.gradient(concentration_field, dim=(0, 1))
        gradient_magnitude = torch.sqrt(gradient_field[0]**2 + gradient_field[1]**2)
        
        # Compute diffusion increment
        updates = diffusion.forward(
            state={"tracer_concentration": concentration_field},
            time=0.0,
            dt=1.0,
        )
        diffusion_increment = updates["tracer_concentration"]
        assert isinstance(diffusion_increment, UpdateItem)
        assert diffusion_increment.mode is UpdateMode.ACCUMULATE
        assert diffusion_increment.per_timestep is True
        
        # Create spatial state for aggregation
        spatial_state = {
            'concentration': concentration_field,
            'diffusion_increment': diffusion_increment.value,
            'air_temperature': sample_transport_state['air_temperature'][0],
            'wind_speed': sample_transport_state['wind_speed'][0]
        }
        
        # Test spatial aggregation (air_temperature is intensive in the YAML catalog)
        aggregated_result = spatial_aggregator.forward(
            state=spatial_state,
            time=0.0,
            dt=1.0,
            variable_metadata={
                'concentration': {'spatial_basis': 'intensive'},
                'diffusion_increment': {'spatial_basis': 'intensive'},
                'wind_speed': {'spatial_basis': 'intensive'},
            },
        )

        # Check results
        assert len(aggregated_result) == 4  # Every declared spatial variable is processed
        
        for key, value in aggregated_result.items():
            assert isinstance(value, torch.Tensor)
            assert value.dim() <= 1  # Should be aggregated to scalar or 1D
    
    def test_coupling_transport_data_formats(self, spatial_aggregator):
        """Test data format consistency between coupling and transport modules."""
        # Create transport-like data with proper units and dimensions
        transport_data = {
            # Time series data (hours x spatial grid)
            'co2_flux': torch.randn(168, 6, 6) * 2e-6,  # One week, hourly, kg/m2/s
            'heat_flux': torch.randn(168, 6, 6) * 50 + 200,  # W/m2
            'momentum_flux': torch.randn(168, 6, 6) * 0.1,  # N/m2
            
            # Static spatial data
            'surface_elevation': torch.randn(6, 6) * 100 + 500,  # m
            'land_cover_type': torch.randint(0, 5, (6, 6)),  # land cover classes
            'surface_roughness': torch.exp(torch.randn(6, 6) * 0.3) * 0.1  # m
        }
        
        # Test spatial aggregation of static fields
        spatial_data = {
            key: value for key, value in transport_data.items() 
            if key not in ['co2_flux', 'heat_flux', 'momentum_flux']
        }
        
        continuous_metadata = {
            'surface_elevation': {'spatial_basis': 'intensive'},
            'surface_roughness': {'spatial_basis': 'intensive'},
        }
        spatial_result = spatial_aggregator.forward(
            state=spatial_data,
            time=0.0,
            dt=1.0,
            variables=['surface_elevation', 'surface_roughness'],
            variable_metadata=continuous_metadata,
        )

        assert len(spatial_result) == 2  # Both continuous static fields are processed
        # Land-cover classes are categorical: they take the (uniformly weighted) mode, ties to the
        # smallest code, and are never averaged.
        land_cover = spatial_aggregator.forward(
            state=spatial_data,
            time=0.0,
            dt=1.0,
            variables=['land_cover_type'],
            variable_metadata={'land_cover_type': {'spatial_basis': 'categorical'}},
        )['land_cover_type_aggregated']
        codes = spatial_data['land_cover_type'].reshape(-1)
        assert land_cover.dtype == codes.dtype
        assert int(land_cover) == int(torch.bincount(codes).argmax())
        
        # Check unit consistency (all values should be reasonable)
        for key, value in spatial_result.items():
            if 'elevation' in key:
                assert torch.all(value > 0)  # Elevation should be positive
            elif 'roughness' in key:
                assert torch.all(value > 0)  # Roughness should be positive
    
    def test_conservation_across_coupling_transport(self, conservative_remap):
        """Test conservation laws across coupling-transport boundaries."""
        # Create test data representing a conserved quantity (e.g., total CO2)
        fine_grid_data = torch.ones(12, 12) * 1e-6  # kg CO2/m2
        
        # Simulate transport redistribution (advection + diffusion)
        # Simple redistribution: rotate and perturb the field
        transport_perturbation = torch.randn(12, 12) * 1e-8
        transported_data = fine_grid_data + transport_perturbation
        
        # Ensure positivity (physical constraint)
        transported_data = torch.clamp(transported_data, min=0)
        
        # Apply conservative remapping to coarser grid
        state = {'co2_stock': transported_data}
        
        remap_result = conservative_remap.forward(
            state=state,
            time=0.0,
            dt=1.0,
            variables=['co2_stock'],
            **_whole_domain_geometry(144, ['co2_stock']),
        )
        
        if 'co2_stock_remapped' in remap_result:
            # Check mass conservation
            original_total = fine_grid_data.sum()
            transported_total = transported_data.sum()
            remapped_total = remap_result['co2_stock_remapped'].sum()
            
            # All totals should be approximately equal (within numerical precision)
            torch.testing.assert_close(original_total, transported_total, rtol=1e-3, atol=1e-6)
            torch.testing.assert_close(
                transported_total.to(remapped_total.device),
                remapped_total,
                rtol=1e-3,
                atol=1e-6,
            )
            
            # Check positivity constraint
            assert torch.all(remap_result['co2_stock_remapped'] >= 0)


class TestBiosphereCouplingIntegration:
    """Test suite for coupling module integration with biosphere module."""
    
    @pytest.fixture
    def sample_biosphere_state(self):
        """Create sample biosphere state data."""
        # Generate physically consistent carbon fluxes
        gpp = torch.abs(torch.randn(30, 5, 5)) * 5e-6  # kg C/m2/s
        auto_resp = torch.abs(torch.randn(30, 5, 5)) * 2e-6  # kg C/m2/s
        # Ensure NPP <= GPP (NPP = GPP - Ra)
        npp = torch.clamp(gpp - auto_resp, min=0)
        
        return {
            # Daily time series for one month
            'gross_primary_productivity': gpp,
            'net_primary_productivity': npp,
            'total_autotrophic_respiration': auto_resp,
            'heterotrophic_respiration': torch.abs(torch.randn(30, 5, 5)) * 1e-6,
            
            # Environmental drivers
            'ppfd': torch.linspace(400.0, 1200.0, 30).view(30, 1, 1).expand(30, 5, 5).clone(),
            'air_temperature': torch.linspace(283.15, 303.15, 30).view(30, 1, 1).expand(30, 5, 5).clone(),
            'vapor_pressure_deficit': torch.linspace(500.0, 2000.0, 30).view(30, 1, 1).expand(30, 5, 5).clone(),
            'atmospheric_co2_ppm': torch.full((30, 5, 5), 420.0),
            'soil_moisture': torch.clamp(torch.randn(30, 5, 5) * 0.2 + 0.3, 0.1, 0.6),
        }

    def test_photosynthesis_daily_updates_preserve_update_item_contract(
        self,
        sample_biosphere_state,
    ):
        """Verify daily photosynthesis outputs without an obsolete temporal facade."""
        photosynthesis = FarquharPhotosynthesis()

        for day in range(30):
            day_state = {
                'air_temperature': sample_biosphere_state['air_temperature'][day],
                'atmospheric_co2_ppm': sample_biosphere_state[
                    'atmospheric_co2_ppm'
                ][day],
                'ppfd': sample_biosphere_state['ppfd'][day],
                'leaf_area_index': torch.ones_like(sample_biosphere_state['ppfd'][day]),
                'vapor_pressure_deficit': sample_biosphere_state[
                    'vapor_pressure_deficit'
                ][day],
            }
            outputs = photosynthesis.forward(
                day_state,
                86400.0 * day,
                86400.0,
            )
            gpp_rate = outputs['gross_primary_productivity']
            assert isinstance(gpp_rate, UpdateItem)
            assert gpp_rate.mode is UpdateMode.REPLACE
            assert gpp_rate.per_timestep is False

    def test_biosphere_spatial_aggregation(self, sample_biosphere_state):
        """Test spatial aggregation of biosphere variables."""
        # Create spatial aggregator
        aggregator = SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
        
        # Use single day data for spatial aggregation
        single_day_state = {
            key: value[0] for key, value in sample_biosphere_state.items()
        }
        
        # Test spatial aggregation on unit-area cells; catalog entries (GPP, NPP, air
        # temperature, VPD, soil moisture) supply their own basis, the rest is declared here.
        result = aggregator.forward(
            state=single_day_state,
            time=0.0,
            dt=1.0,
            weights=torch.ones(5, 5),
            variable_metadata={
                'total_autotrophic_respiration': {'spatial_basis': 'area_density'},
                'heterotrophic_respiration': {'spatial_basis': 'area_density'},
                'ppfd': {'spatial_basis': 'intensive'},
                'atmospheric_co2_ppm': {'spatial_basis': 'intensive'},
            },
        )

        # Check results
        assert len(result) == 9  # Should process all variables
        
        # Check carbon flux conservation
        if (
            'gross_primary_productivity_aggregated' in result
            and 'net_primary_productivity_aggregated' in result
        ):
            # NPP should be less than GPP (NPP = GPP - autotrophic respiration)
            assert torch.all(
                result['net_primary_productivity_aggregated']
                <= result['gross_primary_productivity_aggregated'] + 1e-10
            )
        
        # Check physical constraints
        for key, value in result.items():
            if 'ppfd' in key:
                assert torch.all(value >= 0)  # PAR should be non-negative
            elif 'air_temperature' in key:
                assert torch.all(value > 273.15)  # Temperature should be above freezing
            elif 'soil_moisture' in key:
                assert torch.all(value >= 0) and torch.all(value <= 1)  # Soil moisture should be [0,1]
    
    def test_biosphere_conservation_remapping(self, sample_biosphere_state):
        """Test conservative remapping of biosphere variables."""
        # Create conservative remapper
        remapper = ConservativeRemap(
            preserve_mass=True,  # Preserve carbon mass
            preserve_energy=True,  # Preserve energy balance
            preserve_positivity=True
        )
        
        # Use single day data
        single_day_state = {
            key: value[0] for key, value in sample_biosphere_state.items()
        }
        
        # Test remapping of carbon fluxes
        result = remapper.forward(
            state=single_day_state,
            time=0.0,
            dt=1.0,
            variables=[
                'gross_primary_productivity',
                'net_primary_productivity',
                'total_autotrophic_respiration',
            ],
            # Flux densities [kg C m-2 s-1] are area densities: on the whole-domain target the
            # destarea remap sum_i x_i A_iJ / A_J equals their area mean.
            **_whole_domain_geometry(
                25,
                ['gross_primary_productivity', 'net_primary_productivity', 'total_autotrophic_respiration'],
                spatial_basis="area_density",
            ),
        )
        
        # Check results
        for var in [
            'gross_primary_productivity',
            'net_primary_productivity',
            'total_autotrophic_respiration',
        ]:
            key = f'{var}_remapped'
            if key in result:
                assert torch.all(result[key] >= 0)  # Positivity constraint
                assert not torch.isnan(result[key]).any()  # No NaN values
                assert not torch.isinf(result[key]).any()  # No infinite values


class TestCrossSphereCoupling:
    """Test suite for cross-sphere coupling scenarios."""
    
    @pytest.fixture
    def multi_sphere_state(self):
        """Create sample multi-sphere state data."""
        return {
            # Atmosphere sphere
            'atmosphere_co2': torch.randn(24, 8, 8) * 50 + 400,  # ppm
            'atmosphere_temp': torch.randn(24, 8, 8) * 10 + 293.15,  # K
            'atmosphere_pressure': torch.ones(24, 8, 8) * 101325,  # Pa
            'precipitation': torch.abs(torch.randn(24, 8, 8)) * 5e-6,  # m/s
            
            # Biosphere sphere
            'gpp': torch.abs(torch.randn(24, 8, 8)) * 5e-6,  # kg C/m2/s
            'transpiration': torch.abs(torch.randn(24, 8, 8)) * 1e-7,  # m/s
            'leaf_area_index': torch.abs(torch.randn(8, 8)) * 3 + 1,  # m2/m2
            
            # Pedosphere sphere
            'soil_moisture': torch.clamp(torch.randn(24, 8, 8) * 0.2 + 0.3, 0.1, 0.6),  # m3/m3
            'soil_temperature': torch.randn(24, 8, 8) * 5 + 288.15,  # K
            'soil_carbon': torch.abs(torch.randn(8, 8)) * 10 + 5,  # kg C/m2
            
            # Hydrosphere sphere
            'surface_runoff': torch.abs(torch.randn(24, 8, 8)) * 1e-8,  # m/s
            'water_table_depth': torch.abs(torch.randn(8, 8)) * 5 + 2,  # m
            'evapotranspiration': torch.abs(torch.randn(24, 8, 8)) * 5e-8  # m/s
        }
    
    def test_biosphere_pedosphere_coupling(self, multi_sphere_state):
        """Test coupling between biosphere and pedosphere spheres."""
        # Create spatial aggregator for cross-sphere coupling
        spatial_aggregator = SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
        
        # Extract spatial (non-temporal) variables from both spheres
        spatial_state = {}
        for key, value in multi_sphere_state.items():
            if value.dim() == 2:  # Spatial only
                spatial_state[key] = value
            elif value.dim() == 3:  # Temporal, use first time step
                spatial_state[key] = value[0]
        
        # Test spatial aggregation across spheres on unit-area cells; names outside the YAML
        # catalog declare their basis explicitly.
        result = spatial_aggregator.forward(
            state=spatial_state,
            time=0.0,
            dt=1.0,
            weights=torch.ones(8, 8),
            variable_metadata={
                'atmosphere_co2': {'spatial_basis': 'intensive'},
                'atmosphere_temp': {'spatial_basis': 'intensive'},
                'atmosphere_pressure': {'spatial_basis': 'intensive'},
                'precipitation': {'spatial_basis': 'area_density'},
                'transpiration': {'spatial_basis': 'area_density'},
                'surface_runoff': {'spatial_basis': 'area_density'},
                'evapotranspiration': {'spatial_basis': 'area_density'},
            },
        )
        assert len(result) == len(spatial_state)
        
        # Check results from both spheres
        biosphere_vars = ['leaf_area_index_aggregated']
        pedosphere_vars = ['soil_moisture_aggregated', 'soil_carbon_aggregated']
        
        for var in biosphere_vars + pedosphere_vars:
            if var in result:
                assert isinstance(result[var], torch.Tensor)
                assert torch.all(result[var] >= 0)  # Non-negative
        
        # Check cross-sphere relationships
        # Soil moisture should be in valid range [0, 1]
        if 'soil_moisture_aggregated' in result:
            assert torch.all(result['soil_moisture_aggregated'] >= 0)
            assert torch.all(result['soil_moisture_aggregated'] <= 1)
        
        # LAI should be reasonable
        if 'leaf_area_index_aggregated' in result:
            assert torch.all(result['leaf_area_index_aggregated'] >= 0)
            assert torch.all(result['leaf_area_index_aggregated'] < 10)  # Reasonable LAI range
    
class TestParameterSystemIntegration:
    """Test suite for parameter system integration across modules."""
    
    def test_cross_module_parameter_access(self):
        """Test parameter access across different modules."""
        # Create functors from different modules
        spatial_agg = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
        conservative_remap = ConservativeRemap()
        
        # Test parameter access with different namespaces
        spatial_param = spatial_agg.get_parameter_value(
            'weighting_type',
            units='dimensionless',
            namespace='coupling.scales.spatial'
        )
        
        conservative_param = conservative_remap.get_parameter_value(
            'clamp_min',
            units='dimensionless',
            namespace='coupling.scales.conservative'
        )
        
        # All parameters should be accessible
        assert spatial_param is not None
        assert conservative_param is not None
    
class TestErrorHandlingAcrossModules:
    """Test suite for error handling across module boundaries."""
    
    def test_coupling_biosphere_dimension_mismatch(self):
        """Test handling of dimension mismatches between coupling and biosphere modules."""
        aggregator = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
        
        # Create data with mismatched dimensions
        mismatched_state = {
            'gpp': torch.randn(10, 10),  # 10x10 grid
            'npp': torch.randn(5, 5),     # 5x5 grid - mismatched!
            'temperature': torch.randn(10, 10),
            'soil_moisture': torch.randn(15, 15)  # 15x15 grid - mismatched!
        }
        
        # Should handle dimension mismatches
        try:
            result = aggregator.forward(
                state=mismatched_state,
                time=0.0,
                dt=1.0
            )
            
            # If successful, results should be consistent
            for key, value in result.items():
                assert isinstance(value, torch.Tensor)
                
        except Exception as e:
            # Should fail gracefully with informative error
            assert isinstance(e, (ValueError, RuntimeError))
            logger.info(f"Expected dimension mismatch error: {e}")
    
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
