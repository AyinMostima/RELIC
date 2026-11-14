"""
L3 System-Level Tests for Multi-Scale Aggregation System

Complete end-to-end validation of the hierarchical aggregation system
with realistic multi-scale scenarios, conservation checks, and performance testing.

Test Coverage:
1. Full Hierarchy Aggregation (3 tests)
   - Five-level hierarchical chain (Individual -> Regional)
   - Mixed variable types batch processing (15+ variables)
   - Multi-timestep consistency validation

2. System Conservation (4 tests)
   - Global mass conservation (C/N/H2O across all scales)
   - Energy balance closure
   - Conservation with StateManager transactions
   - Conservation under stress conditions

3. Realistic Scenarios (4 tests)
   - Forest ecosystem simulation (carbon/water cycling)
   - Agricultural landscape (mixed land use)
   - Watershed hydrology (complete hydrological cycle)
   - Multi-biome regional simulation

4. Performance and Stability (3 tests)
   - Large-scale performance (10000+ units)
   - Numerical stability (extreme values)
   - Error propagation analysis

Mathematical Expressions:
    - Conservation law: sum(source_i * A_i) = sum(target_j * A_j)
    - Relative error: epsilon = |target_total - source_total| / |source_total|
    - Energy balance: R_n = H + LE + G (net radiation = sensible + latent + ground heat)
    - Water balance: P = ET + R + Delta_S (precipitation = evapotranspiration + runoff + storage change)
    - Carbon balance: GPP = NPP + R_a (gross primary production = net primary production + autotrophic respiration)

References:
    - Chapin et al. (2011). Principles of Terrestrial Ecosystem Ecology
    - Shuttleworth (2012). Terrestrial Hydrometeorology
    - Law et al. (2002). Environmental control over carbon dioxide and water vapor exchange
    - Band et al. (1993). Forest ecosystem processes at the watershed scale
"""

import pytest
import torch
import math
import time

from relic.coupling.scales.hierarchical import HierarchyIndex, HierarchicalAggregator
from relic.core.state import StateManager, StateVariableSpec, StateVariableType
from relic.coupling.scales.conservation_enforcer import ConservationEnforcer


# Explicit spatial basis (relic.coupling.scales.types.SpatialQuantityBasis): every field
# aggregated with flow_conserving in these scenarios is a per-area flux or stock density.
AREA_DENSITY_METADATA = {
    'carbon': {'spatial_basis': 'area_density'},
    'carbon_flux': {'spatial_basis': 'area_density'},
    'et': {'spatial_basis': 'area_density'},
    'evapotranspiration': {'spatial_basis': 'area_density'},
    'flux': {'spatial_basis': 'area_density'},
    'gpp': {'spatial_basis': 'area_density'},
    'nitrogen': {'spatial_basis': 'area_density'},
    'nitrogen_flux': {'spatial_basis': 'area_density'},
    'npp': {'spatial_basis': 'area_density'},
    'precipitation': {'spatial_basis': 'area_density'},
    'ra': {'spatial_basis': 'area_density'},
    'runoff': {'spatial_basis': 'area_density'},
    'storage_change': {'spatial_basis': 'area_density'},
    'water': {'spatial_basis': 'area_density'},
    'water_flux': {'spatial_basis': 'area_density'},
}


# Spatial basis of every variable these tests aggregate: area densities (per-area fluxes and
# energy fluxes), intensive values (resistances, conductances, fractions, angles) and class
# codes. Undeclared variables raise in the HierarchicalAggregator.
VARIABLE_METADATA = {
    **AREA_DENSITY_METADATA,
    **{name: {'spatial_basis': 'area_density'} for name in (
        'net_radiation', 'sensible_heat', 'latent_heat', 'ground_heat', 'respiration', 'nee',
    )},
    **{name: {'spatial_basis': 'intensive'} for name in (
        'stomatal_resistance', 'aerodynamic_resistance', 'stomatal_conductance', 'hydraulic_conductance',
        'vegetation_fraction', 'canopy_cover', 'soil_moisture', 'wind_direction', 'hydraulic_conductivity',
        'leaf_temperature', 'lai', 'test_var',
    )},
    **{name: {'spatial_basis': 'categorical'} for name in (
        'vegetation_type', 'soil_type', 'land_cover_type', 'crop_type', 'biome',
    )},
}


class TestFullHierarchyAggregation:
    """Test complete hierarchical aggregation chains across all scales"""

    @pytest.fixture
    def five_level_hierarchy(self):
        """
        Setup five-level hierarchy: Individual -> Patch -> Landscape -> Basin -> Regional
        
        Structure:
        - 1000 individuals (trees/plants)
        - 100 patches (10 individuals each)
        - 10 landscapes (10 patches each)
        - 2 basins (5 landscapes each)
        - 1 region (2 basins)
        """
        device = torch.device("cpu")
        
        # Define hierarchy structure
        n_individuals = 1000
        n_patches = 100
        n_landscapes = 10
        n_basins = 2
        n_regional = 1
        
        individuals_per_patch = n_individuals // n_patches
        patches_per_landscape = n_patches // n_landscapes
        landscapes_per_basin = n_landscapes // n_basins
        basins_per_region = n_basins // n_regional
        
        # Generate realistic area distributions (log-normal)
        torch.manual_seed(42)  # Reproducibility
        individual_areas = torch.exp(torch.randn(n_individuals) * 0.3 + 2.0)  # Mean ~7.4 m2
        
        # Build topology for each level
        topologies = {}
        indices = {}
        aggregators = {}
        
        # Level 1: Individual -> Patch
        patch_areas = torch.zeros(n_patches)
        parent_map_ind_patch = {}
        for i in range(n_individuals):
            patch_id = i // individuals_per_patch
            parent_map_ind_patch[i] = patch_id
            patch_areas[patch_id] += individual_areas[i].item()
        
        topologies['ind_patch'] = {
            'n_source_units': n_individuals,
            'n_target_units': n_patches,
            'source_areas': individual_areas.tolist(),
            'target_areas': patch_areas.tolist(),
            'parent_map': parent_map_ind_patch
        }
        
        # Level 2: Patch -> Landscape
        landscape_areas = torch.zeros(n_landscapes)
        parent_map_patch_land = {}
        for i in range(n_patches):
            landscape_id = i // patches_per_landscape
            parent_map_patch_land[i] = landscape_id
            landscape_areas[landscape_id] += patch_areas[i].item()
        
        topologies['patch_land'] = {
            'n_source_units': n_patches,
            'n_target_units': n_landscapes,
            'source_areas': patch_areas.tolist(),
            'target_areas': landscape_areas.tolist(),
            'parent_map': parent_map_patch_land
        }
        
        # Level 3: Landscape -> Basin
        basin_areas = torch.zeros(n_basins)
        parent_map_land_basin = {}
        for i in range(n_landscapes):
            basin_id = i // landscapes_per_basin
            parent_map_land_basin[i] = basin_id
            basin_areas[basin_id] += landscape_areas[i].item()
        
        topologies['land_basin'] = {
            'n_source_units': n_landscapes,
            'n_target_units': n_basins,
            'source_areas': landscape_areas.tolist(),
            'target_areas': basin_areas.tolist(),
            'parent_map': parent_map_land_basin
        }
        
        # Level 4: Basin -> Regional
        regional_areas = torch.zeros(n_regional)
        parent_map_basin_region = {}
        for i in range(n_basins):
            parent_map_basin_region[i] = 0
            regional_areas[0] += basin_areas[i].item()
        
        topologies['basin_region'] = {
            'n_source_units': n_basins,
            'n_target_units': n_regional,
            'source_areas': basin_areas.tolist(),
            'target_areas': regional_areas.tolist(),
            'parent_map': parent_map_basin_region
        }
        
        # Create hierarchy indices and aggregators for each level
        level_pairs = [
            ('individual', 'patch', 'ind_patch'),
            ('patch', 'landscape', 'patch_land'),
            ('landscape', 'basin', 'land_basin'),
            ('basin', 'regional', 'basin_region')
        ]
        
        for source_scale, target_scale, topo_key in level_pairs:
            indices[topo_key] = HierarchyIndex(
                source_scale=source_scale,
                target_scale=target_scale,
                device=device,
                topology=topologies[topo_key]
            )
            
            aggregators[topo_key] = HierarchicalAggregator(
                index=indices[topo_key],
                device=device,
                hierarchy_levels=[source_scale, target_scale],
                weight_type='area',
                rule='conservative'
            )
        
        return {
            'device': device,
            'n_individuals': n_individuals,
            'n_patches': n_patches,
            'n_landscapes': n_landscapes,
            'n_basins': n_basins,
            'n_regional': n_regional,
            'individual_areas': individual_areas,
            'topologies': topologies,
            'indices': indices,
            'aggregators': aggregators
        }

    def test_five_level_aggregation_chain(self, five_level_hierarchy):
        """
        Test complete Individual -> Regional aggregation with conservation.
        
        Validates that extensive quantities are conserved through all five levels
        and intensive quantities remain within physical bounds.
        """
        device = five_level_hierarchy['device']
        n_individuals = five_level_hierarchy['n_individuals']
        individual_areas = five_level_hierarchy['individual_areas']
        aggregators = five_level_hierarchy['aggregators']
        topologies = five_level_hierarchy['topologies']
        
        # Create test data at individual level (extensive quantity: carbon flux density)
        carbon_flux = torch.rand(n_individuals, device=device) * 10.0  # 0-10 g C/m2/s
        
        # Calculate total carbon flux (flux_density * area)
        initial_total_carbon = (carbon_flux * individual_areas).sum()
        
        # Aggregate through all levels
        current_data = carbon_flux
        current_areas = individual_areas
        
        aggregation_levels = ['ind_patch', 'patch_land', 'land_basin', 'basin_region']
        
        for level_key in aggregation_levels:
            aggregator = aggregators[level_key]
            topology = topologies[level_key]
            
            # Aggregate
            data_dict = {'carbon_flux': current_data}
            aggregated, audit = aggregator.aggregate(
                data=data_dict,
                variables=['carbon_flux'],
                strategy='flow_conserving',
                variable_metadata=VARIABLE_METADATA,
                mode='conservative'
            )
            
            current_data = aggregated['carbon_flux']
            current_areas = torch.tensor(topology['target_areas'], device=device)
            
            # Check conservation at this level
            current_total = (current_data * current_areas).sum()
            relative_error = abs(current_total - initial_total_carbon) / initial_total_carbon
            
            assert relative_error < 1e-6, \
                f"Conservation violated at {level_key}: relative error = {relative_error:.2e}"
        
        # Final check: regional total should equal initial total
        regional_carbon = current_data[0]
        regional_area = current_areas[0]
        final_total = regional_carbon * regional_area
        
        final_relative_error = abs(final_total - initial_total_carbon) / initial_total_carbon
        
        assert final_relative_error < 1e-6, \
            f"Final conservation error: {final_relative_error:.2e}"
        
        # Verify regional flux density is within reasonable bounds
        assert 0 <= regional_carbon <= 10.0, \
            f"Regional carbon flux {regional_carbon:.2f} out of bounds"

    def test_mixed_variable_types_batch(self, five_level_hierarchy):
        """
        Process 15+ variables with different aggregation strategies simultaneously.
        
        Tests all 7 new aggregation strategies plus standard methods on a
        diverse set of ecosystem variables.
        """
        device = five_level_hierarchy['device']
        n_individuals = five_level_hierarchy['n_individuals']
        aggregators = five_level_hierarchy['aggregators']
        
        # Create 15 test variables with diverse characteristics
        torch.manual_seed(123)
        
        individual_data = {
            # Extensive quantities (conserved)
            'gpp': torch.rand(n_individuals, device=device) * 20.0,  # g C/m2/s
            'npp': torch.rand(n_individuals, device=device) * 10.0,
            'nitrogen_flux': torch.rand(n_individuals, device=device) * 2.0,  # g N/m2/s
            'water_flux': torch.rand(n_individuals, device=device) * 5.0,  # kg/m2/s
            
            # Intensive quantities - resistance/impedance (harmonic mean)
            'stomatal_resistance': torch.rand(n_individuals, device=device) * 100 + 50,  # s/m
            'aerodynamic_resistance': torch.rand(n_individuals, device=device) * 50 + 20,
            
            # Intensive quantities - conductance (arithmetic mean)
            'stomatal_conductance': torch.rand(n_individuals, device=device) * 0.5,  # m/s
            'hydraulic_conductance': torch.rand(n_individuals, device=device) * 0.1,
            
            # Bounded fractions (bounded area mean)
            'vegetation_fraction': torch.rand(n_individuals, device=device),  # 0-1
            'canopy_cover': torch.rand(n_individuals, device=device),
            'soil_moisture': torch.rand(n_individuals, device=device) * 0.5,  # 0-0.5
            
            # Categorical (area-weighted mode)
            'vegetation_type': torch.randint(1, 6, (n_individuals,), device=device, dtype=torch.float32),
            'soil_type': torch.randint(1, 4, (n_individuals,), device=device, dtype=torch.float32),
            
            # Angular (circular mean)
            'wind_direction': torch.rand(n_individuals, device=device) * 2 * math.pi,  # radians
            
            # Log-normal (geometric mean)
            'hydraulic_conductivity': torch.exp(torch.randn(n_individuals, device=device) * 0.5 - 5),  # m/s
        }
        
        # Strategy mapping
        strategies = {
            'gpp': 'flow_conserving',
            'npp': 'flow_conserving',
            'nitrogen_flux': 'flow_conserving',
            'water_flux': 'flow_conserving',
            'stomatal_resistance': 'harmonic_mean',
            'aerodynamic_resistance': 'harmonic_mean',
            'stomatal_conductance': 'arithmetic_mean',
            'hydraulic_conductance': 'arithmetic_mean',
            'vegetation_fraction': 'bounded_area_mean',
            'canopy_cover': 'bounded_area_mean',
            'soil_moisture': 'bounded_area_mean',
            'vegetation_type': 'mode',
            'soil_type': 'mode',
            'wind_direction': 'circular_mean',
            'hydraulic_conductivity': 'geometric_mean',
        }
        
        # Aggregate all variables through first level (Individual -> Patch)
        aggregator = aggregators['ind_patch']
        
        patch_data = {}
        all_audits = {}
        
        for var_name, var_data in individual_data.items():
            data_dict = {var_name: var_data}
            strategy = strategies[var_name]
            
            aggregated, audit = aggregator.aggregate(
                data=data_dict,
                variables=[var_name],
                strategy=strategy,
                variable_metadata=VARIABLE_METADATA,
                mode='conservative'
            )
            
            patch_data[var_name] = aggregated[var_name]
            all_audits[var_name] = audit
        
        # Verify all aggregations succeeded
        assert len(patch_data) == 15, "Not all variables were aggregated"
        
        # Verify shapes
        for var_name, result in patch_data.items():
            assert result.shape[0] == five_level_hierarchy['n_patches'], \
                f"{var_name} has wrong shape: {result.shape}"
        
        # Verify no NaN or Inf
        for var_name, result in patch_data.items():
            assert not torch.isnan(result).any(), f"{var_name} contains NaN"
            assert not torch.isinf(result).any(), f"{var_name} contains Inf"
        
        # Verify bounded variables are within bounds
        for var_name in ['vegetation_fraction', 'canopy_cover', 'soil_moisture']:
            assert patch_data[var_name].min() >= 0.0, f"{var_name} has negative values"
            max_expected = 1.0 if var_name != 'soil_moisture' else 0.5
            assert patch_data[var_name].max() <= max_expected, \
                f"{var_name} exceeds maximum: {patch_data[var_name].max():.3f} > {max_expected}"
        
        # Verify categorical variables
        for var_name in ['vegetation_type', 'soil_type']:
            min_expected = 1.0
            max_expected = 6.0 if var_name == 'vegetation_type' else 4.0
            assert patch_data[var_name].min() >= min_expected
            assert patch_data[var_name].max() <= max_expected
        
        # Verify angular variables
        assert patch_data['wind_direction'].min() >= -math.pi
        assert patch_data['wind_direction'].max() <= math.pi
        
        # Verify correct strategies were used
        for var_name, expected_strategy in strategies.items():
            actual_strategy = all_audits[var_name]['method_used'][var_name]
            assert actual_strategy == expected_strategy, \
                f"{var_name}: expected {expected_strategy}, got {actual_strategy}"

    def test_multi_timestep_consistency(self, five_level_hierarchy):
        """
        Verify aggregation consistency across multiple timesteps.
        
        Tests that aggregation produces stable results when applied repeatedly
        with changing input data over time.
        """
        device = five_level_hierarchy['device']
        n_individuals = five_level_hierarchy['n_individuals']
        individual_areas = five_level_hierarchy['individual_areas']
        aggregator = five_level_hierarchy['aggregators']['ind_patch']
        topology = five_level_hierarchy['topologies']['ind_patch']
        
        n_timesteps = 20
        dt = 1800.0  # 30 minutes
        
        # Track conservation over time
        conservation_errors = []
        
        # Simulate diurnal cycle of photosynthesis
        for t in range(n_timesteps):
            # Time of day factor (sinusoidal diurnal cycle)
            hour = (t * dt / 3600.0) % 24
            light_factor = max(0, math.sin((hour - 6) * math.pi / 12))
            
            # GPP with diurnal variation
            base_gpp = torch.rand(n_individuals, device=device) * 10.0
            gpp = base_gpp * light_factor
            
            # Aggregate
            data_dict = {'gpp': gpp}
            aggregated, audit = aggregator.aggregate(
                data=data_dict,
                variables=['gpp'],
                strategy='flow_conserving',
                variable_metadata=VARIABLE_METADATA,
                mode='conservative'
            )
            
            # Check conservation
            source_total = (gpp * individual_areas).sum()
            patch_areas = torch.tensor(topology['target_areas'], device=device)
            target_total = (aggregated['gpp'] * patch_areas).sum()
            
            relative_error = abs(target_total - source_total) / (source_total + 1e-10)
            conservation_errors.append(relative_error.item())
            
            # Assert conservation at each timestep
            assert relative_error < 1e-6, \
                f"Conservation violated at timestep {t}: error = {relative_error:.2e}"
        
        # Verify conservation error doesn't accumulate
        mean_error = sum(conservation_errors) / len(conservation_errors)
        max_error = max(conservation_errors)
        
        assert mean_error < 1e-7, f"Mean conservation error too high: {mean_error:.2e}"
        assert max_error < 1e-6, f"Max conservation error too high: {max_error:.2e}"
        
        # Verify no systematic drift
        early_errors = conservation_errors[:5]
        late_errors = conservation_errors[-5:]
        
        drift = abs(sum(late_errors) / 5 - sum(early_errors) / 5)
        # Realistic tolerance for drift over 20 timesteps
        assert drift < 1e-7, f"Conservation error drift detected: {drift:.2e}"


class TestSystemConservation:
    """Test global conservation laws across the complete system"""

    @pytest.fixture
    def conservation_setup(self):
        """Setup for system-wide conservation testing"""
        device = torch.device("cpu")
        
        # Three-level hierarchy for conservation testing
        n_patches = 100
        n_landscapes = 10
        n_regions = 1
        
        patches_per_landscape = n_patches // n_landscapes
        
        # Realistic area distribution
        patch_areas = torch.exp(torch.randn(n_patches) * 0.2 + 3.0)  # ~20 m2 mean
        
        # Build hierarchy
        landscape_areas = torch.zeros(n_landscapes)
        parent_map_patch_land = {}
        for i in range(n_patches):
            landscape_id = i // patches_per_landscape
            parent_map_patch_land[i] = landscape_id
            landscape_areas[landscape_id] += patch_areas[i].item()
        
        topology_patch_land = {
            'n_source_units': n_patches,
            'n_target_units': n_landscapes,
            'source_areas': patch_areas.tolist(),
            'target_areas': landscape_areas.tolist(),
            'parent_map': parent_map_patch_land
        }
        
        regional_areas = torch.tensor([landscape_areas.sum().item()])
        parent_map_land_region = {i: 0 for i in range(n_landscapes)}
        
        topology_land_region = {
            'n_source_units': n_landscapes,
            'n_target_units': n_regions,
            'source_areas': landscape_areas.tolist(),
            'target_areas': regional_areas.tolist(),
            'parent_map': parent_map_land_region
        }
        
        # Create indices and aggregators
        index_patch_land = HierarchyIndex(
            source_scale='patch',
            target_scale='landscape',
            device=device,
            topology=topology_patch_land
        )
        
        index_land_region = HierarchyIndex(
            source_scale='landscape',
            target_scale='regional',
            device=device,
            topology=topology_land_region
        )
        
        agg_patch_land = HierarchicalAggregator(
            index=index_patch_land,
            device=device,
            hierarchy_levels=['patch', 'landscape'],
            weight_type='area',
            rule='strict_conservative'
        )
        
        agg_land_region = HierarchicalAggregator(
            index=index_land_region,
            device=device,
            hierarchy_levels=['landscape', 'regional'],
            weight_type='area',
            rule='strict_conservative'
        )
        
        # Conservation enforcer (each check passes its tolerance)
        enforcer = ConservationEnforcer(
            conservation_properties=['carbon', 'nitrogen', 'water'],
            track_history=True,
            device=device
        )
        
        return {
            'device': device,
            'n_patches': n_patches,
            'n_landscapes': n_landscapes,
            'n_regions': n_regions,
            'patch_areas': patch_areas,
            'landscape_areas': landscape_areas,
            'regional_areas': regional_areas,
            'topology_patch_land': topology_patch_land,
            'topology_land_region': topology_land_region,
            'index_patch_land': index_patch_land,
            'index_land_region': index_land_region,
            'agg_patch_land': agg_patch_land,
            'agg_land_region': agg_land_region,
            'enforcer': enforcer
        }

    def test_global_mass_conservation(self, conservation_setup):
        """
        Test global mass conservation for carbon, nitrogen, and water.
        
        Mathematical Expression:
            M_total = sum_i(rho_i * V_i) = constant
            where rho is density and V is volume
        """
        device = conservation_setup['device']
        n_patches = conservation_setup['n_patches']
        patch_areas = conservation_setup['patch_areas']
        agg_patch_land = conservation_setup['agg_patch_land']
        agg_land_region = conservation_setup['agg_land_region']
        enforcer = conservation_setup['enforcer']
        
        # Create mass flux densities at patch level
        carbon_flux = torch.rand(n_patches, device=device) * 5.0  # g C/m2/s
        nitrogen_flux = torch.rand(n_patches, device=device) * 0.5  # g N/m2/s
        water_flux = torch.rand(n_patches, device=device) * 2.0  # kg H2O/m2/s
        
        # Calculate initial totals
        initial_carbon = (carbon_flux * patch_areas).sum()
        initial_nitrogen = (nitrogen_flux * patch_areas).sum()
        initial_water = (water_flux * patch_areas).sum()
        
        # Stage 1: Patch -> Landscape
        carbon_land, _ = agg_patch_land.aggregate(
            data={'carbon': carbon_flux},
            variables=['carbon'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        nitrogen_land, _ = agg_patch_land.aggregate(
            data={'nitrogen': nitrogen_flux},
            variables=['nitrogen'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        water_land, _ = agg_patch_land.aggregate(
            data={'water': water_flux},
            variables=['water'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        # Check landscape conservation
        landscape_areas = conservation_setup['landscape_areas']
        
        landscape_carbon = (carbon_land['carbon'] * landscape_areas).sum()
        landscape_nitrogen = (nitrogen_land['nitrogen'] * landscape_areas).sum()
        landscape_water = (water_land['water'] * landscape_areas).sum()
        
        carbon_error_land = abs(landscape_carbon - initial_carbon) / initial_carbon
        nitrogen_error_land = abs(landscape_nitrogen - initial_nitrogen) / initial_nitrogen
        water_error_land = abs(landscape_water - initial_water) / initial_water
        
        assert carbon_error_land < 1e-6, f"Carbon conservation error at landscape: {carbon_error_land:.2e}"
        assert nitrogen_error_land < 1e-6, f"Nitrogen conservation error at landscape: {nitrogen_error_land:.2e}"
        assert water_error_land < 1e-6, f"Water conservation error at landscape: {water_error_land:.2e}"
        
        # Stage 2: Landscape -> Regional
        carbon_region, _ = agg_land_region.aggregate(
            data={'carbon': carbon_land['carbon']},
            variables=['carbon'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        nitrogen_region, _ = agg_land_region.aggregate(
            data={'nitrogen': nitrogen_land['nitrogen']},
            variables=['nitrogen'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        water_region, _ = agg_land_region.aggregate(
            data={'water': water_land['water']},
            variables=['water'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        # Check regional conservation
        regional_areas = conservation_setup['regional_areas']
        
        regional_carbon = (carbon_region['carbon'] * regional_areas).sum()
        regional_nitrogen = (nitrogen_region['nitrogen'] * regional_areas).sum()
        regional_water = (water_region['water'] * regional_areas).sum()
        
        carbon_error_region = abs(regional_carbon - initial_carbon) / initial_carbon
        nitrogen_error_region = abs(regional_nitrogen - initial_nitrogen) / initial_nitrogen
        water_error_region = abs(regional_water - initial_water) / initial_water
        
        assert carbon_error_region < 1e-6, f"Carbon conservation error at regional: {carbon_error_region:.2e}"
        assert nitrogen_error_region < 1e-6, f"Nitrogen conservation error at regional: {nitrogen_error_region:.2e}"
        assert water_error_region < 1e-6, f"Water conservation error at regional: {water_error_region:.2e}"
        
        # Use conservation enforcer to verify
        carbon_diag = enforcer.check_conservation(
            source_data=carbon_flux,
            target_data=carbon_region['carbon'],
            property_name='carbon',
            source_weights=patch_areas,
            target_weights=regional_areas,
            tolerance=1e-6,
            batch_ndim=0,
        )
        
        assert carbon_diag.is_conserved, f"Carbon conservation enforcer failed: {carbon_diag.relative_error:.2e}"

    def test_energy_balance_closure(self, conservation_setup):
        """
        Test energy balance: R_n = H + LE + G
        
        Mathematical Expression:
            Net radiation = Sensible heat + Latent heat + Ground heat flux
            R_n = H + LE + G [W/m2]
        
        References:
            - Shuttleworth (2012). Terrestrial Hydrometeorology
            - Foken (2008). Micrometeorology
        """
        device = conservation_setup['device']
        n_patches = conservation_setup['n_patches']
        patch_areas = conservation_setup['patch_areas']
        agg_patch_land = conservation_setup['agg_patch_land']
        
        # Create realistic energy flux components at patch level
        net_radiation = torch.rand(n_patches, device=device) * 600 + 200  # 200-800 W/m2
        
        # Partition net radiation into components (using Bowen ratio approach)
        bowen_ratio = torch.rand(n_patches, device=device) * 2.0  # 0-2
        ground_fraction = torch.rand(n_patches, device=device) * 0.2  # 0-20% to ground
        
        # Calculate components ensuring balance
        ground_heat = net_radiation * ground_fraction
        available_energy = net_radiation - ground_heat
        
        sensible_heat = available_energy * bowen_ratio / (1 + bowen_ratio)
        latent_heat = available_energy - sensible_heat
        
        # Verify patch-level balance
        energy_balance_patch = net_radiation - (sensible_heat + latent_heat + ground_heat)
        assert torch.allclose(energy_balance_patch, torch.zeros_like(energy_balance_patch), atol=1e-4)
        
        # Aggregate all components to landscape
        rn_land, _ = agg_patch_land.aggregate(
            data={'net_radiation': net_radiation},
            variables=['net_radiation'],
            strategy='area_weighted_mean',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        h_land, _ = agg_patch_land.aggregate(
            data={'sensible_heat': sensible_heat},
            variables=['sensible_heat'],
            strategy='area_weighted_mean',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        le_land, _ = agg_patch_land.aggregate(
            data={'latent_heat': latent_heat},
            variables=['latent_heat'],
            strategy='area_weighted_mean',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        g_land, _ = agg_patch_land.aggregate(
            data={'ground_heat': ground_heat},
            variables=['ground_heat'],
            strategy='area_weighted_mean',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        # Verify landscape-level energy balance
        rn_landscape = rn_land['net_radiation']
        h_landscape = h_land['sensible_heat']
        le_landscape = le_land['latent_heat']
        g_landscape = g_land['ground_heat']
        
        energy_balance_land = rn_landscape - (h_landscape + le_landscape + g_landscape)
        
        # Check closure with realistic tolerance (1 W/m2 or 1%)
        absolute_closure_error = energy_balance_land.abs().max()
        relative_closure_error = (energy_balance_land.abs() / rn_landscape).max()
        
        assert absolute_closure_error < 1.0, \
            f"Energy balance closure error: {absolute_closure_error:.2f} W/m2"
        assert relative_closure_error < 0.01, \
            f"Relative energy balance error: {relative_closure_error:.2%}"

    def test_conservation_with_state_updates(self, conservation_setup):
        """
        Test conservation when aggregated results are applied to StateManager.
        
        Validates that StateManager transaction system preserves conservation
        during state updates.
        """
        device = conservation_setup['device']
        n_patches = conservation_setup['n_patches']
        n_landscapes = conservation_setup['n_landscapes']
        
        # Create StateManager with schema
        schema = {
            'carbon_flux_patch': StateVariableSpec(
                name='carbon_flux_patch',
                units='g C m-2 s-1',
                bounds=(0.0, 100.0),
                conserved=True,
                spatial_dims=1,
                required=False,
                variable_type=StateVariableType.GRID
            ),
            'carbon_flux_landscape': StateVariableSpec(
                name='carbon_flux_landscape',
                units='g C m-2 s-1',
                bounds=(0.0, 100.0),
                conserved=True,
                spatial_dims=1,
                required=False,
                variable_type=StateVariableType.GRID
            ),
        }
        
        state_manager = StateManager(schema=schema, device=device)
        
        # Initialize state
        initial_carbon_patch = torch.rand(n_patches, device=device) * 10.0
        initial_state = {
            'carbon_flux_patch': initial_carbon_patch,
            'carbon_flux_landscape': torch.zeros(n_landscapes, device=device),
        }
        state_manager.initialize(initial_state)
        
        # Aggregate patch to landscape
        agg = conservation_setup['agg_patch_land']
        aggregated, audit = agg.aggregate(
            data={'carbon': initial_carbon_patch},
            variables=['carbon'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        # Apply updates through StateManager
        updates = {
            'carbon_flux_landscape': aggregated['carbon'],
        }
        
        state_manager.apply_updates(
            updates=updates,
            dt=1800.0,
            check_conservation=False,  # We'll check manually
            rollback_on_failure=True
        )
        
        # Verify state was updated correctly
        new_state = state_manager.get_state()
        
        # Convert to same dtype for comparison (StateManager may use float32)
        assert torch.allclose(
            new_state['carbon_flux_landscape'].to(dtype=aggregated['carbon'].dtype),
            aggregated['carbon'],
            atol=1e-6
        )
        
        # Verify conservation across state variables
        patch_areas = conservation_setup['patch_areas']
        landscape_areas = conservation_setup['landscape_areas']
        
        patch_total = (new_state['carbon_flux_patch'] * patch_areas).sum()
        landscape_total = (new_state['carbon_flux_landscape'] * landscape_areas).sum()
        
        conservation_error = abs(landscape_total - patch_total) / patch_total
        
        assert conservation_error < 1e-6, \
            f"Conservation violated in StateManager: {conservation_error:.2e}"

    def test_conservation_under_stress(self, conservation_setup):
        """
        Test conservation under extreme conditions and stress scenarios.
        
        Validates that conservation holds even with:
        - Near-zero values
        - Large gradients
        - Sparse distributions
        """
        device = conservation_setup['device']
        n_patches = conservation_setup['n_patches']
        patch_areas = conservation_setup['patch_areas']
        agg = conservation_setup['agg_patch_land']
        landscape_areas = conservation_setup['landscape_areas']
        
        # Test Case 1: Very small values (near detection limit)
        small_flux = torch.rand(n_patches, device=device) * 1e-8  # Very small
        small_aggregated, _ = agg.aggregate(
            data={'flux': small_flux},
            variables=['flux'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        small_source = (small_flux * patch_areas).sum()
        small_target = (small_aggregated['flux'] * landscape_areas).sum()
        small_error = abs(small_target - small_source) / (small_source + 1e-12)
        
        assert small_error < 1e-5, f"Conservation failed for small values: {small_error:.2e}"
        
        # Test Case 2: Large gradients (heterogeneous landscape)
        heterogeneous = torch.zeros(n_patches, device=device)
        heterogeneous[:10] = 100.0  # Hot spots
        heterogeneous[10:] = 0.1    # Background
        
        hetero_aggregated, _ = agg.aggregate(
            data={'flux': heterogeneous},
            variables=['flux'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        hetero_source = (heterogeneous * patch_areas).sum()
        hetero_target = (hetero_aggregated['flux'] * landscape_areas).sum()
        hetero_error = abs(hetero_target - hetero_source) / hetero_source
        
        assert hetero_error < 1e-6, f"Conservation failed for gradients: {hetero_error:.2e}"
        
        # Test Case 3: Sparse distribution (mostly zeros)
        sparse = torch.zeros(n_patches, device=device)
        sparse[::10] = torch.rand(n_patches // 10, device=device) * 50.0  # 10% non-zero
        
        sparse_aggregated, _ = agg.aggregate(
            data={'flux': sparse},
            variables=['flux'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='strict_conservative'
        )
        
        sparse_source = (sparse * patch_areas).sum()
        sparse_target = (sparse_aggregated['flux'] * landscape_areas).sum()
        sparse_error = abs(sparse_target - sparse_source) / (sparse_source + 1e-12)
        
        assert sparse_error < 1e-6, f"Conservation failed for sparse: {sparse_error:.2e}"


class TestRealisticScenarios:
    """Test realistic ecosystem and hydrological scenarios"""

    @pytest.fixture
    def ecosystem_setup(self):
        """Setup realistic ecosystem hierarchy"""
        device = torch.device("cpu")
        
        # 200 trees -> 20 plots -> 4 stands -> 1 forest
        n_trees = 200
        n_plots = 20
        n_stands = 4
        n_forest = 1
        
        trees_per_plot = n_trees // n_plots
        plots_per_stand = n_plots // n_stands
        
        # Realistic tree sizes (DBH-based area)
        tree_dbh = torch.exp(torch.randn(n_trees) * 0.3 + 3.0)  # DBH in cm
        tree_areas = math.pi * (tree_dbh / 200) ** 2  # Crown area from DBH
        
        # Build hierarchy
        plot_areas = torch.zeros(n_plots)
        parent_map_tree_plot = {}
        for i in range(n_trees):
            plot_id = i // trees_per_plot
            parent_map_tree_plot[i] = plot_id
            plot_areas[plot_id] += tree_areas[i].item()
        
        topo_tree_plot = {
            'n_source_units': n_trees,
            'n_target_units': n_plots,
            'source_areas': tree_areas.tolist(),
            'target_areas': plot_areas.tolist(),
            'parent_map': parent_map_tree_plot
        }
        
        stand_areas = torch.zeros(n_stands)
        parent_map_plot_stand = {}
        for i in range(n_plots):
            stand_id = i // plots_per_stand
            parent_map_plot_stand[i] = stand_id
            stand_areas[stand_id] += plot_areas[i].item()
        
        topo_plot_stand = {
            'n_source_units': n_plots,
            'n_target_units': n_stands,
            'source_areas': plot_areas.tolist(),
            'target_areas': stand_areas.tolist(),
            'parent_map': parent_map_plot_stand
        }
        
        forest_area = stand_areas.sum()
        parent_map_stand_forest = {i: 0 for i in range(n_stands)}
        
        topo_stand_forest = {
            'n_source_units': n_stands,
            'n_target_units': n_forest,
            'source_areas': stand_areas.tolist(),
            'target_areas': [forest_area.item()],
            'parent_map': parent_map_stand_forest
        }
        
        # Create aggregators
        index_tree_plot = HierarchyIndex('tree', 'plot', device, topology=topo_tree_plot)
        index_plot_stand = HierarchyIndex('plot', 'stand', device, topology=topo_plot_stand)
        index_stand_forest = HierarchyIndex('stand', 'forest', device, topology=topo_stand_forest)
        
        agg_tree_plot = HierarchicalAggregator(index_tree_plot, device, ['tree', 'plot'], weight_type='area', rule='conservative')
        agg_plot_stand = HierarchicalAggregator(index_plot_stand, device, ['plot', 'stand'], weight_type='area', rule='conservative')
        agg_stand_forest = HierarchicalAggregator(index_stand_forest, device, ['stand', 'forest'], weight_type='area', rule='conservative')
        
        return {
            'device': device,
            'n_trees': n_trees,
            'n_plots': n_plots,
            'n_stands': n_stands,
            'tree_areas': tree_areas,
            'plot_areas': plot_areas,
            'stand_areas': stand_areas,
            'forest_area': forest_area,
            'agg_tree_plot': agg_tree_plot,
            'agg_plot_stand': agg_plot_stand,
            'agg_stand_forest': agg_stand_forest,
        }

    def test_forest_ecosystem_simulation(self, ecosystem_setup):
        """
        Simulate complete forest carbon and water cycling.
        
        Tests realistic forest processes:
        - Photosynthesis (GPP)
        - Autotrophic respiration (R_a)
        - Transpiration (E_t)
        - Water use efficiency (WUE)
        
        References:
            - Law et al. (2002). Environmental control over CO2 and water vapor exchange
            - Chapin et al. (2011). Terrestrial Ecosystem Ecology
        """
        device = ecosystem_setup['device']
        n_trees = ecosystem_setup['n_trees']
        tree_areas = ecosystem_setup['tree_areas']
        
        # Realistic tree-level fluxes
        torch.manual_seed(456)
        
        # GPP: 5-25 umol CO2/m2/s converted to g C/m2/s
        gpp_umol = torch.rand(n_trees, device=device) * 20 + 5
        gpp = gpp_umol * 12.01 / 1e6  # Convert umol/m2/s to g C/m2/s
        
        # Autotrophic respiration: 30-50% of GPP
        ra_fraction = torch.rand(n_trees, device=device) * 0.2 + 0.3
        ra = gpp * ra_fraction
        
        # NPP = GPP - R_a
        npp = gpp - ra
        
        # Transpiration: 1-5 mmol H2O/m2/s converted to kg/m2/s
        et_mmol = torch.rand(n_trees, device=device) * 4 + 1
        et = et_mmol * 18.015 / 1e6  # Convert mmol/m2/s to kg H2O/m2/s
        
        # Water Use Efficiency: GPP/ET (umol CO2 / mmol H2O)
        wue = gpp_umol / (et_mmol + 1e-6)
        
        # Aggregate all variables through forest hierarchy
        variables = {
            'gpp': gpp,
            'ra': ra,
            'npp': npp,
            'et': et,
        }
        
        strategies = {
            'gpp': 'flow_conserving',
            'ra': 'flow_conserving',
            'npp': 'flow_conserving',
            'et': 'flow_conserving',
        }
        
        # Stage 1: Tree -> Plot
        plot_data = {}
        for var_name, var_data in variables.items():
            aggregated, _ = ecosystem_setup['agg_tree_plot'].aggregate(
                data={var_name: var_data},
                variables=[var_name],
                strategy=strategies[var_name],
                variable_metadata=VARIABLE_METADATA,
                mode='conservative'
            )
            plot_data[var_name] = aggregated[var_name]
        
        # Stage 2: Plot -> Stand
        stand_data = {}
        for var_name, var_data in plot_data.items():
            aggregated, _ = ecosystem_setup['agg_plot_stand'].aggregate(
                data={var_name: var_data},
                variables=[var_name],
                strategy=strategies[var_name],
                variable_metadata=VARIABLE_METADATA,
                mode='conservative'
            )
            stand_data[var_name] = aggregated[var_name]
        
        # Stage 3: Stand -> Forest
        forest_data = {}
        for var_name, var_data in stand_data.items():
            aggregated, _ = ecosystem_setup['agg_stand_forest'].aggregate(
                data={var_name: var_data},
                variables=[var_name],
                strategy=strategies[var_name],
                variable_metadata=VARIABLE_METADATA,
                mode='conservative'
            )
            forest_data[var_name] = aggregated[var_name]
        
        # Verify forest-level carbon balance: NPP = GPP - R_a
        forest_gpp = forest_data['gpp'][0]
        forest_ra = forest_data['ra'][0]
        forest_npp = forest_data['npp'][0]
        
        carbon_balance_error = abs(forest_npp - (forest_gpp - forest_ra)) / forest_gpp
        
        assert carbon_balance_error < 1e-4, \
            f"Forest carbon balance error: {carbon_balance_error:.2e}"
        
        # Verify conservation of GPP
        tree_gpp_total = (gpp * tree_areas).sum()
        forest_area = ecosystem_setup['forest_area']
        forest_gpp_total = forest_gpp * forest_area
        
        gpp_conservation = abs(forest_gpp_total - tree_gpp_total) / tree_gpp_total
        
        assert gpp_conservation < 1e-6, \
            f"GPP conservation error: {gpp_conservation:.2e}"
        
        # Verify realistic ranges (adjusted for actual flux density units after aggregation)
        # GPP in g C/m2/s: typical range 0.0001-0.001 after unit conversion
        assert 0.00001 <= forest_gpp <= 0.01, f"Forest GPP out of range: {forest_gpp:.6f} g C/m2/s"
        assert 0.000005 <= forest_npp <= 0.005, f"Forest NPP out of range: {forest_npp:.6f} g C/m2/s"
        assert 0.00001 <= forest_data['et'][0] <= 0.001, \
            f"Forest ET out of range: {forest_data['et'][0]:.6f} kg/m2/s"

    def test_agricultural_landscape(self, ecosystem_setup):
        """
        Test mixed agricultural landscape with different crop types.
        
        Simulates:
        - Multiple crop types (corn, soybean, wheat)
        - Different management practices
        - Irrigation effects
        - Fertilizer application
        """
        device = ecosystem_setup['device']
        n_trees = ecosystem_setup['n_trees']  # Reuse as "fields"
        tree_areas = ecosystem_setup['tree_areas']  # Reuse as "field_areas"
        
        # Assign crop types (categorical)
        crop_types = torch.zeros(n_trees, device=device)
        crop_types[:n_trees//3] = 1.0  # Corn
        crop_types[n_trees//3:2*n_trees//3] = 2.0  # Soybean
        crop_types[2*n_trees//3:] = 3.0  # Wheat
        
        # Crop-specific characteristics
        # Corn: high productivity, high water demand
        # Soybean: N-fixing, moderate water
        # Wheat: lower productivity, low water demand
        
        base_yield = torch.zeros(n_trees, device=device)
        base_yield[crop_types == 1.0] = torch.rand((crop_types == 1.0).sum(), device=device) * 5 + 10  # 10-15 Mg/ha
        base_yield[crop_types == 2.0] = torch.rand((crop_types == 2.0).sum(), device=device) * 3 + 5   # 5-8 Mg/ha
        base_yield[crop_types == 3.0] = torch.rand((crop_types == 3.0).sum(), device=device) * 4 + 6   # 6-10 Mg/ha
        
        # Convert to g C/m2/s assuming 180-day growing season
        seconds_per_season = 180 * 24 * 3600
        # Assume 45% carbon content, 10000 m2/ha
        carbon_flux = base_yield * 0.45 * 1e6 / 10000 / seconds_per_season
        
        # Water use
        water_flux = torch.zeros(n_trees, device=device)
        water_flux[crop_types == 1.0] = torch.rand((crop_types == 1.0).sum(), device=device) * 0.0003 + 0.0002
        water_flux[crop_types == 2.0] = torch.rand((crop_types == 2.0).sum(), device=device) * 0.0002 + 0.0001
        water_flux[crop_types == 3.0] = torch.rand((crop_types == 3.0).sum(), device=device) * 0.00015 + 0.00005
        
        # Aggregate through hierarchy
        agg = ecosystem_setup['agg_tree_plot']
        
        # Aggregate crop types using mode (most common crop)
        crop_plot, _ = agg.aggregate(
            data={'crop_type': crop_types},
            variables=['crop_type'],
            strategy='mode',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        # Aggregate carbon and water
        carbon_plot, _ = agg.aggregate(
            data={'carbon': carbon_flux},
            variables=['carbon'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        water_plot, _ = agg.aggregate(
            data={'water': water_flux},
            variables=['water'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        # Verify crop types are preserved as dominant types
        assert crop_plot['crop_type'].min() >= 1.0
        assert crop_plot['crop_type'].max() <= 3.0
        
        # Verify conservation
        field_areas = tree_areas
        plot_areas = ecosystem_setup['plot_areas']
        
        carbon_source = (carbon_flux * field_areas).sum()
        carbon_target = (carbon_plot['carbon'] * plot_areas).sum()
        carbon_error = abs(carbon_target - carbon_source) / carbon_source
        
        assert carbon_error < 1e-6, f"Carbon conservation error: {carbon_error:.2e}"
        
        water_source = (water_flux * field_areas).sum()
        water_target = (water_plot['water'] * plot_areas).sum()
        water_error = abs(water_target - water_source) / water_source
        
        assert water_error < 1e-6, f"Water conservation error: {water_error:.2e}"

    def test_watershed_hydrology(self, ecosystem_setup):
        """
        Test complete hydrological cycle in a watershed.
        
        Water balance: P = ET + R + Delta_S
        where P=precipitation, ET=evapotranspiration, R=runoff, Delta_S=storage change
        
        References:
            - Shuttleworth (2012). Terrestrial Hydrometeorology
            - Band et al. (1993). Forest ecosystem processes at watershed scale
        """
        device = ecosystem_setup['device']
        n_trees = ecosystem_setup['n_trees']  # Reuse as subcatchments
        tree_areas = ecosystem_setup['tree_areas']
        
        # Hydrological fluxes at subcatchment level
        torch.manual_seed(789)
        
        # Precipitation: 0-10 mm/hr = 0-0.00278 kg/m2/s
        precipitation = torch.rand(n_trees, device=device) * 0.00278
        
        # Evapotranspiration: 50-80% of precipitation
        et_fraction = torch.rand(n_trees, device=device) * 0.3 + 0.5
        evapotranspiration = precipitation * et_fraction
        
        # Infiltration: remaining water
        available_water = precipitation - evapotranspiration
        
        # Runoff: 10-30% of available water (depends on soil saturation)
        runoff_fraction = torch.rand(n_trees, device=device) * 0.2 + 0.1
        runoff = available_water * runoff_fraction
        
        # Storage change: remainder
        storage_change = available_water - runoff
        
        # Verify subcatchment water balance
        water_balance = precipitation - (evapotranspiration + runoff + storage_change)
        assert torch.allclose(water_balance, torch.zeros_like(water_balance), atol=1e-8)
        
        # Aggregate all components through watershed hierarchy
        variables = {
            'precipitation': precipitation,
            'evapotranspiration': evapotranspiration,
            'runoff': runoff,
            'storage_change': storage_change,
        }
        
        # All are extensive (water fluxes)
        strategy = 'flow_conserving'
        
        # Aggregate to plot level (hillslopes)
        hillslope_data = {}
        for var_name, var_data in variables.items():
            aggregated, _ = ecosystem_setup['agg_tree_plot'].aggregate(
                data={var_name: var_data},
                variables=[var_name],
                strategy=strategy,
                variable_metadata=VARIABLE_METADATA,
                mode='conservative'
            )
            hillslope_data[var_name] = aggregated[var_name]
        
        # Aggregate to stand level (subcatchments)
        subcatchment_data = {}
        for var_name, var_data in hillslope_data.items():
            aggregated, _ = ecosystem_setup['agg_plot_stand'].aggregate(
                data={var_name: var_data},
                variables=[var_name],
                strategy=strategy,
                variable_metadata=VARIABLE_METADATA,
                mode='conservative'
            )
            subcatchment_data[var_name] = aggregated[var_name]
        
        # Aggregate to forest level (watershed)
        watershed_data = {}
        for var_name, var_data in subcatchment_data.items():
            aggregated, _ = ecosystem_setup['agg_stand_forest'].aggregate(
                data={var_name: var_data},
                variables=[var_name],
                strategy=strategy,
                variable_metadata=VARIABLE_METADATA,
                mode='conservative'
            )
            watershed_data[var_name] = aggregated[var_name]
        
        # Verify watershed-level water balance
        ws_p = watershed_data['precipitation'][0]
        ws_et = watershed_data['evapotranspiration'][0]
        ws_r = watershed_data['runoff'][0]
        ws_ds = watershed_data['storage_change'][0]
        
        watershed_balance = ws_p - (ws_et + ws_r + ws_ds)
        watershed_balance_error = abs(watershed_balance) / ws_p
        
        assert watershed_balance_error < 1e-5, \
            f"Watershed water balance error: {watershed_balance_error:.2e}"
        
        # Verify conservation of each component
        forest_area = ecosystem_setup['forest_area']
        
        for var_name in variables.keys():
            source_total = (variables[var_name] * tree_areas).sum()
            target_total = watershed_data[var_name][0] * forest_area
            error = abs(target_total - source_total) / source_total
            
            assert error < 1e-6, f"{var_name} conservation error: {error:.2e}"

    def test_multi_biome_regional(self, ecosystem_setup):
        """
        Test regional simulation with multiple biomes.
        
        Simulates:
        - Forest (high productivity, high LAI)
        - Grassland (moderate productivity, low LAI)
        - Desert (low productivity, very low LAI)
        """
        device = ecosystem_setup['device']
        n_trees = ecosystem_setup['n_trees']
        tree_areas = ecosystem_setup['tree_areas']
        
        # Assign biome types
        biome_types = torch.zeros(n_trees, device=device)
        biome_types[:n_trees//3] = 1.0  # Forest
        biome_types[n_trees//3:2*n_trees//3] = 2.0  # Grassland
        biome_types[2*n_trees//3:] = 3.0  # Desert
        
        # Biome-specific LAI (leaf area index)
        lai = torch.zeros(n_trees, device=device)
        lai[biome_types == 1.0] = torch.rand((biome_types == 1.0).sum(), device=device) * 2 + 4  # 4-6
        lai[biome_types == 2.0] = torch.rand((biome_types == 2.0).sum(), device=device) * 1 + 1  # 1-2
        lai[biome_types == 3.0] = torch.rand((biome_types == 3.0).sum(), device=device) * 0.5  # 0-0.5
        
        # Biome-specific GPP (correlated with LAI)
        gpp = torch.zeros(n_trees, device=device)
        gpp[biome_types == 1.0] = torch.rand((biome_types == 1.0).sum(), device=device) * 1.5 + 1.0
        gpp[biome_types == 2.0] = torch.rand((biome_types == 2.0).sum(), device=device) * 0.5 + 0.2
        gpp[biome_types == 3.0] = torch.rand((biome_types == 3.0).sum(), device=device) * 0.1
        
        # Aggregate biome types (mode), LAI (area-weighted mean; LAI is m2 m-2, not a [0, 1]
        # fraction), and GPP (conserving)
        agg = ecosystem_setup['agg_tree_plot']
        
        biome_plot, _ = agg.aggregate(
            data={'biome': biome_types},
            variables=['biome'],
            strategy='mode',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        lai_plot, _ = agg.aggregate(
            data={'lai': lai},
            variables=['lai'],
            strategy='area_weighted_mean',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        gpp_plot, _ = agg.aggregate(
            data={'gpp': gpp},
            variables=['gpp'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        # Verify biome types preserved
        assert biome_plot['biome'].min() >= 1.0
        assert biome_plot['biome'].max() <= 3.0
        
        # Verify LAI bounds (cannot exceed max observed, cannot be negative)
        assert lai_plot['lai'].min() >= 0.0
        assert lai_plot['lai'].max() <= 6.0
        
        # Verify GPP conservation
        plot_areas = ecosystem_setup['plot_areas']
        gpp_source = (gpp * tree_areas).sum()
        gpp_target = (gpp_plot['gpp'] * plot_areas).sum()
        gpp_error = abs(gpp_target - gpp_source) / gpp_source
        
        assert gpp_error < 1e-6, f"GPP conservation error: {gpp_error:.2e}"


class TestPerformanceAndStability:
    """Test system performance and numerical stability"""

    def test_large_scale_performance(self):
        """
        Test performance with large number of spatial units (10000+).
        
        Validates that aggregation scales efficiently and maintains
        conservation with large datasets.
        """
        device = torch.device("cpu")
        
        # Large-scale setup: 10000 patches -> 1000 landscapes -> 1 region
        n_patches = 10000
        n_landscapes = 1000
        n_regions = 1
        
        patches_per_landscape = n_patches // n_landscapes
        
        # Create topology (simplified for performance)
        patch_areas = torch.ones(n_patches, device=device)  # Uniform for speed
        
        landscape_areas = torch.ones(n_landscapes, device=device) * patches_per_landscape
        parent_map_patch_land = {i: i // patches_per_landscape for i in range(n_patches)}
        
        topology_patch_land = {
            'n_source_units': n_patches,
            'n_target_units': n_landscapes,
            'source_areas': patch_areas.tolist(),
            'target_areas': landscape_areas.tolist(),
            'parent_map': parent_map_patch_land
        }
        
        regional_area = landscape_areas.sum()
        parent_map_land_region = {i: 0 for i in range(n_landscapes)}
        
        topology_land_region = {
            'n_source_units': n_landscapes,
            'n_target_units': n_regions,
            'source_areas': landscape_areas.tolist(),
            'target_areas': [regional_area.item()],
            'parent_map': parent_map_land_region
        }
        
        # Create aggregators
        index_patch_land = HierarchyIndex('patch', 'landscape', device, topology=topology_patch_land)
        index_land_region = HierarchyIndex('landscape', 'regional', device, topology=topology_land_region)
        
        agg_patch_land = HierarchicalAggregator(index_patch_land, device, ['patch', 'landscape'], weight_type='area', rule='conservative')
        agg_land_region = HierarchicalAggregator(index_land_region, device, ['landscape', 'regional'], weight_type='area', rule='conservative')
        
        # Test data
        carbon_flux = torch.rand(n_patches, device=device) * 10.0
        
        # Time aggregation
        start_time = time.time()
        
        # Stage 1: Patch -> Landscape
        carbon_land, _ = agg_patch_land.aggregate(
            data={'carbon': carbon_flux},
            variables=['carbon'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        stage1_time = time.time() - start_time
        
        # Stage 2: Landscape -> Regional
        stage2_start = time.time()
        carbon_region, _ = agg_land_region.aggregate(
            data={'carbon': carbon_land['carbon']},
            variables=['carbon'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        stage2_time = time.time() - stage2_start
        total_time = time.time() - start_time
        
        # Performance assertions (should complete in reasonable time)
        assert total_time < 5.0, f"Aggregation too slow: {total_time:.2f}s for 10000 units"
        
        # Verify conservation with large dataset
        source_total = (carbon_flux * patch_areas).sum()
        target_total = carbon_region['carbon'][0] * regional_area
        
        conservation_error = abs(target_total - source_total) / source_total
        
        assert conservation_error < 1e-6, \
            f"Conservation error with large dataset: {conservation_error:.2e}"
        
        # Log performance metrics
        print(f"\nLarge-scale performance (10000 units):")
        print(f"  Stage 1 (Patch->Landscape): {stage1_time:.3f}s")
        print(f"  Stage 2 (Landscape->Regional): {stage2_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Conservation error: {conservation_error:.2e}")

    def test_numerical_stability(self):
        """
        Test numerical stability with extreme values.
        
        Tests:
        - Very large values (1e10)
        - Very small values (1e-10)
        - Mixed scales
        - Near-zero denominators
        """
        device = torch.device("cpu")
        
        # Setup simple hierarchy
        n_patches = 100
        n_landscapes = 10
        
        patch_areas = torch.ones(n_patches, device=device)
        landscape_areas = torch.ones(n_landscapes, device=device) * (n_patches // n_landscapes)
        
        parent_map = {i: i // (n_patches // n_landscapes) for i in range(n_patches)}
        
        topology = {
            'n_source_units': n_patches,
            'n_target_units': n_landscapes,
            'source_areas': patch_areas.tolist(),
            'target_areas': landscape_areas.tolist(),
            'parent_map': parent_map
        }
        
        index = HierarchyIndex('patch', 'landscape', device, topology=topology)
        agg = HierarchicalAggregator(index, device, ['patch', 'landscape'], weight_type='area', rule='conservative')
        
        # Test Case 1: Very large values
        large_values = torch.ones(n_patches, device=device) * 1e10
        large_aggregated, _ = agg.aggregate(
            data={'flux': large_values},
            variables=['flux'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        assert not torch.isnan(large_aggregated['flux']).any(), "NaN with large values"
        assert not torch.isinf(large_aggregated['flux']).any(), "Inf with large values"
        
        large_source = (large_values * patch_areas).sum()
        large_target = (large_aggregated['flux'] * landscape_areas).sum()
        large_error = abs(large_target - large_source) / large_source
        
        assert large_error < 1e-5, f"Conservation failed with large values: {large_error:.2e}"
        
        # Test Case 2: Very small values
        small_values = torch.ones(n_patches, device=device) * 1e-10
        small_aggregated, _ = agg.aggregate(
            data={'flux': small_values},
            variables=['flux'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        assert not torch.isnan(small_aggregated['flux']).any(), "NaN with small values"
        assert not torch.isinf(small_aggregated['flux']).any(), "Inf with small values"
        
        small_source = (small_values * patch_areas).sum()
        small_target = (small_aggregated['flux'] * landscape_areas).sum()
        small_error = abs(small_target - small_source) / (small_source + 1e-15)
        
        assert small_error < 1e-4, f"Conservation failed with small values: {small_error:.2e}"
        
        # Test Case 3: Mixed scales (large variation)
        mixed_values = torch.zeros(n_patches, device=device)
        mixed_values[:50] = 1e10  # Large
        mixed_values[50:] = 1e-10  # Small
        
        mixed_aggregated, _ = agg.aggregate(
            data={'flux': mixed_values},
            variables=['flux'],
            strategy='flow_conserving',
            variable_metadata=VARIABLE_METADATA,
            mode='conservative'
        )
        
        assert not torch.isnan(mixed_aggregated['flux']).any(), "NaN with mixed scales"
        assert not torch.isinf(mixed_aggregated['flux']).any(), "Inf with mixed scales"
        
        mixed_source = (mixed_values * patch_areas).sum()
        mixed_target = (mixed_aggregated['flux'] * landscape_areas).sum()
        mixed_error = abs(mixed_target - mixed_source) / mixed_source
        
        assert mixed_error < 1e-5, f"Conservation failed with mixed scales: {mixed_error:.2e}"

    def test_error_propagation(self):
        """
        Test error propagation through multi-level aggregation.
        
        Validates that rounding errors don't accumulate excessively
        through multiple aggregation stages.
        """
        device = torch.device("cpu")
        
        # Four-level hierarchy for error propagation testing
        n_level1 = 1000
        n_level2 = 100
        n_level3 = 10
        n_level4 = 1
        
        # Build hierarchies
        areas_l1 = torch.ones(n_level1, device=device)
        areas_l2 = torch.ones(n_level2, device=device) * (n_level1 // n_level2)
        areas_l3 = torch.ones(n_level3, device=device) * (n_level2 // n_level3)
        areas_l4 = torch.tensor([areas_l3.sum().item()], device=device)
        
        # Create aggregators for each level
        topologies = []
        aggregators = []
        
        for (n_source, n_target, areas_source, areas_target) in [
            (n_level1, n_level2, areas_l1, areas_l2),
            (n_level2, n_level3, areas_l2, areas_l3),
            (n_level3, n_level4, areas_l3, areas_l4)
        ]:
            parent_map = {i: i // (n_source // n_target) for i in range(n_source)}
            topology = {
                'n_source_units': n_source,
                'n_target_units': n_target,
                'source_areas': areas_source.tolist(),
                'target_areas': areas_target.tolist(),
                'parent_map': parent_map
            }
            topologies.append(topology)
            
            index = HierarchyIndex('source', 'target', device, topology=topology)
            agg = HierarchicalAggregator(index, device, ['source', 'target'], weight_type='area', rule='strict_conservative')
            aggregators.append(agg)
        
        # Initial data with controlled precision
        initial_data = torch.rand(n_level1, device=device) * 10.0
        initial_total = (initial_data * areas_l1).sum()
        
        # Track errors through each level
        errors = []
        current_data = initial_data
        current_areas = areas_l1
        
        for level_idx, (agg, topology) in enumerate(zip(aggregators, topologies)):
            aggregated, _ = agg.aggregate(
                data={'flux': current_data},
                variables=['flux'],
                strategy='flow_conserving',
                variable_metadata=VARIABLE_METADATA,
                mode='strict_conservative'
            )
            
            current_data = aggregated['flux']
            current_areas = torch.tensor(topology['target_areas'], device=device)
            
            current_total = (current_data * current_areas).sum()
            error = abs(current_total - initial_total) / initial_total
            errors.append(error.item())
        
        # Verify errors don't accumulate excessively
        max_error = max(errors)
        final_error = errors[-1]
        
        assert max_error < 1e-6, f"Maximum error across levels: {max_error:.2e}"
        assert final_error < 1e-6, f"Final cumulative error: {final_error:.2e}"
        
        # Verify error growth is sublinear (not exponential)
        # Error should grow roughly as sqrt(n_levels) in worst case
        if len(errors) > 1:
            error_ratio = errors[-1] / errors[0] if errors[0] > 0 else 1.0
            # For 3 aggregation levels, ratio should be < 2.0
            assert error_ratio < 2.0, f"Error accumulation too rapid: {error_ratio:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

