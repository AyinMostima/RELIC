"""
L2 Integration Tests for Anthroposphere Module

This module tests the integration between different anthroposphere submodules:
- Management -> Emissions
- Land Use transformation
- Prescribed surface rates -> downstream consumers
- Management -> Land Use
- Cross-module data flow and conservation

Test Categories:
- L2.1: Module Interface consistency
- L2.2: Data Flow Validation
- L2.3: Cross-Module Conservation Laws
- L2.4: Dependency Resolution
- L2.5: Error Propagation and Recovery
- L2.6: Performance and Scalability
"""

import pytest
import torch
import numpy as np
from typing import Dict
import time

from relic.core.state import UpdateItem, UpdateMode
# Import the modules under test
from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator
from relic.scientific.anthroposphere.emissions.sector_allocation import SectorAllocation
from relic.scientific.anthroposphere.emissions.species_mapping import SpeciesMapper
from relic.scientific.anthroposphere.land_use.mapping import (
    DIAGNOSTIC_PREFIX,
    LandUseParameterMapping,
)
from relic.scientific.anthroposphere.land_use.transitions import LandUseTransitions


class TestAnthroposphereIntegrationL2:
    """L2 Integration Tests for Anthroposphere Module"""

    @pytest.fixture
    def integrated_config(self):
        """Integrated configuration for all anthroposphere modules"""
        return {
            "management": {
                "name": "test_management_coordinator",
                "enabled": True,
                "calendar_type": "noleap",
                "fertilization": {
                    "application_efficiency": 0.7,
                    "leaching_fraction": 0.1
                },
                "irrigation": {
                    "application_efficiency": 0.75,
                    "water_requirement_factor": 1.2
                },
                "tillage": {
                    "disturbance_intensity": 0.8,
                    "carbon_loss_rate": 0.02
                }
            },
            "emissions": {
                "name": "test_emissions",
                "enabled": True,
                "sector_allocation": {
                    "allocation_method": "spatial_proximity",
                    "temporal_resolution": "daily"
                },
                "species_mapping": {
                    "chemical_mechanism": "CB05",
                    "voc_speciation": "detailed"
                },
                "coupling": {
                    "atmospheric_layer_height": 1000,  # m
                    "mixing_coefficient": 0.1
                }
            },
            "land_use": {
                "name": "test_land_use",
                "enabled": True,
                "mapping": {
                    "vegetation_types": ["forest", "grassland", "cropland", "urban"],
                    "spatial_resolution": 1.0  # km
                },
                "transitions": {
                    "transition_rates": {
                        "forest_to_cropland": 0.01,
                        "cropland_to_forest": 0.005
                    }
                }
            }
        }

    @pytest.fixture
    def comprehensive_state(self):
        """Comprehensive state dictionary for integration testing"""
        batch_size, height, width = 2, 10, 10
        device = torch.device("cpu")
        land_use_fractions = torch.rand(batch_size, 9, height, width, device=device)
        land_use_fractions = land_use_fractions / torch.sum(land_use_fractions, dim=1, keepdim=True)
        
        return {
            # Prescribed test-only surface-rate fields
            "energy_consumption": torch.rand(batch_size, 3, height, width, device=device) * 1000,  # MJ day-1
            "industrial_production": torch.rand(batch_size, 4, height, width, device=device) * 100,  # tons day-1
            "agricultural_production": torch.rand(batch_size, 3, height, width, device=device) * 50,  # tons day-1
            "transport_activity": torch.rand(batch_size, 4, height, width, device=device) * 10000,  # km day-1
            
            # Management state variables
            "soil_moisture": torch.rand(batch_size, height, width, device=device) * 0.3 + 0.2,  # m3/m3
            "soil_nitrogen": torch.rand(batch_size, height, width, device=device) * 0.01 + 0.005,  # kg N m-2
            "soil_organic_carbon": torch.rand(batch_size, height, width, device=device) * 3.0 + 3.0,  # kg C m-2
            "crop_maturity": torch.rand(batch_size, height, width, device=device),  # dimensionless
            "aboveground_biomass": torch.rand(batch_size, height, width, device=device),  # kg DM m-2
            "air_temperature": torch.rand(batch_size, height, width, device=device) * 20.0 + 283.15,  # K
            "irrigation_flux": torch.zeros(batch_size, height, width, device=device),
            "harvest_flux": torch.zeros(batch_size, height, width, device=device),
            "grazing_flux": torch.zeros(batch_size, height, width, device=device),
            "co2_emission": torch.zeros(batch_size, height, width, device=device),
            "field_capacity": torch.full((batch_size, height, width), 0.35, device=device),
            "wilting_point": torch.full((batch_size, height, width), 0.15, device=device),
            "evapotranspiration_flux": torch.full((batch_size, height, width), 3.0e-6, device=device),
            "precipitation_flux": torch.zeros(batch_size, height, width, device=device),
            "soil_temperature": torch.rand(batch_size, height, width, device=device) * 20 + 283.15,  # K
            "crop_biomass": torch.rand(batch_size, height, width, device=device) * 10,  # t/ha
            "soil_organic_matter": torch.rand(batch_size, height, width, device=device) * 0.05,  # kg/kg
            "fertilizer_rate": torch.zeros(batch_size, height, width, device=device),  # kg N/ha day-1
            "irrigation_rate": torch.zeros(batch_size, height, width, device=device),  # mm day-1
            "tillage_intensity": torch.zeros(batch_size, height, width, device=device),  # dimensionless
            
            # Emissions state variables
            "base_emissions": torch.rand(batch_size, 5, height, width, device=device) * 10,  # kg day-1
            "sector_emissions": torch.zeros(batch_size, 5, height, width, device=device),  # kg day-1
            "species_emissions": torch.zeros(batch_size, 20, height, width, device=device),  # kg day-1
            "atmospheric_concentrations": torch.rand(batch_size, 20, height, width, device=device) * 1e-9,  # mol/mol
            "anthropogenic_nox_flux": torch.zeros(batch_size, height, width, device=device),
            "anthropogenic_co_flux": torch.zeros(batch_size, height, width, device=device),
            "anthropogenic_so2_flux": torch.zeros(batch_size, height, width, device=device),
            "anthropogenic_voc_flux": torch.zeros(batch_size, height, width, device=device),
            "anthropogenic_nh3_flux": torch.zeros(batch_size, height, width, device=device),
            "anthropogenic_pm25_flux": torch.zeros(batch_size, height, width, device=device),
            
            # Land use state variables
            "land_use_fraction": land_use_fractions,
            "land_use_fractions": land_use_fractions,
            "vegetation_parameters": torch.rand(batch_size, 9, height, width, device=device),
            "transition_probabilities": torch.rand(batch_size, 9, 9, height, width, device=device),
            
            # Cross-module state variables
            "total_carbon_stock": torch.rand(batch_size, height, width, device=device) * 100,  # tC/ha
            "nitrogen_balance": torch.rand(batch_size, height, width, device=device) * 10,  # kgN/ha
            "water_balance": torch.rand(batch_size, height, width, device=device) * 100,  # mm
            "energy_fluxes": torch.rand(batch_size, 3, height, width, device=device) * 100,  # W/m2
            "population_density": torch.rand(batch_size, height, width, device=device) * 1000,  # people/km2
            # Land-use transition driver (formerly supplied by the environment mapping).
            "gdp_per_capita": torch.rand(batch_size, height, width, device=device) * 50000,  # USD/year
            "gdp_density": torch.rand(batch_size, height, width, device=device) * 50000,  # USD/km2
            "road_density": torch.rand(batch_size, height, width, device=device) * 5,  # km/km2
            "time_index": torch.zeros(batch_size, height, width, device=device),
            "year": torch.full((batch_size, height, width), 2020.0, device=device)
        }

    @staticmethod
    def _prepare_sector_allocation_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Preserve canonical leading batch and land-use class axes."""
        state_copy = TestAnthroposphereIntegrationL2._require_prescribed_surface_rates(
            state
        )
        reference = state_copy["anthropogenic_nox_flux"]
        for key in (
            "population_density",
            "gdp_density",
            "road_density",
            "land_use_fractions",
        ):
            value = state_copy.get(key)
            if isinstance(value, torch.Tensor):
                state_copy[key] = value.to(
                    device=reference.device,
                    dtype=reference.dtype,
                )
        fractions = state_copy.get("land_use_fractions")
        assert isinstance(fractions, torch.Tensor)
        assert fractions.ndim == state_copy["anthropogenic_nox_flux"].ndim + 1
        assert fractions.shape[-2:] == state_copy["anthropogenic_nox_flux"].shape[-2:]
        return state_copy

    @staticmethod
    def _require_prescribed_surface_rates(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Require explicit test-only prescribed surface-rate fields."""
        state_copy = dict(state)

        required = (
            "anthropogenic_nox_flux",
            "anthropogenic_co_flux",
            "anthropogenic_so2_flux",
            "anthropogenic_voc_flux",
            "anthropogenic_nh3_flux",
            "anthropogenic_pm25_flux",
        )
        missing = [key for key in required if not isinstance(state_copy.get(key), torch.Tensor)]
        assert not missing, f"Missing canonical atmosphere emission ports: {missing}"

        return state_copy

    @staticmethod
    def _prepare_land_use_transition_state(
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compose the canonical transition state without runtime aliases."""
        fractions = state.get("land_use_fraction")
        if isinstance(fractions, UpdateItem):
            fractions = fractions.value
        assert isinstance(fractions, torch.Tensor)
        assert fractions.ndim == 4 and fractions.shape[1] == 9
        prepared = {
            "land_use_fraction": fractions,
            "time_index": torch.zeros((), device=fractions.device),
        }
        for name in LandUseTransitions.DRIVER_INPUTS:
            value = state.get(name)
            if isinstance(value, torch.Tensor):
                prepared[name] = value
        return prepared

    @classmethod
    def _prepare_species_mapping_state(
        cls,
        state: Dict[str, torch.Tensor],
        allocator: SectorAllocation,
        time_value: float,
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        """Route prescribed canonical rate fields through SectorAllocation."""
        allocated = allocator.forward(
            cls._prepare_sector_allocation_state(state),
            time=time_value,
            dt=dt,
        )
        item = allocated['sector_emission_flux']
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        return {'sector_emission_flux': item.value}

    @pytest.fixture
    def initialized_modules(self, integrated_config):
        """Initialize all anthroposphere modules for integration testing"""
        modules = {
            'management_coordinator': ManagementCoordinator(config=integrated_config["management"]),
            'sector_allocation': SectorAllocation(config=integrated_config["emissions"]),
            'species_mapping': SpeciesMapper(name="test_species_mapper"),
            'land_use_mapping': LandUseParameterMapping(config=integrated_config["land_use"]),
            'land_use_transitions': LandUseTransitions(config=integrated_config["land_use"])
        }
        
        return modules

    # L2.1: Module Interface consistency
    def test_L2_1_management_land_use_interface(self, initialized_modules, comprehensive_state):
        """Test interface consistency between management and land use modules"""
        management = initialized_modules['management_coordinator']
        land_use_mapping = initialized_modules['land_use_mapping']
        
        # Run management module
        management_updates = management.forward(
            comprehensive_state, time=0.0, dt=3600.0,
            calendar_type="noleap"
        )
        
        # Update state with management results
        updated_state = comprehensive_state.copy()
        for key, value in management_updates.items():
            updated_state[key] = value
        
        # Run land use mapping module
        land_use_updates = land_use_mapping.forward(
            updated_state, time=0.0, dt=3600.0
        )
        
        # Mapping reports observations and does not override process parameters.
        for name in (
            f"{DIAGNOSTIC_PREFIX}.vegetation.lai_max",
            f"{DIAGNOSTIC_PREFIX}.vegetation.vcmax25",
        ):
            assert isinstance(land_use_updates[name], UpdateItem)
            assert land_use_updates[name].mode is UpdateMode.REPLACE
            assert land_use_updates[name].per_timestep is False
        
        # The irrigation requirement reads the declared state fluxes (the former check
        # tested keys the coordinator never emits): F_net = max(E - P, 0).
        demand = management_updates["irrigation_demand_flux"].value
        expected = torch.clamp(
            comprehensive_state["evapotranspiration_flux"]
            - comprehensive_state["precipitation_flux"],
            min=0.0,
        ).to(demand.dtype)
        torch.testing.assert_close(demand, expected)

    def test_L2_1_cross_module_device_consistency(self, initialized_modules, comprehensive_state):
        """Test all modules maintain device consistency"""
        device = torch.device("cpu")
        
        # Move all data to device
        state_device = {k: v.to(device) for k, v in comprehensive_state.items()}
        
        for name, module in initialized_modules.items():
            if hasattr(module, "to"):
                module.to(device)
            
            try:
                state_input = state_device
                if name == "sector_allocation":
                    state_input = self._prepare_sector_allocation_state(state_input)
                if name == "species_mapping":
                    state_input = self._prepare_species_mapping_state(
                        state_input,
                        initialized_modules['sector_allocation'],
                        0.0,
                        3600.0,
                    )
                if name == "land_use_transitions":
                    state_input = self._prepare_land_use_transition_state(
                        state_input,
                    )
                    updates = module.forward(state_input, time=0.0, dt=3600.0)
                else:
                    # Every module reads the state only (no environment keyword).
                    updates = module.forward(state_input, time=0.0, dt=3600.0)

                # Check all outputs are on correct device
                for key, value in updates.items():
                    tensor_value = value.value if isinstance(value, UpdateItem) else value
                    if torch.is_tensor(tensor_value):
                        assert tensor_value.device == device, f"Module {name} output {key} not on correct device"
                    
            except Exception as e:
                pytest.fail(f"Module {name} failed with device consistency: {e}")

    # L2.2: Data Flow Validation
    def test_L2_2_management_to_land_use_flow(self, initialized_modules, comprehensive_state):
        """Test data flow from management to land use modules"""
        management = initialized_modules['management_coordinator']
        land_use_mapping = initialized_modules['land_use_mapping']
        land_use_transitions = initialized_modules['land_use_transitions']
        
        # Run management -> land use mapping -> land use transitions chain
        management_updates = management.forward(
            comprehensive_state, time=0.0, dt=3600.0
        )
        
        state_1 = comprehensive_state.copy()
        state_1.update({
            key: item.value if isinstance(item, UpdateItem) else item
            for key, item in management_updates.items()
        })
        
        mapping_updates = land_use_mapping.forward(
            state_1, time=0.0, dt=3600.0
        )
        
        state_2 = state_1.copy()
        state_2.update(mapping_updates)
        
        transition_updates = land_use_transitions.forward(
            self._prepare_land_use_transition_state(
                state_2,
            ),
            time=0.0,
            dt=3600.0,
        )
        
        # Validate data flow
        assert "fertilizer_product_flux" in management_updates
        assert "irrigation_demand_flux" in management_updates
        assert (
            f"{DIAGNOSTIC_PREFIX}.vegetation.lai_max" in mapping_updates
        )
        assert isinstance(transition_updates["land_use_fraction"], UpdateItem)
        assert transition_updates["land_use_fraction"].mode is UpdateMode.REPLACE

    def test_L2_2_temporal_data_consistency(self, initialized_modules, comprehensive_state):
        """Test temporal consistency of data flow"""
        management = initialized_modules['management_coordinator']
        
        # Run multiple time steps
        previous_state = comprehensive_state.copy()
        cumulative_changes = {}
        
        for step in range(5):
            updates = management.forward(
                previous_state, time=step*3600.0, dt=3600.0
            )
            
            # Accumulate changes
            for key, value in updates.items():
                if not torch.is_tensor(value):
                    continue
                if key not in cumulative_changes or cumulative_changes[key].shape != value.shape:
                    cumulative_changes[key] = torch.zeros_like(value)
                cumulative_changes[key] = cumulative_changes[key] + value
            
            # Update state for next step
            for key, value in updates.items():
                if key in previous_state and torch.is_tensor(previous_state[key]) and torch.is_tensor(value):
                    if previous_state[key].shape == value.shape:
                        previous_state[key] = previous_state[key] + value
                    else:
                        previous_state[key] = value
                else:
                    previous_state[key] = value
        
        # Check that cumulative changes are reasonable
        for key, change in cumulative_changes.items():
            assert not torch.any(torch.isnan(change)), f"Cumulative change in {key} contains NaN"
            assert torch.all(torch.isfinite(change)), f"Cumulative change in {key} contains infinite values"

    # L2.3: Cross-Module Conservation Laws
    def test_L2_3_water_balance_across_modules(self, initialized_modules, comprehensive_state):
        """The net irrigation requirement closes the root-zone water deficit left by
        precipitation: F_net = max(E - P, 0) from the declared state fluxes (FAO-24 net
        irrigation requirement; Allen et al. 1998, FAO-56 Eq. 85), so F_net + P >= E in
        every cell and F_net = 0 where precipitation covers the evapotranspiration.
        (The former check added an environment precipitation that no module read and
        bounded the change by keys the modules never emit.)"""
        management = initialized_modules['management_coordinator']
        land_use_mapping = initialized_modules['land_use_mapping']

        state = dict(comprehensive_state)
        evapotranspiration = state["evapotranspiration_flux"]
        precipitation = torch.zeros_like(evapotranspiration)
        precipitation[..., ::2] = 2.0 * evapotranspiration[..., ::2]  # wet columns
        state["precipitation_flux"] = precipitation

        management_updates = management.forward(state, time=0.0, dt=3600.0)
        demand = management_updates["irrigation_demand_flux"].value
        expected = torch.clamp(evapotranspiration - precipitation, min=0.0).to(demand.dtype)
        torch.testing.assert_close(demand, expected)
        assert torch.all(demand + precipitation - evapotranspiration >= 0.0)
        assert torch.all(demand[..., ::2] == 0.0)
        assert torch.all(demand[..., 1::2] > 0.0)

        land_use_updates = land_use_mapping.forward(state, time=0.0, dt=3600.0)
        for update in land_use_updates.values():
            assert torch.all(torch.isfinite(update.value))

    def test_L2_4_dependency_graph_consistency(self, initialized_modules):
        """Test dependency graph consistency across modules"""
        # Collect all dependencies
        all_dependencies = {}
        
        for name, module in initialized_modules.items():
            deps_raw = module.get_dependencies() if hasattr(module, "get_dependencies") else []
            deps_list = []
            if isinstance(deps_raw, dict):
                deps_list.extend(deps_raw.get("required", []))
            elif isinstance(deps_raw, (list, tuple, set)):
                deps_list.extend(list(deps_raw))
            optional = set(getattr(module, "optional_dependencies", []))
            deps_filtered = [dep for dep in deps_list if dep in initialized_modules and dep not in optional and dep != name]
            all_dependencies[name] = deps_filtered
        
        # Check for circular dependencies
        def has_cycle(graph, node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True
            
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        if has_cycle(graph, neighbor, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        return True
            
            rec_stack[node] = False
            return False
        
        visited = {}
        rec_stack = {}
        for module_name in all_dependencies:
            if module_name not in visited:
                assert not has_cycle(all_dependencies, module_name, visited, rec_stack), \
                    f"Circular dependency detected involving {module_name}"

    def test_L2_4_module_availability_check(self, initialized_modules):
        """Test that required modules are available"""
        required_modules = [
            'management_coordinator',
            'sector_allocation', 'species_mapping',
            'land_use_mapping', 'land_use_transitions'
        ]
        
        for module_name in required_modules:
            assert module_name in initialized_modules, f"Required module {module_name} not available"

    # L2.5: Error Propagation and Recovery
    def test_L2_5_recovery_from_temporary_failures(self, initialized_modules, comprehensive_state):
        """A heat episode enters through the declared air_temperature state (350 K). The
        environmental priority reads it, so its weight rises, and every management
        output stays finite and non-negative. (The former test put the temperature in an
        environment mapping that the coordinator never read, and checked an output key
        it never emits.)"""
        management = initialized_modules['management_coordinator']

        stress_state = dict(comprehensive_state)
        stress_state["air_temperature"] = torch.full_like(
            comprehensive_state["air_temperature"], 350.0
        )
        normal_priorities = management.calculate_priority_weights(comprehensive_state)
        stress_priorities = management.calculate_priority_weights(stress_state)
        assert stress_priorities["environmental"] > normal_priorities["environmental"]

        stress_updates = management.forward(stress_state, time=0.0, dt=3600.0)
        for name in ("fertilizer_product_flux", "irrigation_demand_flux", "tillage_intensity"):
            value = stress_updates[name].value
            assert torch.all(torch.isfinite(value)), name
            assert torch.all(value >= 0.0), name

    def test_L2_6_performance_benchmarks(self, initialized_modules, comprehensive_state):
        """Test performance benchmarks for integrated modules"""
        import time
        
        # Test individual module performance
        performance_results = {}
        
        for name, module in initialized_modules.items():
            start_time = time.time()
            
            try:
                state_input = comprehensive_state
                if name == "sector_allocation":
                    state_input = self._prepare_sector_allocation_state(state_input)
                if name == "species_mapping":
                    state_input = self._prepare_species_mapping_state(
                        state_input,
                        initialized_modules['sector_allocation'],
                        0.0,
                        3600.0,
                    )
                if name == "land_use_transitions":
                    state_input = self._prepare_land_use_transition_state(
                        state_input,
                    )
                    updates = module.forward(state_input, time=0.0, dt=3600.0)
                else:
                    updates = module.forward(
                        state_input,
                        time=0.0,
                        dt=3600.0,
                    )
                end_time = time.time()
                
                performance_results[name] = end_time - start_time
                
                # Should complete within reasonable time
                assert performance_results[name] < 1.0, \
                    f"Module {name} took too long: {performance_results[name]}s"
                    
            except Exception as e:
                performance_results[name] = f"Error: {e}"
        
        # Check overall performance
        total_time = sum(t for t in performance_results.values() if isinstance(t, float))
        assert total_time < 10.0, f"Total execution time too long: {total_time}s"

    def test_L2_6_memory_usage_scaling(self, initialized_modules):
        """Test memory usage scales appropriately with problem size"""
        import torch.profiler
        
        # Test with different problem sizes
        sizes = [(2, 5, 5), (2, 10, 10), (2, 20, 20)]
        memory_usage = {}
        
        for batch, height, width in sizes:
            # Create test data
            test_state = {
                "energy_consumption": torch.rand(batch, 3, height, width),
                "land_use_fractions": torch.full(
                    (batch, 9, height, width),
                    1.0 / 9.0,
                ),
                "soil_moisture": torch.rand(batch, height, width)
            }
            
            # Profile memory usage
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True
            ) as prof:
                
                for name, module in initialized_modules.items():
                    try:
                        state_input = test_state
                        if name == "species_mapping":
                            state_input = {
                                "sector_emission_flux": torch.rand(
                                    5,
                                    6,
                                    batch,
                                    height,
                                    width,
                                )
                            }
                        if name == "land_use_transitions":
                            state_input = self._prepare_land_use_transition_state(
                                state_input,
                            )
                            module.forward(state_input, time=0.0, dt=3600.0)
                        else:
                            module.forward(
                                state_input,
                                time=0.0,
                                dt=3600.0,
                            )
                    except Exception:
                        pass  # Ignore errors for memory profiling
            
            # Calculate memory usage
            memory_usage[(batch, height, width)] = sum(
                item.cpu_usage if hasattr(item, 'cpu_usage') else 0
                for item in prof.key_averages()
            )
        
        # Check memory scaling
        sizes_list = list(memory_usage.keys())
        for i in range(1, len(sizes_list)):
            prev_size = np.prod(sizes_list[i-1])
            curr_size = np.prod(sizes_list[i])
            size_ratio = curr_size / prev_size
            
            prev_memory = memory_usage[sizes_list[i-1]]
            curr_memory = memory_usage[sizes_list[i]]
            
            if prev_memory > 0 and curr_memory > 0:
                memory_ratio = curr_memory / prev_memory
                # Memory should not scale faster than problem size
                assert memory_ratio <= size_ratio * 2, \
                    f"Memory scaling too fast: {memory_ratio} vs size ratio {size_ratio}"

    def test_L2_6_scalability_limits(self, initialized_modules, comprehensive_state):
        """Test scalability limits and graceful degradation"""
        # Test with progressively larger problems
        max_successful_size = None
        
        for scale_factor in [1, 2, 4]:
            try:
                # Scale up the problem
                scaled_state = {}
                for key, value in comprehensive_state.items():
                    if len(value.shape) >= 3:  # Has spatial dimensions
                        new_shape = list(value.shape)
                        new_shape[-2] = int(new_shape[-2] * scale_factor)
                        new_shape[-1] = int(new_shape[-1] * scale_factor)
                        scaled_state[key] = torch.rand(new_shape, dtype=value.dtype, device=value.device)
                    else:
                        scaled_state[key] = value
                
                # Try running modules
                succeeded = False
                for name, module in initialized_modules.items():
                    start_time = time.time()
                    state_input = scaled_state
                    if name == "sector_allocation":
                        state_input = self._prepare_sector_allocation_state(state_input)
                    if name == "species_mapping":
                        state_input = self._prepare_species_mapping_state(
                            state_input,
                            initialized_modules['sector_allocation'],
                            0.0,
                            3600.0,
                        )
                    try:
                        if name == "land_use_transitions":
                            state_input = self._prepare_land_use_transition_state(
                                state_input,
                            )
                            fractions = state_input["land_use_fraction"]
                            state_input["land_use_fraction"] = fractions / fractions.sum(
                                dim=1,
                                keepdim=True,
                            )
                            module.forward(state_input, time=0.0, dt=3600.0)
                        else:
                            module.forward(
                                state_input,
                                time=0.0,
                                dt=3600.0,
                            )
                        succeeded = True
                    except Exception:
                        continue
                    end_time = time.time()
                    
                    # Should complete in reasonable time
                    assert end_time - start_time < 30.0, \
                        f"Module {name} at scale {scale_factor} took too long: {end_time - start_time}s"
                
                if succeeded:
                    max_successful_size = scale_factor
                
            except Exception as e:
                # Should fail gracefully with informative error
                assert "memory" in str(e).lower() or "size" in str(e).lower() or "scale" in str(e).lower(), \
                    f"Scalability error should be informative: {e}"
                break
        
        # Should be able to handle at least 2x scaling
        assert max_successful_size is not None and max_successful_size >= 2, \
            f"Should handle at least 2x scaling, only managed: {max_successful_size}"


def test_anthroposphere_forwards_take_no_environment_mapping():
    """Item 113: the coordinator, land-use mapping and sector allocation read their inputs
    from the declared state; an environment keyword raises instead of being dropped."""
    import inspect

    for cls, expected in (
        (ManagementCoordinator, ["self", "state", "time", "dt", "calendar_type"]),
        (LandUseParameterMapping, ["self", "state", "time", "dt"]),
        (SectorAllocation, ["self", "state", "time", "dt"]),
    ):
        assert list(inspect.signature(cls.forward).parameters) == expected, cls.__name__
