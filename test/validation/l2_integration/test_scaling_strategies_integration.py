"""
L2 Integration Tests for Scaling Strategies Integration

Tests the integration of 7 new aggregation strategies with HierarchyIndex
and StateManager in complete simulation workflows.

Test Coverage:
1. HierarchyIndex Integration (3 tests)
   - Hierarchy matrix construction with aggregation strategies
   - parent_map and children_map consistency
   - Multi-level aggregation (Individual -> Patch -> Landscape)

2. StateManager Integration (3 tests)
   - Aggregated results applied to state updates
   - Conservation checks with new strategies
   - Transaction rollback mechanisms

3. Batch Aggregation Performance (2 tests)
   - Multi-variable simultaneous aggregation
   - Mixed strategy batch operations

4. End-to-End Flows (2 tests)
   - Metadata-driven strategy selection
   - Complete Individual -> Patch -> Landscape aggregation chain

Mathematical Expressions:
    - Harmonic mean: r_eff = 1 / sum(w_i / r_i)
    - Arithmetic mean: g_eff = sum(w_i * g_i)
    - Bounded mean: x_j = clip(sum(x_i * A_i) / sum(A_i), 0, 1)
    - Mode: mode_j = argmax_c(sum_{i: x_i=c} A_i)
    - Circular mean: theta_j = atan2(sum(w_i*sin(theta_i)), sum(w_i*cos(theta_i)))
    - Component-wise: V_j = sum(w_i * V_i) for each component
    - Geometric mean: y_j = exp(sum(w_i * ln(x_i)))

References:
    - Monteith & Unsworth (2013). Principles of Environmental Physics
    - Jarvis & McNaughton (1986). Stomatal control and water use
    - Pielke (2002). Mesoscale Meteorological Modeling
    - Mardia & Jupp (2000). Directional Statistics
"""

import pytest
import torch
import math

from relic.coupling.scales.hierarchical import HierarchyIndex, HierarchicalAggregator
from relic.core.state import (
    StateManager,
    StateVariableSpec,
    StateVariableType,
    UpdateItem,
    UpdateMode,
)


# Spatial basis of every variable these tests aggregate: area densities (per-area fluxes and
# energy fluxes), intensive values (resistances, conductances, fractions, angles) and class
# codes. Undeclared variables raise in the HierarchicalAggregator.
VARIABLE_METADATA = {
    **{name: {'spatial_basis': 'area_density'} for name in (
        'carbon', 'carbon_flux', 'et', 'evapotranspiration', 'flux', 'gpp', 'nitrogen', 'nitrogen_flux',
        'npp', 'precipitation', 'ra', 'runoff', 'storage_change', 'water', 'water_flux',
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


class TestHierarchyIndexIntegration:
    """Test integration between new aggregation strategies and HierarchyIndex"""

    @pytest.fixture
    def hierarchy_setup(self):
        """Setup hierarchical structure with 100 individuals, 10 patches, 1 landscape"""
        device = torch.device("cpu")

        # Create realistic hierarchy: 100 individuals -> 10 patches -> 1 landscape
        n_individuals = 100
        n_patches = 10
        n_landscape = 1
        individuals_per_patch = n_individuals // n_patches

        # Topology for individuals -> patches
        topology_ind_patch = {
            "n_source_units": n_individuals,
            "n_target_units": n_patches,
            "source_areas": torch.rand(n_individuals).tolist(),
            "target_areas": torch.zeros(n_patches).tolist(),
            "parent_map": {i: i // individuals_per_patch for i in range(n_individuals)}
        }

        # Calculate patch areas from individual areas
        for i, patch_id in topology_ind_patch["parent_map"].items():
            topology_ind_patch["target_areas"][patch_id] += topology_ind_patch["source_areas"][i]

        # Topology for patches -> landscape
        topology_patch_land = {
            "n_source_units": n_patches,
            "n_target_units": n_landscape,
            "source_areas": topology_ind_patch["target_areas"],
            "target_areas": [sum(topology_ind_patch["target_areas"])],
            "parent_map": {i: 0 for i in range(n_patches)}
        }

        # Create hierarchy indices
        index_ind_patch = HierarchyIndex(
            source_scale="individual",
            target_scale="patch",
            device=device,
            topology=topology_ind_patch
        )

        index_patch_land = HierarchyIndex(
            source_scale="patch",
            target_scale="landscape",
            device=device,
            topology=topology_patch_land
        )

        return {
            "device": device,
            "n_individuals": n_individuals,
            "n_patches": n_patches,
            "n_landscape": n_landscape,
            "topology_ind_patch": topology_ind_patch,
            "topology_patch_land": topology_patch_land,
            "index_ind_patch": index_ind_patch,
            "index_patch_land": index_patch_land
        }

    def test_hierarchy_matrix_with_harmonic_mean(self, hierarchy_setup):
        """Test resistance network aggregation with hierarchy matrix"""
        # Test harmonic mean aggregation with proper hierarchy matrix handling
        index = hierarchy_setup["index_ind_patch"]
        device = hierarchy_setup["device"]
        n_individuals = hierarchy_setup["n_individuals"]

        # Create aggregator with harmonic mean strategy
        aggregator = HierarchicalAggregator(
            index=index,
            device=device,
            hierarchy_levels=["individual", "patch"],
            weight_type="area",
            rule="conservative"
        )

        # Test data: stomatal resistance values (realistic range)
        resistance = torch.rand(n_individuals, device=device) * 100 + 50  # 50-150 s/m

        # Aggregate using harmonic mean
        data_dict = {"stomatal_resistance": resistance}
        aggregated_data, audit_info = aggregator.aggregate(
            data=data_dict,
            variables=["stomatal_resistance"],
            strategy="harmonic_mean",
            variable_metadata=VARIABLE_METADATA,
            mode="conservative"
        )

        # Verify results
        assert "stomatal_resistance" in aggregated_data
        aggregated_resistance = aggregated_data["stomatal_resistance"]

        # Check shape
        assert aggregated_resistance.shape[0] == hierarchy_setup["n_patches"]

        # Check that harmonic mean was used
        assert audit_info["method_used"]["stomatal_resistance"] == "harmonic_mean"

        # Area-weighted harmonic mean of per-area resistances (parallel paths over the parent
        # area): 1 / r_j = sum_i f_ij / r_i with f_ij = A_i / sum_k A_k, so r_j lies between the
        # smallest and the largest member resistance (a mean, not a parallel-sum bound).
        parent_map = hierarchy_setup["topology_ind_patch"]["parent_map"]
        areas = torch.as_tensor(hierarchy_setup["topology_ind_patch"]["source_areas"], dtype=resistance.dtype)
        for patch_id in range(hierarchy_setup["n_patches"]):
            individual_ids = [i for i, p in parent_map.items() if p == patch_id]
            if individual_ids:
                members = resistance[individual_ids]
                fractions = areas[individual_ids] / areas[individual_ids].sum()
                expected = 1.0 / (fractions / members).sum()
                torch.testing.assert_close(aggregated_resistance[patch_id].to(expected.dtype), expected, rtol=1e-5, atol=1e-6)
                assert members.min() - 1e-4 <= aggregated_resistance[patch_id] <= members.max() + 1e-4

        # Verify no NaN or Inf
        assert not torch.isnan(aggregated_resistance).any()
        assert not torch.isinf(aggregated_resistance).any()

    def test_multi_level_aggregation(self, hierarchy_setup):
        """Test individual -> patch -> landscape with mixed strategies"""
        device = hierarchy_setup["device"]
        n_individuals = hierarchy_setup["n_individuals"]

        # Create aggregators for two-level hierarchy
        agg_ind_patch = HierarchicalAggregator(
            index=hierarchy_setup["index_ind_patch"],
            device=device,
            hierarchy_levels=["individual", "patch"],
            weight_type="area",
            rule="conservative"
        )

        agg_patch_land = HierarchicalAggregator(
            index=hierarchy_setup["index_patch_land"],
            device=device,
            hierarchy_levels=["patch", "landscape"],
            weight_type="area",
            rule="conservative"
        )

        # Test data at individual level
        test_variables = {
            # Intensive variables
            "leaf_temperature": torch.rand(n_individuals, device=device) * 10 + 288,  # 288-298 K
            "stomatal_conductance": torch.rand(n_individuals, device=device) * 0.5,  # 0-0.5 m/s
            "vegetation_fraction": torch.rand(n_individuals, device=device),  # 0-1
            "land_cover_type": torch.randint(1, 5, (n_individuals,), device=device, dtype=torch.float32),
            "wind_direction": torch.rand(n_individuals, device=device) * 2 * math.pi,  # 0-2pi rad

            # Extensive variables
            "carbon_flux": torch.rand(n_individuals, device=device) * 10,  # g C/m2/s
        }

        # Strategy mapping for each variable
        strategies = {
            "leaf_temperature": "area_weighted_mean",
            "stomatal_conductance": "arithmetic_mean",
            "vegetation_fraction": "bounded_area_mean",
            "land_cover_type": "mode",
            "wind_direction": "circular_mean",
            "carbon_flux": "flow_conserving",
        }

        # Stage 1: Individual -> Patch aggregation
        patch_data = {}
        for var_name, var_data in test_variables.items():
            data_dict = {var_name: var_data}
            aggregated, audit = agg_ind_patch.aggregate(
                data=data_dict,
                variables=[var_name],
                strategy=strategies[var_name],
                variable_metadata=VARIABLE_METADATA,
                mode="conservative"
            )
            patch_data[var_name] = aggregated[var_name]

            # Verify correct strategy was used
            assert audit["method_used"][var_name] == strategies[var_name]

            # Verify shape
            assert patch_data[var_name].shape[0] == hierarchy_setup["n_patches"]

        # Stage 2: Patch -> Landscape aggregation
        landscape_data = {}
        for var_name, var_data in patch_data.items():
            data_dict = {var_name: var_data}
            aggregated, audit = agg_patch_land.aggregate(
                data=data_dict,
                variables=[var_name],
                strategy=strategies[var_name],
                variable_metadata=VARIABLE_METADATA,
                mode="conservative"
            )
            landscape_data[var_name] = aggregated[var_name]

            # Verify shape (scalar landscape)
            assert landscape_data[var_name].shape[0] == hierarchy_setup["n_landscape"]

        # Verify landscape values are reasonable
        assert 288 <= landscape_data["leaf_temperature"][0] <= 298
        assert 0 <= landscape_data["stomatal_conductance"][0] <= 0.5
        assert 0 <= landscape_data["vegetation_fraction"][0] <= 1
        assert 1 <= landscape_data["land_cover_type"][0] <= 4
        assert -math.pi <= landscape_data["wind_direction"][0] <= math.pi
        assert landscape_data["carbon_flux"][0] >= 0

        # Verify no NaN or Inf in final results
        for var_name, var_data in landscape_data.items():
            assert not torch.isnan(var_data).any(), f"{var_name} contains NaN"
            assert not torch.isinf(var_data).any(), f"{var_name} contains Inf"

    def test_hierarchy_consistency_checks(self, hierarchy_setup):
        """Verify parent-child mapping consistency"""
        index = hierarchy_setup["index_ind_patch"]
        topology = hierarchy_setup["topology_ind_patch"]

        # Verify parent_map consistency with hierarchy matrix
        parent_map = topology["parent_map"]
        hierarchy_matrix = index.get_hierarchy_matrix()

        for child_id, parent_id in parent_map.items():
            # Verify hierarchy matrix entry
            assert hierarchy_matrix[child_id, parent_id] > 0, \
                f"Missing hierarchy matrix entry for child {child_id} -> parent {parent_id}"

        # Verify each individual belongs to exactly one patch
        row_sums = hierarchy_matrix.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
            "Each individual should belong to exactly one patch"

        # Verify area conservation
        source_areas = index.get_source_areas()
        target_areas = index.get_target_areas()

        total_source_area = source_areas.sum()
        total_target_area = target_areas.sum()

        assert torch.allclose(total_source_area, total_target_area, atol=1e-6), \
            f"Area not conserved: source={total_source_area:.6f}, target={total_target_area:.6f}"


class TestStateManagerIntegration:
    """Test integration with StateManager"""

    @pytest.fixture
    def state_manager_setup(self):
        """Setup StateManager with realistic schema"""
        device = torch.device("cpu")

        # Create schema for state variables
        schema = {
            # Extensive variables (conserved)
            "carbon_flux_patch": StateVariableSpec(
                name="carbon_flux_patch",
                units="g C m-2 s-1",
                bounds=(0.0, 100.0),
                conserved=True,
                spatial_dims=1,
                required=False,
                variable_type=StateVariableType.GRID
            ),
            "nitrogen_flux_patch": StateVariableSpec(
                name="nitrogen_flux_patch",
                units="g N m-2 s-1",
                bounds=(0.0, 10.0),
                conserved=True,
                spatial_dims=1,
                required=False,
                variable_type=StateVariableType.GRID
            ),

            # Intensive variables (non-conserved)
            "leaf_temperature_patch": StateVariableSpec(
                name="leaf_temperature_patch",
                units="K",
                bounds=(273.0, 320.0),
                conserved=False,
                spatial_dims=1,
                required=False,
                variable_type=StateVariableType.GRID
            ),
            "stomatal_conductance_patch": StateVariableSpec(
                name="stomatal_conductance_patch",
                units="m s-1",
                bounds=(0.0, 1.0),
                conserved=False,
                spatial_dims=1,
                required=False,
                variable_type=StateVariableType.GRID
            ),

            # Landscape-scale aggregated variables
            "carbon_flux_landscape": StateVariableSpec(
                name="carbon_flux_landscape",
                units="g C m-2 s-1",
                bounds=(0.0, 100.0),
                conserved=True,
                spatial_dims=0,
                required=False,
                variable_type=StateVariableType.SCALAR
            ),
            "leaf_temperature_landscape": StateVariableSpec(
                name="leaf_temperature_landscape",
                units="K",
                bounds=(273.0, 320.0),
                conserved=False,
                spatial_dims=0,
                required=False,
                variable_type=StateVariableType.SCALAR
            ),
        }

        # Create state manager
        state_manager = StateManager(schema=schema, device=device)

        # Initialize with patch-level data (10 patches)
        n_patches = 10
        initial_state = {
            "carbon_flux_patch": torch.rand(n_patches, device=device) * 10,
            "nitrogen_flux_patch": torch.rand(n_patches, device=device) * 1,
            "leaf_temperature_patch": torch.rand(n_patches, device=device) * 10 + 288,
            "stomatal_conductance_patch": torch.rand(n_patches, device=device) * 0.5,
            "carbon_flux_landscape": torch.tensor([0.0], device=device),
            "leaf_temperature_landscape": torch.tensor([288.0], device=device),
        }

        state_manager.initialize(initial_state)

        return {
            "state_manager": state_manager,
            "device": device,
            "n_patches": n_patches,
            "schema": schema
        }

    def test_aggregated_state_updates(self, state_manager_setup):
        """Test applying aggregated results to state"""
        state_manager = state_manager_setup["state_manager"]
        device = state_manager_setup["device"]
        n_patches = state_manager_setup["n_patches"]

        # Create simple patch->landscape hierarchy
        topology = {
            "n_source_units": n_patches,
            "n_target_units": 1,
            "source_areas": torch.rand(n_patches).tolist(),
            "target_areas": [0.0],
            "parent_map": {i: 0 for i in range(n_patches)}
        }
        topology["target_areas"][0] = sum(topology["source_areas"])

        index = HierarchyIndex(
            source_scale="patch",
            target_scale="landscape",
            device=device,
            topology=topology
        )

        aggregator = HierarchicalAggregator(
            index=index,
            device=device,
            hierarchy_levels=["patch", "landscape"],
            weight_type="area",
            rule="conservative"
        )

        # Get current patch-level state
        current_state = state_manager.get_state()
        carbon_flux_patch = current_state["carbon_flux_patch"]
        leaf_temp_patch = current_state["leaf_temperature_patch"]

        # Aggregate to landscape scale
        carbon_aggregated, _ = aggregator.aggregate(
            data={"carbon_flux": carbon_flux_patch},
            variables=["carbon_flux"],
            strategy="flow_conserving",
            variable_metadata=VARIABLE_METADATA,
            mode="conservative"
        )

        temp_aggregated, _ = aggregator.aggregate(
            data={"leaf_temperature": leaf_temp_patch},
            variables=["leaf_temperature"],
            strategy="area_weighted_mean",
            variable_metadata=VARIABLE_METADATA,
            mode="conservative"
        )

        # Create state updates
        updates = {
            "carbon_flux_landscape": UpdateItem(
                value=carbon_aggregated["carbon_flux"],
                mode=UpdateMode.REPLACE,
                per_timestep=True,
            ),
            "leaf_temperature_landscape": UpdateItem(
                value=temp_aggregated["leaf_temperature"],
                mode=UpdateMode.REPLACE,
                per_timestep=True,
            ),
        }

        # Apply updates through StateManager
        state_manager.apply_updates(
            updates=updates,
            dt=1800.0,
            check_conservation=False,  # Not checking conservation for non-conserved updates
            rollback_on_failure=False
        )

        # Verify state was updated
        new_state = state_manager.get_state()
        assert torch.allclose(
            new_state["carbon_flux_landscape"],
            carbon_aggregated["carbon_flux"],
            atol=1e-6
        )
        assert torch.allclose(
            new_state["leaf_temperature_landscape"],
            temp_aggregated["leaf_temperature"],
            atol=1e-6
        )

        # Verify values are within bounds
        assert 273.0 <= new_state["leaf_temperature_landscape"][0] <= 320.0
        assert 0.0 <= new_state["carbon_flux_landscape"][0] <= 100.0

    def test_conservation_verification(self, state_manager_setup):
        """Test conservation checks with new strategies"""
        state_manager = state_manager_setup["state_manager"]
        device = state_manager_setup["device"]
        n_patches = state_manager_setup["n_patches"]

        # Set the strictest accepted conservation tolerance: the declared lower bound.
        from relic.parameters import get_global_manager

        state_manager.conservation_tolerance = get_global_manager().get_parameter(
            "validation.conservation.conservation_tolerance"
        ).bounds[0]

        # Create topology with explicit areas
        patch_areas = torch.rand(n_patches, device=device) + 1.0  # Areas 1-2
        landscape_area = patch_areas.sum()

        topology = {
            "n_source_units": n_patches,
            "n_target_units": 1,
            "source_areas": patch_areas.tolist(),
            "target_areas": [landscape_area.item()],
            "parent_map": {i: 0 for i in range(n_patches)}
        }

        index = HierarchyIndex(
            source_scale="patch",
            target_scale="landscape",
            device=device,
            topology=topology
        )

        aggregator = HierarchicalAggregator(
            index=index,
            device=device,
            hierarchy_levels=["patch", "landscape"],
            weight_type="area",
            rule="strict_conservative"
        )

        # Test extensive variable (carbon flux density)
        carbon_flux = torch.rand(n_patches, device=device) * 10  # g C/m2/s

        # Aggregate with strict conservation
        aggregated, audit = aggregator.aggregate(
            data={"carbon_flux": carbon_flux},
            variables=["carbon_flux"],
            strategy="flow_conserving",
            variable_metadata=VARIABLE_METADATA,
            mode="strict_conservative"
        )

        # Verify conservation: total_flux = flux_density * area
        source_total = (carbon_flux * patch_areas).sum()
        target_total = aggregated["carbon_flux"][0] * landscape_area

        conservation_error = abs(source_total - target_total)
        relative_error = conservation_error / source_total

        # Check conservation error is within tolerance
        assert relative_error < 1e-12, \
            f"Conservation violated: relative error = {relative_error:.2e}"

        # Verify audit info includes conservation check
        assert "conservation_errors" in audit
        assert "carbon_flux" in audit["conservation_errors"]

    def test_transaction_rollback(self, state_manager_setup):
        """Test rollback on conservation failure"""
        state_manager = state_manager_setup["state_manager"]
        device = state_manager_setup["device"]

        # Get initial state snapshot
        initial_state = state_manager.get_state()
        initial_carbon = initial_state["carbon_flux_patch"].clone()

        # Create intentionally violating update (negative carbon flux)
        violating_update = {
            "carbon_flux_patch": torch.tensor([-100.0] * state_manager_setup["n_patches"], device=device)
        }

        # Attempt to apply update with rollback enabled
        with pytest.raises(ValueError):
            state_manager.apply_updates(
                updates=violating_update,
                dt=1800.0,
                check_conservation=True,
                rollback_on_failure=True
            )

        # Verify state was rolled back
        current_state = state_manager.get_state()
        assert torch.allclose(
            current_state["carbon_flux_patch"],
            initial_carbon,
            atol=1e-10
        ), "State was not properly rolled back"


class TestBatchAggregation:
    """Test batch processing of multiple variables"""

    @pytest.fixture
    def batch_setup(self):
        """Setup for batch aggregation tests"""
        device = torch.device("cpu")
        n_individuals = 100
        n_patches = 10

        topology = {
            "n_source_units": n_individuals,
            "n_target_units": n_patches,
            "source_areas": torch.rand(n_individuals).tolist(),
            "target_areas": torch.zeros(n_patches).tolist(),
            "parent_map": {i: i // (n_individuals // n_patches) for i in range(n_individuals)}
        }

        for i, patch_id in topology["parent_map"].items():
            topology["target_areas"][patch_id] += topology["source_areas"][i]

        index = HierarchyIndex(
            source_scale="individual",
            target_scale="patch",
            device=device,
            topology=topology
        )

        aggregator = HierarchicalAggregator(
            index=index,
            device=device,
            hierarchy_levels=["individual", "patch"],
            weight_type="area",
            rule="conservative"
        )

        return {
            "device": device,
            "n_individuals": n_individuals,
            "n_patches": n_patches,
            "index": index,
            "aggregator": aggregator
        }

    def test_mixed_strategy_batch(self, batch_setup):
        """Aggregate multiple variables with different strategies simultaneously"""
        aggregator = batch_setup["aggregator"]
        n_individuals = batch_setup["n_individuals"]
        device = batch_setup["device"]

        # Create test dataset with all 7 strategy types
        test_data = {
            "stomatal_resistance": torch.rand(n_individuals, device=device) * 100 + 50,  # harmonic_mean
            "stomatal_conductance": torch.rand(n_individuals, device=device) * 0.5,  # arithmetic_mean
            "vegetation_fraction": torch.rand(n_individuals, device=device),  # bounded_area_mean
            "land_cover_type": torch.randint(1, 5, (n_individuals,), device=device, dtype=torch.float32),  # mode
            "wind_direction": torch.rand(n_individuals, device=device) * 2 * math.pi,  # circular_mean
            "hydraulic_conductivity": torch.rand(n_individuals, device=device) * 1e-5,  # geometric_mean
        }

        # Strategy mapping
        strategies_map = {
            "stomatal_resistance": "harmonic_mean",
            "stomatal_conductance": "arithmetic_mean",
            "vegetation_fraction": "bounded_area_mean",
            "land_cover_type": "mode",
            "wind_direction": "circular_mean",
            "hydraulic_conductivity": "geometric_mean",
        }

        # Batch aggregate all variables
        all_results = {}
        all_audits = {}

        for var_name, var_data in test_data.items():
            data_dict = {var_name: var_data}
            strategy = strategies_map[var_name]

            aggregated, audit = aggregator.aggregate(
                data=data_dict,
                variables=[var_name],
                strategy=strategy,
                variable_metadata=VARIABLE_METADATA,
                mode="conservative"
            )

            all_results[var_name] = aggregated[var_name]
            all_audits[var_name] = audit

        # Verify all results
        for var_name, result in all_results.items():
            # Check shape
            assert result.shape[0] == batch_setup["n_patches"]

            # Check no NaN or Inf
            assert not torch.isnan(result).any(), f"{var_name} contains NaN"
            assert not torch.isinf(result).any(), f"{var_name} contains Inf"

            # Check correct strategy was used
            assert all_audits[var_name]["method_used"][var_name] == strategies_map[var_name]

        # Check specific variable constraints
        assert all_results["vegetation_fraction"].min() >= 0.0
        assert all_results["vegetation_fraction"].max() <= 1.0
        assert (all_results["wind_direction"] >= -math.pi).all()
        assert (all_results["wind_direction"] <= math.pi).all()
        assert all_results["land_cover_type"].min() >= 1.0
        assert all_results["land_cover_type"].max() <= 4.0

    def test_performance_consistency(self, batch_setup):
        """Verify batch aggregation maintains same accuracy as individual"""
        aggregator = batch_setup["aggregator"]
        n_individuals = batch_setup["n_individuals"]
        device = batch_setup["device"]

        # Create test data
        test_var = torch.rand(n_individuals, device=device) * 10

        # Individual aggregation
        individual_result, _ = aggregator.aggregate(
            data={"test_var": test_var},
            variables=["test_var"],
            strategy="area_weighted_mean",
            variable_metadata=VARIABLE_METADATA,
            mode="conservative"
        )

        # Batch aggregation (simulated by aggregating same variable multiple times)
        batch_results = []
        for _ in range(5):
            result, _ = aggregator.aggregate(
                data={"test_var": test_var},
                variables=["test_var"],
                strategy="area_weighted_mean",
                variable_metadata=VARIABLE_METADATA,
                mode="conservative"
            )
            batch_results.append(result["test_var"])

        # Verify consistency: all results should be identical
        for batch_result in batch_results:
            assert torch.allclose(
                individual_result["test_var"],
                batch_result,
                atol=1e-10
            ), "Batch aggregation results differ from individual"


class TestEndToEndFlow:
    """Test complete aggregation workflows"""

    @pytest.fixture
    def end_to_end_setup(self):
        """Setup complete Individual -> Patch -> Landscape pipeline"""
        device = torch.device("cpu")
        n_individuals = 100
        n_patches = 10
        n_landscape = 1

        # Create complete hierarchy
        ind_areas = torch.rand(n_individuals) + 0.5

        topology_ind_patch = {
            "n_source_units": n_individuals,
            "n_target_units": n_patches,
            "source_areas": ind_areas.tolist(),
            "target_areas": torch.zeros(n_patches).tolist(),
            "parent_map": {i: i // (n_individuals // n_patches) for i in range(n_individuals)}
        }

        for i, patch_id in topology_ind_patch["parent_map"].items():
            topology_ind_patch["target_areas"][patch_id] += topology_ind_patch["source_areas"][i]

        topology_patch_land = {
            "n_source_units": n_patches,
            "n_target_units": n_landscape,
            "source_areas": topology_ind_patch["target_areas"],
            "target_areas": [sum(topology_ind_patch["target_areas"])],
            "parent_map": {i: 0 for i in range(n_patches)}
        }

        index_ind_patch = HierarchyIndex(
            source_scale="individual",
            target_scale="patch",
            device=device,
            topology=topology_ind_patch
        )

        index_patch_land = HierarchyIndex(
            source_scale="patch",
            target_scale="landscape",
            device=device,
            topology=topology_patch_land
        )

        agg_ind_patch = HierarchicalAggregator(
            index=index_ind_patch,
            device=device,
            hierarchy_levels=["individual", "patch"],
            weight_type="area",
            rule="strict_conservative"
        )

        agg_patch_land = HierarchicalAggregator(
            index=index_patch_land,
            device=device,
            hierarchy_levels=["patch", "landscape"],
            weight_type="area",
            rule="strict_conservative"
        )

        return {
            "device": device,
            "n_individuals": n_individuals,
            "n_patches": n_patches,
            "n_landscape": n_landscape,
            "topology_ind_patch": topology_ind_patch,
            "topology_patch_land": topology_patch_land,
            "index_ind_patch": index_ind_patch,
            "index_patch_land": index_patch_land,
            "agg_ind_patch": agg_ind_patch,
            "agg_patch_land": agg_patch_land
        }

    def test_metadata_driven_strategy_selection(self, end_to_end_setup):
        """Use model metadata to automatically select strategies"""
        device = end_to_end_setup["device"]
        n_individuals = end_to_end_setup["n_individuals"]

        # Define variable metadata with strategy hints
        variable_metadata = {
            "stomatal_resistance": {
                "units": "s m-1",
                "type": "intensive",
                "physical_meaning": "resistance_network",
                "recommended_strategy": "harmonic_mean"
            },
            "stomatal_conductance": {
                "units": "m s-1",
                "type": "intensive",
                "physical_meaning": "conductance",
                "recommended_strategy": "arithmetic_mean"
            },
            "carbon_flux": {
                "units": "g C m-2 s-1",
                "type": "extensive",
                "physical_meaning": "flux",
                "recommended_strategy": "flow_conserving"
            },
            "vegetation_fraction": {
                "units": "dimensionless",
                "type": "intensive",
                "physical_meaning": "fraction",
                "recommended_strategy": "bounded_area_mean"
            },
        }

        # Generate test data
        test_data = {
            "stomatal_resistance": torch.rand(n_individuals, device=device) * 100 + 50,
            "stomatal_conductance": torch.rand(n_individuals, device=device) * 0.5,
            "carbon_flux": torch.rand(n_individuals, device=device) * 10,
            "vegetation_fraction": torch.rand(n_individuals, device=device),
        }

        # Aggregate using metadata-driven strategy selection
        agg = end_to_end_setup["agg_ind_patch"]

        results = {}
        for var_name, var_data in test_data.items():
            metadata = variable_metadata[var_name]
            strategy = metadata["recommended_strategy"]

            data_dict = {var_name: var_data}
            aggregated, audit = agg.aggregate(
                data=data_dict,
                variables=[var_name],
                strategy=strategy,
                variable_metadata=VARIABLE_METADATA,
                mode="strict_conservative"
            )

            results[var_name] = aggregated[var_name]

            # Verify correct strategy was applied
            assert audit["method_used"][var_name] == strategy

        # Verify all results are valid
        for var_name, result in results.items():
            assert result.shape[0] == end_to_end_setup["n_patches"]
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()

    def test_complete_multiscale_pipeline(self, end_to_end_setup):
        """Full Individual -> Patch -> Landscape with all 7 strategies"""
        device = end_to_end_setup["device"]
        n_individuals = end_to_end_setup["n_individuals"]
        agg_ind_patch = end_to_end_setup["agg_ind_patch"]
        agg_patch_land = end_to_end_setup["agg_patch_land"]

        # Individual-level data with all 7 strategy types
        individual_data = {
            "stomatal_resistance": torch.rand(n_individuals, device=device) * 100 + 50,
            "stomatal_conductance": torch.rand(n_individuals, device=device) * 0.5,
            "vegetation_fraction": torch.rand(n_individuals, device=device),
            "land_cover_type": torch.randint(1, 5, (n_individuals,), device=device, dtype=torch.float32),
            "wind_direction": torch.rand(n_individuals, device=device) * 2 * math.pi,
            "carbon_flux": torch.rand(n_individuals, device=device) * 10,
            "hydraulic_conductivity": torch.rand(n_individuals, device=device) * 1e-5,
        }

        strategies = {
            "stomatal_resistance": "harmonic_mean",
            "stomatal_conductance": "arithmetic_mean",
            "vegetation_fraction": "bounded_area_mean",
            "land_cover_type": "mode",
            "wind_direction": "circular_mean",
            "carbon_flux": "flow_conserving",
            "hydraulic_conductivity": "geometric_mean",
        }

        # Stage 1: Individual -> Patch
        patch_data = {}
        for var_name, var_data in individual_data.items():
            data_dict = {var_name: var_data}
            aggregated, audit = agg_ind_patch.aggregate(
                data=data_dict,
                variables=[var_name],
                strategy=strategies[var_name],
                variable_metadata=VARIABLE_METADATA,
                mode="strict_conservative"
            )
            patch_data[var_name] = aggregated[var_name]

            # Verify patch-level results
            assert patch_data[var_name].shape[0] == end_to_end_setup["n_patches"]
            assert not torch.isnan(patch_data[var_name]).any()
            assert not torch.isinf(patch_data[var_name]).any()

        # Stage 2: Patch -> Landscape
        landscape_data = {}
        for var_name, var_data in patch_data.items():
            data_dict = {var_name: var_data}
            aggregated, audit = agg_patch_land.aggregate(
                data=data_dict,
                variables=[var_name],
                strategy=strategies[var_name],
                variable_metadata=VARIABLE_METADATA,
                mode="strict_conservative"
            )
            landscape_data[var_name] = aggregated[var_name]

            # Verify landscape-level results
            assert landscape_data[var_name].shape[0] == end_to_end_setup["n_landscape"]
            assert not torch.isnan(landscape_data[var_name]).any()
            assert not torch.isinf(landscape_data[var_name]).any()

        # Verify landscape values are within expected ranges
        assert 50 <= landscape_data["stomatal_resistance"][0] <= 150
        assert 0 <= landscape_data["stomatal_conductance"][0] <= 0.5
        assert 0 <= landscape_data["vegetation_fraction"][0] <= 1
        assert 1 <= landscape_data["land_cover_type"][0] <= 4
        assert -math.pi <= landscape_data["wind_direction"][0] <= math.pi
        assert landscape_data["carbon_flux"][0] >= 0
        assert landscape_data["hydraulic_conductivity"][0] > 0

        # Verify conservation for extensive variables (carbon flux)
        ind_areas = torch.tensor(end_to_end_setup["topology_ind_patch"]["source_areas"], device=device)
        land_area = torch.tensor(end_to_end_setup["topology_patch_land"]["target_areas"], device=device)

        total_carbon_ind = (individual_data["carbon_flux"] * ind_areas).sum()
        total_carbon_land = landscape_data["carbon_flux"][0] * land_area[0]

        conservation_error = abs(total_carbon_ind - total_carbon_land) / total_carbon_ind
        assert conservation_error < 1e-12, \
            f"Carbon conservation violated: relative error = {conservation_error:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

