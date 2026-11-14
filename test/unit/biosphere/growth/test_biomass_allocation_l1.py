"""
L1 Unit Tests for Biomass Allocation Module

Comprehensive testing of the carbon allocation model including PFT-based allocation,
environmental controls, phenological modifications, and allometric constraints.

Test Coverage:
- Basic instantiation and parameter initialization
- Base allocation fraction validation
- Environmental control mechanisms (temperature, water stress)
- Phenological control mechanisms (LAI, growth phase)
- Canonical NPP allocation without a second construction-respiration law
- Allocation normalization and mass balance
- PFT parameter loading and validation
- Parameter sensitivity analysis
- Boundary conditions and robustness
- BiologicalProcess interface consistency

Scientific Validation:
- Allocation fractions: f_leaf + f_stem + f_root = 1
- Growth rates: dB_i/dt = f_i 脳 NPP 脳 畏_i (construction efficiency)
- Environmental modification: f_i = f_i,base 脳 M_env(T, W, N)
- Phenological control: allocation shifts based on growth phase
- Mass conservation: NPP = 危(growth_i + respiration_i)

References:
- Gao et al. (2025). pnetr: An R package for the PnET model family
- Zhang et al. (2011). CASA-CNP carbon allocation routines
- Sands & Landsberg (2002). 3-PG allocation strategies
- Franklin et al. (2012). Modeling carbon allocation in trees
- Poorter et al. (2012). Biomass allocation to leaves, stems and roots
"""

import torch
import pytest
from typing import Dict
from unittest.mock import Mock, patch

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation


CANONICAL_OUTPUTS = {
    "growth_rate_leaf",
    "growth_rate_stem",
    "growth_rate_root",
    "leaf_biomass_c",
    "wood_biomass_c",
    "root_biomass_c",
    "diagnostics.biomass_allocation.total_growth_rate",
    "diagnostics.biomass_allocation.allocation_fraction_leaf",
    "diagnostics.biomass_allocation.allocation_fraction_stem",
    "diagnostics.biomass_allocation.allocation_fraction_root",
    "diagnostics.biomass_allocation.allocation_sum",
    "diagnostics.biomass_allocation.environmental_modifier",
    "diagnostics.biomass_allocation.phenology_modifier",
    "diagnostics.biomass_allocation.carbon_balance_residual",
}


def output_values(output: Dict[str, UpdateItem]) -> Dict[str, torch.Tensor]:
    """Return values after tests have separately validated update semantics."""
    return {key: item.value for key, item in output.items()}


def replace_update_value(output, key, value):
    """Replace one UpdateItem value while preserving its runtime contract."""
    item = output[key]
    output[key] = UpdateItem(
        value=value,
        mode=item.mode,
        unit=item.unit,
        per_timestep=item.per_timestep,
        element=item.element,
        source_functor=item.source_functor,
    )


class TestBiomassAllocationL1:
    """L1 unit tests for biomass allocation model."""
    
    @pytest.fixture
    def allocation_model(self):
        """Create basic allocation model for testing."""
        return BiomassAllocation(
            name="test_biomass_allocation",
            enable_environmental_control=True,
            enable_phenology_control=True,
            enable_allometry_control=False,
        )
    
    def test_basic_instantiation(self, allocation_model):
        """Test L1.1: Basic instantiation and parameter initialization."""
        assert isinstance(allocation_model, BiomassAllocation)
        
        # Check base allocation fractions
        assert hasattr(allocation_model, 'f_leaf_base')
        assert hasattr(allocation_model, 'f_stem_base')
        assert hasattr(allocation_model, 'f_root_base')
        
        # Verify allocation fractions sum to 1.0
        total_allocation = (
            allocation_model.f_leaf_base +
            allocation_model.f_stem_base +
            allocation_model.f_root_base
        )
        assert total_allocation.item() == pytest.approx(1.0, abs=0.001)
        
        # Check control flags
        assert allocation_model.enable_environmental_control == True
        assert allocation_model.enable_phenology_control == True
        assert allocation_model.enable_allometry_control == False
        assert set(allocation_model.metadata.conservation_laws) == {
            "carbon_partition"
        }
        
        # GrowthRespiration is the only production construction-respiration law.
        assert not hasattr(allocation_model, "construction_eff_leaf")
        assert not hasattr(allocation_model, "growth_respiration_model")
        
    def test_dependencies_and_provides(self, allocation_model):
        """Test L1.2: Dependencies and provides interface."""
        deps = allocation_model.get_dependencies()
        required_inputs = allocation_model.get_required_inputs()
        provides = allocation_model.get_provides()
        
        # Check required dependencies
        assert deps == [
            "autotrophic_respiration",
            "water_stress",
            "gsi_phenology",
        ]
        assert 'net_primary_productivity' in required_inputs
        assert 'air_temperature' in required_inputs
        assert 'water_stress_factor' in required_inputs
        assert 'leaf_area_index' in required_inputs  # Phenology control enabled
        assert 'phenology_phase' in required_inputs
        assert "phenological_factor" not in required_inputs
        
        # Check provided variables
        assert 'growth_rate_leaf' in provides
        assert 'growth_rate_stem' in provides
        assert 'growth_rate_root' in provides
        assert set(provides) == CANONICAL_OUTPUTS
        assert set(allocation_model.metadata.output_variables) == CANONICAL_OUTPUTS
    
    def test_base_allocation_mechanics(self, allocation_model):
        """Test L1.3: Base allocation mechanics without controls."""
        # Disable controls for pure base allocation test
        allocation_model.enable_environmental_control = False
        allocation_model.enable_phenology_control = False
        
        # Standard NPP input
        state = {
            'net_primary_productivity': torch.tensor(2.0e-7),
        }
        
        raw_results = allocation_model.forward(state, time=0.0, dt=3600.0)
        results = output_values(raw_results)
        
        # Check output structure
        assert 'growth_rate_leaf' in results
        assert 'growth_rate_stem' in results
        assert 'growth_rate_root' in results
        assert 'diagnostics.biomass_allocation.total_growth_rate' in results
        
        # Check allocation fractions
        f_leaf = results['diagnostics.biomass_allocation.allocation_fraction_leaf']
        f_stem = results['diagnostics.biomass_allocation.allocation_fraction_stem']
        f_root = results['diagnostics.biomass_allocation.allocation_fraction_root']
        
        # Should equal base fractions when no controls are applied
        assert f_leaf.item() == pytest.approx(allocation_model.f_leaf_base.item(), rel=0.01)
        assert f_stem.item() == pytest.approx(allocation_model.f_stem_base.item(), rel=0.01)
        assert f_root.item() == pytest.approx(allocation_model.f_root_base.item(), rel=0.01)
        
        # Fractions should sum to 1.0
        total_fractions = f_leaf + f_stem + f_root
        assert total_fractions.item() == pytest.approx(1.0, abs=0.001)
        
        # Check growth rates proportional to allocation fractions
        npp = 2.0e-7
        assert results['growth_rate_leaf'].item() == pytest.approx(npp * f_leaf.item(), rel=1e-6)
        assert results['growth_rate_stem'].item() == pytest.approx(npp * f_stem.item(), rel=1e-6)
        assert results['growth_rate_root'].item() == pytest.approx(npp * f_root.item(), rel=1e-6)
        assert results['leaf_biomass_c'].item() == pytest.approx(
            results['growth_rate_leaf'].item() * 3600.0,
            rel=1e-6,
        )
    
    def test_environmental_controls_temperature(self, allocation_model):
        """Test L1.4: Temperature-dependent environmental controls."""
        base_state = {
            'net_primary_productivity': torch.tensor(2.0e-7),
            'water_stress_factor': torch.tensor(1.0),
        }
        
        # Cold temperature
        state_cold = base_state.copy()
        state_cold['air_temperature'] = torch.tensor(283.15)
        
        # Warm temperature
        state_warm = base_state.copy()
        state_warm['air_temperature'] = torch.tensor(313.15)
        
        # Disable phenology for pure temperature test
        allocation_model.enable_phenology_control = False
        
        results_cold = output_values(
            allocation_model.forward(state_cold, time=0.0, dt=3600.0)
        )
        results_warm = output_values(
            allocation_model.forward(state_warm, time=0.0, dt=3600.0)
        )
        
        # Extract allocation fractions
        f_root_cold = results_cold['diagnostics.biomass_allocation.allocation_fraction_root'].item()
        f_root_warm = results_warm['diagnostics.biomass_allocation.allocation_fraction_root'].item()
        
        # Higher temperature should increase root allocation (drought adaptation)
        assert f_root_warm > f_root_cold, "Higher temperature should increase root allocation"
        
        # Check environmental modifier
        env_mod_cold = results_cold['diagnostics.biomass_allocation.environmental_modifier']
        env_mod_warm = results_warm['diagnostics.biomass_allocation.environmental_modifier']
        
        # Should show temperature effect
        assert torch.isfinite(env_mod_cold).all()
        assert torch.isfinite(env_mod_warm).all()
        
        # Verify mass conservation
        total_cold = (
            results_cold['diagnostics.biomass_allocation.allocation_fraction_leaf']
            + results_cold['diagnostics.biomass_allocation.allocation_fraction_stem']
            + results_cold['diagnostics.biomass_allocation.allocation_fraction_root']
        )
        total_warm = (
            results_warm['diagnostics.biomass_allocation.allocation_fraction_leaf']
            + results_warm['diagnostics.biomass_allocation.allocation_fraction_stem']
            + results_warm['diagnostics.biomass_allocation.allocation_fraction_root']
        )
        
        assert total_cold.item() == pytest.approx(1.0, abs=0.01)
        assert total_warm.item() == pytest.approx(1.0, abs=0.01)
    
    def test_environmental_controls_water_stress(self, allocation_model):
        """Test L1.5: Water stress effects on allocation."""
        base_state = {
            'net_primary_productivity': torch.tensor(2.0e-7),
            'air_temperature': torch.tensor(298.15),
        }
        
        # No water stress
        state_unstressed = base_state.copy()
        state_unstressed['water_stress_factor'] = torch.tensor(1.0)
        
        # High water stress
        state_stressed = base_state.copy()
        state_stressed['water_stress_factor'] = torch.tensor(0.3)
        
        # Disable phenology for pure water stress test
        allocation_model.enable_phenology_control = False
        
        results_unstressed = output_values(
            allocation_model.forward(state_unstressed, time=0.0, dt=3600.0)
        )
        results_stressed = output_values(
            allocation_model.forward(state_stressed, time=0.0, dt=3600.0)
        )
        
        # Water stress should increase root allocation
        f_root_unstressed = results_unstressed['diagnostics.biomass_allocation.allocation_fraction_root'].item()
        f_root_stressed = results_stressed['diagnostics.biomass_allocation.allocation_fraction_root'].item()
        
        assert f_root_stressed > f_root_unstressed, "Water stress should increase root allocation"
        
        # Should decrease leaf allocation under stress
        f_leaf_unstressed = results_unstressed['diagnostics.biomass_allocation.allocation_fraction_leaf'].item()
        f_leaf_stressed = results_stressed['diagnostics.biomass_allocation.allocation_fraction_leaf'].item()
        
        assert f_leaf_stressed < f_leaf_unstressed, "Water stress should decrease leaf allocation"
        
        # Check environmental modifier reflects stress
        env_mod_unstressed = results_unstressed['diagnostics.biomass_allocation.environmental_modifier'].item()
        env_mod_stressed = results_stressed['diagnostics.biomass_allocation.environmental_modifier'].item()
        
        assert env_mod_stressed < env_mod_unstressed, "Environmental modifier should reflect stress"
    
    def test_phenological_controls_phases(self, allocation_model):
        """Test L1.6: Phenological phase effects on allocation."""
        base_state = {
            'net_primary_productivity': torch.tensor(2.0e-7),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(1.0),
            'leaf_area_index': torch.tensor(3.0),
        }
        
        # Disable environmental control for pure phenology test
        allocation_model.enable_environmental_control = False
        
        # Test different phenological phases
        # Dormant phase (0)
        state_dormant = base_state.copy()
        state_dormant['phenology_phase'] = torch.tensor(0)
        
        # Greenup phase (1)
        state_greenup = base_state.copy()
        state_greenup['phenology_phase'] = torch.tensor(1)
        
        # Mature phase (2)
        state_mature = base_state.copy()
        state_mature['phenology_phase'] = torch.tensor(2)
        
        results_dormant = output_values(
            allocation_model.forward(state_dormant, time=0.0, dt=3600.0)
        )
        results_greenup = output_values(
            allocation_model.forward(state_greenup, time=0.0, dt=3600.0)
        )
        results_mature = output_values(
            allocation_model.forward(state_mature, time=0.0, dt=3600.0)
        )
        
        # Dormant phase should have no leaf allocation
        f_leaf_dormant = results_dormant['diagnostics.biomass_allocation.allocation_fraction_leaf'].item()
        assert f_leaf_dormant == pytest.approx(0.0, abs=0.01), "Dormant phase should have no leaf allocation"
        
        # Greenup phase should have high leaf allocation
        f_leaf_greenup = results_greenup['diagnostics.biomass_allocation.allocation_fraction_leaf'].item()
        f_leaf_mature = results_mature['diagnostics.biomass_allocation.allocation_fraction_leaf'].item()
        
        assert f_leaf_greenup > f_leaf_mature, "Greenup phase should have higher leaf allocation"
        assert f_leaf_greenup >= 0.7, "Greenup phase should have high leaf allocation (>70%)"
        
        # Check phenology modifiers
        pheno_mod_dormant = results_dormant['diagnostics.biomass_allocation.phenology_modifier'].item()
        pheno_mod_greenup = results_greenup['diagnostics.biomass_allocation.phenology_modifier'].item()
        pheno_mod_mature = results_mature['diagnostics.biomass_allocation.phenology_modifier'].item()
        
        assert pheno_mod_dormant < pheno_mod_mature < pheno_mod_greenup
        
        # Mass conservation should be maintained in all phases
        for results in [results_dormant, results_greenup, results_mature]:
            total = (
                results['diagnostics.biomass_allocation.allocation_fraction_leaf']
                + results['diagnostics.biomass_allocation.allocation_fraction_stem']
                + results['diagnostics.biomass_allocation.allocation_fraction_root']
            )
            assert total.item() == pytest.approx(1.0, abs=0.01)

    def test_phenology_control_preserves_the_npp_carbon_ledger(self, allocation_model):
        """Phenology changes allocation fractions without discarding NPP carbon."""
        state = {
            'net_primary_productivity': torch.tensor(2.0e-7),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(1.0),
            'leaf_area_index': torch.tensor(3.0),
            'phenology_phase': torch.tensor(0),
        }

        results = output_values(
            allocation_model.forward(state, time=0.0, dt=3600.0)
        )
        total_growth = (
            results['growth_rate_leaf']
            + results['growth_rate_stem']
            + results['growth_rate_root']
        )

        assert results['diagnostics.biomass_allocation.phenology_modifier'].item() < 1.0
        assert total_growth.item() == pytest.approx(2.0e-7, rel=1e-6)
        assert results['diagnostics.biomass_allocation.carbon_balance_residual'].item() == pytest.approx(0.0, abs=1e-12)
    
    def test_lai_dependent_controls(self, allocation_model):
        """Test L1.7: LAI-dependent allocation controls."""
        base_state = {
            'net_primary_productivity': torch.tensor(2.0e-7),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(1.0),
            'phenology_phase': torch.tensor(2),
        }
        
        # Disable environmental control
        allocation_model.enable_environmental_control = False
        
        # Low LAI (below minimum threshold)
        state_low_lai = base_state.copy()
        state_low_lai['leaf_area_index'] = torch.tensor(0.5)
        
        # High LAI (above maximum threshold)
        state_high_lai = base_state.copy() 
        state_high_lai['leaf_area_index'] = torch.tensor(7.0)
        
        # Normal LAI
        state_normal_lai = base_state.copy()
        state_normal_lai['leaf_area_index'] = torch.tensor(3.0)
        
        results_low = output_values(
            allocation_model.forward(state_low_lai, time=0.0, dt=3600.0)
        )
        results_high = output_values(
            allocation_model.forward(state_high_lai, time=0.0, dt=3600.0)
        )
        results_normal = output_values(
            allocation_model.forward(state_normal_lai, time=0.0, dt=3600.0)
        )
        
        # Low LAI should increase leaf allocation
        f_leaf_low = results_low['diagnostics.biomass_allocation.allocation_fraction_leaf'].item()
        f_leaf_normal = results_normal['diagnostics.biomass_allocation.allocation_fraction_leaf'].item()
        
        assert f_leaf_low > f_leaf_normal, "Low LAI should increase leaf allocation"
        
        # High LAI should stop leaf allocation
        f_leaf_high = results_high['diagnostics.biomass_allocation.allocation_fraction_leaf'].item()
        assert f_leaf_high == pytest.approx(0.0, abs=0.01), "High LAI should stop leaf allocation"
        
        # Mass conservation check
        for results in [results_low, results_high, results_normal]:
            total = (
                results['diagnostics.biomass_allocation.allocation_fraction_leaf']
                + results['diagnostics.biomass_allocation.allocation_fraction_stem']
                + results['diagnostics.biomass_allocation.allocation_fraction_root']
            )
            assert total.item() == pytest.approx(1.0, abs=0.01)
    
    def test_no_second_construction_respiration_law(self, allocation_model):
        """Biomass allocation consumes NPP after GrowthRespiration is deducted."""
        state = {
            'net_primary_productivity': torch.tensor(3.0e-7),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(1.0),
            'leaf_area_index': torch.tensor(3.0),
            'phenology_phase': torch.tensor(2),
        }
        
        raw_results = allocation_model.forward(state, time=0.0, dt=3600.0)
        results = output_values(raw_results)
        forbidden = {
            'construction_respiration_leaf',
            'construction_respiration_stem',
            'construction_respiration_root',
            'total_construction_respiration',
            'gross_allocation_leaf',
            'gross_allocation_stem',
            'gross_allocation_root',
        }
        assert forbidden.isdisjoint(raw_results)
        total_growth = sum(results[key] for key in (
            'growth_rate_leaf',
            'growth_rate_stem',
            'growth_rate_root',
        ))
        assert total_growth.item() == pytest.approx(3.0e-7, rel=1e-6)
    
    def test_removed_growth_respiration_selector_is_rejected(self):
        """The production runtime must not expose a second respiration law."""
        with pytest.raises(TypeError):
            BiomassAllocation(growth_respiration_model="casa")

    def test_missing_npp_fails_fast(self, allocation_model):
        """NPP is the required carbon budget and cannot be synthesized as zero."""
        with pytest.raises(KeyError, match="net_primary_productivity"):
            allocation_model.forward({}, time=0.0, dt=86400.0)

    def test_bvoc_carbon_allocation_is_an_explicit_compile_time_port(self):
        """The selected BVOC closure is a required dependency and carbon input."""
        model = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False,
            bvoc_carbon_allocation_owner="mechanistic_isoprene_synthesis",
        )

        assert model.get_dependencies() == [
            "autotrophic_respiration",
            "mechanistic_isoprene_synthesis",
        ]
        assert model.get_required_inputs() == [
            "net_primary_productivity",
            "bvoc_carbon_allocation_flux",
        ]
        assert "bvoc_carbon_allocation_flux" in model.metadata.input_variables

        npp = torch.tensor(2.0e-7)
        bvoc_allocation = torch.tensor(5.0e-8)
        dt = 3600.0
        output = model.forward(
            {
                "net_primary_productivity": npp,
                "bvoc_carbon_allocation_flux": bvoc_allocation,
            },
            time=0.0,
            dt=dt,
        )
        values = output_values(output)
        total_growth = sum(
            values[key]
            for key in ("growth_rate_leaf", "growth_rate_stem", "growth_rate_root")
        )
        total_biomass_delta = sum(
            values[key]
            for key in ("leaf_biomass_c", "wood_biomass_c", "root_biomass_c")
        )

        expected_growth = npp - bvoc_allocation
        assert total_growth.item() == pytest.approx(
            expected_growth.item(),
            rel=1e-6,
        )
        assert total_biomass_delta.item() == pytest.approx(
            expected_growth.item() * dt,
            rel=1e-6,
        )
        assert values[
            "diagnostics.biomass_allocation.total_growth_rate"
        ].item() == pytest.approx(expected_growth.item(), rel=1e-6)
        assert values[
            "diagnostics.biomass_allocation.carbon_balance_residual"
        ].item() == pytest.approx(0.0, abs=1e-12)
        partition_residual = model.check_conservation_laws(
            {
                "net_primary_productivity": npp,
                "bvoc_carbon_allocation_flux": bvoc_allocation,
            },
            output,
        )["carbon_partition"]
        runtime_dtype = output["growth_rate_leaf"].value.dtype
        assert partition_residual <= torch.finfo(runtime_dtype).eps

        tampered = dict(output)
        growth_leaf = tampered["growth_rate_leaf"].value
        replace_update_value(
            tampered,
            "growth_rate_leaf",
            growth_leaf + growth_leaf.new_tensor(1.0e-8),
        )
        assert model.check_conservation_laws(
            {
                "net_primary_productivity": npp,
                "bvoc_carbon_allocation_flux": bvoc_allocation,
            },
            tampered,
        )["carbon_partition"] == pytest.approx(0.05, rel=1.0e-6)

    @pytest.mark.parametrize(
        ("key", "invalid_value", "error_type", "message"),
        [
            (
                "growth_rate_leaf",
                1.0,
                TypeError,
                "must be a torch.Tensor",
            ),
            (
                "growth_rate_leaf",
                torch.empty(0),
                ValueError,
                "must be non-empty",
            ),
            (
                "growth_rate_leaf",
                torch.tensor(1, dtype=torch.int64),
                TypeError,
                "must be floating",
            ),
            (
                "growth_rate_stem",
                torch.tensor(float("nan")),
                ValueError,
                "must be finite",
            ),
            (
                "growth_rate_root",
                torch.tensor(float("inf")),
                ValueError,
                "must be finite",
            ),
            (
                "growth_rate_root",
                torch.tensor(-1.0e-8),
                ValueError,
                "must be non-negative",
            ),
        ],
    )
    def test_carbon_partition_rejects_invalid_outputs(
        self,
        key,
        invalid_value,
        error_type,
        message,
    ):
        """Invalid allocation evidence must fail before residual evaluation."""
        model = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False,
        )
        state = {"net_primary_productivity": torch.tensor(2.0e-7)}
        output = model.forward(state, time=0.0, dt=3600.0)
        replace_update_value(output, key, invalid_value)

        with pytest.raises(error_type, match=message):
            model.check_conservation_laws(state, output)

    @pytest.mark.parametrize(
        "missing_key",
        ["growth_rate_leaf", "growth_rate_stem", "growth_rate_root"],
    )
    def test_carbon_partition_requires_all_growth_ports(self, missing_key):
        """Every organ-growth port is mandatory conservation evidence."""
        model = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False,
        )
        state = {"net_primary_productivity": torch.tensor(2.0e-7)}
        output = model.forward(state, time=0.0, dt=3600.0)
        output.pop(missing_key)

        with pytest.raises(KeyError, match=missing_key):
            model.check_conservation_laws(state, output)

    def test_carbon_partition_requires_matching_shapes(self):
        """NPP and all partition fluxes must have identical spatial shapes."""
        model = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False,
        )
        state = {"net_primary_productivity": torch.tensor(2.0e-7)}
        output = model.forward(state, time=0.0, dt=3600.0)
        growth_leaf = output["growth_rate_leaf"].value
        replace_update_value(
            output,
            "growth_rate_leaf",
            growth_leaf.reshape(1).expand(2).clone(),
        )

        with pytest.raises(ValueError, match="identical shapes"):
            model.check_conservation_laws(state, output)

    @pytest.mark.parametrize(
        ("state", "error_type", "message"),
        [
            ({}, KeyError, "net_primary_productivity"),
            (
                {"net_primary_productivity": 2.0e-7},
                TypeError,
                "must be a torch.Tensor",
            ),
            (
                {"net_primary_productivity": torch.empty(0)},
                ValueError,
                "must be non-empty",
            ),
            (
                {"net_primary_productivity": torch.tensor(1, dtype=torch.int64)},
                TypeError,
                "must be floating",
            ),
            (
                {"net_primary_productivity": torch.tensor(float("nan"))},
                ValueError,
                "must be finite",
            ),
            (
                {"net_primary_productivity": torch.tensor(-1.0e-8)},
                ValueError,
                "must be non-negative",
            ),
        ],
    )
    def test_carbon_partition_rejects_invalid_npp(
        self,
        state,
        error_type,
        message,
    ):
        """The upstream carbon budget must be explicit and physically valid."""
        model = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False,
        )
        valid_state = {"net_primary_productivity": torch.tensor(2.0e-7)}
        output = model.forward(valid_state, time=0.0, dt=3600.0)

        with pytest.raises(error_type, match=message):
            model.check_conservation_laws(state, output)

    @pytest.mark.parametrize(
        ("state_update", "error_type", "message"),
        [
            ({}, KeyError, "bvoc_carbon_allocation_flux"),
            (
                {"bvoc_carbon_allocation_flux": 1.0e-8},
                TypeError,
                "torch.Tensor",
            ),
            (
                {"bvoc_carbon_allocation_flux": torch.tensor(float("nan"))},
                ValueError,
                "finite",
            ),
            (
                {"bvoc_carbon_allocation_flux": torch.tensor(-1.0e-8)},
                ValueError,
                "non-negative",
            ),
            (
                {"bvoc_carbon_allocation_flux": torch.tensor(3.0e-7)},
                ValueError,
                "exceed net_primary_productivity",
            ),
        ],
    )
    def test_selected_bvoc_carbon_port_fails_fast(
        self,
        state_update,
        error_type,
        message,
    ):
        """The selected BVOC debit cannot use a hidden default or carbon reserve."""
        model = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False,
            bvoc_carbon_allocation_owner="megan_bvoc_model",
        )
        state = {"net_primary_productivity": torch.tensor(2.0e-7)}
        state.update(state_update)

        with pytest.raises(error_type, match=message):
            model.forward(state, time=0.0, dt=3600.0)

    def test_unselected_bvoc_carbon_port_cannot_be_silently_ignored(self):
        """A graph containing the shared BVOC port must select its upstream owner."""
        model = BiomassAllocation(
            enable_environmental_control=False,
            enable_phenology_control=False,
        )
        assert "bvoc_carbon_allocation_flux" not in model.get_required_inputs()
        assert "bvoc_carbon_allocation_flux" not in model.metadata.input_variables

        with pytest.raises(ValueError, match="bvoc_carbon_allocation_owner"):
            model.forward(
                {
                    "net_primary_productivity": torch.tensor(2.0e-7),
                    "bvoc_carbon_allocation_flux": torch.tensor(5.0e-8),
                },
                time=0.0,
                dt=3600.0,
            )
    
    def test_allocation_normalization(self, allocation_model):
        """Test L1.10: Allocation fraction normalization."""
        # Test with extreme conditions that might break normalization
        state = {
            'net_primary_productivity': torch.tensor(1.5e-7),
            'air_temperature': torch.tensor(330.15),
            'water_stress_factor': torch.tensor(0.1),
            'leaf_area_index': torch.tensor(0.2),
            'phenology_phase': torch.tensor(1),
        }
        
        results = output_values(
            allocation_model.forward(state, time=0.0, dt=3600.0)
        )
        
        # Check that fractions are normalized
        f_leaf = results['diagnostics.biomass_allocation.allocation_fraction_leaf']
        f_stem = results['diagnostics.biomass_allocation.allocation_fraction_stem']
        f_root = results['diagnostics.biomass_allocation.allocation_fraction_root']
        
        total = f_leaf + f_stem + f_root
        assert total.item() == pytest.approx(1.0, abs=0.001), "Allocation fractions must sum to 1.0"
        
        # Check that fractions are within bounds (allowing zero for specific biological cases)
        # In greenup phase, herbaceous plants may have zero stem allocation (biologically valid)
        # Only enforce minimum bounds for non-zero allocations
        
        f_leaf_value = f_leaf.item()
        f_stem_value = f_stem.item()
        f_root_value = f_root.item()
        if f_stem_value > 1e-6:  # If stem allocation is non-zero
            assert f_stem_value >= allocation_model.min_allocation_frac, "Non-zero stem allocation should be >= minimum"
        
        # Leaf and root should always have some allocation in normal conditions
        assert f_leaf_value >= 0.0  # Allow zero for dormant phases
        assert f_root_value >= allocation_model.min_allocation_frac  # Root allocation should always be >= minimum
        
        assert f_leaf_value <= allocation_model.max_allocation_frac
        assert f_stem_value <= allocation_model.max_allocation_frac
        assert f_root_value <= allocation_model.max_allocation_frac

    def test_allocation_fractions_bounded_under_severe_water_stress(self, allocation_model):
        """Verify allocation fractions remain physical under severe water stress."""
        state = {
            'net_primary_productivity': torch.tensor(1.0e-7),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(0.0),
            'leaf_area_index': torch.tensor(3.0),
            'phenology_phase': torch.tensor(2),
        }

        results = output_values(
            allocation_model.forward(state, time=0.0, dt=3600.0)
        )

        fractions = [
            results['diagnostics.biomass_allocation.allocation_fraction_leaf'],
            results['diagnostics.biomass_allocation.allocation_fraction_stem'],
            results['diagnostics.biomass_allocation.allocation_fraction_root'],
        ]
        total = sum(fractions)

        assert total.item() == pytest.approx(1.0, abs=0.001)
        for fraction in fractions:
            assert fraction.item() >= 0.0
            assert fraction.item() <= allocation_model.max_allocation_frac
    
    def test_pft_parameter_loading(self):
        """Test L1.11: the configured PFT's allocation triple is read from the
        ParameterManager only; the PFT registry is not consulted (MA-21 15.3)."""
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        pft = "broadleaf_deciduous_temperate"
        expected = [
            float(manager.get(f"biosphere.pft.{pft}.allocation.{fraction}"))
            for fraction in ("f_leaf", "f_stem", "f_root")
        ]
        total = sum(expected)
        with patch('relic.scientific.biosphere.pft.pft_registry.get_pft') as mock_get_pft:
            pft_model = BiomassAllocation(pft_name=pft)
        mock_get_pft.assert_not_called()
        assert pft_model.f_leaf_base.item() == pytest.approx(expected[0] / total)
        assert pft_model.f_stem_base.item() == pytest.approx(expected[1] / total)
        assert pft_model.f_root_base.item() == pytest.approx(expected[2] / total)
        assert not hasattr(pft_model, "construction_eff_leaf")

    def test_unknown_pft_raises_and_lists_the_known_pfts(self):
        """MA-21 15.3: an unknown PFT has no allocation keys in the ParameterManager and
        raises, naming the known PFTs; no registry fallback path exists."""
        import inspect

        assert "get_pft(" not in inspect.getsource(BiomassAllocation._load_pft_parameters)
        with pytest.raises(KeyError, match="broadleaf_deciduous_temperate"):
            BiomassAllocation(pft_name="test_pft")
    
    def test_boundary_conditions_robustness(self, allocation_model):
        """Test L1.12: Boundary conditions and robustness."""
        # Test zero NPP
        state_zero_npp = {
            'net_primary_productivity': torch.tensor(0.0),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(1.0),
            'leaf_area_index': torch.tensor(3.0),
            'phenology_phase': torch.tensor(2),
        }
        
        results_zero = output_values(
            allocation_model.forward(state_zero_npp, time=0.0, dt=3600.0)
        )
        
        # Should handle zero NPP gracefully
        total_key = 'diagnostics.biomass_allocation.total_growth_rate'
        assert torch.isfinite(results_zero[total_key]).all()
        assert results_zero[total_key].item() == pytest.approx(0.0, abs=1e-12)
        
        # Negative NPP violates this allocation process domain.
        state_negative_npp = {
            'net_primary_productivity': torch.tensor(-5.0e-8),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(1.0),
            'leaf_area_index': torch.tensor(3.0),
            'phenology_phase': torch.tensor(2),
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            allocation_model.forward(state_negative_npp, time=0.0, dt=86400.0)
        
        # Test extreme environmental conditions
        state_extreme = {
            'net_primary_productivity': torch.tensor(5.0e-7),
            'air_temperature': torch.tensor(250.15),
            'water_stress_factor': torch.tensor(0.0),
            'leaf_area_index': torch.tensor(20.0),
            'phenology_phase': torch.tensor(0),
        }
        
        results_extreme = output_values(
            allocation_model.forward(state_extreme, time=0.0, dt=3600.0)
        )
        
        # Should handle extreme conditions gracefully
        assert torch.isfinite(results_extreme[total_key]).all()
        assert results_extreme[total_key].item() >= 0
        
        # Allocation fractions should still sum to 1.0
        total_alloc = (
            results_extreme['diagnostics.biomass_allocation.allocation_fraction_leaf']
            + results_extreme['diagnostics.biomass_allocation.allocation_fraction_stem']
            + results_extreme['diagnostics.biomass_allocation.allocation_fraction_root']
        )
        assert total_alloc.item() == pytest.approx(1.0, abs=0.01)
    
    def test_forward_only_canonical_output_contract(self, allocation_model):
        """Test L1.13: expose one forward entry point and canonical output names."""
        state = {
            'net_primary_productivity': torch.tensor(2.0e-7),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(0.8),
        }
        environment = {
            'leaf_area_index': torch.tensor(3.0),
            'phenology_phase': torch.tensor(2),
        }
        
        output = allocation_model.forward(
            {**state, **environment},
            time=0.0,
            dt=86400.0,
        )

        assert set(output) == CANONICAL_OUTPUTS
        assert all(isinstance(item, UpdateItem) for item in output.values())
        for key in ('growth_rate_leaf', 'growth_rate_stem', 'growth_rate_root'):
            item = output[key]
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == 'kg C m-2 s-1'
            assert item.source_functor == allocation_model.name
        for key in ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c'):
            item = output[key]
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == 'kg C m-2'
            assert item.source_functor == allocation_model.name

        rate = output['diagnostics.biomass_allocation.total_growth_rate'].value
        assert torch.isfinite(rate).all()
        assert rate.item() >= 0

        noncanonical_aliases = {
            'rate',
            'carbon_allocation_rate',
            'leaf_allocation',
            'stem_allocation',
            'root_allocation',
            'leaf_growth',
            'stem_growth',
            'root_growth',
            'total_growth_rate',
            'effective_net_primary_production',
            'leaf_biomass_c_increment',
            'wood_biomass_c_increment',
            'root_biomass_c_increment',
            'construction_respiration_leaf',
            'total_construction_respiration',
        }
        assert noncanonical_aliases.isdisjoint(output)
        assert not hasattr(allocation_model, '_compute_rate_core')
    
    def test_allocation_summary_functionality(self, allocation_model):
        """Test L1.14: Allocation summary and diagnostics."""
        summary = allocation_model.get_allocation_summary()
        
        # Check summary structure
        assert 'base_allocation_leaf' in summary
        assert 'base_allocation_stem' in summary
        assert 'base_allocation_root' in summary
        assert 'environmental_control' in summary
        assert 'phenology_control' in summary
        assert 'pft_name' in summary
        
        # Verify values match model parameters
        assert summary['base_allocation_leaf'] == pytest.approx(
            allocation_model.f_leaf_base.item(), rel=0.01
        )
        assert 'construction_efficiency_leaf' not in summary
        assert summary['environmental_control'] == allocation_model.enable_environmental_control
        assert summary['phenology_control'] == allocation_model.enable_phenology_control
        
        # Base allocations should sum to 1.0
        total_base = (summary['base_allocation_leaf'] + 
                     summary['base_allocation_stem'] + 
                     summary['base_allocation_root'])
        assert total_base == pytest.approx(1.0, abs=0.01)
    
    def test_mass_conservation_validation(self, allocation_model):
        """Test L1.15: Comprehensive mass conservation validation."""
        state = {
            'net_primary_productivity': torch.tensor(2.5e-7),
            'air_temperature': torch.tensor(303.15),
            'water_stress_factor': torch.tensor(0.7),
            'leaf_area_index': torch.tensor(4.0),
            'phenology_phase': torch.tensor(2),
        }
        
        dt = 3600.0
        results = output_values(
            allocation_model.forward(state, time=0.0, dt=dt)
        )
        
        npp = 2.5e-7
        total_growth = results['diagnostics.biomass_allocation.total_growth_rate'].item()
        assert total_growth == pytest.approx(npp, rel=1e-6)
        
        # Component mass balance
        net_leaf = results['growth_rate_leaf'].item()
        net_stem = results['growth_rate_stem'].item()
        net_root = results['growth_rate_root'].item()
        
        component_total = net_leaf + net_stem + net_root
        assert component_total == pytest.approx(total_growth, rel=1e-6)
        assert results['leaf_biomass_c'].item() == pytest.approx(net_leaf * dt, rel=1e-6)
        assert results['wood_biomass_c'].item() == pytest.approx(net_stem * dt, rel=1e-6)
        assert results['root_biomass_c'].item() == pytest.approx(net_root * dt, rel=1e-6)
        assert results['diagnostics.biomass_allocation.carbon_balance_residual'].item() == pytest.approx(0.0, abs=1e-12)


    def test_allometric_constraints_accept_gridded_states_in_governed_calls(self, caplog):
        """Test L1.x: vector NPP and biomass pools pass through allometric control and its logging."""
        import logging

        allocation = BiomassAllocation(enable_allometry_control=True)
        allocation.enable_environmental_control = False
        allocation.enable_phenology_control = False
        state = {
            "net_primary_productivity": torch.tensor([2.0e-7, 1.0e-7, 3.0e-7]),
            "leaf_biomass_c": torch.tensor([0.20, 0.05, 0.40]),
            "wood_biomass_c": torch.tensor([1.50, 0.10, 4.00]),
            "root_biomass_c": torch.tensor([0.40, 0.02, 0.90]),
            # Metadata-required inputs; environmental and phenological controls are disabled above.
            "air_temperature": torch.tensor([288.15, 290.15, 286.15]),
            "water_stress_factor": torch.tensor([1.0, 1.0, 1.0]),
            "leaf_area_index": torch.tensor([3.0, 1.0, 5.0]),
            "phenology_phase": torch.tensor([2, 2, 2]),
        }
        with caplog.at_level(logging.DEBUG, logger="relic.scientific.biosphere.growth.biomass_allocation"):
            raw_results = allocation(state, time=0.0, dt=3600.0)
        results = output_values(raw_results)
        leaf = results["diagnostics.biomass_allocation.allocation_fraction_leaf"]
        stem = results["diagnostics.biomass_allocation.allocation_fraction_stem"]
        root = results["diagnostics.biomass_allocation.allocation_fraction_root"]
        assert tuple(leaf.shape) == (3,)
        assert tuple(stem.shape) == (3,)
        assert tuple(root.shape) == (3,)
        torch.testing.assert_close(leaf + stem + root, torch.ones_like(leaf), atol=1.0e-5, rtol=0.0)
        # The allometric targets differ between cells, so the blended fractions must not collapse to one value.
        assert not torch.allclose(stem, stem[0].expand_as(stem))
        growth_total = (
            results["growth_rate_leaf"] + results["growth_rate_stem"] + results["growth_rate_root"]
        )
        torch.testing.assert_close(
            growth_total,
            state["net_primary_productivity"].to(growth_total.dtype),
            atol=1.0e-12,
            rtol=1.0e-5,
        )
        messages = [record.getMessage() for record in caplog.records if "Allometric constraints applied" in record.getMessage()]
        assert len(messages) == 1
        assert "(n=3)" in messages[0]

    def test_forward_keeps_the_runtime_state_dtype(self):
        """Test L1.x: float64 states stay float64 and float32 states stay float32 through allocation."""
        allocation = BiomassAllocation()
        allocation.enable_environmental_control = False
        allocation.enable_phenology_control = False
        base = {
            "net_primary_productivity": 2.0e-7,
            "air_temperature": 288.15,
            "water_stress_factor": 1.0,
            "leaf_area_index": 3.0,
            "phenology_phase": 2,
        }
        outputs = {}
        for dtype in (torch.float64, torch.float32):
            state = {
                key: torch.tensor(value, dtype=dtype) if key != "phenology_phase" else torch.tensor(value)
                for key, value in base.items()
            }
            result = output_values(allocation(state, time=0.0, dt=3600.0))
            for key in ("growth_rate_leaf", "growth_rate_stem", "growth_rate_root", "leaf_biomass_c"):
                assert result[key].dtype == dtype, (key, dtype, result[key].dtype)
            outputs[dtype] = result
        # References: RELIC runtime-precision contract; PyTorch type promotion.
        # Mathematical expression: r_i = f_i NPP is dtype-invariant up to float32 rounding.
        # Parameter meanings: f_i allocation fractions, NPP the input flux.
        # Units: r_i, NPP [kg C m-2 s-1].
        # Domain/constraints: identical inputs in the two precisions.
        # Sign convention: not applicable.
        for key in ("growth_rate_leaf", "growth_rate_stem", "growth_rate_root"):
            torch.testing.assert_close(
                outputs[torch.float32][key].double(),
                outputs[torch.float64][key],
                rtol=1.0e-6,
                atol=0.0,
            )


def test_certificate_parameter_keys_resolve_in_the_parameter_manager():
    """MA-21 14.2: every certificate parameter key resolves in the ParameterManager or
    is a '*' template of the certificate schema; the PFT allocation triple the Functor
    reads for its configured PFT is the template biosphere.pft.*.allocation.f_*, which
    covers every PFT of the parameter database; no <pft> placeholder remains."""
    import fnmatch
    from pathlib import Path

    from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
    from relic.parameters import get_global_manager

    root = Path(__file__).resolve().parents[4]
    certificate = load_iphs_equivalence_certificate(
        root / "docs" / "source" / "theory" / "iphs_certificates"
        / "biosphere_growth_biomass_allocation_biomass_allocation.yaml"
    )
    assert certificate.iphs_level < 4  # a strict certificate admits no wildcard key
    manager = get_global_manager()
    keys = set(certificate.parameter_keys) | set(certificate.parameter_management_map)
    templates = {key for key in keys if "*" in key}
    assert templates == {
        f"biosphere.pft.*.allocation.{fraction}" for fraction in ("f_leaf", "f_stem", "f_root")
    }
    assert not [key for key in keys - templates if manager.get_parameter(key) is None]
    text = str(keys) + str(dict(certificate.parameter_map))
    assert "<pft>" not in text
    pft_dir = root / "relic" / "parameters" / "database" / "relic" / "biosphere" / "pft"
    pfts = sorted(
        path.stem for path in pft_dir.glob("*.yaml") if path.stem not in ("__init__", "metadata")
    )
    assert len(pfts) == 15
    for pft in pfts:
        for fraction in ("f_leaf", "f_stem", "f_root"):
            key = f"biosphere.pft.{pft}.allocation.{fraction}"
            assert manager.get_parameter(key) is not None, key
            assert [pattern for pattern in templates if fnmatch.fnmatchcase(key, pattern)], key


if __name__ == "__main__":
    pytest.main([__file__])


