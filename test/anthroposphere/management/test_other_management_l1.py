#!/usr/bin/env python3
"""
L1 Unit Tests for Other Anthroposphere Management Modules

Tests for:
- TillageManagement
- HarvestManagement  
- GrazingManagement
- FireManagement

Test Categories:
- L1.1: Class instantiation and inheritance
- L1.2: Parameter management system validation
- L1.3: Scientific calculation accuracy
- L1.4: Conservation laws validation
- L1.5: Abstract base class compliance

Scientific References:
- Lal, R. (2004). Soil carbon sequestration to mitigate climate change. Geoderma 123(1-2), 1-22.
- IPCC (2019). 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories.
- FAO (2010). Carbon sequestration in dryland soils.

Test Framework: pytest
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import modules to test
from relic.scientific.anthroposphere.management.tillage import TillageManagement
from relic.scientific.anthroposphere.management.harvest import (
    HarvestEvent,
    HarvestManagement,
)
from relic.scientific.anthroposphere.management.grazing import GrazingManagement
from relic.scientific.anthroposphere.management.fire_management import FireManagement

from relic.core.abstractions import (
    BiologicalProcess,
    TransportMechanism
)
from relic.core.functor import (
    BiologicalProcessFunctor,
    TransportMechanismFunctor
)
from test.utils.forward_call import call_forward
from test.utils.functor_minimal_construction import construct_minimal

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 4
TOLERANCE = 1e-6


class TestTillageManagementL1:
    """L1 Unit Tests for TillageManagement"""
    
    @pytest.fixture
    def tillage_manager(self):
        """Create tillage manager instance for testing"""
        return TillageManagement(config={"calendar_type": "noleap"})
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'soil_moisture': torch.tensor([0.2, 0.25, 0.3, 0.35], device=DEVICE).reshape(batch_shape),
            'soil_temperature': torch.tensor([288.15, 293.15, 298.15, 303.15], device=DEVICE).reshape(batch_shape),
            'soil_bulk_density': torch.tensor([1.2, 1.3, 1.4, 1.5], device=DEVICE).reshape(batch_shape),
            'soil_organic_carbon': torch.tensor([0.015, 0.020, 0.025, 0.030], device=DEVICE).reshape(batch_shape),
            'residue_cover': torch.tensor([0.3, 0.5, 0.7, 0.2], device=DEVICE).reshape(batch_shape),
            'days_since_tillage': torch.full(batch_shape, 365.0, device=DEVICE),
        }
    
    def test_class_instantiation_and_inheritance(self, tillage_manager):
        """Test class instantiation and inheritance"""
        assert tillage_manager is not None
        assert isinstance(tillage_manager, TillageManagement)
        assert isinstance(tillage_manager, TransportMechanismFunctor)
        assert isinstance(tillage_manager, TransportMechanism)
        
        # Current Phase 8 modules expose the unified FunctorBase forward lifecycle.
        assert hasattr(tillage_manager, 'forward')
        assert callable(tillage_manager.forward)
    
    def test_parameter_management(self, tillage_manager):
        """Test parameter management system"""
        # Test accessing tillage parameters
        # Every tillage parameter is read from the database with its namespace; the
        # removed keys tillage_depth and mixing_efficiency have no substitute default.
        disturbance_factor = tillage_manager.get_parameter_value(
            "disturbance_factor", namespace="anthroposphere.management.tillage", required=True
        )
        assert isinstance(disturbance_factor, (float, torch.Tensor))
        assert 0.1 <= disturbance_factor <= 1.0
        for removed in ("tillage_depth", "mixing_efficiency"):
            assert tillage_manager.get_parameter_value(
                removed, namespace="anthroposphere.management.tillage", required=False
            ) is None
    
    def test_soil_mixing_flux_calculation(self, tillage_manager, sample_state):
        """Test soil mixing flux calculation"""
        # Exercise the single state-driven Functor lifecycle.
        fluxes = tillage_manager.forward(sample_state, time=0.0, dt=3600.0)
        
        assert isinstance(fluxes, dict)
        assert len(fluxes) > 0
        
        # Should have tillage transport and disturbance outputs.
        flux_keys = list(fluxes.keys())
        expected_fluxes = {
            "soil_disturbance_rate",
            "residue_incorporation_rate",
            "carbon_loss_rate",
        }
        assert expected_fluxes.intersection(flux_keys)
        
        # All fluxes should be tensors
        for flux_value in fluxes.values():
            value = flux_value.value if hasattr(flux_value, "value") else flux_value
            assert isinstance(value, torch.Tensor)
    
    def test_residue_incorporation(self, tillage_manager, sample_state):
        """Test residue incorporation during tillage"""
        # Test forward method
        outputs = tillage_manager.forward(sample_state, time=0, dt=3600)
        
        assert isinstance(outputs, dict)
        
        # Should have residue-related outputs
        if "residue_incorporation_rate" in outputs:
            residue_rate = outputs["residue_incorporation_rate"].value
            assert torch.all(residue_rate >= 0)
        
        if "soil_disturbance_rate" in outputs:
            disturbance_rate = outputs["soil_disturbance_rate"].value
            assert torch.all(disturbance_rate >= 0)
    
    def test_carbon_sequestration_impact(self, tillage_manager, sample_state):
        """Test tillage impact on carbon sequestration"""
        # Test with different carbon levels
        low_carbon_state = sample_state.copy()
        low_carbon_state['soil_organic_carbon'] = torch.tensor([[0.010], [0.012], [0.015], [0.018]], device=DEVICE)
        
        high_carbon_state = sample_state.copy()
        high_carbon_state['soil_organic_carbon'] = torch.tensor([[0.040], [0.045], [0.050], [0.055]], device=DEVICE)
        
        low_outputs = tillage_manager.forward(low_carbon_state, time=0, dt=3600)
        high_outputs = tillage_manager.forward(high_carbon_state, time=0, dt=3600)
        
        # Should handle different carbon levels appropriately
        assert isinstance(low_outputs, dict)
        assert isinstance(high_outputs, dict)


class TestHarvestManagementL1:
    """L1 Unit Tests for HarvestManagement"""
    
    @pytest.fixture
    def harvest_manager(self):
        """Create harvest manager instance for testing"""
        return HarvestManagement(config={"calendar_type": "noleap"})
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'aboveground_biomass': torch.tensor(
                [0.5, 0.6, 0.7, 0.8], device=DEVICE
            ).reshape(batch_shape),
            'belowground_biomass': torch.tensor(
                [0.1, 0.12, 0.14, 0.16], device=DEVICE
            ).reshape(batch_shape),
            'crop_maturity': torch.ones(batch_shape, device=DEVICE),
            'residue_cover': torch.full(batch_shape, 0.3, device=DEVICE),
        }
    
    def test_class_instantiation_and_inheritance(self, harvest_manager):
        """Test class instantiation and inheritance"""
        assert harvest_manager is not None
        assert isinstance(harvest_manager, HarvestManagement)
        assert isinstance(harvest_manager, BiologicalProcessFunctor)
        assert isinstance(harvest_manager, BiologicalProcess)
        
        # Current Phase 8 modules expose the unified FunctorBase forward lifecycle.
        assert hasattr(harvest_manager, 'forward')
        assert callable(harvest_manager.forward)
    
    def test_parameter_management(self, harvest_manager):
        """Test parameter management system"""
        # Test accessing harvest parameters
        harvest_efficiency = harvest_manager.get_parameter_value("harvest_efficiency")
        assert isinstance(harvest_efficiency, (float, torch.Tensor))
        assert 0.7 <= harvest_efficiency <= 0.95
        
        residue_removal_rate = harvest_manager.get_parameter_value(
            "residue_removal_rate", required=True
        )
        assert isinstance(residue_removal_rate, (float, torch.Tensor))
        assert 0.0 <= residue_removal_rate <= 0.8
    
    def test_harvest_rate_calculation(self, harvest_manager, sample_state):
        """Test the declared harvest partition kernel."""
        rates = harvest_manager._canonical_iphs_harvest_partition(
            sample_state,
            event=HarvestEvent(),
        )
        
        assert isinstance(rates, dict)
        assert len(rates) > 0
        
        assert torch.all(rates["grain_yield_rate"] >= 0.0)
        assert torch.all(rates["residue_generation_rate"] >= 0.0)
        torch.testing.assert_close(
            rates["biomass_removal_rate"],
            rates["grain_yield_rate"] + rates["residue_generation_rate"],
        )
    
    def test_biomass_partitioning(self, harvest_manager, sample_state):
        """A scheduled event emits only canonical dry-matter and element ports."""
        harvest_manager.harvest_events = [HarvestEvent(harvest_date=1)]
        outputs = harvest_manager.forward(sample_state, time=0, dt=86400)
        
        assert isinstance(outputs, dict)
        
        assert torch.any(outputs["harvest_flux"].value > 0.0)
        assert "harvested_biomass" not in outputs
        assert "removed_biomass" not in outputs
    
    def test_mass_conservation(self, harvest_manager, sample_state):
        """Test the executable carbon and nitrogen boundary ledgers."""
        harvest_manager.harvest_events = [HarvestEvent(harvest_date=1)]
        outputs = harvest_manager.forward(sample_state, time=0, dt=86400)

        carbon_residual = (
            outputs["harvest_biomass_carbon_input"].value
            - outputs["harvest_carbon_export_flux"].value
            - outputs["residue_c_flux"].value
        )
        nitrogen_residual = (
            outputs["harvest_biomass_nitrogen_input"].value
            - outputs["harvest_nitrogen_export_flux"].value
            - outputs["residue_n_flux"].value
        )
        torch.testing.assert_close(carbon_residual, torch.zeros_like(carbon_residual))
        torch.testing.assert_close(
            nitrogen_residual,
            torch.zeros_like(nitrogen_residual),
        )


class TestGrazingManagementL1:
    """L1 Unit Tests for GrazingManagement"""
    
    @pytest.fixture
    def grazing_manager(self):
        """Create grazing manager instance for testing"""
        return GrazingManagement(config={"calendar_type": "noleap"})
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'aboveground_biomass': torch.tensor(
                [2.0, 2.5, 3.0, 3.5], device=DEVICE
            ).reshape(batch_shape),
            'belowground_biomass': torch.tensor(
                [800.0, 1000.0, 1200.0, 1400.0], device=DEVICE
            ).reshape(batch_shape),
            'vegetation_quality': torch.tensor([0.6, 0.7, 0.8, 0.9], device=DEVICE).reshape(batch_shape),
            'soil_moisture': torch.tensor([0.2, 0.25, 0.3, 0.35], device=DEVICE).reshape(batch_shape),
            'soil_compaction': torch.tensor([1.1, 1.2, 1.3, 1.4], device=DEVICE).reshape(batch_shape)
        }
    
    def test_class_instantiation_and_inheritance(self, grazing_manager):
        """Test class instantiation and inheritance"""
        assert grazing_manager is not None
        assert isinstance(grazing_manager, GrazingManagement)
        assert isinstance(grazing_manager, BiologicalProcessFunctor)
        assert isinstance(grazing_manager, BiologicalProcess)
        
        # Current Phase 8 modules expose the unified FunctorBase forward lifecycle.
        assert hasattr(grazing_manager, 'forward')
        assert callable(grazing_manager.forward)
    
    def test_parameter_management(self, grazing_manager):
        """Test parameter management system"""
        # Test accessing grazing parameters
        max_utilization = grazing_manager.get_parameter_value(
            "max_utilization",
            namespace="anthroposphere.management.grazing",
            required=True,
        )
        assert isinstance(max_utilization, (float, torch.Tensor))
        assert 0.0 < max_utilization <= 1.0
    
    def test_grazing_rate_calculation(self, grazing_manager, sample_state):
        """Test grazing rate calculation"""
        rates = grazing_manager.forward(
            sample_state,
            time=0,
            dt=86400,
            day_of_year=150,
        )
        
        assert isinstance(rates, dict)
        assert len(rates) > 0
        
        expected_rates = {
            "grazing_flux",
            "dung_c_flux",
            "dung_n_flux",
            "urine_n_flux",
            "livestock_carbon_export_flux",
            "livestock_nitrogen_export_flux",
        }
        assert expected_rates.issubset(rates)
        for rate_name in expected_rates:
            assert torch.all(rates[rate_name].value >= 0)
    
    def test_animal_impact_on_vegetation(self, grazing_manager, sample_state):
        """Test animal impact on vegetation"""
        outputs = grazing_manager.forward(
            sample_state,
            time=0,
            dt=86400,
            day_of_year=150,
        )
        
        assert isinstance(outputs, dict)
        
        assert "aboveground_biomass" in outputs
        assert torch.all(outputs["aboveground_biomass"].value <= 0)
        assert torch.allclose(
            -outputs["aboveground_biomass"].value,
            outputs["cumulative_intake"].value,
        )
        assert "diagnostics.grazing_pressure" in outputs
    
    def test_stocking_density_effects(self, grazing_manager, sample_state):
        """Test effects of different stocking densities"""
        # The stocking density is the active event stocking rate (item 116).
        low_density_manager = GrazingManagement(config={"calendar_type": "noleap"})
        low_density_manager.grazing_events[0].stocking_rate = 0.3
        high_density_manager = GrazingManagement(config={"calendar_type": "noleap"})
        high_density_manager.grazing_events[0].stocking_rate = 3.0

        low_rates = low_density_manager.forward(
            sample_state.copy(),
            time=0,
            dt=86400,
            day_of_year=150,
        )
        high_rates = high_density_manager.forward(
            sample_state.copy(),
            time=0,
            dt=86400,
            day_of_year=150,
        )

        assert torch.mean(high_rates["grazing_flux"].value) > torch.mean(
            low_rates["grazing_flux"].value
        )
        # Below the utilization and storage limits the intake I = a W^b D Q (...) is
        # linear in D: a tenfold stocking rate gives a tenfold grazing flux.
        torch.testing.assert_close(
            high_rates["grazing_flux"].value,
            10.0 * low_rates["grazing_flux"].value,
            rtol=1.0e-5,
            atol=0.0,
        )


class TestFireManagementL1:
    """L1 Unit Tests for FireManagement"""
    
    @pytest.fixture
    def fire_manager(self):
        """Create fire manager instance for testing"""
        return FireManagement(config={"calendar_type": "noleap", "seed": 0})
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'fuel_load': torch.tensor([5.0, 10.0, 15.0, 20.0], device=DEVICE).reshape(batch_shape),
            'fuel_moisture': torch.tensor([0.05, 0.10, 0.15, 0.20], device=DEVICE).reshape(batch_shape),
            'days_since_fire': torch.full(batch_shape, 30.0, device=DEVICE),
            'soil_carbon': torch.tensor([50.0, 60.0, 70.0, 80.0], device=DEVICE).reshape(batch_shape),
            'vegetation_cover': torch.tensor([0.6, 0.7, 0.8, 0.9], device=DEVICE).reshape(batch_shape),
            'air_temperature': torch.tensor([298.15, 303.15, 308.15, 313.15], device=DEVICE).reshape(batch_shape),
            'aboveground_biomass': torch.full(batch_shape, 0.3, device=DEVICE),
            'litter_mass': torch.full(batch_shape, 0.3, device=DEVICE),
            'temperature': torch.full(batch_shape, 288.15, device=DEVICE),
        }
    
    def test_class_instantiation_and_inheritance(self, fire_manager):
        """Test class instantiation and inheritance"""
        assert fire_manager is not None
        assert isinstance(fire_manager, FireManagement)
        assert isinstance(fire_manager, BiologicalProcessFunctor)
        assert isinstance(fire_manager, BiologicalProcess)
        
        # Current Phase 8 modules expose the unified FunctorBase forward lifecycle.
        assert hasattr(fire_manager, 'forward')
        assert callable(fire_manager.forward)
    
    def test_parameter_management(self, fire_manager):
        """Test parameter management system"""
        # Test accessing fire parameters
        combustion_efficiency = fire_manager.get_parameter_value("combustion_efficiency")
        assert isinstance(combustion_efficiency, (float, torch.Tensor))
        assert 0.8 <= combustion_efficiency <= 0.98
        
        fire_frequency = fire_manager.get_parameter_value("fire_frequency")
        assert isinstance(fire_frequency, (float, torch.Tensor))
        assert 0.0 <= fire_frequency <= 1.0
    
    def test_fire_emission_calculation(self, fire_manager, sample_state):
        """Test fire emission calculation"""
        rates = call_forward(fire_manager, sample_state)
        
        assert isinstance(rates, dict)
        assert len(rates) > 0
        
        # Should have canonical fire reaction-network outputs.
        expected_rates = ["combustion_flux", "co2_emission", "carbon_loss"]
        found_rates = [rate for rate in expected_rates if rate in rates]
        
        assert len(found_rates) > 0, "Should have at least one fire-related rate"
        
        # All rates should be non-negative
        for rate_name, rate_value in rates.items():
            if "rate" in rate_name:
                assert torch.all(rate_value >= 0), f"Rate {rate_name} should be non-negative"
    
    def test_carbon_emissions(self, fire_manager, sample_state):
        """Test carbon emissions from fire"""
        outputs = fire_manager.forward(sample_state, time=0, dt=86400)
        
        assert isinstance(outputs, dict)
        
        # Should have emission outputs
        expected_outputs = ["co2_emissions", "carbon_loss", "combustion_flux"]
        found_outputs = [output for output in expected_outputs if output in outputs]
        
        assert len(found_outputs) > 0, "Should have at least one emission output"
    
    def test_fuel_moisture_effects(self, fire_manager, sample_state):
        """Test effects of fuel moisture on fire behavior"""
        # Test dry vs wet conditions
        dry_state = sample_state.copy()
        dry_state['fuel_moisture'] = torch.tensor([[0.03], [0.04], [0.05], [0.06]], device=DEVICE)
        
        wet_state = sample_state.copy()
        wet_state['fuel_moisture'] = torch.tensor([[0.25], [0.30], [0.35], [0.40]], device=DEVICE)
        
        dry_rates = call_forward(fire_manager, dry_state)
        wet_rates = call_forward(fire_manager, wet_state)
        
        # Dry conditions should generally result in higher fire rates
        if "combustion_rate" in dry_rates and "combustion_rate" in wet_rates:
            assert torch.mean(dry_rates["combustion_rate"]) > torch.mean(wet_rates["combustion_rate"])


class TestManagementModulesIntegration:
    """Integration tests for all management modules"""
    
    def test_all_modules_instantiation(self):
        """Test that all management modules can be instantiated"""
        modules = [
            TillageManagement(),
            HarvestManagement(),
            GrazingManagement(),
            FireManagement(config={"seed": 0})
        ]
        
        for module in modules:
            assert module is not None
            assert hasattr(module, 'name')
            assert hasattr(module, 'forward')
            assert hasattr(module, 'forward')
    
    def test_parameter_system_consistency(self):
        """Test parameter system consistency across modules"""
        modules = {
            'tillage': TillageManagement(),
            'harvest': HarvestManagement(),
            'grazing': GrazingManagement(),
            'fire': FireManagement(config={"seed": 0})
        }
        
        # Every module reads one of its own database keys with its namespace.
        keys = {
            'tillage': ("disturbance_factor", "anthroposphere.management.tillage"),
            'harvest': ("harvest_efficiency", "anthroposphere.management.harvest"),
            'grazing': ("base_efficiency", "anthroposphere.management.grazing"),
            'fire': ("fuel_consumption_fraction", "anthroposphere.management.fire"),
        }
        for name, module in modules.items():
            key, namespace = keys[name]
            test_param = module.get_parameter_value(key, namespace=namespace, required=True)
            assert isinstance(test_param, (float, torch.Tensor)), name
    
    def test_abstract_base_class_compliance(self):
        """Test that all modules comply with their abstract base classes"""
        expected_inheritance = {
            TillageManagement: TransportMechanismFunctor,
            HarvestManagement: BiologicalProcessFunctor,
            GrazingManagement: BiologicalProcessFunctor,
            FireManagement: BiologicalProcessFunctor
        }
        
        for module_class, expected_base in expected_inheritance.items():
            instance = construct_minimal(f"{module_class.__module__}:{module_class.__name__}")
            assert isinstance(instance, expected_base), f"{module_class.__name__} should inherit from {expected_base.__name__}"
    
    def test_metadata_completeness(self):
        """Test that all modules have complete metadata"""
        modules = [
            TillageManagement(),
            HarvestManagement(),
            GrazingManagement(),
            FireManagement(config={"seed": 0})
        ]
        
        for module in modules:
            metadata = module.get_metadata()
            assert metadata is not None
            assert hasattr(metadata, 'name')
            assert hasattr(metadata, 'description')
            assert hasattr(metadata, 'references')
            assert len(metadata.references) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

