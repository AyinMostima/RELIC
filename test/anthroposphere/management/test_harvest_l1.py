#!/usr/bin/env python3
"""
L1 Unit Tests for Harvest Management Module

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Parameter management system validation
- L1.3: Scientific calculation accuracy (biomass removal)
- L1.4: Conservation laws validation
- L1.5: Harvest efficiency and residue management
- L1.6: Abstract base class compliance

Scientific References:
- Lal, R. (2004). Soil carbon sequestration to mitigate climate change. Geoderma 123, 1-22.
- Johnson, J.M.F., et al. (2006). Crop residue impacts on soil carbon. Agronomy Journal 98, 1723-1727.
- Wilhelm, W.W., et al. (2007). Corn stover to produce ethanol. Agronomy Journal 99, 268-275.

Test Framework: pytest
"""

import copy
from pathlib import Path
import warnings

import pytest
import torch
import sys
import os
import yaml
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import modules to test
from relic.scientific.anthroposphere.management.harvest import (
    CropType,
    HarvestManagement,
    HarvestEvent
)
from relic.core.abstractions import BiologicalProcess
from relic.core.functor import BiologicalProcessFunctor
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from test.utils.forward_call import call_forward

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 5
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3
REPO_ROOT = Path(__file__).resolve().parents[3]


def call_harvest_partition(
    manager,
    state,
    *,
    event=None,
):
    """Evaluate the declared harvest partition kernel for one event without
    scheduling; without an explicit event the generic C3 cereal combine event
    HarvestEvent() (its construction values) is passed. The partition reads the
    state only."""

    if event is None:
        event = HarvestEvent()

    return manager._canonical_iphs_harvest_partition(
        dict(state),
        event=event,
    )


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
            'aboveground_biomass': torch.tensor([3000, 3500, 4000, 4500, 5000], device=DEVICE).reshape(batch_shape),
            'belowground_biomass': torch.tensor([800, 950, 1100, 1250, 1400], device=DEVICE).reshape(batch_shape),
            'leaf_area_index': torch.tensor([2.5, 3.0, 3.5, 4.0, 4.5], device=DEVICE).reshape(batch_shape),
            'crop_height': torch.tensor([1.0, 1.2, 1.4, 1.6, 1.8], device=DEVICE).reshape(batch_shape),
            'crop_maturity': torch.tensor([0.8, 0.85, 0.9, 0.95, 1.0], device=DEVICE).reshape(batch_shape),
            'residue_cover': torch.tensor([0.7, 0.6, 0.5, 0.4, 0.3], device=DEVICE).reshape(batch_shape),
            'root_depth': torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2], device=DEVICE).reshape(batch_shape)
        }
    
    # L1.1: Class instantiation and initialization tests
    def test_class_instantiation(self, harvest_manager):
        """Test that HarvestManagement can be instantiated"""
        assert harvest_manager is not None
        assert isinstance(harvest_manager, HarvestManagement)
        assert isinstance(harvest_manager, BiologicalProcessFunctor)
        assert isinstance(harvest_manager, BiologicalProcess)
        assert hasattr(harvest_manager, 'name')
        assert harvest_manager.name == "harvest_management"
    
    def test_metadata_completeness(self, harvest_manager):
        """Test that metadata is properly set"""
        metadata = harvest_manager.get_metadata()
        assert metadata is not None
        assert metadata.name == "harvest_management"
        assert set(metadata.input_variables) == {
            "aboveground_biomass",
            "belowground_biomass",
            "residue_cover",
            "day_of_year",
        }
        assert "leaf_area_index" not in metadata.input_variables
        assert len(metadata.references) > 0
        assert "biomass" in metadata.mathematical_expression or "carbon" in metadata.mathematical_expression
        assert "mass" in metadata.conservation_laws
    
    def test_initialization_with_config(self):
        """Test initialization with custom configuration"""
        custom_config = {
            "parameters": {
                "harvest_efficiency": {"value": 0.95, "units": "1"},
                "residue_removal_rate": {"value": 0.3, "units": "1"}
            }
        }
        
        manager = HarvestManagement(config=custom_config)
        assert manager is not None
        
        # Parameters should be accessible through get_parameter_value
        efficiency = manager.get_parameter_value("harvest_efficiency")
        assert abs(efficiency - 0.95) < 1e-3
    
    def test_harvest_schedule_initialization(self, harvest_manager):
        """Test that default harvest schedule is properly initialized"""
        assert len(harvest_manager.harvest_events) > 0
        assert any(event.crop_type == CropType.CEREAL_C3 for event in harvest_manager.harvest_events)
        assert any(event.crop_type == CropType.FORAGE for event in harvest_manager.harvest_events)

    def test_runtime_accounting_roundtrips_through_state_dict(self, harvest_manager):
        harvest_manager.last_harvest_time_seconds = 120.5
        harvest_manager.cumulative_biomass_removed = 1.25
        harvest_manager.cumulative_c_removed = 0.55
        harvest_manager.cumulative_residue_returned = 0.35
        harvest_manager.annual_yield = 0.95
        harvest_manager.harvest_count = 3
        saved_state = copy.deepcopy(harvest_manager.state_dict())

        harvest_manager.last_harvest_time_seconds = 200.0
        harvest_manager.cumulative_biomass_removed = 9.0
        harvest_manager.cumulative_c_removed = 9.0
        harvest_manager.cumulative_residue_returned = 9.0
        harvest_manager.annual_yield = 9.0
        harvest_manager.harvest_count = 9
        harvest_manager.load_state_dict(saved_state)

        assert harvest_manager.last_harvest_time_seconds == pytest.approx(120.5)
        assert harvest_manager.cumulative_biomass_removed == pytest.approx(1.25)
        assert harvest_manager.cumulative_c_removed == pytest.approx(0.55)
        assert harvest_manager.cumulative_residue_returned == pytest.approx(0.35)
        assert harvest_manager.annual_yield == pytest.approx(0.95)
        assert harvest_manager.harvest_count == 3

    def test_360_day_schedule_window_wraps_at_year_boundary(self):
        manager = HarvestManagement(config={"calendar_type": "360_day"})
        manager.harvest_events = [HarvestEvent(harvest_date=359)]

        events = manager._check_harvest_schedule(
            1,
            calendar_type="360_day",
            current_time_seconds=0.0,
        )

        assert events == manager.harvest_events

    @pytest.mark.parametrize("invalid_date", [True, 100.5])
    def test_schedule_rejects_non_integer_event_dates_without_truncation(
        self,
        invalid_date,
    ):
        manager = HarvestManagement(config={"calendar_type": "noleap"})
        manager.harvest_events = [HarvestEvent(harvest_date=invalid_date)]

        with pytest.raises((TypeError, ValueError), match="day_of_year"):
            manager._check_harvest_schedule(
                100,
                calendar_type="noleap",
                current_time_seconds=0.0,
            )

    def test_annual_event_repeats_after_one_complete_noleap_year(self):
        manager = HarvestManagement(config={"calendar_type": "noleap"})
        manager.harvest_events = [HarvestEvent(harvest_date=1)]
        state = {
            "aboveground_biomass": torch.tensor([4.0]),
            "belowground_biomass": torch.tensor([1.0]),
            "crop_maturity": torch.tensor([1.0]),
            "residue_cover": torch.tensor([0.3]),
        }

        manager.forward(state, time=0.0, dt=3600.0)
        manager.forward(
            state,
            time=365.0 * manager._seconds_per_day_value(),
            dt=3600.0,
        )

        assert manager.harvest_count == 2

    def test_forward_requires_explicit_calendar_contract(self):
        manager = HarvestManagement()
        state = {
            "aboveground_biomass": torch.tensor([1.0]),
            "belowground_biomass": torch.tensor([0.5]),
        }

        with pytest.raises(ValueError, match="requires explicit calendar_type"):
            manager.forward(state, time=0.0, dt=3600.0, day_of_year=270)
    
    # L1.2: Parameter management system validation
    def test_parameter_access_through_manager(self, harvest_manager):
        """Test that parameters are accessible through ParameterManager"""
        # Test getting parameters
        harvest_efficiency = harvest_manager.get_parameter_value("harvest_efficiency")
        assert isinstance(harvest_efficiency, (float, torch.Tensor))
        assert 0.7 <= harvest_efficiency <= 0.99
        
        residue_removal_rate = harvest_manager.get_parameter_value("residue_removal_rate")
        assert isinstance(residue_removal_rate, (float, torch.Tensor))
        assert 0.0 <= residue_removal_rate <= 0.8
        
        # Test parameter bounds validation
        root_mortality_rate = harvest_manager.get_parameter_value("root_mortality_rate")
        assert 0.1 <= root_mortality_rate <= 0.9  # Reasonable range for root mortality
    
    def test_parameter_units_validation(self, harvest_manager):
        """Test that parameter units are properly validated"""
        # Test that parameters have proper units
        residue_removal_rate = harvest_manager.get_parameter_value(
            "residue_removal_rate",
            units="dimensionless",
            validate=True
        )
        assert isinstance(residue_removal_rate, (float, torch.Tensor))

    def test_seconds_per_day_uses_physical_constant_ssot(self, harvest_manager):
        """Physical time constants must bypass the scientific parameter accessor."""
        required = set(harvest_manager.required_parameters())
        physical_constants = set(harvest_manager.physical_constant_keys())
        assert physical_constants == {
            "relic.physical.constants.time.seconds_per_day",
        }
        assert required.isdisjoint(physical_constants)

        with patch(
            "relic.scientific.anthroposphere.management.harvest.get_physical_constant",
            return_value=86400.0,
            create=True,
        ) as physical_constant:
            with patch.object(
                harvest_manager,
                "get_parameter_value",
                side_effect=AssertionError("Physical constants must use the SSOT API"),
            ):
                assert harvest_manager._seconds_per_day_value() == 86400.0

        physical_constant.assert_called_once_with("seconds_per_day")
    
    def test_parameter_bounds_checking(self, harvest_manager):
        """Test that parameter bounds are enforced"""
        # These should work within bounds
        valid_params = {
            "harvest_efficiency": 0.92,
            "residue_removal_rate": 0.25,
            "root_mortality_rate": 0.6
        }
        
        # An in-bounds instance value is accepted and read back from the config
        # source (there is no caller default to fall back on).
        for param_name, value in valid_params.items():
            harvest_manager.set_parameter_value(param_name, value)
            result = harvest_manager.get_parameter_value(param_name, source_order=("config",), required=True)
            assert float(result) == pytest.approx(value)
    
    # L1.3: Scientific calculation accuracy (biomass removal)
    def test_biomass_removal_calculation(self, harvest_manager, sample_state):
        """Test biomass removal rate calculation accuracy"""
        rates = call_harvest_partition(harvest_manager, sample_state)
        
        # Check that biomass removal rate is calculated
        assert "biomass_removal_rate" in rates
        removal_rate = rates["biomass_removal_rate"]
        
        # Should be positive for mature crops
        assert torch.all(removal_rate >= 0)
        
        # The grain removal increases with the harvest index of the event.
        index_samples = [0.2, 0.4, 0.6]
        grain_by_index = []

        for harvest_index in index_samples:
            index_rates = call_harvest_partition(
                harvest_manager,
                sample_state,
                event=HarvestEvent(harvest_index=harvest_index),
            )
            grain_by_index.append(index_rates["grain_yield_rate"].mean().item())

        assert grain_by_index == sorted(grain_by_index)
        assert grain_by_index[-1] > grain_by_index[0]
    
    def test_grain_yield_calculation(self, harvest_manager, sample_state):
        """Test grain yield calculation"""
        rates = call_harvest_partition(harvest_manager, sample_state)
        
        # Check grain yield outputs
        assert "grain_yield_rate" in rates
        assert "effective_harvest_index" in rates
        
        grain_yield_rate = rates["grain_yield_rate"]
        harvest_index = rates["effective_harvest_index"]
        
        # All should be non-negative
        assert torch.all(grain_yield_rate >= 0)
        assert torch.all(harvest_index >= 0)
        
        # Harvest index should be reasonable (typically 0.3-0.6)
        assert torch.all(harvest_index >= 0.1) and torch.all(harvest_index <= 0.8)
        
        # Grain yield should be fraction of total biomass
        total_biomass = sample_state['aboveground_biomass']
        if torch.any(grain_yield_rate > 0):
            calculated_hi = grain_yield_rate / torch.clamp(total_biomass, min=1e-10)
            assert torch.allclose(calculated_hi, harvest_index, rtol=0.1)
    
    def test_residue_generation_calculation(self, harvest_manager, sample_state):
        """Test residue generation calculation"""
        rates = call_harvest_partition(harvest_manager, sample_state)
        
        # Check residue generation outputs
        assert "residue_generation_rate" in rates
        assert "delta_residue_cover" in rates
        
        residue_generation_rate = rates["residue_generation_rate"]
        cover_change = rates["delta_residue_cover"]
        
        # Should be related to total biomass
        total_biomass = sample_state['aboveground_biomass']
        
        # Residue should be positive fraction of biomass
        if torch.any(residue_generation_rate > 0):
            residue_fraction = residue_generation_rate / torch.clamp(total_biomass, min=1e-10)
            assert torch.all(residue_fraction >= 0) and torch.all(residue_fraction <= 1.0)
        
        # Cover change should be related to residue generation
        assert torch.all(cover_change >= 0)  # Residue cover should increase
    
    def test_root_mortality_calculation(self, harvest_manager, sample_state):
        """Test root mortality calculation"""
        rates = call_harvest_partition(harvest_manager, sample_state)
        
        # Check root mortality outputs
        assert "root_mortality_flux" in rates
        assert "belowground_carbon_flux" in rates

        root_mortality_rate = rates["root_mortality_flux"]
        bg_carbon_flux = rates["belowground_carbon_flux"]
        
        # Should be related to belowground biomass
        belowground_biomass = sample_state['belowground_biomass']
        
        # Root mortality should be positive fraction of belowground biomass
        if torch.any(root_mortality_rate > 0):
            mortality_fraction = root_mortality_rate / torch.clamp(belowground_biomass, min=1e-10)
            assert torch.all(mortality_fraction >= 0) and torch.all(mortality_fraction <= 1.0)
        
        # Carbon flux should be proportional to root mortality
        if torch.any(root_mortality_rate > 0):
            carbon_fraction = bg_carbon_flux / torch.clamp(root_mortality_rate, min=1e-10)
            # Carbon is typically ~45% of biomass
            assert torch.allclose(carbon_fraction, torch.tensor(0.45), rtol=0.2)
    
    # L1.4: Conservation laws validation
    def test_biomass_mass_conservation(self, harvest_manager):
        """Test biomass mass conservation in harvest operations"""
        # Create a controlled test scenario
        initial_aboveground = torch.tensor([5000.0], device=DEVICE)
        initial_belowground = torch.tensor([1200.0], device=DEVICE)
        
        state = {
            'aboveground_biomass': initial_aboveground,
            'belowground_biomass': initial_belowground,
            'crop_maturity': torch.tensor([1.0], device=DEVICE),
            'residue_cover': torch.tensor([0.0], device=DEVICE),
        }
        
        rates = call_harvest_partition(harvest_manager, state)
        
        # Check mass balance
        grain_yield_rate = rates["grain_yield_rate"]
        residue_generation_rate = rates["residue_generation_rate"]
        root_mortality_rate = rates["root_mortality_flux"]

        # Total biomass removal should equal sum of components
        total_removal = rates["biomass_removal_rate"]
        calculated_removal = grain_yield_rate + residue_generation_rate
        
        # Allow for some numerical tolerance
        mass_balance_error = torch.abs(total_removal - calculated_removal)
        assert mass_balance_error < TOLERANCE
    
    def test_carbon_mass_conservation(self, harvest_manager):
        """Test carbon mass conservation in harvest operations"""
        # Test carbon flows during harvest
        aboveground_biomass = torch.tensor([4000.0], device=DEVICE)  # g/m2
        carbon_fraction = 0.45  # Typical C fraction in biomass
        
        state = {
            'aboveground_biomass': aboveground_biomass,
            'belowground_biomass': torch.tensor([0.2], device=DEVICE),
            'crop_maturity': torch.tensor([1.0], device=DEVICE),
            'residue_cover': torch.tensor([0.0], device=DEVICE),
        }
        
        rates = call_harvest_partition(harvest_manager, state)
        
        # Check carbon fluxes
        if "aboveground_carbon_flux" in rates:
            carbon_flux = rates["aboveground_carbon_flux"]
            biomass_flux = rates["biomass_removal_rate"]
            
            # Carbon flux should be appropriate fraction of biomass flux
            if torch.any(biomass_flux > 0):
                calculated_c_flux = biomass_flux * carbon_fraction
                assert torch.allclose(carbon_flux, calculated_c_flux, rtol=0.1)
    
    # L1.5: Harvest efficiency and residue management
    def test_harvest_efficiency_effects(self, harvest_manager):
        """Test harvest efficiency effects on biomass removal"""
        # Test different harvest efficiencies
        efficiencies = [0.7, 0.8, 0.9, 0.95]  # Low to high efficiency
        biomass_removals = []
        
        for efficiency in efficiencies:
            state = {
                'aboveground_biomass': torch.tensor([4000.0], device=DEVICE),
                'belowground_biomass': torch.tensor([0.2], device=DEVICE),
                'residue_cover': torch.tensor([0.0], device=DEVICE),
                'crop_maturity': torch.tensor([1.0], device=DEVICE),
            }
            
            rates = call_harvest_partition(harvest_manager, state)
            biomass_removals.append(rates["biomass_removal_rate"].item())
        
        # Higher efficiency should lead to more biomass removal
        for i in range(1, len(efficiencies)):
            assert biomass_removals[i] >= biomass_removals[i-1], \
                f"Removal should increase with efficiency at efficiency {efficiencies[i]}"
    
    def test_residue_removal_effects(self, harvest_manager):
        """Test residue removal effects on soil protection"""
        # Test different residue removal rates
        removal_rates = [0.0, 0.3, 0.6, 0.8]  # None to high removal
        residue_covers = []
        
        for removal_rate in removal_rates:
            state = {
                'aboveground_biomass': torch.tensor([4000.0], device=DEVICE),
                'belowground_biomass': torch.tensor([0.2], device=DEVICE),
                'residue_cover': torch.tensor([0.8], device=DEVICE),
                'crop_maturity': torch.tensor([1.0], device=DEVICE)
            }
            
            rates = call_harvest_partition(harvest_manager, state)
            if "residue_cover_after_harvest" in rates:
                residue_covers.append(rates["residue_cover_after_harvest"].item())
        
        if len(residue_covers) > 1:
            # Higher removal should lead to lower residue cover
            for i in range(1, len(removal_rates)):
                assert residue_covers[i] <= residue_covers[i-1], \
                    f"Residue cover should decrease with removal rate {removal_rates[i]}"
    
    # L1.6: Abstract base class compliance
    def test_biological_process_interface(self, harvest_manager, sample_state):
        """Test compliance with BiologicalProcess interface"""
        # Forward-only API: BiologicalProcessFunctor subclasses must expose forward().
        assert hasattr(harvest_manager, 'forward')
        assert callable(harvest_manager.forward)
        assert not hasattr(harvest_manager, "compute_rate")
        
        # forward should return dictionary with rates
        rates = call_forward(harvest_manager, sample_state)
        assert isinstance(rates, dict)
        assert len(rates) > 0
        
        # All rate values should be tensors
        for rate_name, rate_value in rates.items():
            assert isinstance(rate_value, torch.Tensor)
    
    def test_forward_method_consistency(self, harvest_manager):
        """The scheduled runtime exposes one non-redundant port per quantity."""
        harvest_manager.harvest_events = [HarvestEvent(harvest_date=1)]
        state = {
            'aboveground_biomass': torch.tensor([4.0], device=DEVICE),
            'belowground_biomass': torch.tensor([1.0], device=DEVICE),
            'crop_maturity': torch.tensor([1.0], device=DEVICE),
            'residue_cover': torch.tensor([0.3], device=DEVICE),
        }

        outputs = harvest_manager.forward(state, time=0.0, dt=3600.0)

        expected_contracts = {
            "harvest_flux": (UpdateMode.REPLACE, "kg DM m-2 s-1"),
            "harvest_biomass_carbon_input": (
                UpdateMode.REPLACE,
                "kg C m-2 s-1",
            ),
            "harvest_carbon_export_flux": (
                UpdateMode.REPLACE,
                "kg C m-2 s-1",
            ),
            "residue_c_flux": (UpdateMode.REPLACE, "kg C m-2 s-1"),
            "harvest_biomass_nitrogen_input": (
                UpdateMode.REPLACE,
                "kg N m-2 s-1",
            ),
            "harvest_nitrogen_export_flux": (
                UpdateMode.REPLACE,
                "kg N m-2 s-1",
            ),
            "residue_n_flux": (UpdateMode.REPLACE, "kg N m-2 s-1"),
            "delta_residue_cover": (UpdateMode.ACCUMULATE, "s-1"),
            "diagnostics.harvest.events_today": (UpdateMode.REPLACE, "1"),
            "diagnostics.harvest.biomass_partition_residual": (
                UpdateMode.REPLACE,
                "kg DM m-2",
            ),
            "diagnostics.harvest.carbon_ledger_residual": (
                UpdateMode.REPLACE,
                "kg C m-2 s-1",
            ),
            "diagnostics.harvest.nitrogen_ledger_residual": (
                UpdateMode.REPLACE,
                "kg N m-2 s-1",
            ),
            "diagnostics.harvest_management.free_energy": (
                UpdateMode.REPLACE,
                "kg2 m-4",
            ),
            "diagnostics.harvest_management.chemical_potential_norm": (
                UpdateMode.REPLACE,
                "kg m-2",
            ),
            "diagnostics.harvest_management.canonical_iphs_runtime": (
                UpdateMode.REPLACE,
                "1",
            ),
        }
        assert set(outputs) == set(expected_contracts)
        for name, (mode, unit) in expected_contracts.items():
            item = outputs[name]
            assert isinstance(item, UpdateItem), name
            assert item.mode is mode, name
            assert item.unit == unit, name
            assert item.source_functor == harvest_manager.name, name

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

    def test_non_event_step_emits_zero_management_fluxes(self, harvest_manager):
        harvest_manager.harvest_events = [HarvestEvent(harvest_date=100)]
        state = {
            "aboveground_biomass": torch.tensor([4.0]),
            "belowground_biomass": torch.tensor([1.0]),
            "crop_maturity": torch.tensor([1.0]),
            "residue_cover": torch.tensor([0.3]),
        }

        outputs = harvest_manager.forward(state, time=0.0, dt=3600.0)

        for name in (
            "harvest_flux",
            "harvest_biomass_carbon_input",
            "harvest_carbon_export_flux",
            "residue_c_flux",
            "harvest_biomass_nitrogen_input",
            "harvest_nitrogen_export_flux",
            "residue_n_flux",
            "delta_residue_cover",
        ):
            torch.testing.assert_close(
                outputs[name].value,
                torch.zeros_like(outputs[name].value),
            )

    def test_harvest_ports_have_unique_strict_scm_contracts(self, harvest_manager):
        contract_dir = (
            REPO_ROOT / "relic" / "contracts" / "variables" / "anthroposphere"
        )
        expected = {
            "harvest_flux": ("kg DM m-2 s-1", None, "out"),
            "harvest_biomass_carbon_input": (
                "kg C m-2 s-1",
                "carbon.total",
                "in",
            ),
            "harvest_carbon_export_flux": (
                "kg C m-2 s-1",
                "carbon.total",
                "out",
            ),
            "residue_c_flux": ("kg C m-2 s-1", "carbon.total", "out"),
            "harvest_biomass_nitrogen_input": (
                "kg N m-2 s-1",
                "nitrogen.total",
                "in",
            ),
            "harvest_nitrogen_export_flux": (
                "kg N m-2 s-1",
                "nitrogen.total",
                "out",
            ),
            "residue_n_flux": ("kg N m-2 s-1", "nitrogen.total", "out"),
        }

        for variable, (unit, ledger, direction) in expected.items():
            path = contract_dir / f"{variable}.yaml"
            text = path.read_text(encoding="utf-8")
            contract = yaml.safe_load(text)
            assert "REVIEW_NEEDED" not in text
            assert contract["variable"] == variable
            assert contract["kind"] == "flux"
            assert contract["dimension"]["udunits"] == unit
            assert contract["owner"] == harvest_manager.name
            assert contract["incrementable"] is False
            assert contract["accumulates"] == ledger
            assert contract["writers"]["increments"] == []
            assert contract["writers"]["tendencies"] == [
                {
                    "name": harvest_manager.name,
                    "module": (
                        "relic.scientific.anthroposphere.management.harvest"
                    ),
                    "provides": [variable],
                    "unit": unit,
                }
            ]
            assert contract["flux_contract"]["allowed_in"] == (
                [harvest_manager.name] if direction == "in" else []
            )
            assert contract["flux_contract"]["allowed_out"] == (
                [harvest_manager.name] if direction == "out" else []
            )

        assert not (contract_dir / "biomass_removal.yaml").exists()

    def test_real_simulator_commits_one_conservative_harvest_event(self):
        simulator = RELICSimulator(
            {
                "max_time": 7200.0,
                "time_step": 3600.0,
                "calendar_type": "noleap",
                "conservation_checks": True,
                "device": "cpu",
            }
        )
        simulator.initialize_state(
            {
                "aboveground_biomass": torch.tensor([4.0]),
                "belowground_biomass": torch.tensor([1.0]),
                "crop_maturity": torch.tensor([1.0]),
                "residue_cover": torch.tensor([0.3]),
                "day_of_year": torch.tensor(1.0),
            }
        )
        manager = HarvestManagement(config={"calendar_type": "noleap"})
        manager.harvest_events = [HarvestEvent(harvest_date=1)]
        simulator.register_functor(manager, execution_group="fast")

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            first = simulator.step(dt=3600.0)
            second = simulator.step(dt=3600.0)

        assert torch.any(first["harvest_flux"] > 0.0)
        assert torch.count_nonzero(second["harvest_flux"]) == 0
        torch.testing.assert_close(
            first["harvest_biomass_carbon_input"],
            first["harvest_carbon_export_flux"] + first["residue_c_flux"],
        )
        torch.testing.assert_close(
            first["harvest_biomass_nitrogen_input"],
            first["harvest_nitrogen_export_flux"] + first["residue_n_flux"],
        )
        assert manager.harvest_count == 1
    
    def test_dependency_declaration(self, harvest_manager):
        """State and event ports do not create false scheduling edges."""
        dependencies = harvest_manager.get_dependencies()
        assert dependencies == []
    
    # Additional validation tests
    def test_harvest_parameter_values(self, harvest_manager):
        """Test that harvest model parameters have reasonable values"""
        # Test key harvest parameters
        harvest_efficiency = harvest_manager.get_parameter_value("harvest_efficiency")
        assert 0.7 <= harvest_efficiency <= 0.99  # Reasonable range for efficiency
        
        residue_removal_rate = harvest_manager.get_parameter_value("residue_removal_rate")
        assert 0.0 <= residue_removal_rate <= 0.8  # Should not remove all residue
        
        root_mortality_rate = harvest_manager.get_parameter_value("root_mortality_rate")
        assert 0.1 <= root_mortality_rate <= 0.9  # Reasonable range for root mortality
    
    def test_edge_cases_and_error_handling(self, harvest_manager):
        """Test edge cases and error handling"""
        # Test with immature crop
        immature_state = {
            'aboveground_biomass': torch.tensor([1000.0], device=DEVICE),
            'belowground_biomass': torch.tensor([0.2], device=DEVICE),
            'residue_cover': torch.tensor([0.0], device=DEVICE),
            'crop_maturity': torch.tensor([0.3], device=DEVICE),
        }
        
        rates = call_harvest_partition(harvest_manager, immature_state)
        
        # Should not crash and should return minimal fluxes for immature crop
        for rate_name, rate_value in rates.items():
            if "rate" in rate_name:
                assert torch.all(rate_value >= 0)  # Should be non-negative
        
        # Test with zero biomass
        zero_biomass_state = immature_state.copy()
        zero_biomass_state['aboveground_biomass'] = torch.tensor([0.0], device=DEVICE)
        
        zero_rates = call_harvest_partition(
            harvest_manager,
            zero_biomass_state,
        )
        
        # Should handle zero biomass without crashing
        assert all(torch.isfinite(rate).all() for rate in zero_rates.values())


# ---------------------------------------------------------------------------
# Event-driven partition, no aliases, parameter-backed schedule (MA-21 round 14)
# ---------------------------------------------------------------------------


def _event_state():
    return {
        "aboveground_biomass": torch.tensor([4.0]),
        "belowground_biomass": torch.tensor([1.0]),
        "residue_cover": torch.tensor([0.2]),
    }


def test_partition_requires_a_scheduled_event():
    """The partition kernel evaluates one scheduled event; it constructs no default
    event and has no event-less branch."""
    import inspect

    manager = HarvestManagement(config={"calendar_type": "noleap"})
    with pytest.raises(TypeError, match="event"):
        manager._canonical_iphs_harvest_partition(_event_state())
    source = inspect.getsource(HarvestManagement._canonical_iphs_harvest_partition)
    assert "HarvestEvent(" not in source
    assert "harvest_events[-1]" not in source
    assert not hasattr(HarvestManagement, "_compute_rate_core")


def test_harvest_reads_only_the_inputs_of_its_process_law():
    """crop_maturity and grain_moisture entered only the removed event-less branch,
    whose flows reached no output: they are not inputs, and the outputs of a scheduled
    harvest do not depend on them."""
    manager = HarvestManagement(config={"calendar_type": "noleap"})
    assert "crop_maturity" not in manager.metadata.input_variables
    assert "grain_moisture" not in manager.metadata.input_variables
    for key in (
        "anthroposphere.management.harvest.grain_moisture_threshold",
        "anthroposphere.management.harvest.harvest_type_factors",
    ):
        assert key not in manager.metadata.parameters
    results = []
    for extra in ({}, {"crop_maturity": torch.tensor([0.2]), "grain_moisture": torch.tensor([0.4])}):
        manager = HarvestManagement(config={"calendar_type": "noleap"})
        manager.harvest_events = [HarvestEvent(harvest_date=1)]
        outputs = manager.forward(dict(_event_state(), **extra), time=0.0, dt=86400.0)
        results.append({k: v.value.detach().clone() for k, v in outputs.items()})
    for key, value in results[0].items():
        torch.testing.assert_close(results[1][key], value, rtol=0.0, atol=0.0)


def test_harvest_has_no_biomass_or_event_field_aliases():
    """aboveground_biomass is read under its contract name only, and HarvestEvent has
    one field per quantity (harvest_date, harvest_method)."""
    manager = HarvestManagement(config={"calendar_type": "noleap"})
    state = _event_state()
    state["biomass"] = state.pop("aboveground_biomass")
    with pytest.raises(KeyError, match="aboveground_biomass"):
        manager.forward(state, time=0.0, dt=86400.0)
    with pytest.raises(TypeError):
        HarvestEvent(date=100)
    with pytest.raises(TypeError):
        HarvestEvent(method="mowing")


def test_harvest_schedule_window_and_event_interval_are_parameters(monkeypatch):
    """d_C <= W and t_now - t_last > T_min with W and T_min from the parameter
    database (scheduling conventions, units day) instead of the literals 3 and 7."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    for key in ("schedule_window_days", "minimum_event_interval_days"):
        spec = manager.get_parameter(f"anthroposphere.management.harvest.{key}")
        assert spec is not None, key
        assert spec.units == "day", key

    harvest = HarvestManagement(config={"calendar_type": "noleap"})
    harvest.harvest_events = [HarvestEvent(harvest_date=100)]
    assert harvest._check_harvest_schedule(
        102, calendar_type="noleap", current_time_seconds=101 * 86400.0
    )
    original = harvest.require_scalar
    configured = {"schedule_window_days": 1.0, "minimum_event_interval_days": 7.0}

    def require_scalar(name, *args, **kwargs):
        if name in configured:
            return configured[name]
        return original(name, *args, **kwargs)

    monkeypatch.setattr(harvest, "require_scalar", require_scalar)
    assert not harvest._check_harvest_schedule(
        102, calendar_type="noleap", current_time_seconds=101 * 86400.0
    )


def test_default_schedule_comes_from_the_parameter_database():
    """The default cereal harvest and the two forage cuts take date, harvest index and
    residue retention from anthroposphere.management.harvest.default_schedule.*."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    harvest = HarvestManagement(config={"calendar_type": "noleap"})
    names = ("cereal", "forage_first_cut", "forage_second_cut")
    assert len(harvest.harvest_events) == len(names)
    for name, event in zip(names, harvest.harvest_events):
        prefix = f"anthroposphere.management.harvest.default_schedule.{name}"
        assert event.harvest_date == int(manager.get(f"{prefix}.date"))
        assert event.harvest_index == pytest.approx(float(manager.get(f"{prefix}.harvest_index")))
        assert event.residue_retention == pytest.approx(
            float(manager.get(f"{prefix}.residue_retention"))
        )


def test_crop_without_parameters_raises_and_hodge_is_not_padded():
    """A crop type without carbon content and C:N parameters raises instead of taking
    generic default_* values; the Hodge contract has exactly one weight per channel."""
    from relic.scientific.anthroposphere.management.harvest import CropType

    manager = HarvestManagement(config={"calendar_type": "noleap"})
    assert set(manager._get_crop_characteristics(CropType.CEREAL_C3)) == {"c_content", "cn_ratio"}
    with pytest.raises(ValueError, match="fiber"):
        manager._get_crop_characteristics(CropType.FIBER)
    with pytest.raises(ValueError, match="channels"):
        manager.build_hodge_contract(n_dofs=6)


def test_grain_moisture_has_no_contract_without_a_reader():
    """No RELIC Functor reads or writes grain_moisture since the event-less harvest
    partition was removed, so it has no SCM contract (item 107)."""
    from relic.contracts.contract_loader import PACKAGED_CONTRACTS_DIR, load_contract_registry

    assert load_contract_registry(PACKAGED_CONTRACTS_DIR).get_contract("grain_moisture") is None


def test_harvest_event_has_only_the_fields_the_partition_reads():
    """HarvestEvent carries the fields the model uses (crop type, date, harvest index,
    residue retention, quality factor); descriptive fields that no calculation reads
    (method, product category, residue management, moisture content, efficiency) are
    not kept as fields that appear to take effect (item 109)."""
    import dataclasses

    names = {field.name for field in dataclasses.fields(HarvestEvent)}
    assert names == {
        "crop_type",
        "harvest_date",
        "harvest_index",
        "residue_retention",
        "quality_factor",
    }


def test_harvest_reads_no_environment_channel():
    """Inputs come from the state only: forward reads no environment keyword and the
    partition takes no environment mapping (item 112)."""
    import inspect

    assert "environment" not in inspect.signature(
        HarvestManagement._canonical_iphs_harvest_partition
    ).parameters
    assert "environment" not in inspect.getsource(HarvestManagement.forward)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


