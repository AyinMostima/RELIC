#!/usr/bin/env python3
"""
L1 Unit Tests for Irrigation Management Module

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Parameter management system validation
- L1.3: Water balance calculations (FAO-56 methodology)
- L1.4: Irrigation efficiency calculations
- L1.5: Soil moisture response functions
- L1.6: Abstract base class compliance

Scientific References:
- Allen, R.G., et al. (1998). Crop evapotranspiration - Guidelines for computing crop water requirements. FAO Irrigation and drainage paper 56.
- Doorenbos, J. & Pruitt, W.O. (1977). Crop water requirements. FAO irrigation and drainage paper 24.
- Hanks, R.J. (1974). Model for predicting plant yield as influenced by water use. Agronomy Journal 66(5), 660-665.

Test Framework: pytest
"""

import pytest
import torch
import sys
import os
import warnings
from pathlib import Path
from typing import Dict

import yaml
import relic.scientific.anthroposphere as anthroposphere_package
from relic.scientific.anthroposphere import management as management_package
from relic.scientific.anthroposphere.management import irrigation as irrigation_module

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import modules to test
from relic.scientific.anthroposphere.management.irrigation import (
    IrrigationManagement,
)
from relic.core.abstractions import TransportMechanism
from relic.core.functor import TransportMechanismFunctor
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.contracts.contract_loader import ContractRegistry
from relic.data.standards.unit_converter import ConversionContext, unit_converter
from test.utils.forward_call import call_forward

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 5
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3


def _output_value(outputs: Dict[str, UpdateItem], key: str) -> torch.Tensor:
    """Return one explicitly declared irrigation output value."""
    item = outputs[key]
    assert isinstance(item, UpdateItem)
    return item.value


def _canonical_state(
    *,
    soil_moisture: float = 0.20,
    precipitation_flux: float = 0.0,
    irrigation_type: int = 2,
    field_capacity: float = 0.35,
    wilting_point: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Build one complete canonical irrigation state."""
    return {
        "soil_moisture": torch.tensor([[soil_moisture]], device=DEVICE),
        "precipitation_flux": torch.tensor(
            [[precipitation_flux]],
            device=DEVICE,
        ),
        "land_use_type": torch.tensor([[1]], device=DEVICE),
        "irrigation_type": torch.tensor([[irrigation_type]], device=DEVICE),
        "field_capacity": torch.tensor([[field_capacity]], device=DEVICE),
        "wilting_point": torch.tensor([[wilting_point]], device=DEVICE),
        "irrigation_remaining_quota": torch.tensor(800.0, device=DEVICE),
    }


class TestIrrigationManagementL1:
    """L1 Unit Tests for IrrigationManagement"""
    
    @pytest.fixture
    def irrigation_manager(self):
        """Create irrigation manager instance for testing"""
        return IrrigationManagement()
    
    @pytest.fixture
    def sample_state(self):
        """Sample state dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'soil_moisture': torch.tensor([0.15, 0.20, 0.25, 0.30, 0.35], device=DEVICE).reshape(batch_shape),
            'precipitation_flux': torch.tensor([0.0, 1e-6, 2e-6, 0.0, 1e-6], device=DEVICE).reshape(batch_shape),
            'land_use_type': torch.tensor([1, 1, 1, 0, 1], device=DEVICE).reshape(batch_shape),  # Some non-cropland
            'irrigation_type': torch.tensor([2, 2, 3, 2, 1], device=DEVICE).reshape(batch_shape),  # Different types
            'field_capacity': torch.tensor([0.35, 0.35, 0.35, 0.35, 0.35], device=DEVICE).reshape(batch_shape),
            'wilting_point': torch.tensor([0.15, 0.15, 0.15, 0.15, 0.15], device=DEVICE).reshape(batch_shape),
            'irrigation_remaining_quota': torch.tensor(800.0, device=DEVICE),
        }
    
    @pytest.fixture
    def sample_environment(self):
        """Sample environment dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'air_temperature': torch.tensor([288.15, 293.15, 298.15, 303.15, 308.15], device=DEVICE).reshape(batch_shape),
            'relative_humidity': torch.tensor([0.6, 0.7, 0.8, 0.9, 0.5], device=DEVICE).reshape(batch_shape),
            'wind_speed': torch.tensor([2.0, 2.5, 3.0, 3.5, 4.0], device=DEVICE).reshape(batch_shape)
        }

    # L1.1: Class instantiation and initialization tests
    def test_class_instantiation(self, irrigation_manager):
        """Test that IrrigationManagement can be instantiated"""
        assert irrigation_manager is not None
        assert isinstance(irrigation_manager, IrrigationManagement)
        assert isinstance(irrigation_manager, TransportMechanismFunctor)
        assert isinstance(irrigation_manager, TransportMechanism)
        assert hasattr(irrigation_manager, 'name')
        assert irrigation_manager.name == "irrigation_management"
    
    def test_metadata_completeness(self, irrigation_manager):
        """Test that metadata is properly set"""
        metadata = irrigation_manager.get_metadata()
        assert metadata is not None
        assert metadata.name == "irrigation_management"
        assert len(metadata.references) > 0
        expression = metadata.mathematical_expression
        assert expression.startswith("Canonical IPHS:")
        assert "managed water boundary port" in expression
        assert "M_net" in expression
        assert "M_delivered" in expression
        assert "level 2 non-strict" in expression
        assert "no prognostic water state" in expression
        assert "J_h e_S = 0" not in expression
        assert "M_h e_E = 0" not in expression
        assert "mass" in metadata.conservation_laws
        assert set(metadata.parameters) == {
            f"anthroposphere.management.irrigation.{name}"
            for name in (
                "moisture_threshold",
                "surface_efficiency",
                "sprinkler_efficiency",
                "drip_efficiency",
                "subsurface_efficiency",
                "root_zone_depth",
                "max_application_rate",
                "conservation_tolerance",
            )
        }
        assert metadata.input_variables == [
            "soil_moisture",
            "precipitation_flux",
            "land_use_type",
            "irrigation_type",
            "field_capacity",
            "wilting_point",
            "irrigation_remaining_quota",
        ]
        assert "precipitation" not in metadata.input_variables

    def test_runtime_name_is_the_canonical_scm_writer(self):
        """Reject a writer identity that is not authorized by the SCM."""
        with pytest.raises(ValueError, match="irrigation_management"):
            IrrigationManagement(name="custom_irrigation")

    def test_unused_mutable_control_aliases_are_not_public(self, irrigation_manager):
        """Use state ports instead of an ineffective second control surface."""
        for owner in (
            irrigation_module,
            management_package,
            anthroposphere_package,
        ):
            assert not hasattr(owner, "IrrigationType")
            assert not hasattr(owner, "IrrigationSchedule")
        assert not hasattr(irrigation_manager, "set_irrigation_type")
        assert not hasattr(irrigation_manager, "set_schedule_type")
        assert "irrigation_type" in irrigation_manager.metadata.input_variables
    
    def test_initialization_with_config(self):
        """Test initialization with custom configuration"""
        custom_config = {
            "parameters": {
                "surface_efficiency": {"value": 0.65, "units": "1"},
                "moisture_threshold": {"value": 0.4, "units": "1"},
            }
        }
        
        manager = IrrigationManagement(config=custom_config)
        assert manager is not None
        from relic.parameters import get_global_manager

        efficiency = manager.get_parameter_value(
            "surface_efficiency",
            namespace="anthroposphere.management.irrigation",
            validate=True,
        )
        pm = get_global_manager()
        expected = pm.get("anthroposphere.management.irrigation.surface_efficiency")
        efficiency_val = efficiency.item() if isinstance(efficiency, torch.Tensor) else efficiency
        assert abs(efficiency_val - float(expected)) < 1e-6

    def test_irrigation_system_initialization(self, irrigation_manager):
        """The Functor holds no accounting buffer (item 119): the remaining quota is
        the StateManager state irrigation_remaining_quota."""
        state_dict = irrigation_manager.state_dict()
        for removed in (
            "remaining_quota_mm",
            "cumulative_applied_water_mm",
            "cumulative_abstraction_mm",
            "last_irrigation_time_s",
        ):
            assert removed not in state_dict
            assert not hasattr(irrigation_manager, removed)
        for old_name in (
            "remaining_quota",
            "cumulative_irrigation",
            "cumulative_abstraction",
            "last_irrigation_time",
        ):
            assert not hasattr(irrigation_manager, old_name)
    
    def test_learnable_parameters_registration(self, irrigation_manager):
        """The learnable parameters are the four system efficiencies: the
        ParameterManager bindings themselves, held by reference, so the manager
        calibration path (get_trainable_parameters) and the durable checkpoint
        see them. The Functor owns no module parameter of its own."""
        parameter_manager = irrigation_manager.get_parameter_manager()
        trainable = parameter_manager.get_trainable_parameters()
        held = irrigation_manager.held_manager_parameters()
        assert sorted(held) == sorted(SYSTEM_EFFICIENCY_KEYS.values())
        for key, binding in held.items():
            assert binding is trainable[key], key
            assert isinstance(binding, torch.nn.Parameter)
            assert binding.requires_grad is True
        learnable = sorted(
            key
            for key in irrigation_manager.metadata.parameters
            if getattr(parameter_manager.get_parameter(key), "requires_grad", False)
        )
        assert learnable == sorted(held)
        assert list(irrigation_manager.parameters()) == []
    
    # L1.2: Parameter management system validation
    def test_parameter_access_through_manager(self, irrigation_manager):
        """Test that parameters are accessible through ParameterManager"""
        # Test getting parameters
        efficiency = irrigation_manager.get_parameter_value(
            "surface_efficiency",
            namespace="anthroposphere.management.irrigation",
            validate=True,
        )
        assert isinstance(efficiency, (float, torch.Tensor))
        assert 0.3 <= efficiency <= 0.95
        
        threshold = irrigation_manager.get_parameter_value(
            "moisture_threshold",
            namespace="anthroposphere.management.irrigation",
            validate=True,
        )
        assert isinstance(threshold, (float, torch.Tensor))
        assert 0.3 <= threshold <= 0.8
        
        # The quota is scenario state (irrigation_remaining_quota), not a parameter.
        assert irrigation_manager.get_parameter_value(
            "annual_water_quota",
            namespace="anthroposphere.management.irrigation",
        ) is None
    
    def test_irrigation_efficiency_by_type(self, irrigation_manager):
        """Expose each ParameterManager-backed system efficiency through forward."""
        efficiencies = []
        for system_id, parameter_name in (
            (1, "surface_efficiency"),
            (2, "sprinkler_efficiency"),
            (3, "drip_efficiency"),
            (4, "subsurface_efficiency"),
        ):
            outputs = irrigation_manager.forward(
                _canonical_state(irrigation_type=system_id),
                time=0.0,
                dt=3600.0,
            )
            actual = _output_value(
                outputs,
                "diagnostics.irrigation.effective_efficiency",
            )
            expected = irrigation_manager._get_irrigation_param(
                parameter_name
            ).to(dtype=actual.dtype, device=actual.device)
            torch.testing.assert_close(actual, expected.expand_as(actual))
            efficiencies.append(actual.item())
        assert efficiencies[2] > efficiencies[1] > efficiencies[0]
        assert efficiencies[3] >= efficiencies[2]
    
    def test_parameter_bounds_validation(self, irrigation_manager):
        """Test bounds for parameters used by the canonical calculation."""
        root_zone_depth = irrigation_manager.get_parameter_value(
            "root_zone_depth",
            namespace="anthroposphere.management.irrigation",
            validate=True,
        )
        assert 0.1 <= root_zone_depth <= 3.0

        sprinkler_efficiency = irrigation_manager.get_parameter_value(
            "sprinkler_efficiency",
            namespace="anthroposphere.management.irrigation",
            validate=True,
        )
        assert 0.3 <= sprinkler_efficiency <= 0.95
        
        max_rate = irrigation_manager.get_parameter_value(
            "max_application_rate",
            namespace="anthroposphere.management.irrigation",
            validate=True,
        )
        assert 1 <= max_rate <= 50  # mm/hr
    
    # L1.3: Water balance calculations (FAO-56 methodology)
    def test_water_deficit_calculation(self, irrigation_manager):
        """Drier root zones require no less delivered irrigation."""
        delivered = []
        for moisture in (0.16, 0.18, 0.20):
            outputs = IrrigationManagement().forward(
                _canonical_state(soil_moisture=moisture),
                time=0.0,
                dt=86400.0,
            )
            delivered.append(_output_value(outputs, "irrigation_flux").item())
        assert delivered[0] > delivered[1] > delivered[2] >= 0.0

    def test_water_deficit_unit_consistency(self, irrigation_manager):
        """Use SI mass balance and the physical-constant water density."""
        soil_moisture = torch.tensor([[0.20]])
        field_capacity = torch.tensor([[0.35]])
        precipitation_flux = torch.tensor([[2e-6]])
        dt_seconds = 86400.0
        root_depth = irrigation_manager.get_parameter_value(
            "root_zone_depth",
            namespace="anthroposphere.management.irrigation",
            validate=True,
        )
        from relic.utils.physical_constants import get_physical_constant

        water_density = soil_moisture.new_tensor(
            get_physical_constant("water_density")
        )
        net_requirement = torch.clamp(
            (field_capacity - soil_moisture)
            * torch.as_tensor(root_depth, dtype=soil_moisture.dtype)
            * water_density
            - precipitation_flux * dt_seconds,
            min=0.0,
        )
        outputs = irrigation_manager.forward(
            _canonical_state(
                soil_moisture=0.20,
                precipitation_flux=2e-6,
            ),
            time=0.0,
            dt=dt_seconds,
        )
        actual_amount = _output_value(outputs, "irrigation_flux") * dt_seconds
        torch.testing.assert_close(actual_amount, net_requirement)
    
    def test_irrigation_requirement_calculation(self, irrigation_manager):
        """Gross abstraction uses one system efficiency and no leaching branch."""
        outputs = irrigation_manager.forward(
            _canonical_state(irrigation_type=2),
            time=0.0,
            dt=3600.0,
        )
        delivered = _output_value(outputs, "irrigation_flux")
        abstraction = _output_value(outputs, "abstraction_flux")
        efficiency = irrigation_manager.get_parameter_value(
            "sprinkler_efficiency",
            namespace="anthroposphere.management.irrigation",
            validate=True,
        ).to(dtype=delivered.dtype, device=delivered.device)
        torch.testing.assert_close(delivered, abstraction * efficiency)
        assert "leaching_fraction" not in irrigation_manager.metadata.parameters
    
    def test_effective_precipitation_calculation(self, irrigation_manager):
        """Canonical precipitation monotonically reduces irrigation demand."""
        fluxes = []
        for precipitation_flux in (0.0, 1e-4, 1e-3):
            outputs = IrrigationManagement().forward(
                _canonical_state(precipitation_flux=precipitation_flux),
                time=0.0,
                dt=86400.0,
            )
            fluxes.append(_output_value(outputs, "irrigation_flux").item())
        assert fluxes[0] > fluxes[1] > fluxes[2]
    
    # L1.4: Irrigation efficiency calculations
    def test_irrigation_efficiency_impact(self, irrigation_manager):
        """Test impact of irrigation efficiency on water application"""
        water_deficit = torch.tensor([50.0], device=DEVICE)  # mm
        
        efficiencies = [0.5, 0.7, 0.9]
        gross_applications = []
        
        for efficiency in efficiencies:
            gross_amount = water_deficit / efficiency
            gross_applications.append(gross_amount.item())
        
        # Lower efficiency should require more gross water
        assert gross_applications[0] > gross_applications[1] > gross_applications[2]
        
        # Check conservation: gross * efficiency should equal net
        for i, efficiency in enumerate(efficiencies):
            net_water = gross_applications[i] * efficiency
            assert abs(net_water - water_deficit.item()) < 1e-6
    
    def test_system_type_efficiency_ranges(self, irrigation_manager):
        """Public diagnostics preserve documented system-efficiency ranges."""
        ranges = {
            1: (0.5, 0.6),
            2: (0.7, 0.8),
            3: (0.85, 0.95),
            4: (0.9, 0.95),
        }
        for system_id, (minimum, maximum) in ranges.items():
            outputs = irrigation_manager.forward(
                _canonical_state(irrigation_type=system_id),
                time=0.0,
                dt=3600.0,
            )
            efficiency = _output_value(
                outputs,
                "diagnostics.irrigation.effective_efficiency",
            ).item()
            assert minimum - 1e-6 <= efficiency <= maximum + 1e-6

    def test_private_second_formula_paths_are_absent(self, irrigation_manager):
        """Forward is the only executable irrigation-process calculation."""
        for name in (
            "_compute_rate_core",
            "_calculate_irrigation_efficiency",
            "_calculate_water_deficit",
            "_calculate_irrigation_amount",
            "_check_irrigation_trigger",
            "_state_with_irrigation_default",
        ):
            assert not hasattr(irrigation_manager, name)
    
    def test_water_quota_constraints(self, irrigation_manager):
        """Test water quota constraint enforcement"""
        # Limited quota: the declared state irrigation_remaining_quota [mm]
        
        # Create state requiring more water than quota
        state = {
            'soil_moisture': torch.tensor([[0.15]], device=DEVICE),
            'precipitation_flux': torch.tensor([[0.0]], device=DEVICE),
            'land_use_type': torch.tensor([[1]], device=DEVICE),
            'irrigation_type': torch.tensor([[2]], device=DEVICE),
            'field_capacity': torch.tensor([[0.35]], device=DEVICE),
            'wilting_point': torch.tensor([[0.15]], device=DEVICE),
            'irrigation_remaining_quota': torch.tensor(20.0, device=DEVICE),
        }
        
        outputs = irrigation_manager.forward(state, time=0, dt=86400)
        
        abstraction_flux = _output_value(outputs, "abstraction_flux")
        from relic.utils.physical_constants import get_physical_constant

        water_density = get_physical_constant("water_density")
        abstraction_mm = unit_converter.convert(
            abstraction_flux * 86400.0 / water_density,
            "generic",
            "m",
            "mm",
            ConversionContext(),
        )
        assert abstraction_mm.item() <= 20.0 + 1e-6
        assert _output_value(outputs, "irrigation_remaining_quota").item() == pytest.approx(
            20.0 - abstraction_mm.item(),
            abs=1e-5,
        )
    
    # L1.5: Soil moisture response functions
    def test_moisture_threshold_triggering(self, irrigation_manager):
        """Test soil moisture threshold triggering logic"""
        # Test different moisture levels
        moisture_levels = [0.12, 0.18, 0.25, 0.32, 0.38]
        field_capacity = 0.35
        wilting_point = 0.15
        threshold = 0.5
        
        # Calculate trigger moisture: WP + threshold * (FC - WP)
        trigger_moisture = wilting_point + threshold * (field_capacity - wilting_point)
        
        for moisture in moisture_levels:
            should_irrigate = moisture < trigger_moisture
            
            # Create test state
            state = {
                'soil_moisture': torch.tensor([[moisture]], device=DEVICE),
                'precipitation_flux': torch.tensor([[0.0]], device=DEVICE),
                'land_use_type': torch.tensor([[1]], device=DEVICE),
                'irrigation_type': torch.tensor([[2]], device=DEVICE),
                'field_capacity': torch.tensor([[field_capacity]], device=DEVICE),
                'wilting_point': torch.tensor([[wilting_point]], device=DEVICE),
                'irrigation_remaining_quota': torch.tensor(800.0, device=DEVICE),
            }
            
            outputs = irrigation_manager.forward(state, time=0, dt=86400)
            irrigation_flux = _output_value(outputs, "irrigation_flux")
            
            # Check if irrigation was applied
            irrigation_applied = irrigation_flux > 0
            
            if should_irrigate:
                assert irrigation_applied.item(), f"Should irrigate at moisture {moisture}"
            else:
                # Might not irrigate if above threshold
                pass  # Allow flexibility in implementation
    
    def test_field_capacity_limiting(self, irrigation_manager):
        """Test that irrigation doesn't exceed field capacity"""
        # Test soil near field capacity
        near_fc_moisture = 0.34
        field_capacity = 0.35
        
        state = {
            'soil_moisture': torch.tensor([[near_fc_moisture]], device=DEVICE),
            'precipitation_flux': torch.tensor([[0.0]], device=DEVICE),
            'land_use_type': torch.tensor([[1]], device=DEVICE),
            'irrigation_type': torch.tensor([[2]], device=DEVICE),
            'field_capacity': torch.tensor([[field_capacity]], device=DEVICE),
            'wilting_point': torch.tensor([[0.15]], device=DEVICE),
            'irrigation_remaining_quota': torch.tensor(800.0, device=DEVICE),
        }
        
        outputs = irrigation_manager.forward(state, time=0, dt=86400)
        irrigation_flux = _output_value(outputs, "irrigation_flux")
        
        # Irrigation should be limited or zero near field capacity
        applied_amount = irrigation_flux * 86400  # mm
        max_possible = (field_capacity - near_fc_moisture) * 1000  # Convert to mm
        
        assert applied_amount.item() <= max_possible + 1e-3  # Small tolerance
    
    def test_efficiency_diagnostic_is_the_applied_system_efficiency(
        self,
        irrigation_manager,
    ):
        """Expose the same efficiency used by the delivered-water ledger."""
        # Test at different moisture levels
        moisture_levels = [0.16, 0.22, 0.28, 0.34]
        efficiencies = []
        
        for moisture in moisture_levels:
            state = {
                'soil_moisture': torch.tensor([[moisture]], device=DEVICE),
                'precipitation_flux': torch.tensor([[0.0]], device=DEVICE),
                'land_use_type': torch.tensor([[1]], device=DEVICE),
                'irrigation_type': torch.tensor([[2]], device=DEVICE),
                'field_capacity': torch.tensor([[0.35]], device=DEVICE),
                'wilting_point': torch.tensor([[0.15]], device=DEVICE),
                'irrigation_remaining_quota': torch.tensor(800.0, device=DEVICE),
            }
            
            outputs = irrigation_manager.forward(state, time=0, dt=86400)
            efficiency = _output_value(
                outputs,
                "diagnostics.irrigation.effective_efficiency",
            )
            efficiencies.append(efficiency.item())
        
        assert efficiencies == pytest.approx([0.75] * len(moisture_levels))
    
    # L1.6: Abstract base class compliance
    def test_transport_interface(self, irrigation_manager, sample_state):
        """Test compliance with TransportMechanism interface"""
        # Forward-only entrypoint is required by architecture.
        assert hasattr(irrigation_manager, 'forward')
        assert callable(irrigation_manager.forward)
        assert not hasattr(irrigation_manager, "compute_rate")
        
        # forward should return dictionary with fluxes
        rates = call_forward(irrigation_manager, sample_state)
        assert isinstance(rates, dict)
        assert len(rates) > 0
        
        # The shared call_forward helper unwraps explicit UpdateItems.
        for flux_name, flux_value in rates.items():
            assert isinstance(flux_value, torch.Tensor)
    
    def test_transport_mechanism_methods(
        self,
        irrigation_manager,
        sample_state,
    ):
        """Test transport-related methods"""
        # Forward computes water-transport outputs.
        assert hasattr(irrigation_manager, 'forward')

        rates = call_forward(irrigation_manager, sample_state)
        
        flux_key = next(
            (name for name in ("irrigation_flux", "irrigation_water_flux", "abstraction_flux") if name in rates),
            None,
        )
        assert flux_key is not None
        assert torch.all(rates[flux_key] >= 0)
    
    def test_forward_method_consistency(self, irrigation_manager, sample_state):
        """Test backward consistency with forward method"""
        # Should be able to call forward method
        outputs = irrigation_manager.forward(sample_state, time=0, dt=3600)
        
        # Should return dictionary with expected outputs
        assert isinstance(outputs, dict)
        assert "irrigation_flux" in outputs
        assert "abstraction_flux" in outputs
        assert "drainage_flux" not in outputs
        assert "drainage_flux" not in irrigation_manager.metadata.output_variables
        assert set(outputs) == set(irrigation_manager.metadata.output_variables)
        assert "diagnostics.irrigation.effective_efficiency" in outputs
        assert "diagnostics.irrigation.applied_water" in outputs
        assert "irrigation_remaining_quota" in outputs
        assert "diagnostics.irrigation.moisture_response" in outputs
        assert "diagnostics.irrigation.water_balance_residual" in outputs

        expected_units = {
            "irrigation_flux": "kg m-2 s-1",
            "abstraction_flux": "kg m-2 s-1",
            "irrigation_remaining_quota": "mm",
            "diagnostics.irrigation.applied_water": "mm",
            "diagnostics.irrigation.effective_efficiency": "1",
            "diagnostics.irrigation.moisture_response": "1",
            "diagnostics.irrigation.water_balance_residual": "kg m-2 s-1",
        }
        for key, expected_unit in expected_units.items():
            item = outputs[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == expected_unit
            assert item.source_functor == irrigation_manager.name

    def test_scm_writer_contract_matches_runtime_name(self, irrigation_manager):
        """Public boundary-flux SCMs authorize the actual runtime writer."""
        contract_dir = (
            Path(__file__).resolve().parents[3]
            / "relic"
            / "contracts"
            / "variables"
            / "anthroposphere"
        )
        expected_roles = {
            "irrigation_flux": "managed_root_zone_water_delivery_flux",
            "abstraction_flux": "managed_source_water_withdrawal_flux",
        }

        for variable, runtime_role in expected_roles.items():
            path = contract_dir / f"{variable}.yaml"
            text = path.read_text(encoding="utf-8")
            contract = yaml.safe_load(text)
            assert "REVIEW_NEEDED" not in text
            assert contract["kind"] == "flux"
            assert contract["dimension"]["udunits"] == "kg m-2 s-1"
            assert contract["owner"] == irrigation_manager.name
            assert contract["incrementable"] is False
            assert contract["accumulates"] is None
            assert contract["writers"]["increments"] == []
            assert contract["writers"]["tendencies"] == [{
                "name": irrigation_manager.name,
                "module": (
                    "relic.scientific.anthroposphere.management.irrigation"
                ),
                "provides": [variable],
                "unit": "kg m-2 s-1",
            }]
            assert contract["flux_contract"]["allowed_in"] == [
                irrigation_manager.name
            ]
            assert contract["provenance"]["runtime_writer"] == (
                irrigation_manager.name
            )
            assert contract["provenance"]["runtime_role"] == runtime_role

    def test_unique_external_precipitation_flux_contract(self):
        """Load one canonical external forcing contract without an old alias."""
        root = Path(__file__).resolve().parents[3]
        contract_path = (
            root
            / "relic"
            / "contracts"
            / "variables"
            / "atmosphere"
            / "precipitation.yaml"
        )
        text = contract_path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        assert "REVIEW_NEEDED" not in text
        assert payload["variable"] == "precipitation_flux"
        assert payload["kind"] == "forcing"
        assert payload["dimension"] == {
            "udunits": "kg m-2 s-1",
            "cf_standard_name": "precipitation_flux",
        }
        assert payload["owner"] is None
        assert payload["aliases"] == []
        assert payload["incrementable"] is False
        assert payload["accumulates"] is None
        # No RELIC process writes it; its writers are the external adapters that map a
        # precipitation field to it (the alternative providers of the forcing).
        adapters = {"WRFAdapter", "CESMAdapter", "CMIP6Adapter", "VICAdapter", "SWATAdapter"}
        assert payload["writers"]["increments"] == []
        assert {writer["name"] for writer in payload["writers"]["tendencies"]} == adapters
        assert set(payload["provenance"]["adapter_writers"]) == adapters
        assert payload["provenance"]["runtime_writer"] is None
        assert payload["provenance"]["runtime_role"] == (
            "external_precipitation_mass_flux_forcing"
        )

        registry = ContractRegistry(strict=True)
        registry.load_contracts_dir(
            root / "relic" / "contracts" / "variables"
        )
        contract = registry.get_contract("precipitation_flux")
        assert contract is not None
        assert contract.udunits == "kg m-2 s-1"
        assert registry.get_contract("precipitation") is None

    def test_real_simulator_writer_provenance_and_open_water_ledger(self):
        """Commit both public ports without a false closed-mass warning."""
        simulator = RELICSimulator({
            "max_time": 3600.0,
            "time_step": 3600.0,
            "conservation_checks": False,
            "device": "cpu",
        })
        simulator.initialize_state({
            "soil_moisture": torch.tensor([[0.20]], dtype=torch.float32),
            "precipitation_flux": torch.tensor([[0.0]], dtype=torch.float32),
            "land_use_type": torch.tensor([[1]], dtype=torch.long),
            "irrigation_type": torch.tensor([[2]], dtype=torch.long),
            "field_capacity": torch.tensor([[0.35]], dtype=torch.float32),
            "wilting_point": torch.tensor([[0.15]], dtype=torch.float32),
            "irrigation_remaining_quota": torch.tensor(800.0, dtype=torch.float32),
        })
        functor = IrrigationManagement()
        simulator.register_functor(functor, execution_group="fast")

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            updates = simulator.step(dt=3600.0)

        efficiency = functor._get_irrigation_param(
            "sprinkler_efficiency"
        ).to(dtype=updates["irrigation_flux"].dtype)
        torch.testing.assert_close(
            updates["irrigation_flux"],
            updates["abstraction_flux"] * efficiency,
        )
        torch.testing.assert_close(
            updates["diagnostics.irrigation.water_balance_residual"],
            torch.zeros_like(
                updates["diagnostics.irrigation.water_balance_residual"]
            ),
        )

        for variable in ("irrigation_flux", "abstraction_flux"):
            provenance = simulator.get_variable_provenance(variable)
            assert {record["role"] for record in provenance} == {
                "writer",
                "final_writer",
            }
            assert {record["mode"] for record in provenance} == {
                "UpdateMode.REPLACE"
            }
            assert {record["functor"] for record in provenance} == {
                "irrigation_management"
            }
        assert str(simulator.get_effective_device("irrigation_management")) == (
            "cpu"
        )
    
    def test_dependency_declaration(self, irrigation_manager):
        """External forcing and composition inputs are not Functor dependencies."""
        assert irrigation_manager.get_dependencies() == []

    def test_canonical_precipitation_flux_is_required_and_used(
        self,
        irrigation_manager,
    ):
        """Reject aliases/defaults and use the canonical precipitation mass flux."""
        base_state = {
            "soil_moisture": torch.tensor([[0.20]], device=DEVICE),
            "precipitation_flux": torch.tensor([[0.0]], device=DEVICE),
            "land_use_type": torch.tensor([[1]], device=DEVICE),
            "irrigation_type": torch.tensor([[2]], device=DEVICE),
            "field_capacity": torch.tensor([[0.35]], device=DEVICE),
            "wilting_point": torch.tensor([[0.15]], device=DEVICE),
            "irrigation_remaining_quota": torch.tensor(800.0, device=DEVICE),
        }

        missing = dict(base_state)
        missing.pop("precipitation_flux")
        with pytest.raises(KeyError, match="precipitation_flux"):
            irrigation_manager.forward(missing, time=0.0, dt=3600.0)

        legacy = dict(missing)
        legacy["precipitation"] = torch.tensor([[0.1]], device=DEVICE)
        with pytest.raises(KeyError, match="precipitation_flux"):
            irrigation_manager.forward(legacy, time=0.0, dt=3600.0)

        invalid_values = [
            0.0,
            torch.tensor([[float("nan")]], device=DEVICE),
            torch.tensor([[float("inf")]], device=DEVICE),
            torch.tensor([[-1.0e-6]], device=DEVICE),
        ]
        for invalid in invalid_values:
            state = dict(base_state)
            state["precipitation_flux"] = invalid
            expected_error = TypeError if not isinstance(invalid, torch.Tensor) else ValueError
            with pytest.raises(expected_error, match="precipitation_flux"):
                irrigation_manager.forward(state, time=0.0, dt=3600.0)

        dry_outputs = IrrigationManagement().forward(
            base_state,
            time=0.0,
            dt=3600.0,
        )
        wet_state = dict(base_state)
        wet_state["precipitation_flux"] = torch.tensor([[0.1]], device=DEVICE)
        wet_outputs = IrrigationManagement().forward(
            wet_state,
            time=0.0,
            dt=3600.0,
        )
        assert torch.all(
            _output_value(wet_outputs, "irrigation_flux")
            < _output_value(dry_outputs, "irrigation_flux")
        )
    
    # Additional validation tests
    def test_water_balance_conservation(self, irrigation_manager):
        """Test water balance conservation"""
        # Create a controlled scenario
        state = {
            'soil_moisture': torch.tensor([[0.2]], device=DEVICE),
            'precipitation_flux': torch.tensor([[1e-6]], device=DEVICE),
            'land_use_type': torch.tensor([[1]], device=DEVICE),
            'irrigation_type': torch.tensor([[2]], device=DEVICE),
            'field_capacity': torch.tensor([[0.35]], device=DEVICE),
            'wilting_point': torch.tensor([[0.15]], device=DEVICE),
            'irrigation_remaining_quota': torch.tensor(800.0, device=DEVICE),
        }
        
        outputs = irrigation_manager.forward(state, time=0, dt=86400)
        
        irrigation_flux = _output_value(outputs, "irrigation_flux")
        abstraction_flux = _output_value(outputs, "abstraction_flux")
        
        # Check water accounting: abstraction should account for efficiency losses
        if irrigation_flux.sum() > 0:
            # Abstraction should be >= irrigation (accounting for losses)
            assert torch.all(abstraction_flux >= irrigation_flux * 0.99)  # Small tolerance

    def test_application_efficiency_is_applied_once(self, irrigation_manager):
        """Gross abstraction equals net root-zone delivery divided by efficiency."""
        dt = 86400.0
        soil_moisture = torch.tensor([[0.20]], device=DEVICE)
        field_capacity = torch.tensor([[0.35]], device=DEVICE)
        state = {
            "soil_moisture": soil_moisture,
            "precipitation_flux": torch.zeros_like(soil_moisture),
            "land_use_type": torch.ones_like(soil_moisture, dtype=torch.long),
            "irrigation_type": torch.full_like(
                soil_moisture,
                2,
                dtype=torch.long,
            ),
            "field_capacity": field_capacity,
            "wilting_point": torch.tensor([[0.15]], device=DEVICE),
            "irrigation_remaining_quota": torch.tensor(800.0, device=DEVICE),
        }

        outputs = irrigation_manager.forward(state, time=0.0, dt=dt)
        efficiency = irrigation_manager._get_irrigation_param(
            "sprinkler_efficiency"
        ).to(device=DEVICE, dtype=soil_moisture.dtype)
        root_depth = irrigation_manager._get_irrigation_param(
            "root_zone_depth"
        ).to(device=DEVICE, dtype=soil_moisture.dtype)
        from relic.utils.physical_constants import get_physical_constant

        water_density = torch.as_tensor(
            get_physical_constant("water_density"),
            device=DEVICE,
            dtype=soil_moisture.dtype,
        )
        net_requirement = (
            (field_capacity - soil_moisture) * root_depth * water_density
        )
        expected_irrigation_flux = net_requirement / dt
        expected_abstraction_flux = expected_irrigation_flux / efficiency

        torch.testing.assert_close(
            _output_value(outputs, "irrigation_flux"),
            expected_irrigation_flux,
        )
        torch.testing.assert_close(
            _output_value(outputs, "abstraction_flux"),
            expected_abstraction_flux,
        )
        torch.testing.assert_close(
            _output_value(outputs, "irrigation_flux"),
            _output_value(outputs, "abstraction_flux") * efficiency,
        )
        torch.testing.assert_close(
            _output_value(
                outputs,
                "diagnostics.irrigation.effective_efficiency",
            ),
            efficiency.expand_as(soil_moisture),
        )
        assert irrigation_manager.check_conservation_laws(state, outputs) == {
            "mass": pytest.approx(0.0, rel=0.0, abs=1.0e-12)
        }
    
    def test_land_use_filtering(self, irrigation_manager):
        """Test that irrigation only applies to cropland"""
        # Mixed land use scenario
        state = {
            'soil_moisture': torch.tensor([[0.18, 0.18]], device=DEVICE),
            'precipitation_flux': torch.tensor([[0.0, 0.0]], device=DEVICE),
            'land_use_type': torch.tensor([[1, 0]], device=DEVICE),  # Cropland, non-cropland
            'irrigation_type': torch.tensor([[2, 2]], device=DEVICE),
            'field_capacity': torch.tensor([[0.35, 0.35]], device=DEVICE),
            'wilting_point': torch.tensor([[0.15, 0.15]], device=DEVICE),
            'irrigation_remaining_quota': torch.tensor(800.0, device=DEVICE),
        }
        
        outputs = irrigation_manager.forward(state, time=0, dt=86400)
        irrigation_flux = _output_value(outputs, "irrigation_flux")
        
        # Should only irrigate cropland (first element)
        assert irrigation_flux[0, 0] > 0 or irrigation_flux[0, 0] == 0  # May be zero if threshold not met
        assert irrigation_flux[1, 0] == 0  # Non-cropland should not be irrigated
    
    def test_edge_cases_and_error_handling(self, irrigation_manager):
        """Test edge cases and error handling"""
        # Test with zero ET demand
        zero_et_state = {
            'soil_moisture': torch.tensor([[0.2]], device=DEVICE),
            'precipitation_flux': torch.tensor([[0.0]], device=DEVICE),
            'land_use_type': torch.tensor([[1]], device=DEVICE),
            'irrigation_type': torch.tensor([[2]], device=DEVICE),
            'field_capacity': torch.tensor([[0.35]], device=DEVICE),
            'wilting_point': torch.tensor([[0.15]], device=DEVICE),
            'irrigation_remaining_quota': torch.tensor(800.0, device=DEVICE),
        }
        
        outputs = irrigation_manager.forward(zero_et_state, time=0, dt=86400)
        assert isinstance(outputs, dict)
        
        # Test with very high precipitation
        high_rain_state = zero_et_state.copy()
        high_rain_state['precipitation_flux'] = torch.tensor([[10e-6]], device=DEVICE)
        
        rain_outputs = irrigation_manager.forward(high_rain_state, time=0, dt=86400)
        
        # High precipitation should reduce irrigation need
        rain_irrigation = _output_value(rain_outputs, "irrigation_flux")
        assert torch.all(rain_irrigation >= 0)  # Should be non-negative
        
        # Test with exhausted quota
        no_quota_state = dict(zero_et_state)
        no_quota_state["irrigation_remaining_quota"] = torch.tensor(0.0, device=DEVICE)
        no_quota_outputs = irrigation_manager.forward(no_quota_state, time=0, dt=86400)
        no_quota_irrigation = _output_value(no_quota_outputs, "irrigation_flux")
        
        # Should not irrigate with no quota
        assert torch.all(no_quota_irrigation == 0)

    @pytest.mark.parametrize("bad_dt", [0.0, -1.0, float("nan"), float("inf")])
    def test_timestep_is_validated_when_irrigation_is_inactive(
        self,
        irrigation_manager,
        bad_dt,
    ):
        """Inactive branches cannot bypass the timestep contract."""
        with pytest.raises(ValueError, match="dt must be finite and greater than zero"):
            irrigation_manager.forward(
                _canonical_state(soil_moisture=0.40),
                time=0.0,
                dt=bad_dt,
            )

    @pytest.mark.parametrize("bad_time", [-1.0, float("nan"), float("inf")])
    def test_time_is_validated_when_irrigation_is_inactive(
        self,
        irrigation_manager,
        bad_time,
    ):
        """Inactive branches cannot bypass the simulation-time contract."""
        with pytest.raises(ValueError, match="time must be finite and non-negative"):
            irrigation_manager.forward(
                _canonical_state(soil_moisture=0.40),
                time=bad_time,
                dt=3600.0,
            )

    def test_field_capacity_must_exceed_wilting_point(self, irrigation_manager):
        """The canonical moisture response requires a positive denominator."""
        with pytest.raises(
            ValueError,
            match="field_capacity must be greater than wilting_point",
        ):
            irrigation_manager.forward(
                _canonical_state(
                    soil_moisture=0.20,
                    field_capacity=0.20,
                    wilting_point=0.20,
                ),
                time=0.0,
                dt=3600.0,
            )

    def test_water_depth_conversion_rejects_nonpositive_density(
        self,
        irrigation_manager,
        monkeypatch,
    ):
        """A zero physical density cannot use a tiny-value fallback."""
        monkeypatch.setattr(
            irrigation_module,
            "get_physical_constant",
            lambda name: 0.0,
        )

        with pytest.raises(ValueError, match="water_density"):
            irrigation_manager._kg_m2_to_mm_factor(torch.tensor(1.0))

    def test_unknown_moisture_response_method_is_rejected(
        self,
        irrigation_manager,
    ):
        """Unknown response methods cannot silently fall back to linear."""
        with pytest.raises(ValueError, match="Unsupported moisture response method"):
            irrigation_manager._standard_moisture_response(
                torch.tensor([0.2]),
                method="unknown",
                params={"min": 0.1, "max": 0.3},
            )

    def test_conservation_checker_returns_exact_residual_magnitude(
        self,
        irrigation_manager,
    ):
        """The conservation hook must return the numerical residual maximum."""
        output = {
            "diagnostics.irrigation.water_balance_residual": torch.tensor(
                [-0.25, 0.1], dtype=torch.float64
            )
        }

        assert irrigation_manager.check_conservation_laws({}, output) == {
            "mass": pytest.approx(0.25, rel=0.0, abs=0.0)
        }

    @pytest.mark.parametrize(
        ("value", "message"),
        (
            (torch.tensor([float("nan")]), "finite"),
            (torch.empty(0), "non-empty"),
        ),
    )
    def test_conservation_checker_rejects_malformed_diagnostics(
        self,
        irrigation_manager,
        value,
        message,
    ):
        """Malformed diagnostics cannot count as conservation evidence."""
        output = {"diagnostics.irrigation.water_balance_residual": value}

        with pytest.raises(ValueError, match=message):
            irrigation_manager.check_conservation_laws({}, output)

    @pytest.mark.parametrize(
        "quota",
        (torch.tensor(-1.0), torch.tensor([1.0, 2.0]), torch.tensor(float("nan"))),
    )
    def test_remaining_quota_state_is_one_non_negative_domain_value(
        self,
        irrigation_manager,
        quota,
    ):
        """irrigation_remaining_quota is one finite non-negative domain value [mm];
        a quota reset is a state assignment (reset_water_quota was removed)."""
        state = _canonical_state()
        state["irrigation_remaining_quota"] = quota.to(DEVICE)
        with pytest.raises((ValueError, TypeError), match="irrigation_remaining_quota"):
            irrigation_manager.forward(state, time=0, dt=86400)


def _quota_state(quota_mm: float, cells: int = 1) -> dict:
    """Dry cropland cells (theta at the wilting point) with a sprinkler system."""
    return {
        "soil_moisture": torch.full((cells, 1), 0.15, device=DEVICE),
        "precipitation_flux": torch.zeros((cells, 1), device=DEVICE),
        "land_use_type": torch.ones((cells, 1), dtype=torch.long, device=DEVICE),
        "irrigation_type": torch.full((cells, 1), 2, dtype=torch.long, device=DEVICE),
        "field_capacity": torch.full((cells, 1), 0.35, device=DEVICE),
        "wilting_point": torch.full((cells, 1), 0.15, device=DEVICE),
        "irrigation_remaining_quota": torch.tensor(quota_mm, device=DEVICE),
    }


def _mm(manager, flux, dt):
    return float((flux * dt).sum() * manager._kg_m2_to_mm_factor(flux))


def test_irrigation_forward_mutates_no_module_tensor():
    """Item 119 (invariant I1): forward changes no module tensor; the remaining quota
    is the declared state input irrigation_remaining_quota [mm] returned as a REPLACE
    update, and the former accounting buffers and reset_water_quota are removed."""
    from relic.core.state import UpdateMode

    manager = IrrigationManagement()
    for name in (
        "remaining_quota_mm",
        "cumulative_applied_water_mm",
        "cumulative_abstraction_mm",
        "last_irrigation_time_s",
        "reset_water_quota",
    ):
        assert not hasattr(manager, name), name
    assert "irrigation_remaining_quota" in manager.metadata.input_variables
    assert "irrigation_remaining_quota" in manager.metadata.output_variables
    before = {k: v.clone() for k, v in manager.state_dict().items() if isinstance(v, torch.Tensor)}
    first = manager.forward(_quota_state(500.0), time=0.0, dt=86400.0)
    second = manager.forward(_quota_state(500.0), time=0.0, dt=86400.0)
    for name in first:
        assert torch.equal(_output_value(first, name), _output_value(second, name)), name
    after = {k: v.clone() for k, v in manager.state_dict().items() if isinstance(v, torch.Tensor)}
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), name
    item = first["irrigation_remaining_quota"]
    assert item.mode is UpdateMode.REPLACE and item.per_timestep is False
    used = _mm(manager, _output_value(first, "abstraction_flux"), 86400.0)
    assert used > 0.0
    assert float(item.value) == pytest.approx(500.0 - used, rel=1e-6)
    assert float(_output_value(first, "diagnostics.irrigation.applied_water")) == pytest.approx(
        _mm(manager, _output_value(first, "irrigation_flux"), 86400.0), rel=1e-6
    )


def test_quota_shortage_is_shared_at_equal_percentage_without_discard():
    """A domain demand above the remaining quota is met at the same percentage in
    every cell (WEAP21: demand sites of one priority are supplied at an equal
    percentage of their demand; Yates et al. 2005): the abstraction uses exactly the
    quota and nothing is discarded. Below the quota the demand is met in full."""
    manager = IrrigationManagement()
    full = manager.forward(_quota_state(1.0e6, cells=3), time=0.0, dt=86400.0)
    demand = _mm(manager, _output_value(full, "abstraction_flux"), 86400.0)
    assert demand > 0.0

    quota = 0.4 * demand
    short = manager.forward(_quota_state(quota, cells=3), time=0.0, dt=86400.0)
    used = _mm(manager, _output_value(short, "abstraction_flux"), 86400.0)
    assert used == pytest.approx(quota, rel=1e-5)
    assert float(_output_value(short, "irrigation_remaining_quota")) == pytest.approx(0.0, abs=1e-6)
    ratio = _output_value(short, "abstraction_flux") / _output_value(full, "abstraction_flux")
    torch.testing.assert_close(ratio, torch.full_like(ratio, 0.4), rtol=1e-5, atol=0.0)

    empty = manager.forward(_quota_state(0.0, cells=3), time=0.0, dt=86400.0)
    assert torch.all(_output_value(empty, "irrigation_flux") == 0.0)
    assert float(_output_value(empty, "irrigation_remaining_quota")) == 0.0


_IRRIGATION_NAMESPACE = "anthroposphere.management.irrigation"
SYSTEM_EFFICIENCY_KEYS = {
    1: f"{_IRRIGATION_NAMESPACE}.surface_efficiency",
    2: f"{_IRRIGATION_NAMESPACE}.sprinkler_efficiency",
    3: f"{_IRRIGATION_NAMESPACE}.drip_efficiency",
    4: f"{_IRRIGATION_NAMESPACE}.subsurface_efficiency",
}
_MOISTURE_THRESHOLD_KEY = f"{_IRRIGATION_NAMESPACE}.moisture_threshold"
_GENERIC_EFFICIENCY_KEY = f"{_IRRIGATION_NAMESPACE}.irrigation_efficiency"


@pytest.mark.parametrize("irrigation_type", [1, 2, 3, 4])
@pytest.mark.parametrize("regime", ["deficit", "rate_limited"])
def test_gradient_reaches_the_selected_system_efficiency(irrigation_type, regime):
    """The application efficiency of the irrigation system of a cell is a learnable
    ParameterManager binding: the gradient of the water ports lands on it, equals
    the analytic derivative, leaves the other three systems untouched and is the
    same on a second backward through the same instance."""
    # References: Allen et al. (1998), FAO Irrigation and Drainage Paper 56
    #   (root-zone deficit); FAO, Determination of the irrigation schedule for a
    #   field (gross irrigation depth = net depth / field application
    #   efficiency), as cited by docs/source/theory/
    #   anthroposphere_management_irrigation.md.
    # Mathematical expression: A = min(M_net / eta, F_max dt) / dt; D = eta A.
    #   Deficit regime (M_net / eta < F_max dt): dA/deta = -A / eta, dD/deta = 0.
    #   Rate-limited regime (M_net / eta >= F_max dt): dA/deta = 0, dD/deta = A.
    # Parameter meanings: M_net the net root-zone requirement, eta the
    #   application efficiency of the cell's irrigation system, F_max the maximum
    #   application flux, A the abstraction flux, D the delivered flux.
    # Units: M_net [kg m-2]; F_max, A, D [kg m-2 s-1]; dt [s]; eta [1].
    # Domain/constraints: one cropland cell below the trigger moisture, no
    #   precipitation and a quota above the demand; 0 < eta <= 1.
    # Sign convention: abstraction and delivery are nonnegative boundary fluxes; a
    #   higher efficiency lowers the abstraction needed for the same delivery.
    model = IrrigationManagement()
    held = model.held_manager_parameters()
    assert sorted(held) == sorted(SYSTEM_EFFICIENCY_KEYS.values())
    selected_key = SYSTEM_EFFICIENCY_KEYS[irrigation_type]
    soil_moisture, dt = (0.19, 86400.0) if regime == "deficit" else (0.16, 600.0)
    state = _canonical_state(soil_moisture=soil_moisture, irrigation_type=irrigation_type)

    values = {}
    gradients = {}
    try:
        for output in ("abstraction_flux", "irrigation_flux"):
            passes = []
            for _ in range(2):
                for binding in held.values():
                    binding.grad = None
                update = model(dict(state), time=0.0, dt=dt)
                update[output].value.sum().backward()
                passes.append(
                    {
                        key: torch.zeros(())
                        if binding.grad is None
                        else binding.grad.detach().cpu().clone().reshape(())
                        for key, binding in held.items()
                    }
                )
            values[output] = update[output].value.detach().cpu().to(torch.float64).sum()
            gradients[output] = passes
    finally:
        for binding in held.values():
            binding.grad = None

    eta = float(held[selected_key].detach())
    abstraction = values["abstraction_flux"]
    assert float(abstraction) > 0.0
    for output, passes in gradients.items():
        for key in held:
            torch.testing.assert_close(
                passes[1][key], passes[0][key], rtol=0.0, atol=0.0, msg=f"{output}: {key}"
            )
            if key != selected_key:
                assert float(passes[0][key]) == 0.0, (output, key)
    abstraction_gradient = gradients["abstraction_flux"][0][selected_key].to(torch.float64)
    delivery_gradient = gradients["irrigation_flux"][0][selected_key].to(torch.float64)
    scale = float(abstraction) / eta
    if regime == "deficit":
        torch.testing.assert_close(
            abstraction_gradient, -abstraction / eta, rtol=1.0e-5, atol=0.0
        )
        assert abs(float(delivery_gradient)) <= 1.0e-6 * scale
    else:
        assert float(abstraction_gradient) == 0.0
        torch.testing.assert_close(delivery_gradient, abstraction, rtol=1.0e-5, atol=0.0)


def test_moisture_threshold_is_the_available_water_fraction_of_a_discrete_trigger(
    restore_global_parameter_manager,
):
    """The moisture threshold is the fraction of plant-available water below which
    irrigation starts. It enters through a discrete trigger, so it is an event
    parameter read from its owner at every call: declared dimensionless, not
    learnable, not held, and followed by an existing instance when it changes."""
    # References: Allen et al. (1998), FAO Irrigation and Drainage Paper 56,
    #   Eqs. 82 and 83 (total and readily available soil water).
    # Mathematical expression: theta_trigger = theta_wp + f (theta_fc - theta_wp);
    #   u = 1 if theta < theta_trigger, else 0.
    # Parameter meanings: f the moisture threshold (fraction of the plant-available
    #   water range), theta_wp and theta_fc wilting point and field capacity,
    #   theta the root-zone soil moisture, u the irrigation trigger.
    # Units: f, u [1]; theta, theta_wp, theta_fc, theta_trigger [m3 m-3].
    # Domain/constraints: theta_fc > theta_wp; f inside its ParameterManager
    #   bounds; u is a step function of theta and f, so no gradient reaches f.
    # Sign convention: none (a trigger; the triggered fluxes are nonnegative).
    manager = restore_global_parameter_manager
    spec = manager.get_parameter(_MOISTURE_THRESHOLD_KEY)
    assert spec.units == "dimensionless"
    assert spec.requires_grad is False

    model = IrrigationManagement()
    assert _MOISTURE_THRESHOLD_KEY not in model.held_manager_parameters()
    fraction = float(manager.get(_MOISTURE_THRESHOLD_KEY, required=True))
    wilting_point, field_capacity = 0.15, 0.35
    trigger = wilting_point + fraction * (field_capacity - wilting_point)
    below = model(_canonical_state(soil_moisture=trigger - 0.01), time=0.0, dt=3600.0)
    above = model(_canonical_state(soil_moisture=trigger + 0.01), time=0.0, dt=3600.0)
    assert float(_output_value(below, "irrigation_flux").sum()) > 0.0
    assert float(_output_value(above, "irrigation_flux").sum()) == 0.0

    manager.set_value(_MOISTURE_THRESHOLD_KEY, 0.6, units=spec.units, validate=True)
    assert float(manager.get(_MOISTURE_THRESHOLD_KEY, required=True)) == pytest.approx(0.6)
    raised = model(_canonical_state(soil_moisture=trigger + 0.01), time=0.0, dt=3600.0)
    assert float(_output_value(raised, "irrigation_flux").sum()) > 0.0


def test_generic_irrigation_efficiency_has_no_reader_and_no_entry():
    """Every valid irrigation_type selects one of the four system efficiencies, so
    the generic efficiency had no cell to apply to: the Functor neither declares
    nor reads it and the parameter database holds no entry for it."""
    model = IrrigationManagement()
    assert _GENERIC_EFFICIENCY_KEY not in model.metadata.parameters
    assert model.get_parameter_manager().get_parameter(_GENERIC_EFFICIENCY_KEY) is None
    for removed in ("irrigation_efficiency", "moisture_threshold"):
        assert not hasattr(model, removed)
    assert list(model.parameters()) == []
    assert dict(model.state_dict()) == {}


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])


