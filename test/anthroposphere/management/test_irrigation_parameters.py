"""
Test suite for irrigation.py parameter management system
Tests L1-L3: Import, Parameter Loading, Functional Logic

Stage 4.2.1c - irrigation.py parameter system validation
"""

import pytest
import torch
import sys
import inspect
from pathlib import Path

from relic.core.state import UpdateItem, UpdateMode

# Add relic to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "relic"))


def _canonical_state(
    *,
    soil_moisture: float = 0.20,
    precipitation_flux: float = 0.0,
    irrigation_type: int = 1,
) -> dict[str, torch.Tensor]:
    """Build one complete canonical irrigation state."""
    return {
        "soil_moisture": torch.tensor([[soil_moisture]], dtype=torch.float32),
        "precipitation_flux": torch.tensor(
            [[precipitation_flux]],
            dtype=torch.float32,
        ),
        "land_use_type": torch.tensor([[1]], dtype=torch.long),
        "irrigation_type": torch.tensor([[irrigation_type]], dtype=torch.long),
        "field_capacity": torch.tensor([[0.35]], dtype=torch.float32),
        "wilting_point": torch.tensor([[0.15]], dtype=torch.float32),
        # Remaining source-water quota [mm], owned state (item 119).
        "irrigation_remaining_quota": torch.tensor(800.0, dtype=torch.float32),
    }


# ============================================================================
# L1 Tests: Import Validation
# ============================================================================

class TestL1Import:
    """L1: Verify module imports without errors"""

    def test_irrigation_module_import(self):
        """Test that irrigation module can be imported"""
        try:
            from relic.scientific.anthroposphere.management import irrigation
            assert irrigation is not None, "Irrigation module import failed"
        except ImportError as e:
            pytest.fail(f"Failed to import irrigation module: {e}")

    def test_irrigation_functor_import(self):
        """Test that IrrigationManagement can be imported and instantiated"""
        try:
            from relic.scientific.anthroposphere.management.irrigation import IrrigationManagement
            from relic.parameters import get_global_manager

            # Initialize parameter manager first
            pm = get_global_manager()
            assert pm is not None, "Parameter manager initialization failed"

            # Create minimal config for instantiation
            config = {
                "device": "cpu",
                "dtype": "float32"
            }

            functor = IrrigationManagement(config=config)
            assert functor is not None, "IrrigationManagement instantiation failed"

        except Exception as e:
            pytest.fail(f"Failed to import/instantiate IrrigationManagement: {e}")


# ============================================================================
# L2 Tests: Parameter Loading Validation
# ============================================================================

class TestL2ParameterLoading:
    """L2: Verify all parameters load from YAML with correct metadata"""

    @pytest.fixture
    def irrigation_functor(self):
        """Create IrrigationManagement instance for testing"""
        from relic.scientific.anthroposphere.management.irrigation import IrrigationManagement

        config = {
            "device": "cpu",
            "dtype": "float32"
        }

        return IrrigationManagement(config=config)

    def test_parameter_manager_availability(self, irrigation_functor):
        """Test that parameter manager is accessible"""
        from relic.parameters import get_global_manager

        pm = get_global_manager()
        assert pm is not None, "Parameter manager not available"

    def test_generic_irrigation_efficiency_is_retired(self, irrigation_functor):
        """Every valid irrigation_type selects a system efficiency, so the generic
        efficiency had no reader; it is no parameter of the process any more."""
        assert irrigation_functor.get_parameter_value(
            "irrigation_efficiency",
            namespace="anthroposphere.management.irrigation",
        ) is None
        for name in (
            "surface_efficiency",
            "sprinkler_efficiency",
            "drip_efficiency",
            "subsurface_efficiency",
        ):
            param_value = irrigation_functor.get_parameter_value(
                name,
                namespace="anthroposphere.management.irrigation",
                units="dimensionless",
                validate=True,
                required=True,
            )
            assert 0.3 <= param_value <= 1.0, f"{name} {param_value} outside (0.3, 1.0]"

    def test_moisture_threshold_parameter(self, irrigation_functor):
        """Test moisture threshold parameter loads correctly"""
        param_value = irrigation_functor.get_parameter_value(
            "moisture_threshold",
            namespace="anthroposphere.management.irrigation",
            units="dimensionless",
            bounds=(0.15, 0.4),
            validate=True,
            required=True,
        )

        assert param_value is not None, "moisture_threshold not loaded"
        assert 0.15 <= param_value <= 0.4, f"moisture_threshold {param_value} outside bounds [0.15, 0.4]"

    def test_field_capacity_is_a_required_runtime_input(self, irrigation_functor):
        """Keep field capacity in StateManager rather than a hidden default."""
        assert "field_capacity" in irrigation_functor.metadata.input_variables
        forward_source = inspect.getsource(irrigation_functor.forward)
        assert "field_capacity_default" not in forward_source

    def test_wilting_point_is_a_required_runtime_input(self, irrigation_functor):
        """Keep wilting point in StateManager rather than a hidden default."""
        assert "wilting_point" in irrigation_functor.metadata.input_variables
        forward_source = inspect.getsource(irrigation_functor.forward)
        assert "wilting_point_default" not in forward_source

    def test_water_quota_is_state_not_a_parameter(self, irrigation_functor):
        """The remaining quota is the declared state irrigation_remaining_quota [mm]
        (initial value from the scenario); annual_water_quota is no parameter."""
        assert "irrigation_remaining_quota" in irrigation_functor.metadata.input_variables
        assert irrigation_functor.get_parameter_value(
            "annual_water_quota",
            namespace="anthroposphere.management.irrigation",
        ) is None

    def test_root_zone_depth_parameter(self, irrigation_functor):
        """Test the canonical root-zone depth parameter."""
        param_value = irrigation_functor.get_parameter_value(
            "root_zone_depth",
            namespace="anthroposphere.management.irrigation",
            units="m",
            bounds=(0.1, 3.0),
            validate=True,
            required=True,
        )

        assert param_value is not None, "root_zone_depth not loaded"
        assert 0.1 <= param_value <= 3.0

    def test_conservation_tolerance_parameter(self, irrigation_functor):
        """Test the explicit water-ledger tolerance parameter."""
        param_value = irrigation_functor.get_parameter_value(
            "conservation_tolerance",
            namespace="anthroposphere.management.irrigation",
            units="kg m-2 s-1",
            bounds=(0.0, 1.0),
            validate=True,
            required=True,
        )

        assert param_value is not None, "conservation_tolerance not loaded"
        assert 0.0 <= param_value <= 1.0

    def test_physical_water_density_is_not_a_domain_parameter(
        self,
        irrigation_functor,
    ):
        """Keep physical water density outside the irrigation namespace."""
        manager = irrigation_functor.get_parameter_manager()
        assert not manager.has_parameter(
            "anthroposphere.management.irrigation.water_density"
        )
        source = inspect.getsource(type(irrigation_functor))
        assert "get_physical_constant(\"water_density\")" in source
        assert "relic.physical.constants" not in source

    def test_subsurface_efficiency_parameter(self, irrigation_functor):
        """Test the fourth canonical irrigation-system efficiency."""
        param_value = irrigation_functor.get_parameter_value(
            "subsurface_efficiency",
            namespace="anthroposphere.management.irrigation",
            units="dimensionless",
            bounds=(0.3, 0.95),
            validate=True,
            required=True,
        )

        assert param_value is not None, "subsurface_efficiency not loaded"
        assert 0.3 <= param_value <= 0.95

    def test_max_application_rate_parameter(self, irrigation_functor):
        """Test max application rate parameter (newly added)"""
        param_value = irrigation_functor.get_parameter_value(
            "max_application_rate",
            namespace="anthroposphere.management.irrigation",
            units="mm hr-1",
            bounds=(1.0, 50.0),
            validate=True,
            required=True,
        )

        assert param_value is not None, "max_application_rate not loaded"
        assert 1.0 <= param_value <= 50.0, f"max_application_rate {param_value} outside bounds [1.0, 50.0]"

    def test_irrigation_type_efficiencies(self, irrigation_functor):
        """Test all three irrigation type efficiency parameters"""
        types = ["surface", "sprinkler", "drip", "subsurface"]

        for irrigation_type in types:
            param_value = irrigation_functor.get_parameter_value(
                f"{irrigation_type}_efficiency",
                namespace="anthroposphere.management.irrigation",
                units="dimensionless",
                bounds=(0.3, 0.95),
                validate=True,
                required=True,
            )

            assert param_value is not None, f"{irrigation_type}_efficiency not loaded"
            assert 0.3 <= param_value <= 0.95, \
                f"{irrigation_type}_efficiency {param_value} outside bounds [0.3, 0.95]"

    def test_default_values_preloaded(self, irrigation_functor):
        """Required runtime inputs have no cached ParameterManager defaults."""
        for name in (
            "_soil_moisture_default",
            "_et_demand_default",
            "_field_capacity_default",
            "_wilting_point_default",
        ):
            assert not hasattr(irrigation_functor, name)
        forward_source = inspect.getsource(irrigation_functor.forward)
        for stale_key in (
            "soil_moisture_default",
            "et_demand_default",
            "precipitation_default",
            "crop_stage_default",
        ):
            assert stale_key not in forward_source


# ============================================================================
# L3 Tests: Functional Logic Validation
# ============================================================================

class TestL3IrrigationFunctionalLogic:
    """L3: Verify basic irrigation scheduling logic"""

    @pytest.fixture
    def irrigation_functor(self):
        """Create IrrigationManagement instance for testing"""
        from relic.scientific.anthroposphere.management.irrigation import IrrigationManagement

        config = {
            "device": "cpu",
            "dtype": "float32"
        }

        return IrrigationManagement(config=config)

    def test_forward_execution_no_errors(self, irrigation_functor):
        """Test that forward() executes without errors with minimal state"""
        state = _canonical_state(soil_moisture=0.25)

        time = 0.0  # scalar time
        dt = 3600.0  # 1 hour

        try:
            updates = irrigation_functor.forward(state, time, dt)
            assert isinstance(updates, dict), "forward() should return a dict"

        except Exception as e:
            pytest.fail(f"forward() execution failed: {e}")

    def test_irrigation_triggered_when_dry(self, irrigation_functor):
        """Test that irrigation is triggered when soil is dry"""
        state = _canonical_state(soil_moisture=0.20)

        time = 0.0
        dt = 3600.0

        updates = irrigation_functor.forward(state, time, dt)

        # Check that irrigation flux is positive
        if "irrigation_flux" in updates:
            irrigation_flux = updates["irrigation_flux"].value
            assert irrigation_flux.sum() > 0, "Irrigation should be triggered when soil is dry"

    def test_no_irrigation_when_wet(self, irrigation_functor):
        """Test that irrigation is NOT triggered when soil is already wet"""
        state = _canonical_state(soil_moisture=0.40)

        time = 0.0
        dt = 3600.0

        updates = irrigation_functor.forward(state, time, dt)

        # Irrigation should be minimal or zero when soil is already wet
        if "irrigation_flux" in updates:
            irrigation_flux = updates["irrigation_flux"].value
            # Allow small numerical values, but should be close to zero
            assert irrigation_flux.sum() < 1e-6, "Irrigation should not occur when soil is wet"

    def test_water_balance_conservation(self, irrigation_functor):
        """Test that water balance is conserved (inputs = outputs)"""
        state = _canonical_state(
            soil_moisture=0.20,
            precipitation_flux=1e-5,
        )

        time = 0.0
        dt = 3600.0

        updates = irrigation_functor.forward(state, time, dt)

        conservation_error = updates[
            "diagnostics.irrigation.water_balance_residual"
        ].value
        assert abs(conservation_error.item()) < 1e-3, \
            f"Water balance conservation violated: error = {conservation_error.item()}"

    def test_missing_required_state_fails_fast(self, irrigation_functor):
        """Do not replace a missing canonical scientific input with a parameter."""
        with pytest.raises(KeyError, match="soil_moisture"):
            irrigation_functor.forward(
                {"irrigation_type": torch.tensor([[1]], dtype=torch.long)},
                time=0.0,
                dt=3600.0,
            )

    def test_output_units_consistency(self, irrigation_functor):
        """Test that output fluxes have consistent units"""
        state = _canonical_state(soil_moisture=0.20)

        time = 0.0
        dt = 3600.0

        updates = irrigation_functor.forward(state, time, dt)

        # All outputs carry explicit units and source-writer semantics.
        for key, item in updates.items():
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.source_functor == irrigation_functor.name
            if "flux" in key.lower():
                assert item.unit == "kg m-2 s-1"
                assert isinstance(item.value, torch.Tensor), \
                    f"{key} should carry a torch.Tensor, got {type(item.value)}"

                # Flux values should be finite
                assert torch.isfinite(item.value).all(), \
                    f"{key} contains non-finite values"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])

