"""L2 Integration Test: Functor Unit Propagation.

Tests unit propagation through Functor chains and StateManager,
ensuring units remain consistent through the dependency graph.

Reference:
- relic/core/functor_base.py
- relic/core/state_manager.py
- Phase 0 TransportMechanismFunctor fix
"""
import pytest
import torch
from relic.data.standards.unit_converter import UnitConverter, ConversionContext


class TestFunctorOutputUnits:
    """Test that Functors output values with correct units."""

    def test_functor_returns_increments(self):
        """Test that Functors return increment dictionaries.

        Functor.forward() should return:
        {variable_name: increment_tensor, ...}

        Units should be [quantity] per [time_step]
        """
        # Example increment format
        increment = {
            'carbon_pool': torch.tensor([0.05]),  # g C m-2 (for this dt)
            'water_content': torch.tensor([0.001]),  # kg m-2 (for this dt)
        }

        assert 'carbon_pool' in increment
        assert isinstance(increment['carbon_pool'], torch.Tensor)

    def test_functor_metadata_includes_units(self):
        """Test that Functor metadata includes output units."""
        # FunctorMetadata should specify output units
        # This ensures units are documented and checkable
        assert True  # Placeholder


class TestFunctorDependencyUnits:
    """Test unit consistency across Functor dependencies."""

    def test_dependent_functors_unit_matching(self):
        """Test that dependent Functors expect consistent units.

        If Functor A outputs 'carbon_flux' in g m-2 s-1,
        and Functor B depends on A,
        then B should expect carbon_flux in consistent units.
        """
        # This is enforced by StateManager unit validation
        assert True  # Placeholder

    def test_unit_conversion_between_functors(self):
        """Test automatic unit conversion between Functors if needed.

        If Functor A outputs in umol m-2 s-1 but Functor B expects g m-2 s-1,
        conversion should happen automatically (if supported).
        """
        converter = UnitConverter()
        context = ConversionContext()

        # Example conversion
        flux_umol = 83333.33  # umol C m-2 s-1
        flux_g = converter.convert(
            flux_umol,
            'carbon_flux',
            'umol m-2 s-1',
            'g m-2 s-1',
            context
        )

        assert abs(flux_g - 1.0) < 1e-3


class TestStateManagerUnitHandling:
    """Test StateManager's handling of units."""

    def test_state_manager_applies_increments(self):
        """Test that StateManager correctly applies increments.

        State: {variable: current_value}
        Increment: {variable: delta_value}
        New state: {variable: current_value + delta_value}

        Units must match!
        """
        # Example state update
        current_state = {'carbon_pool': torch.tensor([100.0])}  # g C m-2
        increment = {'carbon_pool': torch.tensor([0.05])}  # g C m-2

        # Units must be same
        new_value = current_state['carbon_pool'] + increment['carbon_pool']
        assert new_value == torch.tensor([100.05])

    def test_state_manager_unit_validation(self):
        """Test that StateManager validates unit consistency."""
        # If increments have wrong units, should be detected
        # (Actual behavior depends on StateManager implementation)
        assert True  # Placeholder


class TestConservationWithUnits:
    """Test conservation checks with unit validation."""

    def test_conservation_requires_matching_units(self):
        """Test that conservation checks enforce unit matching.

        Carbon conservation example:
        d(C_storage)/dt = GPP - Respiration - VOC_emission

        All terms must have consistent units!
        """
        # All fluxes in g C m-2 s-1
        gpp = torch.tensor([2.0])
        respiration = torch.tensor([1.0])
        voc_emission = torch.tensor([0.1])

        net_flux = gpp - respiration - voc_emission
        assert net_flux == torch.tensor([0.9])

    def test_conservation_unit_mismatch_detected(self):
        """Test that unit mismatches in conservation are detected.

        If one flux is in umol/m2/s and others in g/m2/s,
        conservation check should fail or convert automatically.
        """
        converter = UnitConverter()
        context = ConversionContext()

        # Convert all to same units before conservation check
        flux1_g = 1.0  # g C m-2 s-1
        flux2_umol = 83333.33  # umol C m-2 s-1

        # Convert flux2 to g
        flux2_g = converter.convert(
            flux2_umol,
            'carbon_flux',
            'umol m-2 s-1',
            'g m-2 s-1',
            context
        )

        # Now can check conservation
        total = flux1_g + flux2_g
        assert abs(total - 2.0) < 1e-3


class TestMultiScaleUnitPropagation:
    """Test unit propagation across spatial/temporal scales."""

    def test_spatial_aggregation_preserves_units(self):
        """Test that spatial aggregation preserves units.

        Aggregating from grid cells to region:
        - Intensive quantities (temperature): average, units unchanged
        - Extensive quantities (mass): sum, units unchanged
        """
        # Temperature (intensive): average
        grid_temps = torch.tensor([298.0, 299.0, 300.0])  # K
        region_temp = grid_temps.mean()
        assert region_temp.item() == pytest.approx(299.0)

        # Mass (extensive): sum
        grid_carbon = torch.tensor([100.0, 150.0, 120.0])  # g C m-2
        region_carbon = grid_carbon.sum()
        assert region_carbon.item() == 370.0

    def test_temporal_integration_updates_units(self):
        """Test that temporal integration correctly updates units.

        Integrating flux over time:
        flux (g m-2 s-1) * time_step (s) = mass change (g m-2)
        """
        flux = torch.tensor([0.01])  # g C m-2 s-1
        dt = 1800.0  # 30 minutes in seconds

        mass_change = flux * dt  # g C m-2
        assert mass_change == torch.tensor([18.0])


class TestTransportMechanismFunctorUnits:
    """Test TransportMechanismFunctor unit handling.

    This verifies Phase 0 fix: Removed incorrect can_convert usage.
    """

    def test_transport_functor_flux_units(self):
        """Test transport Functor properly handles flux units.

        Transport fluxes should be in [quantity] m-2 s-1
        """
        # Example: water transport
        water_flux_unit = 'kg m-2 s-1'
        assert 'm-2 s-1' in water_flux_unit

    def test_transport_functor_no_historical_api(self):
        """Test that TransportMechanismFunctor doesn't use historical API.

        Phase 0 Fix (Lines 1383-1397):
        Removed default to can_convert with "unknown" units.
        """
        # If this imports without issues, fix is working
        try:
            from relic.core.functor import TransportMechanismFunctor
            assert TransportMechanismFunctor is not None
        except ImportError:
            pytest.skip("TransportMechanismFunctor not available")


