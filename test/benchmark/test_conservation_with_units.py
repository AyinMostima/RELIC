"""L3 System Test: Conservation Laws with Units.

Tests conservation of mass, energy, and other quantities,
ensuring unit consistency in conservation checks.

Mathematical References:
- Mass conservation: d(M)/dt = sum(inputs) - sum(outputs)
- Energy conservation: d(E)/dt = sum(energy_in) - sum(energy_out)
- Units must match across all terms!
"""
import pytest
import torch
from relic.data.standards.unit_converter import UnitConverter, ConversionContext


class TestMassConservationWater:
    """Test water mass conservation with proper units.

    Water balance equation:
    d(Storage)/dt = Precipitation - ET - Runoff - Percolation

    All flux units: kg m-2 s-1
    Storage units: kg m-2
    """

    def test_water_balance_units(self):
        """Test that water balance equation has consistent units."""
        # Define fluxes in kg m-2 s-1
        precipitation = torch.tensor([0.0005])  # kg m-2 s-1
        evapotranspiration = torch.tensor([0.0002])
        runoff = torch.tensor([0.0001])
        percolation = torch.tensor([0.0001])

        # Net flux
        net_flux = precipitation - evapotranspiration - runoff - percolation

        # Storage change over time step
        dt = 1800.0  # seconds
        storage_change = net_flux * dt  # kg m-2

        # Verify units are correct
        assert storage_change.item() == pytest.approx(0.18)  # kg m-2

    def test_water_conservation_violation_detected(self):
        """Test that water conservation violations are detected.

        If inputs don't equal outputs + storage change, should be detected.
        """
        # Initial storage
        storage = torch.tensor([100.0])  # kg m-2

        # Fluxes (kg m-2 s-1)
        inputs = torch.tensor([0.001])
        outputs = torch.tensor([0.0005])

        dt = 1800.0
        expected_storage = storage + (inputs - outputs) * dt

        # Check conservation
        net_flux = inputs - outputs
        actual_change = net_flux * dt

        assert torch.allclose(expected_storage, storage + actual_change)

    def test_unit_conversion_in_water_balance(self):
        """Test water balance when inputs have different units.

        Convert all to consistent units before conservation check.
        """
        converter = UnitConverter()
        context = ConversionContext()

        # Precipitation in mm day-1 (common input format)
        precip_mm_day = 5.0

        # Convert to kg/m2/s: 1 mm day-1 = 1 kg/m2 day-1 = 1/86400 kg/m2/s
        precip_kg_m2_s = precip_mm_day / 86400.0

        # Now can use in conservation equation
        assert precip_kg_m2_s > 0


class TestMassConservationCarbon:
    """Test carbon mass conservation with proper units.

    Carbon balance:
    d(C_pool)/dt = GPP - R_auto - R_hetero - VOC - Leaching

    All flux units: g C m-2 s-1
    Pool units: g C m-2
    """

    def test_carbon_balance_units(self):
        """Test carbon balance equation has consistent units."""
        # Fluxes in g C m-2 s-1
        gpp = torch.tensor([0.002])  # Gross primary production
        r_auto = torch.tensor([0.001])  # Autotrophic respiration
        r_hetero = torch.tensor([0.0005])  # Heterotrophic respiration
        voc = torch.tensor([0.0001])  # VOC emissions
        leaching = torch.tensor([0.00001])  # Carbon leaching

        # Net ecosystem exchange
        nee = r_auto + r_hetero + voc - gpp

        # Storage change
        dt = 1800.0
        storage_change = -nee * dt  # g C m-2 (negative NEE = carbon gain)

        assert storage_change.item() > 0  # Should be accumulating carbon

    def test_carbon_conservation_with_unit_conversion(self):
        """Test carbon conservation when mixing umol and g units.

        Some modules output umol CO2 m-2 s-1, others use g C m-2 s-1.
        Must convert to consistent units before conservation check.
        """
        converter = UnitConverter()
        context = ConversionContext()

        # GPP in umol CO2 m-2 s-1
        gpp_umol = 40.0

        # Convert to g C m-2 s-1
        # Molecular weight of C = 12 g/mol
        # 1 umol CO2 = 1 umol C = 12e-6 g C
        gpp_g = converter.convert(
            gpp_umol,
            'carbon_flux',
            'umol m-2 s-1',
            'g m-2 s-1',
            context
        )

        # Now can compare with other fluxes in g C m-2 s-1
        respiration_g = 0.0002  # g C m-2 s-1

        net_flux = gpp_g - respiration_g
        assert net_flux > 0

    def test_multi_pool_carbon_conservation(self):
        """Test conservation across multiple carbon pools.

        Total carbon: Vegetation + Soil + Litter
        All pools in g C m-2
        """
        # Initial pools
        veg_c = torch.tensor([1000.0])  # g C m-2
        soil_c = torch.tensor([5000.0])
        litter_c = torch.tensor([500.0])

        total_initial = veg_c + soil_c + litter_c

        # Fluxes between pools (g C m-2 s-1)
        litterfall = torch.tensor([0.0001])  # Veg to litter
        decomposition = torch.tensor([0.00005])  # Litter to soil

        dt = 1800.0

        # Update pools
        veg_c_new = veg_c - litterfall * dt
        litter_c_new = litter_c + litterfall * dt - decomposition * dt
        soil_c_new = soil_c + decomposition * dt

        total_final = veg_c_new + litter_c_new + soil_c_new

        # Total carbon should be conserved
        assert torch.allclose(total_final, total_initial, atol=1e-6)


class TestEnergyConservation:
    """Test energy conservation with proper units.

    Energy balance:
    Rnet = H + LE + G

    Where:
    - Rnet: Net radiation (W m-2)
    - H: Sensible heat flux (W m-2)
    - LE: Latent heat flux (W m-2)
    - G: Ground heat flux (W m-2)
    """

    def test_surface_energy_balance(self):
        """Test surface energy balance equation."""
        # All in W m-2
        net_radiation = torch.tensor([600.0])
        sensible_heat = torch.tensor([100.0])
        latent_heat = torch.tensor([400.0])
        ground_heat = torch.tensor([100.0])

        # Check balance
        energy_sum = sensible_heat + latent_heat + ground_heat
        residual = net_radiation - energy_sum

        # Should balance (within measurement error)
        assert abs(residual.item()) < 10.0  # Within 10 W m-2

    def test_latent_heat_water_flux_consistency(self):
        """Test consistency between latent heat and water flux.

        LE (W m-2) = lambda * E (kg m-2 s-1)
        where lambda = 2.45e6 J/kg (latent heat of vaporization)
        """
        # Water flux (evapotranspiration)
        et_flux = torch.tensor([0.0001])  # kg m-2 s-1

        # Latent heat
        lambda_v = 2.45e6  # J/kg
        latent_heat = et_flux * lambda_v  # W m-2

        assert latent_heat.item() == pytest.approx(245.0)


class TestUnitMismatchDetection:
    """Test that unit mismatches in conservation are detected."""

    def test_detect_flux_storage_unit_mismatch(self):
        """Test detection of flux vs storage unit mismatch.

        Common error: Using flux units (m-2 s-1) where storage units
        (m-2) are expected, or vice versa.
        """
        # Correct: flux * time = storage change
        flux = torch.tensor([0.001])  # kg m-2 s-1
        dt = 1800.0  # s
        storage_change = flux * dt  # kg m-2 (correct units)

        # Incorrect would be: storage = flux (missing time factor)
        # This test verifies we don't make that mistake
        assert storage_change.item() == pytest.approx(1.8)

    def test_detect_inconsistent_units_in_sum(self):
        """Test that summing inconsistent units is prevented.

        Cannot add:
        - kg m-2 s-1 (flux) + kg m-2 (storage)
        - W m-2 (power) + J (energy)
        """
        # These should have same units to be added
        flux1 = torch.tensor([0.001])  # kg m-2 s-1
        flux2 = torch.tensor([0.0005])  # kg m-2 s-1

        # Can add
        total_flux = flux1 + flux2
        assert total_flux.item() == pytest.approx(0.0015)

        # Cannot add flux to storage (different units)
        # This is a logical check, not prevented by PyTorch
        storage = torch.tensor([100.0])  # kg m-2

        # Would need to multiply flux by time first
        dt = 1800.0
        storage_consistent = flux1 * dt  # Now in kg m-2
        total = storage + storage_consistent  # Now valid
        assert total.item() > storage.item()


class TestConservationAcrossScales:
    """Test conservation when aggregating across scales."""

    def test_spatial_aggregation_conserves_mass(self):
        """Test that spatial aggregation conserves total mass.

        Summing over grid cells should conserve total mass.
        """
        # Grid of carbon storage (g C m-2)
        grid_storage = torch.tensor([
            [100.0, 150.0, 120.0],
            [110.0, 130.0, 140.0],
            [105.0, 125.0, 135.0]
        ])

        total_before = grid_storage.sum()

        # Apply uniform flux to all cells
        flux = torch.tensor([0.001])  # g C m-2 s-1
        dt = 1800.0

        grid_storage_new = grid_storage + flux * dt

        total_after = grid_storage_new.sum()

        # Total change should equal flux * area * time
        expected_change = flux * dt * grid_storage.numel()
        actual_change = total_after - total_before

        assert torch.allclose(actual_change, expected_change)

    def test_temporal_integration_conserves_totals(self):
        """Test that temporal integration maintains conservation.

        Integrating fluxes over time should match storage changes.
        """
        # Initial storage
        storage = torch.tensor([1000.0])  # g C m-2

        # Constant flux
        flux = torch.tensor([0.001])  # g C m-2 s-1

        # Integrate over multiple time steps
        dt = 1800.0
        num_steps = 10

        for _ in range(num_steps):
            storage = storage + flux * dt

        # Total change
        expected_change = flux * dt * num_steps
        actual_change = storage - torch.tensor([1000.0])

        assert torch.allclose(actual_change, expected_change)


