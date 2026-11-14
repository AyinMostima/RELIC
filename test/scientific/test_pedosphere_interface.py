"""
Test Suite for Pedosphere Interface Mapping

This module tests the PedosphereInterfaceMapper class to verify:

1. Proper inheritance from ScaleCouplingFunctor
2. Correct mapping between hydrosphere and pedosphere variables
3. Unit conversions and species splitting
4. Mass balance checking functionality
5. Forward method returns increments only

References:
    - Dentener et al. (2006) Nitrogen and sulfur deposition
    - Mahowald et al. (2008) Atmospheric phosphorus
"""

import pytest
import torch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from relic.scientific.pedosphere.interface_mapping import (
    PedosphereInterfaceMapper,
    get_pedosphere_interface_mapper,
    get_leaching_flux,
    calculate_doc_leaching,
    calculate_nitrate_leaching
)
from relic.core.functor import ScaleCouplingFunctor
from relic.core.abstractions import ScaleCoupling
from relic.core.functor import FunctorBase
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant


class TestPedosphereInterfaceMapper:
    """Test PedosphereInterfaceMapper class."""

    def test_inheritance(self):
        """Verify proper inheritance from ScaleCouplingFunctor."""
        assert issubclass(PedosphereInterfaceMapper, ScaleCouplingFunctor)
        assert issubclass(PedosphereInterfaceMapper, FunctorBase)
        # ScaleCoupling should be in the MRO through ScaleCouplingFunctor
        assert issubclass(PedosphereInterfaceMapper, ScaleCoupling)

    def test_initialization(self):
        """Test initialization of PedosphereInterfaceMapper."""
        mapper = PedosphereInterfaceMapper(name="test_mapper")

        assert mapper.name == "test_mapper"
        assert mapper.source_scale == "hydrosphere"
        assert mapper.target_scale == "pedosphere"
        assert mapper.coupling_type == "spatial"
        assert hasattr(mapper, 'metadata')
        assert mapper.parameter_manager is not None
        assert "seconds_per_year" not in mapper.metadata.parameters
        assert "density_liquid" not in mapper.metadata.parameters
        assert mapper.seconds_per_year == float(
            get_physical_constant(
                "relic.physical.constants.time.seconds_per_year"
            )
        )
        assert mapper.water_density == float(
            get_physical_constant(
                "relic.physical.constants.water.density_liquid"
            )
        )

    def test_get_dependencies(self):
        """Test that get_dependencies returns empty list."""
        mapper = PedosphereInterfaceMapper(
            upscale_bridge=lambda value: value.mean(dim=-1),
            downscale_bridge=lambda value: value.repeat_interleave(2),
        )
        deps = mapper.get_dependencies()
        assert isinstance(deps, list)
        assert deps == []

    def test_retired_hydrosphere_name_table_is_removed(self):
        """Hydrosphere water fluxes enter the pedosphere only through forward
        (y_h = C_h x_h, C_h = I_3; see test_forward_returns_exact_canonical_flux_ports).
        The uncalled name-table remapping, its zero-flux fallback and its retired
        solute names (for example nitrate_concentration) are removed."""
        for retired in (
            "map_hydrosphere_to_pedosphere",
            "WATER_INPUT_MAPPING",
            "SOLUTE_TRANSPORT_MAPPING",
        ):
            assert not hasattr(PedosphereInterfaceMapper, retired), retired

    def test_retired_statemanager_output_table_is_removed(self):
        """Pedosphere Functors emit canonical StateManager names directly; the
        uncalled output renaming table, its removed-alias guard and
        map_pedosphere_to_statemanager are removed."""
        for retired in (
            "map_pedosphere_to_statemanager",
            "PEDOSPHERE_OUTPUT_MAPPING",
            "_REMOVED_METHANE_OUTPUT_ALIASES",
        ):
            assert not hasattr(PedosphereInterfaceMapper, retired), retired

    def test_map_atmosphere_to_pedosphere(self):
        """Test atmospheric deposition mapping and species splitting."""
        mapper = PedosphereInterfaceMapper()

        # Annual deposition values
        state = {
            "wet_n_deposition": torch.tensor(10.0),  # g N m^-2 year^-1
            "dry_n_deposition": torch.tensor(5.0),
            "wet_p_deposition": torch.tensor(1.0),   # g P m^-2 year^-1
        }

        mapped = mapper.map_atmosphere_to_pedosphere(state)

        # Check N species splitting (default 60% NOy, 40% NHx)
        year_to_second = 1.0 / (365.25 * 24 * 3600)

        if "wet_noy_deposition" in mapped:
            expected_wet_noy = 10.0 * year_to_second * 0.6
            assert torch.allclose(mapped["wet_noy_deposition"],
                                torch.tensor(expected_wet_noy), rtol=1e-5)

        if "wet_nhx_deposition" in mapped:
            expected_wet_nhx = 10.0 * year_to_second * 0.4
            assert torch.allclose(mapped["wet_nhx_deposition"],
                                torch.tensor(expected_wet_nhx), rtol=1e-5)

        # Check P species splitting (default 30% PO4, 70% dust)
        if "wet_po4_deposition" in mapped:
            expected_wet_po4 = 1.0 * year_to_second * 0.3
            assert torch.allclose(mapped["wet_po4_deposition"],
                                torch.tensor(expected_wet_po4), rtol=1e-5)

    def test_forward_returns_exact_canonical_flux_ports(self):
        """Forward exposes only the declared hydrosphere-to-soil flux ports."""
        mapper = PedosphereInterfaceMapper()

        state = {
            "drainage_flux": torch.tensor(0.001),
            "subsurface_runoff_flux": torch.tensor(0.0005),
            "surface_runoff_flux": torch.tensor(0.0002),
            "wet_n_deposition": torch.tensor(10.0),
            "leaching_flux": torch.tensor(9.0),
        }

        result = mapper.forward(state, time=0.0, dt=1.0)

        expected = {
            "leaching_flux": state["drainage_flux"],
            "lateral_flow": state["subsurface_runoff_flux"],
            "surface_loss": state["surface_runoff_flux"],
        }
        assert set(result) == set(expected)
        assert set(mapper.metadata.output_variables) == set(expected)
        for key, source_value in expected.items():
            assert isinstance(result[key], UpdateItem)
            assert result[key].mode == UpdateMode.REPLACE
            assert result[key].per_timestep is False
            assert result[key].unit == "kg m-2 s-1"
            torch.testing.assert_close(result[key].value, source_value)


    def test_forward_requires_all_declared_inputs(self):
        """A missing coupling port cannot be replaced by a hidden zero."""
        mapper = PedosphereInterfaceMapper()
        state = {
            "drainage_flux": torch.tensor(0.001),
            "subsurface_runoff_flux": torch.tensor(0.0005),
        }

        with pytest.raises(ValueError, match="surface_runoff_flux"):
            mapper.forward(state, time=0.0, dt=1.0)

    def test_calculate_solute_leaching(self):
        """Test solute leaching calculation with adsorption."""
        mapper = PedosphereInterfaceMapper()

        water_flux = torch.tensor(0.001)  # kg m^-2 s^-1
        concentration = torch.tensor(100.0)  # g m^-3
        adsorption_coef = 1.0  # 50% mobile

        solute_flux = mapper.calculate_solute_leaching(
            water_flux, concentration, adsorption_coef
        )

        # Expected: 0.001 * 0.001 * 100 * 0.5 = 0.00005
        water_flux_m = water_flux * 0.001  # Convert to m/s
        mobile_fraction = 1.0 / (1.0 + adsorption_coef)
        expected = water_flux_m * concentration * mobile_fraction

        assert torch.allclose(solute_flux, expected, rtol=1e-5)

    def test_ensure_mass_balance(self):
        """Test mass balance checking functionality."""
        mapper = PedosphereInterfaceMapper()

        pools_before = {
            "soil_organic_c": torch.tensor(1000.0),  # g C m^-2
            "soil_inorganic_n": torch.tensor(50.0),  # g N m^-2
        }

        pools_after = {
            "soil_organic_c": torch.tensor(995.0),
            "soil_inorganic_n": torch.tensor(48.0),
        }

        fluxes = {
            "heterotrophic_respiration": torch.tensor(0.005),  # g C m^-2 s^-1
            "nitrogen_leaching": torch.tensor(0.002),  # g N m^-2 s^-1
        }

        dt = 1000.0  # seconds

        errors = mapper.ensure_mass_balance(
            pools_before, pools_after, fluxes, dt, tolerance=1e-3
        )

        # Carbon: lost 5 g C, respired 0.005 * 1000 = 5 g C (balanced)
        # Nitrogen: lost 2 g N, leached 0.002 * 1000 = 2 g N (balanced)
        assert len(errors) == 0  # Should be balanced

    def test_upscale_downscale_methods(self):
        """Test that upscale and downscale methods exist."""
        mapper = PedosphereInterfaceMapper(
            upscale_bridge=lambda value: value.mean(dim=-1),
            downscale_bridge=lambda value: value.repeat_interleave(2),
        )

        fine_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        coarse_data = torch.tensor([2.5])

        # Test upscale
        upscaled = mapper.upscale(fine_data)
        assert isinstance(upscaled, torch.Tensor)

        # Test downscale
        downscaled = mapper.downscale(coarse_data)
        assert isinstance(downscaled, torch.Tensor)


class TestConvenienceFunctions:
    """Test convenience functions for pedosphere interface."""

    def test_get_leaching_flux(self):
        """Test get_leaching_flux with various state configurations."""
        # Test with drainage_flux present
        state1 = {"drainage_flux": torch.tensor(0.001)}
        flux1 = get_leaching_flux(state1)
        assert torch.equal(flux1, torch.tensor(0.001))

        # Another flux (subsurface_runoff_flux) or no flux is not a drainage_flux:
        # the canonical input is required and no zero is substituted.
        for state in ({"subsurface_runoff_flux": torch.tensor(0.002)}, {}):
            with pytest.raises(ValueError, match="drainage_flux"):
                get_leaching_flux(state)

    def test_calculate_doc_leaching(self):
        """Test DOC leaching calculation."""
        state = {"drainage_flux": torch.tensor(0.001), "bulk_density": torch.tensor(1300.0), "soil_depth": torch.tensor(1.0)}  # kg m^-2 s^-1, kg m-3, m
        doc_pool = torch.tensor(100.0)  # g C m^-2
        soil_water = torch.tensor(0.3)  # m^3 m^-3

        doc_flux = calculate_doc_leaching(
            state, doc_pool, soil_water, doc_solubility=0.01
        )

        assert isinstance(doc_flux, torch.Tensor)
        assert doc_flux >= 0
        # Should not exceed maximum leaching rate
        max_rate = doc_pool * 0.01 / 86400.0
        assert doc_flux <= max_rate

    def test_calculate_nitrate_leaching(self):
        """Test nitrate leaching calculation."""
        state = {"drainage_flux": torch.tensor(0.001), "bulk_density": torch.tensor(1300.0), "soil_depth": torch.tensor(1.0)}
        no3_pool = torch.tensor(10.0)  # g N m^-2
        soil_water = torch.tensor(0.3)

        no3_flux = calculate_nitrate_leaching(state, no3_pool, soil_water)

        assert isinstance(no3_flux, torch.Tensor)
        assert no3_flux >= 0
        # Should not exceed maximum leaching rate
        max_rate = no3_pool / 3600.0
        assert no3_flux <= max_rate



    def test_map_atmosphere_to_pedosphere_handles_gridded_depositions(self):
        """Vector depositions are converted elementwise instead of being silently dropped."""
        mapper = PedosphereInterfaceMapper()
        wet = torch.tensor([31.5576, 63.1152])  # g N m^-2 year^-1
        dry = torch.tensor([10.0, 20.0])
        wet_p = torch.tensor([1.0, 2.0])
        mapped = mapper.map_atmosphere_to_pedosphere(
            {"wet_n_deposition": wet, "dry_n_deposition": dry, "wet_p_deposition": wet_p}
        )
        # References: Dentener et al. (2006); Mahowald et al. (2008).
        # Mathematical expression: F = D / T_year; F_species = f_species * F.
        # Parameter meanings: D annual deposition, T_year seconds per year, f_species split fraction.
        # Units: D [g m-2 year-1]; F [g m-2 s-1]; f [1].
        # Domain/constraints: elementwise over the grid; f_noy + f_nhx = 1.
        # Sign convention: deposition is a positive input to the soil surface.
        year_to_second = 1.0 / float(get_physical_constant("relic.physical.constants.time.seconds_per_year"))
        assert tuple(mapped["total_wet_n_input"].shape) == (2,)
        torch.testing.assert_close(mapped["total_wet_n_input"], wet * year_to_second)
        torch.testing.assert_close(mapped["total_dry_n_input"], dry * year_to_second)
        torch.testing.assert_close(mapped["total_atmospheric_n_input"], (wet + dry) * year_to_second)
        torch.testing.assert_close(
            mapped["wet_noy_deposition"] + mapped["wet_nhx_deposition"],
            mapped["total_wet_n_input"],
        )
        torch.testing.assert_close(
            mapped["wet_po4_deposition"] + mapped["wet_dust_p_deposition"],
            wet_p * year_to_second,
        )
        assert torch.equal(mapped["wet_noy_soil_input"], mapped["wet_noy_deposition"])
        assert "total_atmospheric_p_input" in mapped

    def test_map_atmosphere_to_pedosphere_rejects_invalid_inputs(self):
        """Negative or non-finite depositions and inconsistent split overrides fail fast."""
        mapper = PedosphereInterfaceMapper()
        with pytest.raises(ValueError, match="wet_n_deposition"):
            mapper.map_atmosphere_to_pedosphere({"wet_n_deposition": torch.tensor([1.0, -1.0])})
        with pytest.raises(ValueError, match="dry_p_deposition"):
            mapper.map_atmosphere_to_pedosphere({"dry_p_deposition": torch.tensor(float("nan"))})
        with pytest.raises(ValueError, match="deposition_n_split"):
            mapper.map_atmosphere_to_pedosphere(
                {"wet_n_deposition": torch.tensor(1.0), "deposition_n_split": torch.tensor([0.5, 0.3, 0.2])}
            )
        with pytest.raises(ValueError, match="deposition_p_split"):
            mapper.map_atmosphere_to_pedosphere(
                {"wet_p_deposition": torch.tensor(1.0), "deposition_p_split": torch.tensor([0.5, 0.6])}
            )

    def test_nitrate_adsorption_converts_kd_liters_to_cubic_meters(self):
        """The nitrate retardation uses rho_b * Kd[L kg-1] * (m3/L) / theta, not the raw L m-3 product."""
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        rho_b = 1300.0  # bulk_density forcing input [kg m-3]
        kd = float(manager.get("pedosphere.transport.nitrate_distribution_coefficient"))
        liter_to_m3 = float(get_physical_constant("relic.physical.constants.conversions.liter_to_cubic_meter"))
        rho_w = float(get_physical_constant("relic.physical.constants.water.density_liquid"))
        drainage = torch.tensor(0.001)  # kg m^-2 s^-1
        no3_pool = torch.tensor(10.0)   # g N m^-2
        theta = torch.tensor(0.3)
        # References: Jury and Horton (2004); van Genuchten and Wierenga (1986).
        # Mathematical expression: R - 1 = rho_b Kd c_L / theta; F = (q / rho_w) (N / theta) / R.
        # Parameter meanings: rho_b bulk density, Kd distribution coefficient, c_L liters to m3,
        #   theta volumetric water content, q drainage mass flux, N nitrate pool.
        # Units: rho_b [kg m-3]; Kd [L kg-1]; c_L [m3 L-1]; q [kg m-2 s-1]; N [g N m-2]; F [g N m-2 s-1].
        # Domain/constraints: theta > 0; the result must stay below the hourly pool cap.
        # Sign convention: leaching is a positive loss from the soil column.
        retardation = 1.0 + rho_b * kd * liter_to_m3 / float(theta)
        expected = (float(drainage) / rho_w) * (float(no3_pool) / float(theta)) / retardation
        geometry = {"bulk_density": torch.tensor(rho_b), "soil_depth": torch.tensor(1.0)}
        result = calculate_nitrate_leaching({"drainage_flux": drainage, **geometry}, no3_pool, theta)
        assert expected < float(no3_pool) / 3600.0
        torch.testing.assert_close(result, torch.tensor(expected, dtype=result.dtype), rtol=1.0e-5, atol=0.0)

    def test_doc_adsorption_uses_the_same_retardation_form(self):
        """DOC mobility is 1/(1 + rho_b Kd c_L / theta); the raw Kd is not a dimensionless fraction."""
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        rho_b = 1300.0  # bulk_density forcing input [kg m-3]
        kd = float(manager.get("pedosphere.transport.doc_distribution_coefficient"))
        liter_to_m3 = float(get_physical_constant("relic.physical.constants.conversions.liter_to_cubic_meter"))
        rho_w = float(get_physical_constant("relic.physical.constants.water.density_liquid"))
        drainage = torch.tensor(0.001)
        doc_pool = torch.tensor(100.0)
        theta = torch.tensor(0.3)
        solubility = 0.01
        retardation = 1.0 + rho_b * kd * liter_to_m3 / float(theta)
        expected = (float(drainage) / rho_w) * (float(doc_pool) * solubility / float(theta)) / retardation
        geometry = {"bulk_density": torch.tensor(rho_b), "soil_depth": torch.tensor(1.0)}
        result = calculate_doc_leaching({"drainage_flux": drainage, **geometry}, doc_pool, theta, doc_solubility=solubility)
        assert expected < float(doc_pool) * solubility / 86400.0
        torch.testing.assert_close(result, torch.tensor(expected, dtype=result.dtype), rtol=1.0e-5, atol=0.0)


class TestUnitConversions:
    """Test unit conversion accuracy."""

    def test_water_flux_units(self):
        """Test water flux unit consistency."""
        mapper = PedosphereInterfaceMapper()

        # 1 kg m^-2 s^-1 water = 1 mm/s = 0.001 m/s
        water_flux_kg = torch.tensor(1.0)  # kg m^-2 s^-1

        # When used in solute calculation
        concentration = torch.tensor(1000.0)  # g m^-3 = 1 g/L

        solute_flux = mapper.calculate_solute_leaching(
            water_flux_kg, concentration, adsorption_coefficient=None
        )

        # Expected: 1 mm/s * 1 g/L = 0.001 m/s * 1000 g/m^3 = 1 g m^-2 s^-1
        expected = torch.tensor(1.0)
        assert torch.allclose(solute_flux, expected, rtol=1e-5)

    def test_deposition_time_conversion(self):
        """Test year to second conversion for deposition."""
        mapper = PedosphereInterfaceMapper()

        state = {
            "wet_n_deposition": torch.tensor(31.5576)  # g N m^-2 year^-1
        }

        mapped = mapper.map_atmosphere_to_pedosphere(state)

        # 31.5576 g/year = 1e-6 g/s (approximately)
        year_to_second = 1.0 / (365.25 * 24 * 3600)
        expected_rate = 31.5576 * year_to_second  # Should be ~1e-6

        if "total_wet_n_input" in mapped:
            assert torch.allclose(mapped["total_wet_n_input"],
                                torch.tensor(expected_rate), rtol=1e-5)


def test_imports():
    """Test that all required classes can be imported."""
    from relic.scientific.pedosphere.interface_mapping import (
        PedosphereInterfaceMapper,
        get_pedosphere_interface_mapper,
        get_leaching_flux,
        calculate_doc_leaching,
        calculate_nitrate_leaching
    )

    assert PedosphereInterfaceMapper is not None
    assert callable(get_pedosphere_interface_mapper)
    assert callable(get_leaching_flux)
    assert callable(calculate_doc_leaching)
    assert callable(calculate_nitrate_leaching)


def test_global_instance():
    """Test that mapper factory returns a valid instance."""
    mapper = get_pedosphere_interface_mapper()

    assert isinstance(mapper, PedosphereInterfaceMapper)
    assert mapper.name == "pedosphere_interface_mapper"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


