"""
L2 Integration Tests for BVOC Module.

Tests module integration with:
- PFT system
- Environmental state management
- Conservation checks
- Coupling with photosynthesis
- Multi-scale emissions

Author: Yuchong Long
Date: 2025-09-28
"""

import logging
import warnings
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from relic.core.simulator import RELICSimulator
from relic.core.simulator.execution_group import _clean_updates_and_collect_diagnostics
from relic.core.state import StateManager, StateVariableSpec
from relic.core.state.types import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.scientific.atmosphere.environmental_forcing import (
    EnvironmentalForcingFunctor,
)
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.biosphere import FarquharPhotosynthesis
from relic.scientific.biosphere.bvoc.megan_model import EmissionPathway, MEGANModel
from relic.scientific.biosphere.bvoc.synthesis import IsopreneSynthesis
from relic.utils.physical_constants import get_physical_constant
from test.utils.forward_call import call_forward


def _canonical_state(
    *,
    air_temperature=298.0,
    ppfd=800.0,
    leaf_area_index=4.0,
    atmospheric_co2_ppm=410.0,
):
    return {
        "air_temperature": torch.as_tensor(air_temperature, dtype=torch.float32),
        "ppfd": torch.as_tensor(ppfd, dtype=torch.float32),
        "leaf_area_index": torch.as_tensor(
            leaf_area_index, dtype=torch.float32
        ),
        "atmospheric_co2_ppm": torch.as_tensor(
            atmospheric_co2_ppm, dtype=torch.float32
        ),
    }


def _raw_megan_output(model, state, *, dt=3600.0):
    output = model.forward(state, time=0.0, dt=dt)
    for key, item in output.items():
        assert isinstance(item, UpdateItem), key
        assert item.mode is UpdateMode.REPLACE, key
        assert item.per_timestep is False, key
        assert item.element is None, key
        assert item.source_functor == model.name, key
    return output


class TestBVOCPFTIntegration:
    """Test BVOC integration with PFT system."""
    
    @pytest.fixture
    def setup_pft_environment(self):
        """Setup PFT-specific test environment."""
        state_manager = StateManager(device="cpu")
        
        # Initialize state with multiple PFTs
        state = {
            **_canonical_state(),
            'pft_fractions': torch.tensor([0.3, 0.4, 0.2, 0.1]),  # 4 PFT types
            'pft_types': ['broadleaf_deciduous_temperate', 'needleleaf_evergreen_temperate',
                         'c3_grass_temperate', 'shrub_deciduous_temperate']
        }
        
        environment = {
            'surface_pressure': torch.tensor(101325.0),
            'specific_humidity': torch.tensor(0.01)
        }
        
        return state_manager, state, environment
    
    def test_pft_specific_emissions(self, setup_pft_environment):
        """Test that different PFTs produce different emission rates."""
        state_manager, state, environment = setup_pft_environment
        
        # PFT closure is selected at construction, not by a hidden state alias.
        emissions_by_pft = []
        for pft_type in state['pft_types']:
            model = MEGANModel(
                config={"default_pft": pft_type},
                emission_pathway=EmissionPathway.STANDARD,
            )
            emissions = call_forward(model, {**state, **environment})
            emissions_by_pft.append(emissions["isoprene_emission"].item())
        
        # Different PFTs should have different emission rates
        assert len(set(emissions_by_pft)) > 1, "All PFTs have identical emissions"
        
        # Broadleaf deciduous should have highest isoprene emissions
        assert emissions_by_pft[0] > emissions_by_pft[2], \
            "Broadleaf deciduous should emit more isoprene than grass"
    
    def test_weighted_pft_emissions(self, setup_pft_environment):
        """Test weighted emissions across multiple PFTs."""
        state_manager, state, environment = setup_pft_environment
        
        # Calculate total weighted emissions
        total_emissions = torch.zeros(1)
        
        for i, (pft_type, fraction) in enumerate(zip(state['pft_types'], 
                                                      state['pft_fractions'])):
            model = MEGANModel(
                config={"default_pft": pft_type},
                emission_pathway=EmissionPathway.STANDARD,
            )
            emissions = call_forward(model, {**state, **environment})
            total_emissions += emissions["isoprene_emission"] * fraction
        
        # Total should be positive and reasonable
        assert total_emissions > 0
        assert total_emissions < 100.0  # mol m-2 s-1


class TestBVOCPhotosynthesisCoupling:
    """Test coupling between BVOC emissions and photosynthesis."""
    
    @pytest.fixture
    def setup_coupled_environment(self):
        """Setup environment with photosynthesis data."""
        state = {
            **_canonical_state(ppfd=1000.0, leaf_area_index=5.0),
            'radiation': torch.tensor(1000.0),
            'lai': torch.tensor(5.0),
            # Photosynthesis outputs
            'gross_photosynthesis': torch.tensor(10.0),  # umol CO2 m-2 s-1
            'stomatal_conductance': torch.tensor(0.2),  # mol H2O m-2 s-1
            'electron_transport_rate': torch.tensor(150.0)  # 渭mol e- m-2 s-1
        }
        
        environment = {
            'co2_concentration': torch.tensor(410.0, dtype=torch.float32),
            'air_temperature': torch.tensor(298.0, dtype=torch.float32),
            'surface_pressure': torch.tensor(101325.0, dtype=torch.float32)
        }
        
        return state, environment
    
    def test_gpp_emission_coupling(self, setup_coupled_environment):
        """Test that BVOC emissions scale with GPP."""
        state, environment = setup_coupled_environment
        
        synthesis = IsopreneSynthesis()
        
        # Test with normal GPP
        emissions_normal = call_forward(synthesis, {**state, **environment})
        
        # Test with doubled GPP
        state['gross_photosynthesis'] = state['gross_photosynthesis'] * 2
        state['electron_transport_rate'] = state['electron_transport_rate'] * 1.8
        emissions_high = call_forward(synthesis, {**state, **environment})

        # Higher GPP should lead to higher emissions.
        emission_key = "isoprene_emission"
        assert "isoprene_emission_mechanistic" not in emissions_normal
        assert "isoprenes_emission" not in emissions_normal
        assert emissions_high[emission_key] > emissions_normal[emission_key]
        
    def test_declared_stress_coupling(self, setup_coupled_environment):
        """Test explicit upstream stress factors without a hidden stomatal law."""
        state, environment = setup_coupled_environment
        state.update(
            {
                "water_stress_factor": torch.tensor(1.0, dtype=torch.float32),
                "temperature_stress_factor": torch.tensor(1.0, dtype=torch.float32),
                "ozone_stress_factor": torch.tensor(1.0, dtype=torch.float32),
            }
        )
        model = MEGANModel(
            config={"use_stress_functors": True},
            emission_pathway=EmissionPathway.STANDARD,
        )
        emissions_normal = call_forward(model, {**state, **environment})

        state["water_stress_factor"] = torch.tensor(0.2, dtype=torch.float32)
        emissions_stressed = call_forward(model, {**state, **environment})

        assert torch.allclose(
            emissions_stressed["monoterpenes_emission"],
            emissions_normal["monoterpenes_emission"] * 0.2,
            rtol=1.0e-6,
        )


class TestBVOCConservation:
    """Test mass and energy conservation in BVOC emissions."""
    
    @pytest.fixture
    def setup_conservation_test(self):
        """Setup for conservation tests."""
        state_manager = StateManager(device="cpu")
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        
        state = {
            **_canonical_state(),
            'biomass': torch.tensor(1000.0),  # g m-2
            'carbon_pool': torch.tensor(500.0)  # g C m-2
        }
        environment = {}
        
        return state_manager, model, state, environment
    
    def test_carbon_conservation(self, setup_conservation_test):
        """Test that BVOC carbon emissions don't exceed available carbon."""
        state_manager, model, state, environment = setup_conservation_test
        
        # Calculate emissions over 1 hour
        emissions = call_forward(model, {**state, **environment})
        
        carbon_allocation_flux = emissions["bvoc_carbon_allocation_flux"]
        expected_carbon_g = torch.zeros_like(carbon_allocation_flux)
        for compound in model.enabled_compounds:
            expected_carbon_g += (
                emissions[f"{compound}_emission"]
                * model.molecular_weights[compound]
                * model._get_carbon_fraction(compound)
            )
        expected_carbon_flux = unit_converter.convert(
            expected_carbon_g,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        )
        assert torch.allclose(
            carbon_allocation_flux,
            expected_carbon_flux,
            rtol=1.0e-6,
            atol=1.0e-15,
        )
        
        # Convert to g C m-2 h-1.
        seconds_per_hour = float(
            get_physical_constant(
                "relic.physical.constants.time.seconds_per_hour"
            )
        )
        total_c_emitted_g = unit_converter.convert(
            carbon_allocation_flux,
            variable_name="carbon_flux",
            source_units="kg C m-2 s-1",
            target_units="g C m-2 s-1",
        ) * seconds_per_hour
        
        # Should be much less than available carbon pool
        assert total_c_emitted_g < state['carbon_pool'] * 0.01  # < 1% per hour
        
    def test_energy_balance(self, setup_conservation_test):
        """Test that BVOC synthesis energy requirements are reasonable."""
        state_manager, model, state, environment = setup_conservation_test
        
        emissions = call_forward(model, {**state, **environment})
        
        # Energy cost of isoprene synthesis ~20 ATP per molecule
        # Energy cost of monoterpene synthesis ~30 ATP per molecule
        
        # Canonical species outputs are mol m-2 s-1.
        isoprene_mol_h = emissions["isoprene_emission"] * 3600.0
        monoterpene_mol_h = emissions["monoterpenes_emission"] * 3600.0
        
        # Stoichiometric ATP demand [mol ATP m-2 h-1].
        atp_required = isoprene_mol_h * 20.0 + monoterpene_mol_h * 30.0
        
        # Should be reasonable compared to photosynthetic ATP supply.
        # Assuming 3 ATP per CO2 fixed and GPP = 10 umol CO2 m-2 s-1.
        available_atp = 10e-6 * 3.0 * 3600.0
        
        assert atp_required < available_atp


class TestBVOCStateManagement:
    """Test state management and updates for BVOC emissions."""
    
    def test_state_updates(self):
        """Test that BVOC emissions properly update state."""
        state_manager = StateManager(
            schema={},
            autoload_contract_registry=False,
            device="cpu",
        )
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        output = _raw_megan_output(model, _canonical_state())
        diagnostics = {}
        runtime = SimpleNamespace(state_manager=state_manager)
        state_updates = _clean_updates_and_collect_diagnostics(
            runtime, model.name, model, output, diagnostics,
        )
        for key, item in state_updates.items():
            state_manager.register_variable(
                StateVariableSpec(
                    name=key,
                    units=item.unit or "dimensionless",
                    default=torch.zeros_like(item.value),
                )
            )
        sources = {key: model.name for key in state_updates}
        state_manager.apply_updates(
            state_updates,
            dt=3600.0,
            check_conservation=True,
            update_sources=sources,
        )
        first = {
            key: state_manager.get_state(key).clone()
            for key in state_updates
        }
        first_diagnostics = state_manager.conservation_history[-1].custom_checks
        state_updates = _clean_updates_and_collect_diagnostics(
            runtime, model.name, model, output, diagnostics,
        )
        state_manager.apply_updates(
            state_updates,
            dt=3600.0,
            check_conservation=True,
            update_sources=sources,
        )
        for key, item in state_updates.items():
            assert torch.equal(first[key], item.value)
            assert torch.equal(state_manager.get_state(key), item.value)
        for key, item in diagnostics.items():
            assert key not in state_manager.get_state()
            diagnostic_name = f"{model.name}.{key.removeprefix('diagnostics.')}"
            assert first_diagnostics[diagnostic_name] == pytest.approx(item.value.item())
            assert state_manager.conservation_history[-1].custom_checks[diagnostic_name] == pytest.approx(item.value.item())
        assert state_manager.get_diagnostics()["update_counter"] == 2
        
    def test_diurnal_cycle(self):
        """Test BVOC emissions over a diurnal cycle."""
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        
        # Simulate 24-hour cycle
        hours = np.arange(24)
        emissions_profile = []
        
        for hour in hours:
            # Simple radiation model
            if 6 <= hour <= 18:
                radiation = 500 * np.sin(np.pi * (hour - 6) / 12)
            else:
                radiation = 0.0
            
            # Temperature model
            temperature = 288 + 10 * np.sin(np.pi * (hour - 3) / 12)
            
            state = _canonical_state(
                air_temperature=float(temperature),
                ppfd=float(radiation),
            )
            emissions = call_forward(model, state)
            emissions_profile.append(emissions["isoprene_emission"].item())
        
        # Check diurnal pattern
        emissions_array = np.array(emissions_profile)
        
        # Daytime emissions should be higher than nighttime
        day_emissions = emissions_array[6:18].mean()
        night_emissions = emissions_array[np.r_[0:6, 18:24]].mean()
        assert day_emissions > night_emissions * 5  # Daytime must exceed nighttime by factor
        
        # Peak should be around midday
        peak_hour = np.argmax(emissions_array)
        assert 10 <= peak_hour <= 15


class TestBVOCMultiScale:
    """Test multi-scale BVOC emissions (leaf to canopy)."""
    
    def test_leaf_to_canopy_scaling(self):
        """Test scaling from leaf-level to canopy-level emissions."""
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        leaf_state = _canonical_state(ppfd=1500.0, leaf_area_index=1.0)
        canopy_state = _canonical_state(ppfd=1500.0, leaf_area_index=5.0)

        leaf_emissions = call_forward(model, leaf_state)
        canopy_emissions = call_forward(model, canopy_state)

        # This reduced closure uses an explicit linear foliar-density scaling.
        assert torch.allclose(
            canopy_emissions["isoprene_emission"],
            leaf_emissions["isoprene_emission"] * 5.0,
            rtol=1.0e-6,
        )
        
    def test_spatial_heterogeneity(self):
        """Test emissions with spatial heterogeneity."""
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        
        # Create spatially variable conditions (e.g., 10x10 grid)
        torch.manual_seed(0)
        grid_size = (10, 10)

        # Variable temperature field
        temperature = torch.randn(grid_size) * 5 + 298.0

        # Variable radiation field (some shading)
        radiation = torch.rand(grid_size) * 1000 + 500
        
        # Variable LAI
        lai = torch.rand(grid_size) * 3 + 2
        
        state = _canonical_state(
            air_temperature=temperature,
            ppfd=radiation,
            leaf_area_index=lai,
            atmospheric_co2_ppm=torch.full(grid_size, 410.0),
        )

        emissions = call_forward(model, state)
        
        # Check spatial variability in emissions
        assert emissions["isoprene_emission"].shape == grid_size
        assert emissions["isoprene_emission"].std() > 0
        
        # High temperature/radiation areas should have higher emissions
        iso_field = emissions["isoprene_emission"]
        activity_score = (
            (temperature - temperature.mean())
            + 0.001 * (radiation - radiation.mean())
            + (lai - lai.mean())
        )
        threshold = activity_score.median()
        high_activity_mask = activity_score >= threshold
        low_activity_mask = activity_score < threshold
        assert high_activity_mask.any(), "High activity mask should select at least one grid cell"
        assert low_activity_mask.any(), "Low activity mask should select at least one grid cell"
        assert iso_field[high_activity_mask].mean() >= iso_field[low_activity_mask].mean()


def test_farquhar_to_mechanistic_isoprene_real_simulator_chain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Canonical forcing and Farquhar ports must drive isoprene export.

    References:
    - Rasulov et al. (2009), Plant Physiology 151, 448-460.
    - Guenther et al. (2012), MEGAN2.1.

    Mathematical expressions:
    - F_iso,mol = convert(F_iso,umol).
    - F_C,alloc = convert(F_iso,mol * MW_iso * f_C, g C to kg C).

    Parameter meanings:
    - MW_iso is the isoprene molecular weight [g mol-1].
    - f_C is the carbon mass fraction of isoprene [1].
    - dt is the simulator step [s].
    """
    dt = 60.0
    caplog.set_level(logging.WARNING)
    isoprene = IsopreneSynthesis(device=torch.device("cpu"))
    assert isoprene.get_required_inputs() == [
        "air_temperature",
        "gross_photosynthesis",
        "electron_transport_rate",
        "atmospheric_co2_ppm",
        "leaf_area_index",
    ]
    simulator = RELICSimulator(
        {
            "max_time": dt,
            "time_step": dt,
            "device": "cpu",
            "conservation_checks": True,
            "strict_dependencies": True,
            "strict_scale_detection": False,
        }
    )
    simulator.initialize_state(
        {
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.6),
            "incoming_shortwave_potential": torch.tensor(1000.0),
            "cos_zenith": torch.tensor(0.8),
            "surface_temperature": torch.tensor(298.15),
            "surface_albedo": torch.tensor(0.2),
            "surface_emissivity": torch.tensor(0.97),
            "cloud_fraction": torch.tensor(0.2),
            "atmospheric_co2_ppm": torch.tensor(400.0),
            "leaf_area_index": torch.tensor(3.0),
            "water_stress_factor": torch.tensor(1.0),
            "temperature_stress_factor": torch.tensor(1.0),
            "ozone_stress_factor": torch.tensor(1.0),
        }
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for functor in (
            EnvironmentalForcingFunctor(device=torch.device("cpu")),
            RadiationFunctor(device=torch.device("cpu")),
            FarquharPhotosynthesis(device=torch.device("cpu")),
            isoprene,
        ):
            simulator.register_functor(functor, execution_group="fast")
        order = simulator.dependency_graph.topological_sort()
        updates = simulator.step(dt=dt)

    assert captured == []
    assert [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ] == []
    assert order.index("environmental_forcing") < order.index(
        "farquhar_photosynthesis"
    )
    assert order.index("atmospheric_radiation") < order.index(
        "farquhar_photosynthesis"
    )
    assert order.index("farquhar_photosynthesis") < order.index(
        "mechanistic_isoprene_synthesis"
    )

    state = simulator.state_manager.get_state()
    expected_values = {
            "vapor_pressure_deficit": 1267.1112060546875,
            "ppfd": 1043.2799072265625,
            "gross_photosynthesis": 15.49929141998291,
            "electron_transport_rate": 155.42617797851562,
            "net_photosynthesis": 13.634803771972656,
            "gross_primary_productivity": 1.8616198360632552e-07,
            "isoprene_emission": 7.834020721020352e-08,
            "bvoc_carbon_allocation_flux": 4.7068242636783e-09,
        }
    for key, expected in expected_values.items():
        actual = state[key]
        assert torch.allclose(
            actual,
            torch.as_tensor(
                expected,
                dtype=actual.dtype,
                device=actual.device,
            ),
            rtol=1.0e-6,
            atol=0.0,
        ), f"{key}: {actual.detach().cpu().tolist()} != {expected}"

    emission = state["isoprene_emission"]
    carbon_allocation = state["bvoc_carbon_allocation_flux"]
    assert torch.all(emission > 0)
    assert torch.all(carbon_allocation > 0)
    expected_carbon_allocation = unit_converter.convert(
        emission * isoprene.isoprene_mw * isoprene.carbon_fraction_isoprene,
        variable_name="carbon_flux",
        source_units="g C m-2 s-1",
        target_units="kg C m-2 s-1",
    )
    torch.testing.assert_close(
        carbon_allocation,
        expected_carbon_allocation,
    )

    expected_units = {
        "gross_photosynthesis": "umol CO2 m-2 s-1",
        "electron_transport_rate": "umol e- m-2 s-1",
        "isoprene_emission": "mol m-2 s-1",
        "bvoc_carbon_allocation_flux": "kg C m-2 s-1",
    }
    registry = simulator.state_manager._contract_registry
    assert registry is not None
    for key, unit in expected_units.items():
        assert isinstance(updates[key], torch.Tensor)
        contract = registry.get_contract(key)
        assert contract is not None
        assert contract.udunits == unit

    expected_writers = {
        "gross_photosynthesis": "farquhar_photosynthesis",
        "electron_transport_rate": "farquhar_photosynthesis",
        "isoprene_emission": "mechanistic_isoprene_synthesis",
        "bvoc_carbon_allocation_flux": "mechanistic_isoprene_synthesis",
    }
    for key, expected_writer in expected_writers.items():
        writers = {
            record["functor"]
            for record in simulator.get_variable_provenance(key)
            if record.get("role") in {"writer", "final_writer"}
        }
        assert writers == {expected_writer}, f"{key}: {writers}"

    for functor_name in (
        "environmental_forcing",
        "atmospheric_radiation",
        "farquhar_photosynthesis",
        "mechanistic_isoprene_synthesis",
    ):
        assert simulator.get_effective_device(functor_name).type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

