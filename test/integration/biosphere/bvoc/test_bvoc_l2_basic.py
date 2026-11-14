"""
Basic L2 Integration Tests for BVOC Module (without full parameter loading).

Tests basic integration functionality.

Author: Yuchong Long
Date: 2025-09-28
"""

import pytest
import torch

from relic.core.state.types import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.scientific.biosphere.bvoc.megan_model import EmissionPathway, MEGANModel
from relic.parameters import get_global_manager


def ensure_basic_bvoc_parameters(param_manager) -> None:
    """Ensure required BVOC parameters are available via the parameter system."""
    required_keys = [
        "biosphere.bvoc.emission_factors.isoprene",
        "biosphere.bvoc.emission_factors.monoterpenes",
        "biosphere.bvoc.emission_factors.sesquiterpenes",
        "biosphere.bvoc.emission_factors.ovocs",
        "biosphere.bvoc.megan.alpha_light",
        "biosphere.bvoc.megan.cl_light",
        "biosphere.bvoc.megan.topt",
        "biosphere.bvoc.megan.ct1_isoprene",
        "biosphere.bvoc.megan.ct2",
        "biosphere.bvoc.megan.eopt_isoprene",
        "relic.physical.constants.universal.gas_constant",
    ]
    for key in required_keys:
        param_manager.get(key, required=True)


def _state(*, air_temperature: float = 298.0, ppfd: float = 1000.0):
    return {
        "air_temperature": torch.tensor(air_temperature),
        "ppfd": torch.tensor(ppfd),
        "leaf_area_index": torch.tensor(3.0),
        "atmospheric_co2_ppm": torch.tensor(400.0),
    }


def _values(output):
    values = {}
    for key, item in output.items():
        assert isinstance(item, UpdateItem), key
        assert item.mode is UpdateMode.REPLACE, key
        assert item.per_timestep is False, key
        assert item.element is None, key
        assert item.source_functor == "megan_bvoc_model", key
        values[key] = item.value
    return values


class TestBVOCBasic:
    """Test basic BVOC functionality without full database loading."""
    
    def test_model_initialization(self):
        """Test that model can initialize."""
        # Load minimal parameters
        param_manager = get_global_manager()
        
        ensure_basic_bvoc_parameters(param_manager)
        
        # Initialize model
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        
        # Check model attributes
        assert hasattr(model, 'isoprene_ef')
        assert model.isoprene_ef > 0
        
    def test_basic_emission(self):
        """Test basic emission calculation."""
        param_manager = get_global_manager()
        
        ensure_basic_bvoc_parameters(param_manager)
        
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        output = _values(model.forward(_state(), time=0.0, dt=3600.0))
        
        # Check output
        assert 'total_bvoc_emission' in output
        assert output['total_bvoc_emission'] > 0
        
    def test_temperature_sensitivity(self):
        """Test temperature sensitivity."""
        param_manager = get_global_manager()
        
        ensure_basic_bvoc_parameters(param_manager)
        
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        
        # Test at different temperatures
        emissions = []
        for temp_c in [15, 25, 35]:  # Celsius
            output = _values(
                model.forward(
                    _state(air_temperature=temp_c + 273.15),
                    time=0.0,
                    dt=3600.0,
                )
            )
            emissions.append(output['total_bvoc_emission'].item())
        
        # Check temperature response
        # Emissions should increase from 15 to 25 degC.
        assert emissions[1] > emissions[0]
        
    def test_light_sensitivity(self):
        """Test light sensitivity."""
        param_manager = get_global_manager()
        
        ensure_basic_bvoc_parameters(param_manager)
        
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        
        # Test at different light levels
        emissions = []
        for light in [0, 500, 1500]:
            output = _values(
                model.forward(
                    _state(ppfd=float(light)),
                    time=0.0,
                    dt=3600.0,
                )
            )
            emissions.append(output['total_bvoc_emission'].item())
        
        # Check light response
        # Emissions should increase with light
        assert emissions[1] > emissions[0]  # Light > Dark
        assert emissions[2] > emissions[1]  # Bright > Medium
        
    def test_compound_outputs(self):
        """Test that different compounds are calculated."""
        param_manager = get_global_manager()
        
        ensure_basic_bvoc_parameters(param_manager)
        
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        output = _values(model.forward(_state(), time=0.0, dt=3600.0))
        
        # Check for compound-specific outputs
        assert 'isoprene_emission' in output
        assert 'monoterpenes_emission' in output
        
        # Isoprene should be dominant
        assert output['isoprene_emission'] > output['monoterpenes_emission']
        
    def test_conservation(self):
        """Test basic conservation properties."""
        param_manager = get_global_manager()
        
        ensure_basic_bvoc_parameters(param_manager)
        
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        output = _values(model.forward(_state(), time=0.0, dt=3600.0))
        
        # Total should equal sum of components
        total = output['total_bvoc_emission']
        
        # Manually sum components
        components_sum = torch.zeros_like(total)
        for compound in ['isoprene', 'monoterpenes', 'sesquiterpenes', 'ovocs']:
            key = f'{compound}_emission'
            if key in output:
                components_sum += output[key]
        
        # Check conservation (within numerical tolerance)
        assert torch.allclose(total, components_sum, rtol=1e-5)
        expected_carbon_g = torch.zeros_like(total)
        for compound in model.enabled_compounds:
            expected_carbon_g += (
                output[f"{compound}_emission"]
                * model.molecular_weights[compound]
                * model._get_carbon_fraction(compound)
            )
        expected_carbon_kg = unit_converter.convert(
            expected_carbon_g,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        )
        assert torch.allclose(
            output["bvoc_carbon_allocation_flux"],
            expected_carbon_kg,
            rtol=1.0e-6,
            atol=1.0e-15,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

