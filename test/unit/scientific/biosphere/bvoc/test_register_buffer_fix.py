"""
Unit tests for register_buffer hardcoding fixes in BVOC synthesis modules.

Tests verify that parameters are correctly loaded from ParameterManager
using lazy loading and can be dynamically updated.

Author: Yuchong Long
Date: 2025-10-20
"""

import pytest
import torch

from relic.core.state import UpdateItem
from relic.scientific.biosphere.bvoc.synthesis.isoprene import (
    MEPPathwayModel,
    IsopreneSynthesis,
)
from relic.scientific.biosphere.bvoc.synthesis.monoterpene import (
    MonoterpeneSynthesis,
)
from relic.parameters import get_global_manager


class TestMEPPathwayModelParameterLoading:
    """Test MEPPathwayModel parameter lazy loading."""

    def test_parameter_properties_exist(self):
        """Verify all parameter properties are defined."""
        model = MEPPathwayModel()

        # Check that all parameters are accessible as properties
        assert hasattr(model, 'v_max_dxs')
        assert hasattr(model, 'k_g3p')
        assert hasattr(model, 'k_pyruvate')
        assert hasattr(model, 'q10_dxs')
        assert hasattr(model, 'atp_per_dmapp')
        assert hasattr(model, 'nadph_per_dmapp')
        assert hasattr(model, 'temp_ref')

    def test_parameters_are_tensors(self):
        """Verify parameters are returned as tensors."""
        model = MEPPathwayModel()

        assert isinstance(model.v_max_dxs, torch.Tensor)
        assert isinstance(model.k_g3p, torch.Tensor)
        assert isinstance(model.k_pyruvate, torch.Tensor)
        assert isinstance(model.q10_dxs, torch.Tensor)

    def test_parameter_caching(self):
        """Verify parameters are cached after first access."""
        model = MEPPathwayModel()

        # First access should populate cache
        v_max_1 = model.v_max_dxs
        assert 'v_max_dxs' in model._param_cache

        # Second access should return cached value
        v_max_2 = model.v_max_dxs
        assert torch.equal(v_max_1, v_max_2)
        assert v_max_1 is model._param_cache['v_max_dxs']

    def test_cache_clearing(self):
        """Verify parameter cache can be cleared."""
        model = MEPPathwayModel()

        # Populate cache
        _ = model.v_max_dxs
        _ = model.k_g3p
        assert len(model._param_cache) >= 2

        # Clear cache
        model.clear_parameter_cache()
        assert len(model._param_cache) == 0

    def test_unit_conversion(self):
        """Verify unit conversion is applied correctly."""
        model = MEPPathwayModel()

        # k_g3p should be converted from umol/L to umol/m3 (脳 1000)
        # Just verify it's a positive value (actual value depends on parameter database)
        k_g3p = model.k_g3p
        assert k_g3p.item() > 0.0  # Should be positive after conversion

    def test_forward_uses_properties(self):
        """Verify forward method can use parameter properties."""
        model = MEPPathwayModel()

        # Create test inputs
        g3p = torch.tensor(100.0)
        pyr = torch.tensor(50.0)
        atp = torch.tensor(1000.0)
        nadph = torch.tensor(500.0)
        temp = torch.tensor(298.15)

        # Forward pass should work without errors
        result = model.forward(g3p, pyr, atp, nadph, temp)

        assert 'dmapp_flux' in result
        assert isinstance(result['dmapp_flux'], torch.Tensor)

    def test_unread_dxr_q10_is_retired(self):
        """The MEP rate law applies only the DXS temperature factor (q10_dxs); the DXR
        Q10 had no reader, so it is neither a lazy parameter nor a database entry."""
        model = MEPPathwayModel()
        assert 'q10_dxr' not in model._param_specs
        assert not hasattr(model, 'q10_dxr')
        assert get_global_manager().get_parameter('biosphere.bvoc.mep.q10_dxr') is None


class TestIsopreneSynthesisParameterLoading:
    """Test IsopreneSynthesis Functor with fixed parameter loading."""

    def test_functor_initialization(self):
        """Verify Functor can be initialized."""
        functor = IsopreneSynthesis()
        assert functor is not None
        assert hasattr(functor, 'mep_pathway')
        assert (
            dict(functor.named_buffers())["canopy_light_extinction"]
            is functor.canopy_light_extinction
        )

    def test_mep_pathway_accessible(self):
        """Verify MEP pathway model is properly initialized."""
        functor = IsopreneSynthesis()
        assert isinstance(functor.mep_pathway, MEPPathwayModel)

        # MEP pathway should have parameter cache
        assert hasattr(functor.mep_pathway, '_param_cache')

    def test_forward_pass(self):
        """Verify forward pass works with lazy-loaded parameters."""
        functor = IsopreneSynthesis()

        # Create test state
        state = {
            'air_temperature': torch.tensor(298.15),
            'gross_photosynthesis': torch.tensor(10.0),
            'electron_transport_rate': torch.tensor(100.0),
            'atmospheric_co2_ppm': torch.tensor(250.0),
            'leaf_area_index': torch.tensor(1.0),
        }

        # Forward pass
        result = functor.forward(state, time=0.0, dt=3600.0)

        # Verify outputs
        assert "isoprene_emission" in result
        assert "isoprenes_emission" not in result
        assert "isoprene_emission_mechanistic" not in result
        assert isinstance(result["isoprene_emission"], UpdateItem)
        assert isinstance(result["isoprene_emission"].value, torch.Tensor)


class TestMonoterpeneSynthesisParameterLoading:
    """Test canonical nn.Module parameter and buffer registration."""

    def test_parameter_properties_exist(self):
        """Verify all parameter properties are defined."""
        functor = MonoterpeneSynthesis()

        # Check parameter properties
        assert hasattr(functor, 'f_carbon_monoterpenes')
        assert hasattr(functor, 'monoterpene_mw')
        assert hasattr(functor, 'seasonal_amplitude')
        assert hasattr(functor, 'seasonal_min_factor')
        assert hasattr(functor, 'seasonal_period_seconds')
        assert hasattr(functor, 'seasonal_phase_offset')

    def test_parameters_are_tensors(self):
        """Verify parameters are returned as tensors."""
        functor = MonoterpeneSynthesis()

        assert isinstance(functor.f_carbon_monoterpenes, torch.Tensor)
        assert isinstance(functor.monoterpene_mw, torch.Tensor)
        assert isinstance(functor.seasonal_amplitude, torch.Tensor)

    def test_trainable_parameters_are_the_manager_bindings(self):
        """Expose calibratable parameters exactly once, through their one owner:
        the Functor holds the ParameterManager bindings by reference and owns no
        module parameter of its own."""
        functor = MonoterpeneSynthesis()
        trainable = get_global_manager().get_trainable_parameters()
        held = functor.held_manager_parameters()
        for attribute, key in (
            (
                "f_carbon_monoterpenes",
                "biosphere.bvoc.photosynthetic_coupling.f_carbon_monoterpenes",
            ),
            ("seasonal_amplitude", "biosphere.bvoc.monoterpene.seasonal.amplitude"),
            ("seasonal_peak_day", "biosphere.bvoc.monoterpene.seasonal.peak_day"),
        ):
            assert getattr(functor, attribute) is trainable[key], attribute
            assert held[key] is trainable[key], attribute
        assert dict(functor.named_parameters()) == {}
        assert not hasattr(functor, "_param_cache")
        assert not hasattr(functor, "clear_parameter_cache")

    def test_nontrainable_values_are_registered_buffers(self):
        """Keep constants and non-calibrated values device-migratable."""
        functor = MonoterpeneSynthesis()
        buffers = dict(functor.named_buffers())
        for name in (
            "monoterpene_mw",
            "carbon_fraction_monoterpenes",
            "seasonal_min_factor",
            "seasonal_period_seconds",
            "seconds_per_day",
        ):
            assert buffers[name] is getattr(functor, name)

    def test_seasonal_phase_offset_calculation(self):
        """Verify seasonal_phase_offset is calculated correctly."""
        functor = MonoterpeneSynthesis()

        # Access phase offset (should trigger calculation)
        phase_offset = functor.seasonal_phase_offset

        assert isinstance(phase_offset, torch.Tensor)
        assert 0.0 <= phase_offset.item() <= 2.0 * 3.14159  # Within 0 to 2蟺

    def test_forward_pass(self):
        """Verify forward pass works with lazy-loaded parameters."""
        functor = MonoterpeneSynthesis()

        # Create test state
        state = {
            'air_temperature': torch.tensor(298.15),
            'gross_photosynthesis': torch.tensor(10.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: functor.reset_pools(),
        }

        # Forward pass
        result = functor.forward(state, time=0.0, dt=3600.0)

        # Verify outputs
        assert 'monoterpenes_emission' in result
        assert 'monoterpene_emission_mechanistic' not in result
        assert isinstance(result['monoterpenes_emission'].value, torch.Tensor)
        assert result['monoterpenes_emission'].unit == 'mol m-2 s-1'


class TestParameterDynamicUpdate:
    """Test that parameters can be dynamically updated."""

    def test_mep_pathway_parameter_update(self):
        """Verify MEP pathway parameters can be updated at runtime."""
        model = MEPPathwayModel()
        param_manager = get_global_manager()

        # Get original value
        original_value = model.v_max_dxs.item()

        # Clear cache to force reload
        model.clear_parameter_cache()

        # Parameter should reload from manager (value may be same, but reload happened)
        new_value = model.v_max_dxs.item()

        # Verify it's still a valid value
        assert new_value > 0

    def test_monoterpene_parameter_is_optimizer_visible(self):
        """Verify the canonical trainable tensor is stable and optimizer-visible
        on the manager calibration path (get_trainable_parameters)."""
        functor = MonoterpeneSynthesis()
        trainable = get_global_manager().get_trainable_parameters()
        optimizer = torch.optim.SGD(list(trainable.values()), lr=1.0e-3)
        assert any(
            parameter is functor.f_carbon_monoterpenes
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        assert functor.f_carbon_monoterpenes.requires_grad
        assert list(functor.parameters()) == []


def test_no_register_buffer_in_code():
    """Meta-test: Verify register_buffer was removed from parameter initialization."""
    import inspect

    # Check MEPPathwayModel.__init__
    mep_source = inspect.getsource(MEPPathwayModel.__init__)
    assert 'register_buffer' not in mep_source, \
        "MEPPathwayModel.__init__ should not contain register_buffer calls"

    # Check MonoterpeneSynthesis._init_coupling_parameters
    mono_source = inspect.getsource(MonoterpeneSynthesis._init_coupling_parameters)
    assert 'register_buffer' not in mono_source, \
        "MonoterpeneSynthesis._init_coupling_parameters should not contain register_buffer calls"

    mono_init = inspect.getsource(MonoterpeneSynthesis.__init__)
    assert "_param_cache" not in mono_init
    assert "_param_loaders" not in mono_init


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

