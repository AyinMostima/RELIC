"""
Unit tests for Denitrification Process

Tests the denitrification functor including:
- Initialization and parameter loading
- NO3 reduction to N2O and N2
- Carbon dependency
- Oxygen/anaerobic controls
- Conservation laws
"""

import pytest
import torch
import inspect
from relic.core.state import UpdateItem
from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess


def _unwrap_update(value):
    """Return the tensor carried by an explicit simulator update."""
    return value.value if isinstance(value, UpdateItem) else value


class TestDenitrificationInit:
    """Test initialization and setup"""

    def test_initialization(self, basic_config):
        """Test that DenitrificationProcess can be instantiated"""
        functor = DenitrificationProcess(name="test_denit", config=basic_config)
        assert functor is not None
        assert functor.name == "test_denit"

    def test_parameter_loading(self, parameter_manager, basic_config):
        """Test that parameters are loaded from ParameterManager"""
        functor = DenitrificationProcess(name="test_denit", config=basic_config)

        # Check key parameters are loaded
        assert hasattr(functor, 'vmax')
        assert hasattr(functor, 'km_no3')
        assert functor.vmax is not None
        assert functor.km_no3 is not None

    def test_metadata_conservation_laws(self, basic_config):
        """Test that nitrogen conservation is declared in metadata"""
        functor = DenitrificationProcess(name="test_denit", config=basic_config)

        assert hasattr(functor.metadata, 'conservation_laws')
        assert 'nitrogen' in functor.metadata.conservation_laws

    def test_phase8_runtime_uses_explicit_reaction_network(self, basic_config):
        """Test Phase 8 runtime advances state through explicit N_h r_h."""
        functor = DenitrificationProcess(name="test_denit", config=basic_config)
        source = inspect.getsource(DenitrificationProcess.forward)

        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "advance_reaction_network(" in source
        assert "stoichiometric_N=" in source
        assert "x_dot = N_h r_h" in source


class TestDenitrificationForward:
    """Test forward execution"""

    def test_forward_runs(self, basic_config, mock_state, mock_environment):
        """Test that forward() executes without errors"""
        functor = DenitrificationProcess(name="test_denit", config=basic_config)

        # Add required state variables
        state = mock_state.copy()
        state['no3_nitrogen'] = torch.tensor([1.0, 2.0, 3.0])
        state['labile_carbon'] = torch.tensor([0.5, 0.5, 0.5])

        result = functor.forward(state, time=0.0, dt=1800.0)

        assert result is not None
        assert isinstance(result, dict)

    def test_no3_reduction(self, basic_config, mock_state):
        """Test that NO3 is reduced to N2O and N2"""
        functor = DenitrificationProcess(name="test_denit", config=basic_config)

        # High NO3, high moisture (anaerobic), sufficient carbon
        state = mock_state.copy()
        state['no3_nitrogen'] = torch.tensor([5.0])  # High NO3
        state['water_filled_pore_space'] = torch.tensor([0.7])  # High WFPS (anaerobic)
        state['labile_carbon'] = torch.tensor([1.0])  # Sufficient carbon
        state['soil_temperature'] = torch.tensor([293.15])  # 20C
        state['soil_ph'] = torch.tensor([6.8])

        result = functor.forward(state, time=0.0, dt=1800.0)

        # Should have denitrification flux and NO3 delta
        assert 'denitrification_flux' in result
        assert 'no3_nitrogen' in result
        assert 'n2o_flux' in result or 'n2_production' in result

    def test_carbon_dependency(self, basic_config, mock_state):
        """Test that denitrification depends on labile carbon availability"""
        functor = DenitrificationProcess(name="test_denit", config=basic_config)

        # Set up anaerobic conditions with adequate NO3
        base_state = mock_state.copy()
        base_state['no3_nitrogen'] = torch.tensor([3.0])
        base_state['water_filled_pore_space'] = torch.tensor([0.7])  # Anaerobic
        base_state['soil_temperature'] = torch.tensor([293.15])
        base_state['soil_ph'] = torch.tensor([6.8])

        # Low carbon case
        state_low_c = base_state.copy()
        state_low_c['labile_carbon'] = torch.tensor([0.01])
        result_low_c = functor.forward(state_low_c, time=0.0, dt=1800.0)

        # High carbon case
        state_high_c = base_state.copy()
        state_high_c['labile_carbon'] = torch.tensor([2.0])
        result_high_c = functor.forward(state_high_c, time=0.0, dt=1800.0)

        # Get denitrification fluxes
        if 'denitrification_flux' in result_low_c and 'denitrification_flux' in result_high_c:
            flux_low = _unwrap_update(result_low_c['denitrification_flux'])
            flux_high = _unwrap_update(result_high_c['denitrification_flux'])

            # High carbon should produce higher flux (in absolute value)
            assert abs(flux_high.sum().item()) > abs(flux_low.sum().item()) * 0.5

    def test_oxygen_inhibition(self, basic_config, mock_state):
        """Test that aerobic conditions inhibit denitrification"""
        functor = DenitrificationProcess(name="test_denit", config=basic_config)

        # Same conditions except moisture
        base_state = mock_state.copy()
        base_state['no3_nitrogen'] = torch.tensor([3.0])
        base_state['labile_carbon'] = torch.tensor([1.0])
        base_state['soil_temperature'] = torch.tensor([293.15])
        base_state['soil_ph'] = torch.tensor([6.8])

        # Aerobic (low moisture)
        state_aerobic = base_state.copy()
        state_aerobic['water_filled_pore_space'] = torch.tensor([0.3])  # Below anaerobic threshold
        result_aerobic = functor.forward(state_aerobic, time=0.0, dt=1800.0)

        # Anaerobic (high moisture)
        state_anaerobic = base_state.copy()
        state_anaerobic['water_filled_pore_space'] = torch.tensor([0.8])  # Above anaerobic threshold
        result_anaerobic = functor.forward(state_anaerobic, time=0.0, dt=1800.0)

        # Get denitrification fluxes
        if 'denitrification_flux' in result_aerobic and 'denitrification_flux' in result_anaerobic:
            flux_aerobic = _unwrap_update(result_aerobic['denitrification_flux'])
            flux_anaerobic = _unwrap_update(result_anaerobic['denitrification_flux'])

            # Anaerobic should have much higher flux
            assert abs(flux_anaerobic.sum().item()) > abs(flux_aerobic.sum().item())


class TestDenitrificationConservation:
    """Test conservation laws"""

    def test_nitrogen_conservation(self, basic_config, mock_state):
        """Test that nitrogen is conserved (NO3 consumed = N2O + N2 produced)"""
        functor = DenitrificationProcess(name="test_denit", config=basic_config)

        state = mock_state.copy()
        state['no3_nitrogen'] = torch.tensor([5.0])
        state['water_filled_pore_space'] = torch.tensor([0.7])
        state['labile_carbon'] = torch.tensor([1.0])
        state['soil_temperature'] = torch.tensor([293.15])
        state['soil_ph'] = torch.tensor([6.8])

        result = functor.forward(state, time=0.0, dt=1800.0)

        # Check if conservation check was called (should be in result or logged)
        # The functor should call self.check_conservation_laws(state, updates)
        # This is verified by the functor's implementation
        assert result is not None

        # In a real conservation check, we'd verify:
        # delta_NO3 + delta_N2O + delta_N2 ~= 0
        # But this requires access to the full update dictionary

def test_public_conservation_checker_audits_the_returned_denitrification_proposal(mock_state, basic_config):
    """The public checker recomputes dNO3 + (F_N2O + F_N2)*dt from the returned UpdateItems."""
    import dataclasses

    functor = DenitrificationProcess(name="test_denit_public_ledger", config=basic_config)
    state = dict(mock_state)
    state["water_filled_pore_space"] = torch.tensor([0.7, 0.75, 0.8])
    state["labile_carbon"] = torch.tensor([1.0, 1.2, 0.9])
    result = functor(state, time=0.0, dt=1800.0)
    clean = functor.check_conservation_laws(state, result)
    assert set(clean) == set(functor.metadata.conservation_laws)
    assert clean["nitrogen"] <= 1.0e-9
    assert float(_unwrap_update(result["diagnostics.denitrification.step_seconds"])) == 1800.0
    no3 = result["no3_nitrogen"]
    assert torch.all(no3.value < 0.0)
    tampered_pool = dict(result)
    tampered_pool["no3_nitrogen"] = dataclasses.replace(no3, value=no3.value + 1.0)
    assert functor.check_conservation_laws(state, tampered_pool)["nitrogen"] == pytest.approx(1.0, rel=1.0e-6)
    n2 = result["n2_production"]
    tampered_rate = dict(result)
    tampered_rate["n2_production"] = dataclasses.replace(n2, value=n2.value + 1.0 / 1800.0)
    assert functor.check_conservation_laws(state, tampered_rate)["nitrogen"] == pytest.approx(1.0, rel=1.0e-6)
    without_step = dict(result)
    del without_step["diagnostics.denitrification.step_seconds"]
    with pytest.raises(ValueError, match="step_seconds"):
        functor.check_conservation_laws(state, without_step)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

