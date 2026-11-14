"""
L1 Unit Tests for Mineralization Process Functor.

Tests organic N -> NH4 conversion and environmental controls.
"""
import pytest
import torch
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess


class TestMineralizationInit:
    """Test initialization."""

    def test_initialization(self, basic_config):
        """Test Functor can be initialized."""
        functor = MineralizationProcess(name="test_min", config=basic_config)
        assert functor is not None
        assert functor.name == "test_min"

    def test_metadata_present(self, basic_config):
        """Test metadata is complete."""
        functor = MineralizationProcess(name="test_min", config=basic_config)
        assert functor.metadata is not None


class TestMineralizationForward:
    """Test forward execution."""

    def test_forward_runs(self, single_state, basic_config):
        """Test forward executes without errors."""
        functor = MineralizationProcess(name="test_min", config=basic_config)
        state = dict(single_state)
        state["carbon_decomposition_rate"] = torch.tensor(1e-6)
        state["organic_nitrogen_input"] = torch.tensor(0.0)
        updates = functor.forward(state=state, time=0.0, dt=3600.0)
        assert isinstance(updates, dict)
        assert {
            "organic_nitrogen",
            "nh4_nitrogen",
            "microbial_nitrogen",
        }.issubset(updates)
        assert not {
            "delta_organic_nitrogen",
            "delta_nh4_nitrogen",
            "delta_microbial_nitrogen",
        }.intersection(updates)
        assert {
            "organic_nitrogen",
            "nh4_nitrogen",
            "microbial_nitrogen",
        }.issubset(functor.metadata.output_variables)
        for name in ("organic_nitrogen", "nh4_nitrogen", "microbial_nitrogen"):
            update = updates[name]
            assert isinstance(update, UpdateItem)
            assert update.mode is UpdateMode.ACCUMULATE
            assert update.per_timestep is True

    def test_organic_to_nh4_conversion(self, single_state, basic_config):
        """Test that organic N is converted to NH4."""
        functor = MineralizationProcess(name="test_min", config=basic_config)
        state = dict(single_state)
        state["carbon_decomposition_rate"] = torch.tensor(1e-6)
        state["organic_nitrogen_input"] = torch.tensor(0.0)
        updates = functor.forward(state=state, time=0.0, dt=3600.0)

        gross_rate = updates["gross_mineralization_flux"]
        assert isinstance(gross_rate, UpdateItem)
        assert torch.all(gross_rate.value >= 0)
        assert "mineralization_rate" not in updates

    def test_temperature_sensitivity(self, single_state, basic_config):
        """Test mineralization increases with temperature."""
        functor = MineralizationProcess(name="test_min", config=basic_config)

        # Cold
        state_cold = single_state.copy()
        state_cold['soil_temperature'] = torch.tensor(278.15)
        state_cold['carbon_decomposition_rate'] = torch.tensor(1e-6)
        state_cold['organic_nitrogen_input'] = torch.tensor(0.0)

        # Warm
        state_warm = single_state.copy()
        state_warm['soil_temperature'] = torch.tensor(298.15)
        state_warm['carbon_decomposition_rate'] = torch.tensor(1e-6)
        state_warm['organic_nitrogen_input'] = torch.tensor(0.0)

        updates_cold = functor.forward(state_cold, 0.0, 3600.0)
        updates_warm = functor.forward(state_warm, 0.0, 3600.0)

        rate_cold = updates_cold['gross_mineralization_flux'].value.item()
        rate_warm = updates_warm['gross_mineralization_flux'].value.item()
        assert rate_warm >= rate_cold, "Warm temperature should increase or maintain mineralization"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

