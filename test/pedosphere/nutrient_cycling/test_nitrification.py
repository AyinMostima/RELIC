"""
L1 Unit Tests for Nitrification Process Functor.

Tests pH response, temperature sensitivity, and NH4 -> NO3 conversion.
"""
import pytest
import torch
import inspect
from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
from relic.core.state import UpdateItem


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


class TestNitrificationInit:
    """Test initialization."""

    def test_initialization(self, basic_config):
        """Test Functor can be initialized."""
        functor = NitrificationProcess(name="test_nit", config=basic_config)
        assert functor is not None
        assert functor.name == "test_nit"

    def test_metadata_conservation_laws(self, basic_config):
        """Test metadata includes nitrogen conservation."""
        functor = NitrificationProcess(name="test_nit", config=basic_config)
        metadata = functor.metadata
        if isinstance(metadata, dict):
            conservation = metadata.get('conservation_laws', [])
        else:
            conservation = getattr(metadata, 'conservation_laws', [])
        assert 'nitrogen' in conservation

    def test_phase8_runtime_uses_explicit_reaction_network(self, basic_config):
        """Test Phase 8 runtime advances state through explicit N_h r_h."""
        functor = NitrificationProcess(name="test_nit", config=basic_config)
        source = inspect.getsource(NitrificationProcess.forward)

        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "advance_reaction_network(" in source
        assert "stoichiometric_N=" in source
        assert "x_dot = N_h r_h" in source


class TestNitrificationForward:
    """Test forward execution."""

    def test_forward_runs(self, single_state, basic_config):
        """Test forward executes without errors."""
        functor = NitrificationProcess(name="test_nit", config=basic_config)
        updates = functor.forward(state=single_state, time=0.0, dt=3600.0)
        assert isinstance(updates, dict)
        assert len(updates) > 0

    def test_nh4_to_no3_conversion(self, single_state, basic_config):
        """Test that NH4 is converted to NO3."""
        functor = NitrificationProcess(name="test_nit", config=basic_config)
        updates = functor.forward(state=single_state, time=0.0, dt=3600.0)

        # Should have NH4 decrease and NO3 increase
        if 'nh4_nitrogen' in updates and 'no3_nitrogen' in updates:
            delta_nh4 = _unwrap_update(updates['nh4_nitrogen']).item()
            delta_no3 = _unwrap_update(updates['no3_nitrogen']).item()
            assert delta_nh4 <= 0, "NH4 should decrease"
            assert delta_no3 >= 0, "NO3 should increase"


class TestNitrificationTemperatureSensitivity:
    """Test temperature response."""

    def test_temperature_response(self, single_state, basic_config):
        """Test nitrification increases with temperature."""
        functor = NitrificationProcess(name="test_nit", config=basic_config)

        # Cold
        state_cold = single_state.copy()
        state_cold['soil_temperature'] = torch.tensor(278.15)  # 5degC

        # Warm
        state_warm = single_state.copy()
        state_warm['soil_temperature'] = torch.tensor(298.15)  # 25degC

        updates_cold = functor.forward(state_cold, 0.0, 3600.0)
        updates_warm = functor.forward(state_warm, 0.0, 3600.0)

        if 'nitrification_flux' in updates_cold and 'nitrification_flux' in updates_warm:
            rate_cold = _unwrap_update(updates_cold['nitrification_flux']).item()
            rate_warm = _unwrap_update(updates_warm['nitrification_flux']).item()
            assert rate_warm > rate_cold, "Warm temperature should increase nitrification"


class TestNitrificationPHResponse:
    """Test pH sensitivity."""

    def test_ph_optimum(self, single_state, basic_config):
        """Test pH affects nitrification rate."""
        functor = NitrificationProcess(name="test_nit", config=basic_config)

        # Acidic
        state_acidic = single_state.copy()
        state_acidic['soil_ph'] = torch.tensor(5.0)

        # Optimal
        state_optimal = single_state.copy()
        state_optimal['soil_ph'] = torch.tensor(7.5)

        updates_acidic = functor.forward(state_acidic, 0.0, 3600.0)
        updates_optimal = functor.forward(state_optimal, 0.0, 3600.0)

        if 'nitrification_flux' in updates_acidic and 'nitrification_flux' in updates_optimal:
            rate_acidic = _unwrap_update(updates_acidic['nitrification_flux']).item()
            rate_optimal = _unwrap_update(updates_optimal['nitrification_flux']).item()
            assert rate_optimal >= rate_acidic, "Optimal pH should have higher nitrification"

class TestNitrificationOxygenLimitation:
    """Round 53: water-filled pore space drives the oxygen limitation of nitrification."""

    def test_oxygen_factor_follows_water_filled_pore_space(self, single_state, basic_config):
        """f_O2 = 1 for WFPS <= WFPS_opt and max((1 - WFPS)/(1 - WFPS_opt), f_O2_min) above.

        Linn and Doran (1984): aerobic microbial activity peaks at 60 percent WFPS and
        declines above it through reduced aeration; the linear decline to saturation and
        the floor f_O2_min are RELIC conventions.
        """
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        optimum = float(manager.get("pedosphere.physical.wfps_nitrification_optimum"))
        floor = float(manager.get("pedosphere.nitrification.minimum_oxygen_factor"))
        assert optimum == pytest.approx(0.6)
        functor = NitrificationProcess(name="test_nit", config=basic_config)
        assert "water_filled_pore_space" in functor.metadata.input_variables
        assert "soil_moisture" not in functor.metadata.input_variables
        wfps = torch.tensor([0.2, 0.6, 0.8, 0.95, 1.0], dtype=torch.float64)
        state = dict(single_state)
        state["water_filled_pore_space"] = wfps
        state["nh4_nitrogen"] = torch.full_like(wfps, 5.0)
        rates = functor._compute_rate_core(state)
        expected = torch.where(
            wfps <= optimum,
            torch.ones_like(wfps),
            torch.clamp((1.0 - wfps) / (1.0 - optimum), min=floor),
        )
        assert torch.allclose(rates["oxygen_factor"].to(torch.float64), expected, atol=1.0e-6)
        assert expected.tolist() == pytest.approx([1.0, 1.0, 0.5, 0.125, 0.1])
        # Volumetric soil moisture no longer stands in for the pore saturation.
        del state["water_filled_pore_space"]
        with pytest.raises(ValueError, match="water_filled_pore_space"):
            functor._compute_rate_core(state)

    @pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
    def test_water_filled_pore_space_outside_the_unit_interval_raises(
        self, single_state, basic_config, value
    ):
        functor = NitrificationProcess(name="test_nit", config=basic_config)
        state = dict(single_state)
        state["water_filled_pore_space"] = torch.tensor(value)
        with pytest.raises(ValueError, match="water_filled_pore_space"):
            functor.forward(state, 0.0, 3600.0)

    def test_certificate_declares_the_water_filled_pore_space_input(self):
        import yaml
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        certificate = yaml.safe_load(
            (
                root / "docs" / "source" / "theory" / "iphs_certificates"
                / "pedosphere_nutrient_cycling_nitrification_nitrification_process.yaml"
            ).read_text(encoding="utf-8")
        )
        for inputs in (
            certificate["input_variables"],
            certificate["interface_map"]["legacy_inputs"],
            certificate["interface_map"]["iphs_inputs"],
        ):
            assert "water_filled_pore_space" in inputs
            assert "soil_moisture" not in inputs
        assert "pedosphere.physical.wfps_nitrification_optimum" in certificate["parameter_keys"]
        for removed in (
            "pedosphere.nitrification.moisture_threshold",
            "pedosphere.nitrification.oxygen_limitation_factor",
        ):
            assert removed not in certificate["parameter_keys"]
            assert removed not in certificate["parameter_map"]


class TestNitrificationAmmoniumEntry:
    """Round 53.3: a negative or non-finite NH4 pool raises at the forward entry."""

    @pytest.mark.parametrize("value", [-0.1, float("nan")])
    def test_invalid_ammonium_pool_raises(self, single_state, basic_config, value):
        functor = NitrificationProcess(name="test_nit", config=basic_config)
        state = dict(single_state)
        state["nh4_nitrogen"] = torch.tensor(value)
        with pytest.raises(ValueError, match="nh4_nitrogen"):
            functor.forward(state, 0.0, 3600.0)


def test_public_conservation_checker_audits_the_returned_nitrification_proposal(single_state, basic_config):
    """The public checker recomputes dNH4 + dNO3 + F_N2O*dt from the returned UpdateItems."""
    import dataclasses

    functor = NitrificationProcess(name="test_nit_public_ledger", config=basic_config)
    result = functor(single_state, time=0.0, dt=3600.0)
    clean = functor.check_conservation_laws(single_state, result)
    assert set(clean) == set(functor.metadata.conservation_laws)
    assert clean["nitrogen"] <= 1.0e-9
    assert float(_unwrap_update(result["diagnostics.nitrification.step_seconds"])) == 3600.0
    nh4 = result["nh4_nitrogen"]
    assert float(nh4.value.detach()) < 0.0
    tampered_pool = dict(result)
    tampered_pool["nh4_nitrogen"] = dataclasses.replace(nh4, value=nh4.value + 1.0)
    assert functor.check_conservation_laws(single_state, tampered_pool)["nitrogen"] == pytest.approx(1.0, rel=1.0e-6)
    n2o = result["n2o_flux"]
    tampered_rate = dict(result)
    tampered_rate["n2o_flux"] = dataclasses.replace(n2o, value=n2o.value + 1.0 / 3600.0)
    assert functor.check_conservation_laws(single_state, tampered_rate)["nitrogen"] == pytest.approx(1.0, rel=1.0e-6)
    without_step = dict(result)
    del without_step["diagnostics.nitrification.step_seconds"]
    with pytest.raises(ValueError, match="step_seconds"):
        functor.check_conservation_laws(single_state, without_step)
    with pytest.raises(ValueError, match="requires outputs"):
        functor.check_conservation_laws(single_state, {"nh4_nitrogen": nh4})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

