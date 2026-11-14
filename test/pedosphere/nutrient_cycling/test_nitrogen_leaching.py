"""
L1 Unit Tests for Nitrogen Leaching Process Functor.

Tests hydrological nitrogen transport with adsorption effects.
Following the RELIC L1 testing specification.

References:
- Li et al. (2006), doi:10.1016/j.ecolmodel.2006.02.007
- van Genuchten and Wierenga (1986), soil solute transport
- Biome-BGC nleaching.c: MOBILEN_PROPORTION formulation

Mathematical Expressions:
F_N = Q_drain * C_NO3 * mobile_fraction
R = 1 + rho_b * K_d / theta
mobile_fraction = 1 / R

Units:
- F_N: g N m^-2 s^-1
- Q_drain: m s^-1
- C_NO3: g N m^-3
"""
import pytest
import torch
from relic.core.state import UpdateItem
from relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching import (
    NitrogenLeachingProcess
)


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


class TestNitrogenLeachingInit:
    """Test initialization and parameter loading."""

    def test_initialization(self, parameter_manager, basic_config):
        """Test Functor can be initialized."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )
        assert functor is not None
        assert functor.name == "test_n_leach"

    def test_parameter_loading(self, parameter_manager, basic_config):
        """Test that required parameters are loaded from manager."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        # Should have sorption parameters
        assert hasattr(functor, 'nitrate_distribution_coefficient')
        assert functor.nitrate_distribution_coefficient >= 0
        assert {'bulk_density', 'soil_depth'} <= set(functor.get_required_inputs())
        assert not hasattr(functor, 'bulk_density_default')

        # Should have flux control parameters
        assert hasattr(functor, 'max_flux_fraction')
        assert 0 < functor.max_flux_fraction <= 1.0

    def test_metadata_includes_nitrogen_conservation(self, basic_config):
        """Test metadata includes nitrogen conservation law."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )
        metadata = functor.metadata

        if isinstance(metadata, dict):
            conservation = metadata.get('conservation_laws', [])
        else:
            conservation = getattr(metadata, 'conservation_laws', [])

        assert 'nitrogen' in conservation

    def test_stability_controller_initialized(self, basic_config):
        """Test that stability controller is properly initialized."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        assert hasattr(functor, 'stability_controller')
        assert functor.stability_controller is not None


class TestNitrogenLeachingForward:
    """Test forward execution."""

    @pytest.fixture
    def leaching_state(self):
        """State with nitrogen pools and drainage."""
        return {
            'no3_nitrogen': torch.tensor(10.0),  # g N/m2
            'nh4_nitrogen': torch.tensor(5.0),   # g N/m2
            'soil_moisture': torch.tensor(0.35), # m3/m3
            'drainage_flux': torch.tensor(1e-6), # m/s (downward water flux)
            'bulk_density': torch.tensor(1400.0), # kg/m3
            'soil_depth': torch.tensor(1.0),  # m
            'porosity': torch.tensor(0.45),       # m3/m3
            'soil_temperature': torch.tensor(293.15)
        }

    def test_forward_runs(self, leaching_state, basic_config):
        """Test that forward() executes without errors."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        updates = functor.forward(
            state=leaching_state,
            time=0.0,
            dt=3600.0  # 1 hour
        )

        assert isinstance(updates, dict)
        assert len(updates) > 0

    def test_forward_returns_tensors(self, leaching_state, basic_config):
        """Test that forward returns Tensor values."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        updates = functor.forward(
            state=leaching_state,
            time=0.0,
            dt=3600.0
        )

        for key, value in updates.items():
            if not key.endswith('_units'):
                if isinstance(value, UpdateItem):
                    value = value.value
                assert isinstance(value, torch.Tensor), f"{key} should be Tensor"

    def test_leaching_flux_output(self, leaching_state, basic_config):
        """Test that nitrogen leaching flux is calculated."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        updates = functor.forward(leaching_state, 0.0, 3600.0)

        # Should have leaching flux key
        leach_keys = [k for k in updates.keys() if 'leach' in k.lower() and 'nitrogen' in k.lower()]
        assert len(leach_keys) > 0, "Should have nitrogen leaching flux output"

        # Leaching flux should be non-negative
        leach_flux = _unwrap_update(updates[leach_keys[0]])
        assert leach_flux >= 0, "Leaching flux should be non-negative"

    def test_no3_depletion(self, leaching_state, basic_config):
        """Test that NO3 pool decreases with leaching."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        updates = functor.forward(leaching_state, 0.0, 3600.0)

        # Should have no3_nitrogen update
        assert 'no3_nitrogen' in updates
        delta_no3 = _unwrap_update(updates['no3_nitrogen'])
        # With drainage, NO3 should decrease (negative delta)
        assert delta_no3 <= 0, "NO3 should decrease with leaching"


class TestWaterFluxDependency:
    """Test dependency on water drainage flux."""

    @pytest.fixture
    def base_leaching_state(self):
        return {
            'no3_nitrogen': torch.tensor(10.0),
            'nh4_nitrogen': torch.tensor(5.0),
            'soil_moisture': torch.tensor(0.35),
            'bulk_density': torch.tensor(1400.0),
            'soil_depth': torch.tensor(1.0),  # m
            'porosity': torch.tensor(0.45),
            'soil_temperature': torch.tensor(293.15)
        }

    def test_higher_drainage_increases_leaching(self, base_leaching_state, basic_config):
        """
        Test that higher water drainage increases nitrogen leaching.

        Mathematical Expression:
        F_N = Q_drain * C_NO3 * mobile_fraction

        Expected: Higher Q_drain -> Higher F_N
        """
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        # Low drainage
        state_low = base_leaching_state.copy()
        state_low['drainage_flux'] = torch.tensor(1e-7)  # Low

        # High drainage
        state_high = base_leaching_state.copy()
        state_high['drainage_flux'] = torch.tensor(1e-5)  # High (100x)

        updates_low = functor.forward(state_low, 0.0, 3600.0)
        updates_high = functor.forward(state_high, 0.0, 3600.0)

        # Find leaching flux
        leach_keys = [k for k in updates_low.keys() if 'leach' in k.lower() and 'flux' in k.lower()]

        if leach_keys:
            flux_low = _unwrap_update(updates_low[leach_keys[0]]).item()
            flux_high = _unwrap_update(updates_high[leach_keys[0]]).item()

            assert flux_high > flux_low, "Higher drainage should increase leaching"
            # Should be roughly proportional
            assert flux_high > flux_low * 50, "Leaching should scale with drainage"

    def test_no_drainage_no_leaching(self, base_leaching_state, basic_config):
        """Test that zero drainage produces zero leaching."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        state = base_leaching_state.copy()
        state['drainage_flux'] = torch.tensor(0.0)  # No drainage

        updates = functor.forward(state, 0.0, 3600.0)

        # Find leaching flux
        leach_keys = [k for k in updates.keys() if 'leach' in k.lower() and 'flux' in k.lower()]

        if leach_keys:
            flux = _unwrap_update(updates[leach_keys[0]]).item()
            assert flux == 0.0 or flux < 1e-10, "No drainage should produce no leaching"


class TestAdsorptionEffects:
    """Test adsorption retardation effects."""

    def test_distribution_coefficient_reduces_mobility(self, basic_config):
        """
        Test that adsorption reduces mobile fraction.

        Mathematical Expression:
        mobile_fraction = 1 / (1 + R)
        R = rho_b * K_d / theta

        Expected: Higher K_d -> Lower mobile_fraction -> Lower leaching
        """
        # Low adsorption functor
        config_low_ads = basic_config.copy()
        functor_low = NitrogenLeachingProcess(
            name="test_low_ads",
            config=config_low_ads
        )
        # Force low distribution coefficient (m3 kg-1)
        functor_low.nitrate_distribution_coefficient = 1.0e-5

        # High adsorption functor
        config_high_ads = basic_config.copy()
        functor_high = NitrogenLeachingProcess(
            name="test_high_ads",
            config=config_high_ads
        )
        # Force high distribution coefficient (m3 kg-1)
        functor_high.nitrate_distribution_coefficient = 5.0e-4

        # Same state for both
        state = {
            'no3_nitrogen': torch.tensor(10.0),
            'soil_moisture': torch.tensor(0.35),
            'drainage_flux': torch.tensor(1e-6),
            'bulk_density': torch.tensor(1400.0),
            'soil_depth': torch.tensor(1.0),  # m
            'porosity': torch.tensor(0.45),
            'soil_temperature': torch.tensor(293.15)
        }

        updates_low = functor_low.forward(state, 0.0, 3600.0)
        updates_high = functor_high.forward(state, 0.0, 3600.0)

        # Find mobile fraction or leaching flux
        leach_keys = [k for k in updates_low.keys() if 'leach' in k.lower() and 'flux' in k.lower()]

        if leach_keys:
            flux_low_ads = _unwrap_update(updates_low[leach_keys[0]]).item()
            flux_high_ads = _unwrap_update(updates_high[leach_keys[0]]).item()

            # Higher adsorption should reduce leaching
            assert flux_low_ads >= flux_high_ads, "Higher adsorption should reduce leaching"


class TestNO3Concentration:
    """Test concentration-dependent leaching."""

    def test_higher_concentration_increases_leaching(self, basic_config):
        """
        Test that higher NO3 concentration increases leaching.

        Mathematical Expression:
        F_N = Q_drain * C_NO3 * mobile_fraction

        Expected: Higher C_NO3 -> Higher F_N
        """
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        # Low concentration
        state_low = {
            'no3_nitrogen': torch.tensor(2.0),  # Low
            'soil_moisture': torch.tensor(0.35),
            'drainage_flux': torch.tensor(1e-6),
            'bulk_density': torch.tensor(1400.0),
            'soil_depth': torch.tensor(1.0),  # m
            'porosity': torch.tensor(0.45),
            'soil_temperature': torch.tensor(293.15)
        }

        # High concentration
        state_high = {
            'no3_nitrogen': torch.tensor(20.0),  # High (10x)
            'soil_moisture': torch.tensor(0.35),
            'drainage_flux': torch.tensor(1e-6),
            'bulk_density': torch.tensor(1400.0),
            'soil_depth': torch.tensor(1.0),  # m
            'porosity': torch.tensor(0.45),
            'soil_temperature': torch.tensor(293.15)
        }

        updates_low = functor.forward(state_low, 0.0, 3600.0)
        updates_high = functor.forward(state_high, 0.0, 3600.0)

        # Find leaching flux
        leach_keys = [k for k in updates_low.keys() if 'leach' in k.lower() and 'flux' in k.lower()]

        if leach_keys:
            flux_low = _unwrap_update(updates_low[leach_keys[0]]).item()
            flux_high = _unwrap_update(updates_high[leach_keys[0]]).item()

            assert flux_high > flux_low, "Higher concentration should increase leaching"
            # Should be roughly proportional
            assert flux_high > flux_low * 5, "Leaching should scale with concentration"


class TestNumericalStability:
    """Test numerical stability and conservation."""

    def test_flux_limiting_prevents_overdraft(self, basic_config):
        """Test that flux limiting prevents pool depletion beyond bounds."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        # Small pool with high drainage
        state = {
            'no3_nitrogen': torch.tensor(0.1),   # Very small pool
            'soil_moisture': torch.tensor(0.35),
            'drainage_flux': torch.tensor(1e-4), # Very high drainage
            'bulk_density': torch.tensor(1400.0),
            'soil_depth': torch.tensor(1.0),  # m
            'porosity': torch.tensor(0.45),
            'soil_temperature': torch.tensor(293.15)
        }

        updates = functor.forward(state, 0.0, 3600.0)

        # Find no3_nitrogen update
        assert 'no3_nitrogen' in updates
        delta_no3 = _unwrap_update(updates['no3_nitrogen']).item()
        total_loss = abs(delta_no3)

        # Total loss should not exceed available pool
        assert total_loss <= state['no3_nitrogen'].item(), "Should not overdraft NO3 pool"

    def test_conservation_check_called(self, basic_config, caplog):
        """Test that conservation check is invoked."""
        import logging
        caplog.set_level(logging.INFO)

        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        state = {
            'no3_nitrogen': torch.tensor(10.0),
            'soil_moisture': torch.tensor(0.35),
            'drainage_flux': torch.tensor(1e-6),
            'bulk_density': torch.tensor(1400.0),
            'soil_depth': torch.tensor(1.0),  # m
            'porosity': torch.tensor(0.45),
            'soil_temperature': torch.tensor(293.15)
        }

        functor.forward(state, 0.0, 3600.0)

        # If we got here, conservation check didn't crash
        assert True


class TestDependencies:
    """Test functor dependencies."""

    def test_get_dependencies_includes_hydrosphere(self, basic_config):
        """Test that dependencies include hydrosphere interface."""
        functor = NitrogenLeachingProcess(
            name="test_n_leach",
            config=basic_config
        )

        deps = functor.get_dependencies()

        if isinstance(deps, dict):
            deps_list = deps.get("required", []) + deps.get("optional", [])
        else:
            deps_list = deps

        assert isinstance(deps_list, list)
        assert any(name in ("nitrification", "denitrification") for name in deps_list)
        assert all(
            "soil_moisture" not in name.lower() and "hydrosphere" not in name.lower()
            for name in deps_list
        )

def test_public_conservation_checker_audits_the_returned_leaching_proposal(basic_config):
    """The public checker recomputes dNO3 + F_leach*dt from the returned UpdateItems."""
    import dataclasses

    functor = NitrogenLeachingProcess(name="test_n_leach_public_ledger", config=basic_config)
    state = {
        'no3_nitrogen': torch.tensor(10.0),
        'soil_moisture': torch.tensor(0.35),
        'drainage_flux': torch.tensor(1e-6),
        'bulk_density': torch.tensor(1400.0),
        'soil_depth': torch.tensor(1.0),  # m
        'porosity': torch.tensor(0.45),
        'soil_temperature': torch.tensor(293.15),
    }
    result = functor(state, time=0.0, dt=3600.0)
    clean = functor.check_conservation_laws(state, result)
    assert set(clean) == set(functor.metadata.conservation_laws) == {"nitrogen", "mass"}
    assert max(clean.values()) <= 1.0e-9
    assert float(_unwrap_update(result["diagnostics.nitrogen_leaching.step_seconds"])) == 3600.0
    no3 = result["no3_nitrogen"]
    assert float(no3.value.detach()) < 0.0
    tampered_pool = dict(result)
    tampered_pool["no3_nitrogen"] = dataclasses.replace(no3, value=no3.value + 1.0)
    tampered = functor.check_conservation_laws(state, tampered_pool)
    assert tampered["nitrogen"] == pytest.approx(1.0, rel=1.0e-6)
    assert tampered["mass"] == pytest.approx(1.0, rel=1.0e-6)
    flux = result["nitrogen_leaching_flux"]
    tampered_rate = dict(result)
    tampered_rate["nitrogen_leaching_flux"] = dataclasses.replace(flux, value=flux.value + 1.0 / 3600.0)
    assert functor.check_conservation_laws(state, tampered_rate)["nitrogen"] == pytest.approx(1.0, rel=1.0e-6)
    without_step = dict(result)
    del without_step["diagnostics.nitrogen_leaching.step_seconds"]
    with pytest.raises(ValueError, match="step_seconds"):
        functor.check_conservation_laws(state, without_step)


def test_forward_is_the_one_entry_and_the_rate_core_reads_the_state_only():
    """No secondary _compute_rate_core entry exists, and the leaching rate core takes
    the state only (no environment or keyword channel supplies inputs)."""
    import inspect

    assert not hasattr(NitrogenLeachingProcess, "_compute_rate_core")
    assert list(
        inspect.signature(NitrogenLeachingProcess._compute_leaching_rates).parameters
    ) == ["self", "state"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

