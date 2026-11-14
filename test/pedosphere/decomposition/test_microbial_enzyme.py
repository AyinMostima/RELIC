"""
Unit tests for Microbial-Enzyme Decomposition Process

Tests the microbial-enzyme decomposition functor including:
- Initialization and parameter loading
- Enzyme kinetics (Michaelis-Menten)
- Microbial biomass growth dynamics
- Substrate limitation effects
- Temperature response (Q10)
- Carbon Use Efficiency (CUE)
- Enzyme allocation patterns
- Conservation laws (carbon/mass)

References:
- Wieder et al. (2013) MIMICS framework
- Allison et al. (2010) Microbial trait-based decomposition
- Schimel & Weintraub (2003) Enzyme activity implications

Mathematical Expressions:
V = (MIC * Vmax(T,theta) * [S]) / (Km(T) + MIC)
dMIC/dt = CUE * V - tau_mic * MIC
"""

import inspect
import math
from pathlib import Path
import warnings

import pytest
import torch
import yaml

from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.pedosphere.decomposition.microbial_enzyme import MicrobialEnzymeDecomposition


STATE_OUTPUTS = {
    "metabolic_litter_c",
    "structural_litter_c",
    "available_som_c",
    "microbial_biomass_c",
}
ENZYME_DIAGNOSTICS = {
    f"diagnostics.microbial_enzyme.{substrate}.{quantity}"
    for substrate in ("metabolic_litter", "structural_litter", "som")
    for quantity in ("vmax", "km", "activity")
}
CANONICAL_OUTPUTS = STATE_OUTPUTS | {
    "heterotrophic_respiration",
    "diagnostics.microbial_enzyme.microbial_growth_rate",
    "diagnostics.microbial_enzyme.microbial_death_rate",
    "diagnostics.microbial_enzyme.microbial_respiration",
    "diagnostics.microbial_enzyme.growth_respiration",
    "diagnostics.microbial_enzyme.maintenance_respiration",
    "diagnostics.microbial_enzyme.metabolic_consumption",
    "diagnostics.microbial_enzyme.structural_consumption",
    "diagnostics.microbial_enzyme.som_consumption",
    "diagnostics.microbial_enzyme.moisture_response",
    "diagnostics.microbial_enzyme.temperature_celsius",
    "diagnostics.microbial_enzyme.microbial_biomass_content",
    "diagnostics.microbial_enzyme.total_substrate_consumed",
    "diagnostics.microbial_enzyme.carbon_use_efficiency_realized",
    "diagnostics.microbial_enzyme.carbon_balance_residual",
    # Declared step length read by the public carbon-rate ledger (round 47).
    "diagnostics.microbial_enzyme.step_seconds",
} | ENZYME_DIAGNOSTICS


def _value(output, key):
    item = output[key]
    return item.value if isinstance(item, UpdateItem) else item


def _canonical_state(dtype=torch.float64):
    return {
        "metabolic_litter_c": torch.tensor(0.5, dtype=dtype),
        "structural_litter_c": torch.tensor(0.5, dtype=dtype),
        "available_som_c": torch.tensor(1.0, dtype=dtype),
        "microbial_biomass_c": torch.tensor(0.15, dtype=dtype),
        "soil_temperature": torch.tensor(298.15, dtype=dtype),
        "soil_moisture": torch.tensor(0.55, dtype=dtype),
    }


class TestCanonicalRuntimeContract:
    """Test the exact state, flux, diagnostic, and fail-fast contract."""

    def test_all_runtime_parameters_have_complete_governance(self):
        manager = get_global_manager()
        required_fields = (
            "units",
            "dtype",
            "bounds",
            "category",
            "description",
            "references",
            "tags",
            "requires_grad",
            "provenance",
        )

        assert len(MicrobialEnzymeDecomposition.PARAMETER_DESCRIPTIONS) == 25
        project_root = Path(__file__).resolve().parents[3]
        raw_parameters = {}
        for relative_path in (
            Path("relic/parameters/database/relic/pedosphere/decomposition.yaml"),
            Path("relic/parameters/database/relic/pedosphere/numerical.yaml"),
        ):
            payload = yaml.safe_load(
                (project_root / relative_path).read_text(encoding="utf-8")
            )
            raw_parameters.update(payload["parameters"])

        for key in MicrobialEnzymeDecomposition.PARAMETER_DESCRIPTIONS:
            spec = manager.get_parameter(key)
            assert spec is not None, key
            for field in required_fields:
                assert getattr(spec, field) is not None, (key, field)
            assert all(math.isfinite(float(limit)) for limit in spec.bounds), key
            assert spec.bounds[0] <= spec.value <= spec.bounds[1], key
            assert "default" not in raw_parameters[key], key

    def test_exact_inputs_outputs_and_signature(self, basic_config):
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)
        assert set(functor.metadata.input_variables) == {
            "metabolic_litter_c",
            "structural_litter_c",
            "available_som_c",
            "microbial_biomass_c",
            "soil_temperature",
            "soil_moisture",
        }
        assert set(functor.metadata.output_variables) == CANONICAL_OUTPUTS
        assert inspect.signature(functor.forward).parameters.keys() == {
            "state",
            "time",
            "dt",
        }

        result = functor.forward(_canonical_state(), time=0.0, dt=1800.0)
        assert set(result) == CANONICAL_OUTPUTS
        assert set(result) & ENZYME_DIAGNOSTICS == ENZYME_DIAGNOSTICS
        assert not any(key.startswith("diagnostics.enzyme_") for key in result)

    def test_state_and_flux_update_items_are_explicit(self, basic_config):
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)
        result = functor.forward(_canonical_state(), time=0.0, dt=1800.0)

        for key in STATE_OUTPUTS:
            item = result[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == "g C m-2"
            assert item.source_functor == functor.name

        respiration = result["heterotrophic_respiration"]
        assert isinstance(respiration, UpdateItem)
        assert respiration.mode is UpdateMode.REPLACE
        assert respiration.per_timestep is False
        assert respiration.unit == "g C m-2 s-1"
        assert respiration.source_functor == functor.name

    def test_canonical_flux_specification(self, basic_config):
        """Test the single non-diagnostic boundary-flux specification."""
        functor = MicrobialEnzymeDecomposition(
            name="test_mic_enzyme",
            config=basic_config,
        )
        specifications = functor.get_output_specifications()

        assert set(specifications) == {"heterotrophic_respiration"}
        assert specifications["heterotrophic_respiration"].units == "g C m-2 s-1"
        assert specifications["heterotrophic_respiration"].is_rate is True

    @pytest.mark.parametrize(
        "missing_key",
        [
            "metabolic_litter_c",
            "structural_litter_c",
            "available_som_c",
            "microbial_biomass_c",
            "soil_temperature",
            "soil_moisture",
        ],
    )
    def test_required_inputs_fail_fast_without_basis_fallback(self, basic_config, missing_key):
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)
        state = _canonical_state()
        state.update(
            {
                "fast_som_c": torch.tensor(5.0),
                "slow_som_c": torch.tensor(6.0),
                "passive_som_c": torch.tensor(7.0),
            }
        )
        state.pop(missing_key)
        with pytest.raises(KeyError, match=missing_key):
            functor.forward(state, time=0.0, dt=1800.0)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("metabolic_litter_c", -1.0),
            ("microbial_biomass_c", float("nan")),
            ("soil_moisture", 1.01),
            ("soil_temperature", float("inf")),
        ],
    )
    def test_invalid_state_fails_before_calculation(self, basic_config, key, value):
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)
        state = _canonical_state()
        state[key] = torch.tensor(value, dtype=torch.float64)
        with pytest.raises((TypeError, ValueError), match=key):
            functor.forward(state, time=0.0, dt=1800.0)

    @pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_dt_fails_before_calculation(self, basic_config, dt):
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)
        with pytest.raises(ValueError, match="dt"):
            functor.forward(_canonical_state(), time=0.0, dt=dt)

    def test_fixed_reference_numerics_and_exact_carbon_ledger(self, basic_config):
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)
        dt = 1800.0
        result = functor.forward(_canonical_state(), time=0.0, dt=dt)

        expected = {
            "metabolic_litter_c": -0.004218749853599466,
            "structural_litter_c": -0.002025000042136922,
            "available_som_c": -0.0031764705541443053,
            "microbial_biomass_c": 0.004549701634277443,
            "heterotrophic_respiration": 2.7058437864462506e-06,
        }
        for key, value in expected.items():
            assert torch.allclose(
                _value(result, key),
                torch.tensor(value, dtype=torch.float64),
                rtol=1.0e-10,
                atol=1.0e-14,
            )

        ledger = sum(_value(result, key) for key in STATE_OUTPUTS)
        ledger = ledger + _value(result, "heterotrophic_respiration") * dt
        assert torch.allclose(ledger, torch.zeros_like(ledger), atol=1.0e-12, rtol=0.0)
        residual = _value(result, "diagnostics.microbial_enzyme.carbon_balance_residual")
        assert torch.equal(residual, ledger)


class TestMicrobialEnzymeInit:
    """Test initialization and setup"""

    def test_initialization(self, basic_config):
        """Test that MicrobialEnzymeDecomposition can be instantiated"""
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)
        assert functor is not None
        assert functor.name == "test_mic_enzyme"

    def test_parameter_loading(self, parameter_manager, basic_config):
        """Test that parameters are loaded from ParameterManager"""
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)

        # Check key parameters are loaded
        assert hasattr(functor, 'kinetic_params')
        assert hasattr(functor, 'microbial_params')
        assert functor.kinetic_params is not None
        assert functor.microbial_params is not None

        # Check substrate-specific parameters
        assert 'metabolic_litter' in functor.kinetic_params
        assert 'structural_litter' in functor.kinetic_params
        assert 'som' in functor.kinetic_params

    def test_metadata_conservation_laws(self, basic_config):
        """Test that carbon conservation is declared in metadata"""
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)

        assert hasattr(functor.metadata, 'conservation_laws')
        assert 'carbon' in functor.metadata.conservation_laws


class TestEnzymeKinetics:
    """Test Michaelis-Menten enzyme kinetics"""

    def test_enzyme_kinetics_michaelis_menten(self, basic_config, single_state):
        """
        Test that enzyme kinetics follow Michaelis-Menten equation.
        
        Mathematical Expression:
        V = (MIC * Vmax * [S]) / (Km + MIC)
        
        Expected behavior for reverse Michaelis-Menten kinetics:
        - At fixed microbial biomass: rate is proportional to substrate
        - At high microbial biomass: rate approaches Vmax * substrate
        - At MIC = Km: rate = 0.5 * Vmax * substrate
        """
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)

        # Test with varying substrate concentrations
        base_state = single_state.copy()
        base_state['microbial_biomass_c'] = torch.tensor(0.1)  # Fixed biomass
        base_state['structural_litter_c'] = torch.tensor(0.1)  # Required state variable
        base_state['available_som_c'] = torch.tensor(0.1)  # Required state variable

        # Low substrate case
        state_low_s = base_state.copy()
        state_low_s['metabolic_litter_c'] = torch.tensor(0.01)
        result_low = functor.forward(state_low_s, time=0.0, dt=1800.0)

        # High substrate case
        state_high_s = base_state.copy()
        state_high_s['metabolic_litter_c'] = torch.tensor(1.0)
        result_high = functor.forward(state_high_s, time=0.0, dt=1800.0)

        # Extract consumption rates
        consumption_low = _value(result_low, 'diagnostics.microbial_enzyme.metabolic_consumption')
        consumption_high = _value(result_high, 'diagnostics.microbial_enzyme.metabolic_consumption')

        # At higher substrate, consumption should be higher
        assert float(consumption_high) > float(consumption_low)

        # Check that consumption is substrate-dependent
        # At low [S], rate should be roughly proportional to [S]
        substrate_ratio = float(state_high_s['metabolic_litter_c'] / state_low_s['metabolic_litter_c'])
        rate_ratio = float(consumption_high / consumption_low)
        
        # Reverse Michaelis-Menten is linear in substrate at fixed biomass.
        assert rate_ratio == pytest.approx(substrate_ratio, rel=1.0e-5)


class TestMicrobialBiomassGrowth:
    """Test microbial biomass dynamics"""

    def test_microbial_biomass_growth(self, basic_config, single_state):
        """
        Test microbial biomass growth dynamics.
        
        Mathematical Expression:
        dMIC/dt = CUE * V - tau_mic * MIC
        
        Where:
        - CUE: Carbon Use Efficiency (0.3-0.6)
        - V: Substrate consumption rate
        - tau_mic: Microbial turnover rate
        """
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)

        # Setup state with substrate and microbial biomass
        state = single_state.copy()
        state['microbial_biomass_c'] = torch.tensor(0.1)
        state['metabolic_litter_c'] = torch.tensor(0.5)
        state['structural_litter_c'] = torch.tensor(0.5)
        state['available_som_c'] = torch.tensor(1.0)

        dt = 1800.0
        result = functor.forward(state, time=0.0, dt=dt)

        # Check that biomass change is calculated
        assert 'microbial_biomass_c' in result
        assert 'diagnostics.microbial_enzyme.microbial_growth_rate' in result
        assert 'diagnostics.microbial_enzyme.microbial_death_rate' in result

        # Extract rates
        growth_rate = _value(result, 'diagnostics.microbial_enzyme.microbial_growth_rate')
        death_rate = _value(result, 'diagnostics.microbial_enzyme.microbial_death_rate')
        net_change = _value(result, 'microbial_biomass_c')

        # Verify relationship: net_change = (growth - death) * dt
        expected_net = (growth_rate - death_rate) * dt
        assert torch.allclose(net_change, expected_net, rtol=1e-3)

        # Growth should be positive when substrate is available
        assert float(growth_rate) > 0


class TestSubstrateLimitation:
    """Test substrate limitation effects"""

    def test_substrate_limitation(self, basic_config, single_state):
        """
        Test that decomposition depends on substrate availability.
        
        Expected behavior:
        - Low substrate -> low consumption rate
        - High substrate -> higher consumption rate (saturation)
        - Zero substrate -> zero substrate consumption (but microbial death still occurs)
        
        Note: total_substrate_consumed includes substrate + microbial death,
        so we test individual substrate consumption rates instead.
        """
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)

        # Base state with adequate microbial biomass
        base_state = single_state.copy()
        base_state['microbial_biomass_c'] = torch.tensor(0.2)

        # Zero substrate case
        state_zero = base_state.copy()
        state_zero['metabolic_litter_c'] = torch.tensor(0.0)
        state_zero['structural_litter_c'] = torch.tensor(0.0)
        state_zero['available_som_c'] = torch.tensor(0.0)
        result_zero = functor.forward(state_zero, time=0.0, dt=1800.0)

        # Low substrate case
        state_low = base_state.copy()
        state_low['metabolic_litter_c'] = torch.tensor(0.01)
        state_low['structural_litter_c'] = torch.tensor(0.02)
        state_low['available_som_c'] = torch.tensor(0.05)
        result_low = functor.forward(state_low, time=0.0, dt=1800.0)

        # High substrate case
        state_high = base_state.copy()
        state_high['metabolic_litter_c'] = torch.tensor(1.0)
        state_high['structural_litter_c'] = torch.tensor(2.0)
        state_high['available_som_c'] = torch.tensor(5.0)
        result_high = functor.forward(state_high, time=0.0, dt=1800.0)

        # Extract substrate consumption rates (not total which includes death)
        consumption_zero = sum(
            _value(result_zero, f'diagnostics.microbial_enzyme.{name}_consumption')
            for name in ('metabolic', 'structural', 'som')
        )
        consumption_low = sum(
            _value(result_low, f'diagnostics.microbial_enzyme.{name}_consumption')
            for name in ('metabolic', 'structural', 'som')
        )
        consumption_high = sum(
            _value(result_high, f'diagnostics.microbial_enzyme.{name}_consumption')
            for name in ('metabolic', 'structural', 'som')
        )

        # Verify substrate limitation
        # Zero substrate should give zero consumption (within numerical tolerance)
        assert float(consumption_zero) < 1e-9
        # More substrate should give more consumption
        assert float(consumption_low) > 0
        assert float(consumption_high) > float(consumption_low)


class TestTemperatureResponse:
    """Test temperature sensitivity"""

    def test_temperature_response_q10(self, basic_config, single_state):
        """
        Test temperature response follows Q10 relationship.
        
        Mathematical Expression:
        Vmax(T) = Vmax_ref * exp(Ea/R * (1/T_ref - 1/T))
        
        Expected Q10 ~= 2.0 for enzyme activity
        
        References:
        - Davidson & Janssens (2006) Temperature sensitivity
        - Arrhenius temperature response
        """
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)

        # Base state
        base_state = single_state.copy()
        base_state['microbial_biomass_c'] = torch.tensor(0.15)
        base_state['metabolic_litter_c'] = torch.tensor(0.5)
        base_state['structural_litter_c'] = torch.tensor(0.5)
        base_state['available_som_c'] = torch.tensor(1.0)

        # Cold temperature (15 degC = 288.15 K)
        state_cold = base_state.copy()
        state_cold['soil_temperature'] = torch.tensor(288.15)
        result_cold = functor.forward(state_cold, time=0.0, dt=1800.0)

        # Warm temperature (25 degC = 298.15 K)
        state_warm = base_state.copy()
        state_warm['soil_temperature'] = torch.tensor(298.15)
        result_warm = functor.forward(state_warm, time=0.0, dt=1800.0)

        # Extract respiration rates (proxy for total activity)
        resp_cold = _value(result_cold, 'diagnostics.microbial_enzyme.microbial_respiration')
        resp_warm = _value(result_warm, 'diagnostics.microbial_enzyme.microbial_respiration')

        # Calculate Q10
        # Q10 = (R2/R1)^(10/(T2-T1))
        temp_diff = 10.0  # 10 degree difference
        rate_ratio = float(resp_warm / (resp_cold + 1e-12))
        
        # Q10 should be around 2.0 (typical for biological processes)
        # Allow range 1.5-3.0 for model variability
        assert 1.5 < rate_ratio < 3.0

        # Warmer temperature should always increase rate
        assert float(resp_warm) > float(resp_cold)


class TestCarbonUseEfficiency:
    """Test Carbon Use Efficiency (CUE)"""

    def test_cue_calculation(self, basic_config, single_state):
        """
        Test Carbon Use Efficiency is in expected range.
        
        Mathematical Expression:
        CUE = Growth / (Growth + Respiration)
        
        Expected range: 0.3-0.6 for soil microbes
        
        References:
        - Manzoni et al. (2012) Environmental controls on CUE
        - Sinsabaugh et al. (2013) Carbon use efficiency
        """
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)

        # Setup state with moderate substrate
        state = single_state.copy()
        state['microbial_biomass_c'] = torch.tensor(0.15)
        state['metabolic_litter_c'] = torch.tensor(0.5)
        state['structural_litter_c'] = torch.tensor(0.5)
        state['available_som_c'] = torch.tensor(1.0)

        result = functor.forward(state, time=0.0, dt=1800.0)

        # Extract realized CUE
        realized_cue = _value(result, 'diagnostics.microbial_enzyme.carbon_use_efficiency_realized')

        # CUE should be in realistic range (0.3-0.6)
        assert 0.2 < float(realized_cue) < 0.7

        # Check relationship with growth and respiration
        growth = _value(result, 'diagnostics.microbial_enzyme.microbial_growth_rate')
        respiration = _value(result, 'diagnostics.microbial_enzyme.growth_respiration')

        # CUE = growth / (growth + respiration)
        expected_cue = growth / (growth + respiration + 1e-10)
        assert torch.allclose(realized_cue, expected_cue, rtol=0.1)


class TestEnzymeAllocation:
    """Test enzyme production and allocation"""

    def test_enzyme_allocation(self, basic_config, single_state):
        """
        Test enzyme allocation across substrate types.
        
        Expected behavior:
        - Different enzyme activities for different substrates
        - Metabolic litter: highest Vmax
        - Structural litter: intermediate Vmax
        - SOM: lowest Vmax
        """
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)

        # Setup equal substrate concentrations
        state = single_state.copy()
        state['microbial_biomass_c'] = torch.tensor(0.15)
        state['metabolic_litter_c'] = torch.tensor(1.0)
        state['structural_litter_c'] = torch.tensor(1.0)
        state['available_som_c'] = torch.tensor(1.0)

        result = functor.forward(state, time=0.0, dt=1800.0)

        # Extract consumption rates
        met_consumption = _value(result, 'diagnostics.microbial_enzyme.metabolic_consumption')
        struct_consumption = _value(result, 'diagnostics.microbial_enzyme.structural_consumption')
        som_consumption = _value(result, 'diagnostics.microbial_enzyme.som_consumption')

        # With equal substrate, metabolic should have highest rate (highest quality)
        assert float(met_consumption) > float(struct_consumption)
        # SOM typically has intermediate rates
        assert float(som_consumption) > 0


class TestConservation:
    """Test conservation laws"""

    def test_conservation_check(self, basic_config, single_state):
        """
        Test that carbon is conserved in decomposition.
        
        Mathematical Expression:
        delta_substrates + delta_MIC + respiration = 0
        
        Conservation law:
        Sum(consumption) = growth + respiration
        """
        functor = MicrobialEnzymeDecomposition(name="test_mic_enzyme", config=basic_config)

        # Setup realistic state
        state = single_state.copy()
        state['microbial_biomass_c'] = torch.tensor(0.15)
        state['metabolic_litter_c'] = torch.tensor(0.5)
        state['structural_litter_c'] = torch.tensor(0.5)
        state['available_som_c'] = torch.tensor(1.0)

        result = functor.forward(state, time=0.0, dt=1800.0)

        # Extract fluxes
        total_consumption = _value(result, 'diagnostics.microbial_enzyme.total_substrate_consumed')
        growth = _value(result, 'diagnostics.microbial_enzyme.microbial_growth_rate')
        respiration = _value(result, 'diagnostics.microbial_enzyme.microbial_respiration')

        growth_respiration = _value(
            result,
            'diagnostics.microbial_enzyme.growth_respiration',
        )
        death = _value(result, 'diagnostics.microbial_enzyme.microbial_death_rate')

        # Substrate uptake partitions into microbial growth and growth respiration.
        assert torch.allclose(
            total_consumption,
            growth + growth_respiration,
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        # Microbial mortality is routed to the respiration port in this closure.
        assert torch.allclose(
            respiration,
            growth_respiration + death,
            rtol=1.0e-6,
            atol=1.0e-12,
        )

        # All fluxes should be non-negative
        assert float(total_consumption) >= 0
        assert float(growth) >= 0
        assert float(respiration) >= 0

    def test_simulator_state_updates_provenance_and_public_carbon_ledger(self):
        """Test canonical simulator updates and the public carbon ledger."""
        state = _canonical_state()
        dt = 1800.0
        direct_functor = MicrobialEnzymeDecomposition(
            name="microbial_enzyme_decomposition",
            config={},
        )
        direct = direct_functor.forward(
            {key: value.clone() for key, value in state.items()},
            time=0.0,
            dt=dt,
        )

        state_manager = StateManager(
            device="cpu",
            conservation_checks={"enabled": False},
        )
        state_manager.set_state(
            {key: value.clone() for key, value in state.items()},
            validate=True,
        )
        simulator = RELICSimulator(
            {
                "device": "cpu",
                "time_step": dt,
                "max_time": dt,
                "conservation_checks": False,
                "strict_dependencies": False,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize(state_manager, DependencyGraphResolver())
        simulator.register_functor(
            MicrobialEnzymeDecomposition(
                name="microbial_enzyme_decomposition",
                config={},
            ),
            execution_group="fast",
        )
        initial = {
            key: state_manager.get_state(key).clone()
            for key in STATE_OUTPUTS
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step_updates = simulator.step(dt=dt)

        assert set(step_updates) == CANONICAL_OUTPUTS
        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, UserWarning)
        ]
        for key in STATE_OUTPUTS:
            actual_delta = state_manager.get_state(key) - initial[key]
            expected_delta = direct[key].value.to(dtype=actual_delta.dtype)
            torch.testing.assert_close(
                actual_delta,
                expected_delta,
                rtol=0.0,
                atol=5.0e-17,
            )
            writers = {
                record["functor"]
                for record in simulator.get_variable_provenance(key)
            }
            assert writers == {"microbial_enzyme_decomposition"}

        violations = direct_functor.check_conservation_laws(
            state,
            direct,
            dt=dt,
        )
        assert violations["carbon"] <= 1.0e-15
        assert violations["mass"] <= 1.0e-15
        assert simulator.get_effective_device(
            "microbial_enzyme_decomposition"
        ) == torch.device("cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


def test_forward_keeps_no_module_state_and_declares_its_step():
    """Round 47 (I1): forward writes no module tensor; two identical calls give
    bit-identical outputs and unchanged parameters and buffers; the step length reaches
    the public ledger as the declared diagnostics.microbial_enzyme.step_seconds output."""
    functor = MicrobialEnzymeDecomposition(name="enzyme_i1", config={})

    def _snapshot():
        return {
            **{f"p.{k}": v.detach().clone() for k, v in functor.named_parameters()},
            **{f"b.{k}": v.detach().clone() for k, v in functor.named_buffers()},
        }

    before = _snapshot()
    first = functor(_canonical_state(), time=0.0, dt=3600.0)
    second = functor(_canonical_state(), time=0.0, dt=3600.0)
    after = _snapshot()
    assert not hasattr(functor, "_last_time_step")
    assert before.keys() == after.keys()
    for key, value in before.items():
        assert torch.equal(value, after[key]), key
    assert first.keys() == second.keys()
    for key in first:
        a = first[key].value if isinstance(first[key], UpdateItem) else first[key]
        b = second[key].value if isinstance(second[key], UpdateItem) else second[key]
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), key
    step = first["diagnostics.microbial_enzyme.step_seconds"]
    assert isinstance(step, UpdateItem) and step.unit == "s"
    assert float(step.value) == 3600.0
    assert "diagnostics.microbial_enzyme.step_seconds" in functor.metadata.output_variables
