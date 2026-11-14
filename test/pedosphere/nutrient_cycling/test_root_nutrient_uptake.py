"""
L1 Unit Tests for Root Nutrient Uptake Process Functor.

Tests Michaelis-Menten kinetics, root distribution, plant demand regulation,
and environmental modifiers (temperature, water stress).
Following the RELIC L1 testing specification.

References:
- Barber (1995) - Soil Nutrient Bioavailability: Mechanistic Approach
- Tinker & Nye (2000) - Solute Movement in the Rhizosphere
- Chapin et al. (1993) - Evolution of trait suites in response to stress
- Jackson et al. (1996) - Global analysis of root distributions

Mathematical Expressions:
U_i = Vmax_i * C_i / (Km + C_i) * f_T(T) * f_W(theta_i) * RootWeight_i
U_total = Sigma_i U_i
U_actual = min(U_total, PlantDemand)
W_i = W0 * exp(-k_root * depth_i)

Units:
- U: mol m^-2 s^-1
- C: mol m^-3
- Vmax: mol m^-2 s^-1
- Km: mol m^-3
"""
import warnings

import pytest
import torch
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
import relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake as uptake_module
from relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake import (
    RootNutrientUptake
)


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _canonical_state(**overrides):
    """Build the complete canonical state contract for root nutrient uptake."""
    state = {
        "no3_nitrogen": torch.tensor(5.0),
        "nh4_nitrogen": torch.tensor(3.0),
        "available_phosphorus": torch.tensor(1.0),
        "soil_water_content": torch.tensor(0.35),
        "soil_temperature": torch.tensor(293.15),
        "soil_depth": torch.tensor(1.0),
        "plant_n_storage": torch.tensor(1.0),
        "plant_p_storage": torch.tensor(1.0),
        "root_density": torch.tensor(1.0),
        "plant_n_demand": torch.tensor(1.0e-3),
        "plant_p_demand": torch.tensor(1.0e-4),
        "field_capacity": torch.tensor(0.45),
        "wilting_point": torch.tensor(0.15),
    }
    state.update(overrides)
    return state


class TestRootNutrientUptakeInit:
    """Test initialization and parameter loading."""

    def test_initialization(self, parameter_manager, basic_config):
        """Test Functor can be initialized."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )
        assert functor is not None
        assert functor.name == "test_root_uptake"

    def test_parameter_loading(self, parameter_manager, basic_config):
        """Test that required parameters are loaded from manager."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        # Should have Michaelis-Menten parameters
        assert hasattr(functor, 'vmax_n')
        assert hasattr(functor, 'vmax_p')
        assert hasattr(functor, 'km_n')
        assert hasattr(functor, 'km_p')

        # Parameters should be positive
        assert functor.vmax_n > 0
        assert functor.vmax_p > 0
        assert functor.km_n > 0
        assert functor.km_p > 0

    def test_root_distribution_parameters(self, parameter_manager, basic_config):
        """Test root distribution parameters."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        # Should have root decay coefficient
        assert hasattr(functor, 'root_decay_coeff')
        assert functor.root_decay_coeff > 0

    def test_temperature_response_parameters(self, parameter_manager, basic_config):
        """Test temperature response parameters."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        # Should have Q10 parameters
        assert hasattr(functor, 'q10_value')
        assert hasattr(functor, 'reference_temperature')

        # Q10 should be > 1
        assert functor.q10_value > 1.0
        # Reference temperature should be reasonable (273-323 K)
        assert 273.15 < functor.reference_temperature < 323.15

    def test_metadata_includes_conservation_laws(self, basic_config):
        """Test metadata includes nitrogen conservation."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )
        metadata = functor.metadata

        if isinstance(metadata, dict):
            conservation = metadata.get('conservation_laws', [])
        else:
            conservation = getattr(metadata, 'conservation_laws', [])

        assert 'nitrogen' in conservation or 'phosphorus' in conservation


class TestRootNutrientUptakeForward:
    """Test forward execution."""

    @pytest.fixture
    def uptake_state(self):
        """State with nutrient pools and plant variables."""
        return _canonical_state()

    def test_forward_runs(self, uptake_state, basic_config):
        """Test that forward() executes without errors."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        updates = functor.forward(
            state=uptake_state,
            time=0.0,
            dt=3600.0  # 1 hour
        )

        assert isinstance(updates, dict)
        assert len(updates) > 0
        state_updates = {
            "no3_nitrogen",
            "nh4_nitrogen",
            "available_phosphorus",
            "plant_n_storage",
            "plant_p_storage",
        }
        assert state_updates.issubset(updates)
        assert state_updates.issubset(functor.metadata.output_variables)
        assert not {
            "delta_no3_nitrogen",
            "delta_nh4_nitrogen",
            "delta_available_phosphorus",
            "delta_plant_n_storage",
            "delta_plant_p_storage",
        }.intersection(updates)

    def test_forward_returns_tensors(self, uptake_state, basic_config):
        """Test that forward returns Tensor values."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        updates = functor.forward(
            state=uptake_state,
            time=0.0,
            dt=3600.0
        )

        for key, value in updates.items():
            if not key.endswith('_units'):
                value = _unwrap_update(value)
                assert isinstance(value, torch.Tensor), f"{key} should be Tensor"

    def test_nitrogen_pool_deltas_use_explicit_accumulate_mode(
        self, uptake_state, basic_config
    ):
        """Root uptake must contribute per-step losses to both SCM ledgers."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config,
        )
        updates = functor.forward(uptake_state, 0.0, 3600.0)
        for key in ("no3_nitrogen", "nh4_nitrogen"):
            assert isinstance(updates[key], UpdateItem)
            assert updates[key].mode is UpdateMode.ACCUMULATE
            assert updates[key].per_timestep is True
            assert updates[key].element == "N"

        assert updates["plant_n_storage"].element == "N"
        assert updates["plant_n_storage"].unit == "mol N m-2"
        assert updates["plant_n_storage"].source_functor == "test_root_uptake"
        for key in (
            "nitrogen_uptake_rate",
            "no3_uptake_by_layer",
            "nh4_uptake_by_layer",
        ):
            assert getattr(updates[key], "element", None) is None

    def test_phosphorus_pool_deltas_use_one_mass_basis(
        self, uptake_state, basic_config
    ):
        """Soil P loss and plant P gain use explicit per-step mass updates."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config,
        )

        updates = functor.forward(uptake_state, 0.0, 3600.0)

        for key in ("available_phosphorus", "plant_p_storage"):
            item = updates[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == "g P m-2"
            assert item.element == "P"
            assert item.source_functor == "test_root_uptake"

        for key in ("phosphorus_uptake_rate", "po4_uptake_by_layer"):
            item = updates[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == "g P m-2 s-1"
            assert item.element is None

    def test_simulator_applies_closed_phosphorus_transfer_once(
        self, uptake_state, basic_config, monkeypatch
    ):
        """StateManager must apply the closed P transfer once with provenance."""
        simulator_state = dict(uptake_state)
        simulator_state["available_phosphorus"] = torch.tensor(1.0e-6)
        simulator_state["plant_p_storage"] = torch.tensor(0.0)
        simulator_state["plant_p_demand"] = torch.tensor(1.0)
        dt = 86400.0
        simulator = RELICSimulator(
            {
                "max_time": dt,
                "time_step": dt,
                "device": "cpu",
                "conservation_checks": False,
                "strict_dependencies": False,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize_state(simulator_state)
        functor = RootNutrientUptake(
            name="root_nutrient_uptake",
            config=basic_config,
        )
        # This isolated StateManager test does not execute the upstream N
        # supply processes; their full dependency chain has a separate L2 test.
        monkeypatch.setattr(functor, "get_dependencies", lambda: [])
        simulator.register_functor(functor, execution_group="fast")

        tracked = ("available_phosphorus", "plant_p_storage")
        before = {
            key: simulator.state_manager.get_state(key).clone()
            for key in tracked
        }
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            updates = simulator.step(dt=dt)

        assert captured == []
        deltas = {}
        for key in tracked:
            deltas[key] = simulator.state_manager.get_state(key) - before[key]
            applied = updates[key]
            if isinstance(applied, UpdateItem):
                applied = applied.value
            assert torch.allclose(deltas[key], applied, rtol=0.0, atol=1.0e-12)
            observed = {
                item["functor"]
                for item in simulator.get_variable_provenance(key)
                if item.get("role") == "increment_source"
            }
            assert observed == {"root_nutrient_uptake"}

        assert torch.allclose(
            deltas["available_phosphorus"] + deltas["plant_p_storage"],
            torch.zeros_like(deltas["available_phosphorus"]),
            rtol=0.0,
            atol=1.0e-12,
        )
        assert simulator.get_effective_device("root_nutrient_uptake").type == "cpu"

    def test_nutrient_transfer_uses_canonical_stoichiometric_channels(
        self, uptake_state, basic_config, monkeypatch
    ):
        """The reaction incidence matrix closes both elemental N and P transfer."""
        captured = {}
        real_advance = uptake_module.advance_reaction_network

        def capture_advance(**kwargs):
            captured.update(kwargs)
            return real_advance(**kwargs)

        monkeypatch.setattr(
            uptake_module,
            "advance_reaction_network",
            capture_advance,
        )
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config,
        )

        functor.forward(uptake_state, 0.0, 3600.0)

        species = captured["species"]
        reactions = captured["reactions"]
        stoichiometric_n = captured["stoichiometric_N"]
        assert reactions == (
            "no3_root_transfer",
            "nh4_root_transfer",
            "po4_root_transfer",
        )
        p_column = reactions.index("po4_root_transfer")
        available_row = species.index("available_phosphorus")
        plant_row = species.index("plant_p_storage")
        assert stoichiometric_n[available_row, p_column] == -1.0
        assert stoichiometric_n[plant_row, p_column] == 1.0
        assert torch.count_nonzero(stoichiometric_n[:, p_column]) == 2
        conservation_matrix = torch.as_tensor(
            captured["conservation_matrix"],
            dtype=stoichiometric_n.dtype,
            device=stoichiometric_n.device,
        )
        assert conservation_matrix.shape == (2, 5)
        assert torch.count_nonzero(conservation_matrix) == 5
        assert torch.allclose(
            conservation_matrix @ stoichiometric_n,
            torch.zeros(
                (conservation_matrix.shape[0], stoichiometric_n.shape[1]),
                dtype=stoichiometric_n.dtype,
                device=stoichiometric_n.device,
            ),
        )

    def test_nitrogen_uptake_output(self, uptake_state, basic_config):
        """Test that nitrogen uptake is calculated."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        updates = functor.forward(uptake_state, 0.0, 3600.0)

        # Should have N uptake keys
        n_keys = [k for k in updates.keys() if 'nitrogen' in k.lower() and 'uptake' in k.lower()]
        assert len(n_keys) > 0, "Should have nitrogen uptake output"

        # Uptake should be non-negative
        n_uptake = _unwrap_update(updates[n_keys[0]])
        assert n_uptake >= 0, "Nitrogen uptake should be non-negative"

    def test_phosphorus_uptake_output(self, uptake_state, basic_config):
        """Test that phosphorus uptake is calculated."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        updates = functor.forward(uptake_state, 0.0, 3600.0)

        # Should have P uptake keys
        p_keys = [k for k in updates.keys() if 'phosphorus' in k.lower() and 'uptake' in k.lower()]

        if len(p_keys) > 0:
            p_uptake = _unwrap_update(updates[p_keys[0]])
            assert p_uptake >= 0, "Phosphorus uptake should be non-negative"

    def test_nutrient_pool_depletion(self, uptake_state, basic_config):
        """Test that nutrient pools decrease with uptake."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        updates = functor.forward(uptake_state, 0.0, 3600.0)

        pool_keys = [k for k in ("no3_nitrogen", "nh4_nitrogen") if k in updates]
        assert pool_keys, "Expected nitrogen pool deltas in updates"
        delta_n = _unwrap_update(updates[pool_keys[0]])
        # With uptake, pool should decrease (negative delta)
        assert delta_n <= 0, "Nitrogen pool should decrease with uptake"


class TestMichaelisMentenKinetics:
    """Test Michaelis-Menten kinetics behavior."""

    @pytest.fixture
    def base_uptake_state(self):
        return _canonical_state(
            nh4_nitrogen=torch.tensor(0.0),
            plant_n_demand=torch.tensor(1.0),
        )

    def test_saturation_kinetics(self, base_uptake_state, basic_config):
        """
        Test Michaelis-Menten saturation behavior.

        Mathematical Expression:
        U = Vmax * C / (Km + C)

        Expected: At low C, uptake ~ linear. At high C, uptake ~ Vmax
        """
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        # Low concentration (linear range)
        state_low = base_uptake_state.copy()
        state_low['no3_nitrogen'] = torch.tensor(0.1)  # Low C << Km

        # Medium concentration
        state_med = base_uptake_state.copy()
        state_med['no3_nitrogen'] = torch.tensor(5.0)  # Medium C ~ Km

        # High concentration (saturation)
        state_high = base_uptake_state.copy()
        state_high['no3_nitrogen'] = torch.tensor(50.0)  # High C >> Km

        updates_low = functor.forward(state_low, 0.0, 3600.0)
        updates_med = functor.forward(state_med, 0.0, 3600.0)
        updates_high = functor.forward(state_high, 0.0, 3600.0)

        # Find uptake keys
        n_keys = [k for k in updates_low.keys() if 'nitrogen' in k.lower() and 'uptake' in k.lower()]

        if n_keys:
            uptake_low = _unwrap_update(updates_low[n_keys[0]]).item()
            uptake_med = _unwrap_update(updates_med[n_keys[0]]).item()
            uptake_high = _unwrap_update(updates_high[n_keys[0]]).item()

            # Uptake should increase with concentration
            assert uptake_med > uptake_low, "Medium conc should have higher uptake than low"
            assert uptake_high > uptake_med, "High conc should have higher uptake than medium"

            # At high concentration, should approach saturation (smaller increments)
            increment_low_to_med = uptake_med - uptake_low
            increment_med_to_high = uptake_high - uptake_med

            # Increment should decrease at higher concentrations (saturation)
            assert increment_low_to_med > increment_med_to_high * 0.5, "Should show saturation at high conc"


class TestTemperatureSensitivity:
    """Test temperature effects on uptake."""

    @pytest.fixture
    def base_temp_state(self):
        return _canonical_state(plant_n_demand=torch.tensor(1.0))

    def test_temperature_response(self, base_temp_state, basic_config):
        """
        Test Q10 temperature response.

        Mathematical Expression:
        f_T(T) = Q10^((T-Tref)/10)

        Expected: Higher temperature -> Higher uptake
        """
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        # Cold
        state_cold = base_temp_state.copy()
        state_cold['soil_temperature'] = torch.tensor(278.15)  # 5 C

        # Warm
        state_warm = base_temp_state.copy()
        state_warm['soil_temperature'] = torch.tensor(303.15)  # 30 C

        updates_cold = functor.forward(state_cold, 0.0, 3600.0)
        updates_warm = functor.forward(state_warm, 0.0, 3600.0)

        # Find uptake keys
        n_keys = [k for k in updates_cold.keys() if 'nitrogen' in k.lower() and 'uptake' in k.lower()]

        if n_keys:
            uptake_cold = _unwrap_update(updates_cold[n_keys[0]]).item()
            uptake_warm = _unwrap_update(updates_warm[n_keys[0]]).item()

            assert uptake_warm > uptake_cold, "Warm temperature should increase uptake"

            # Should follow Q10 relationship approximately
            # Q10 = 2 means doubling per 10K
            # 25K difference -> ~ 2^2.5 = 5.66x increase
            if functor.q10_value == 2.0:
                ratio = uptake_warm / (uptake_cold + 1e-10)
                assert ratio > 3.0, "Q10=2 should give ~5-6x increase over 25K"


class TestWaterStress:
    """Test water stress effects on uptake."""

    @pytest.fixture
    def base_water_state(self):
        return _canonical_state(plant_n_demand=torch.tensor(1.0))

    def test_water_stress_response(self, base_water_state, basic_config):
        """
        Test water stress limitation.

        Mathematical Expression:
        f_W(theta) = (theta - theta_wilt) / (theta_fc - theta_wilt)

        Expected: Dry soil -> Lower uptake
        """
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        # Dry (near wilting point)
        state_dry = base_water_state.copy()
        state_dry['soil_water_content'] = torch.tensor(0.20)  # Near wilting

        # Optimal (field capacity)
        state_optimal = base_water_state.copy()
        state_optimal['soil_water_content'] = torch.tensor(0.40)  # Near FC

        updates_dry = functor.forward(state_dry, 0.0, 3600.0)
        updates_optimal = functor.forward(state_optimal, 0.0, 3600.0)

        # Find uptake keys
        n_keys = [k for k in updates_dry.keys() if 'nitrogen' in k.lower() and 'uptake' in k.lower()]

        if n_keys:
            uptake_dry = _unwrap_update(updates_dry[n_keys[0]]).item()
            uptake_optimal = _unwrap_update(updates_optimal[n_keys[0]]).item()

            assert uptake_optimal > uptake_dry, "Optimal moisture should have higher uptake"


class TestRootDistribution:
    """Test root distribution effects on uptake."""

    def test_root_density_routes_layer_uptake(self, basic_config):
        """
        Test that the declared root-density profile routes layer uptake.

        Mathematical Expression:
        W_i = W0 * exp(-k_root * depth_i) * root_density_i

        Expected: A layer without roots has no root uptake.
        """
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        common = _canonical_state(
            no3_nitrogen=torch.tensor([2.5, 2.5]),
            nh4_nitrogen=torch.tensor([1.5, 1.5]),
            available_phosphorus=torch.tensor([0.5, 0.5]),
            soil_water_content=torch.tensor([0.35, 0.35]),
            soil_depth=torch.tensor([0.25, 0.75]),
            field_capacity=torch.tensor([0.45, 0.45]),
            wilting_point=torch.tensor([0.15, 0.15]),
            plant_n_demand=torch.tensor(1.0),
            plant_p_demand=torch.tensor(1.0),
        )
        shallow_roots = dict(common)
        shallow_roots["root_density"] = torch.tensor([1.0, 0.0])
        deep_roots = dict(common)
        deep_roots["root_density"] = torch.tensor([0.0, 1.0])

        shallow = functor.forward(shallow_roots, 0.0, 3600.0)
        deep = functor.forward(deep_roots, 0.0, 3600.0)
        shallow_profile = shallow["no3_uptake_by_layer"].value
        deep_profile = deep["no3_uptake_by_layer"].value

        assert shallow_profile[0] > 0.0
        assert shallow_profile[1] == 0.0
        assert deep_profile[0] == 0.0
        assert deep_profile[1] > 0.0


class TestPlantDemandRegulation:
    """Test plant demand limitation."""

    def test_demand_limits_uptake(self, basic_config):
        """
        Test that plant demand limits uptake.

        Mathematical Expression:
        U_actual = min(U_potential, PlantDemand)

        Expected: Uptake cannot exceed demand
        """
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        # High soil N, low plant demand
        state_low_demand = _canonical_state(
            no3_nitrogen=torch.tensor(100.0),
            nh4_nitrogen=torch.tensor(0.0),
            plant_n_demand=torch.tensor(1.0e-8),
        )

        # High soil N, high plant demand
        state_high_demand = state_low_demand.copy()
        state_high_demand['plant_n_demand'] = torch.tensor(1.0)  # High demand

        updates_low = functor.forward(state_low_demand, 0.0, 3600.0)
        updates_high = functor.forward(state_high_demand, 0.0, 3600.0)

        # Find uptake keys
        n_keys = [k for k in updates_low.keys() if 'nitrogen' in k.lower() and 'uptake' in k.lower()]

        if n_keys:
            uptake_low_demand = _unwrap_update(updates_low[n_keys[0]]).item()
            uptake_high_demand = _unwrap_update(updates_high[n_keys[0]]).item()

            # Low demand should result in lower uptake
            assert uptake_low_demand < uptake_high_demand, "Plant demand should limit uptake"

    @pytest.mark.parametrize("missing_key", ["available_phosphorus", "plant_p_storage"])
    def test_required_phosphorus_states_fail_fast(self, basic_config, missing_key):
        """Canonical phosphorus states cannot be supplied by defaults or aliases."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        state = _canonical_state(po4_concentration=torch.tensor(100.0))
        state.pop(missing_key)

        with pytest.raises(ValueError, match=missing_key):
            functor.forward(
                state=state,
                time=0.0,
                dt=3600.0,
            )


class TestConservation:
    """Test conservation laws."""

    def test_uptake_does_not_overdraft_pool(self, basic_config):
        """Test that uptake cannot exceed available pool."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config
        )

        # Very small N pool with high demand
        state = _canonical_state(
            no3_nitrogen=torch.tensor(0.01),
            nh4_nitrogen=torch.tensor(0.0),
            plant_n_demand=torch.tensor(10.0),
        )

        updates = functor.forward(state, 0.0, 3600.0)

        if "no3_nitrogen" in updates:
            delta_n = _unwrap_update(updates["no3_nitrogen"]).item()
            total_uptake = abs(delta_n)

            # Total uptake should not exceed available pool
            assert total_uptake <= state['no3_nitrogen'].item() * 1.01, "Should not overdraft N pool"

    def test_phosphorus_transfer_is_pool_limited_and_closed(self, basic_config):
        """A large step transfers no more P mass than the available soil pool."""
        functor = RootNutrientUptake(
            name="test_root_uptake",
            config=basic_config,
        )
        initial_available = torch.tensor(1.0e-8)
        state = _canonical_state(
            no3_nitrogen=torch.tensor(1.0),
            nh4_nitrogen=torch.tensor(1.0),
            available_phosphorus=initial_available,
            plant_p_storage=torch.tensor(0.0),
            soil_water_content=torch.tensor(0.45),
            soil_temperature=torch.tensor(298.15),
            plant_n_demand=torch.tensor(1.0),
            plant_p_demand=torch.tensor(1.0),
        )

        updates = functor.forward(state, 0.0, 86400.0)
        available_delta = updates["available_phosphorus"].value
        plant_delta = updates["plant_p_storage"].value
        final_available = initial_available + available_delta

        assert final_available >= -torch.finfo(final_available.dtype).eps
        assert -available_delta <= initial_available
        assert torch.allclose(plant_delta, -available_delta, rtol=1.0e-6, atol=1.0e-15)
        assert torch.allclose(
            available_delta + plant_delta,
            torch.zeros_like(available_delta),
            rtol=0.0,
            atol=1.0e-15,
        )

def _layered_state():
    return _canonical_state(
        no3_nitrogen=torch.tensor([2.5, 2.5]),
        nh4_nitrogen=torch.tensor([1.5, 1.5]),
        available_phosphorus=torch.tensor([0.5, 0.5]),
        soil_water_content=torch.tensor([0.35, 0.35]),
        soil_depth=torch.tensor([0.25, 0.75]),
        field_capacity=torch.tensor([0.45, 0.45]),
        wilting_point=torch.tensor([0.15, 0.15]),
        root_density=torch.tensor([1.0, 0.5]),
        plant_n_demand=torch.tensor(1.0),
        plant_p_demand=torch.tensor(1.0),
    )


def test_layered_donor_pools_receive_layered_increments_that_close_with_plant_storage(basic_config):
    """Layered NO3/NH4/P pools get per-layer increments whose sums close with the areal plant gains."""
    functor = RootNutrientUptake(name="test_root_uptake_layers", config=basic_config)
    dt = 3600.0
    state = _layered_state()
    updates = functor(state, time=0.0, dt=dt)
    m_n = functor.atomic_weight_nitrogen
    no3 = updates["no3_nitrogen"].value
    nh4 = updates["nh4_nitrogen"].value
    phosphorus = updates["available_phosphorus"].value
    # References: Barber (1995); Tinker and Nye (2000).
    # Mathematical expression: Delta_pool(z) = -M U(z) dt; sum_z Delta_pool(z) / M + Delta S = 0.
    # Parameter meanings: U(z) layer uptake, M molar mass, dt step, S plant storage.
    # Units: Delta_pool [g m-2]; U [mol N m-2 s-1] or [g P m-2 s-1]; dt [s].
    # Domain/constraints: two declared soil layers, both rooted.
    # Sign convention: donor increments negative, plant increments positive.
    assert tuple(no3.shape) == (2,)
    assert tuple(nh4.shape) == (2,)
    assert tuple(phosphorus.shape) == (2,)
    assert torch.all(no3 < 0.0)
    assert torch.all(nh4 < 0.0)
    assert torch.all(phosphorus < 0.0)
    torch.testing.assert_close(no3, -updates["no3_uptake_by_layer"].value * m_n * dt)
    torch.testing.assert_close(nh4, -updates["nh4_uptake_by_layer"].value * m_n * dt)
    torch.testing.assert_close(phosphorus, -updates["po4_uptake_by_layer"].value * dt)
    plant_n = updates["plant_n_storage"].value
    plant_p = updates["plant_p_storage"].value
    assert plant_n.dim() == 0 and plant_p.dim() == 0
    torch.testing.assert_close(
        (no3.sum() + nh4.sum()) / m_n + plant_n, torch.zeros_like(plant_n), rtol=0.0, atol=1.0e-12
    )
    torch.testing.assert_close(phosphorus.sum() + plant_p, torch.zeros_like(plant_p), rtol=0.0, atol=1.0e-12)
    violations = functor.check_conservation_laws(state, updates)
    assert violations["nitrogen"] <= 1.0e-9
    assert violations["phosphorus"] <= 1.0e-9


def test_simulator_commits_layered_nutrient_uptake_once(basic_config, monkeypatch):
    """A real StateManager accepts the layered donor increments and the areal plant gains in one step."""
    dt = 3600.0
    simulator = RELICSimulator(
        {
            "max_time": dt,
            "time_step": dt,
            "device": "cpu",
            "conservation_checks": False,
            "strict_dependencies": False,
            "strict_scale_detection": False,
        }
    )
    simulator.initialize_state(_layered_state())
    functor = RootNutrientUptake(name="root_nutrient_uptake", config=basic_config)
    monkeypatch.setattr(functor, "get_dependencies", lambda: [])
    simulator.register_functor(functor, execution_group="fast")
    tracked = ("no3_nitrogen", "nh4_nitrogen", "available_phosphorus", "plant_n_storage", "plant_p_storage")
    before = {key: simulator.state_manager.get_state(key).clone() for key in tracked}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        updates = simulator.step(dt=dt)
    assert captured == []
    deltas = {}
    for key in tracked:
        deltas[key] = simulator.state_manager.get_state(key) - before[key]
        assert tuple(deltas[key].shape) == tuple(before[key].shape)
        applied = updates[key]
        if isinstance(applied, UpdateItem):
            applied = applied.value
        assert torch.allclose(deltas[key], applied.to(deltas[key].dtype), rtol=0.0, atol=1.0e-12)
    assert torch.all(deltas["no3_nitrogen"] < 0.0)
    assert torch.all(deltas["nh4_nitrogen"] < 0.0)
    m_n = functor.atomic_weight_nitrogen
    nitrogen_closure = (deltas["no3_nitrogen"].sum() + deltas["nh4_nitrogen"].sum()) / m_n + deltas["plant_n_storage"]
    assert torch.allclose(nitrogen_closure, torch.zeros_like(nitrogen_closure), rtol=0.0, atol=1.0e-12)
    phosphorus_closure = deltas["available_phosphorus"].sum() + deltas["plant_p_storage"]
    assert torch.allclose(phosphorus_closure, torch.zeros_like(phosphorus_closure), rtol=0.0, atol=1.0e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

