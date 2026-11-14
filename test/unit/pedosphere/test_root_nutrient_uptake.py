"""
L1 Unit Tests for RootNutrientUptake Functor

Tests root-mediated nutrient acquisition processes including nitrogen (N) and 
phosphorus (P) uptake using Michaelis-Menten kinetics with root weight distribution,
plant demand regulation, and environmental response functions.
"""

import pytest
import torch
import inspect
from typing import Dict, Any

# Import the class under test
from relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake import RootNutrientUptake
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant
from relic.core.state import UpdateItem, UpdateMode

# Import supporting classes  
from relic.core.abstractions import StateDict
from test.utils.forward_call import call_forward

@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _normalize_param_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return value

def _set_param_value(param_manager, key: str, value, *, validate: bool = True) -> None:
    spec = param_manager.get_parameter(key)
    units = spec.units if spec is not None else None
    param_manager.set_value(key, value, units=units, validate=validate, notify=False)


class TestRootNutrientUptakeL1:
    """L1 Unit Tests for RootNutrientUptake functor."""

    @pytest.fixture(autouse=True)
    def _default_param_overrides(self, default_params):
        param_manager = get_global_manager()
        overrides = {
            "pedosphere.nutrient_cycling.vmax_n": default_params["vmax_n"],
            "pedosphere.nutrient_cycling.vmax_p": default_params["vmax_p"],
            "pedosphere.nutrient_cycling.km_n": default_params["km_n"],
            "pedosphere.nutrient_cycling.km_p": default_params["km_p"],
            "pedosphere.nutrient_cycling.root_decay_coeff": default_params["root_decay_coeff"],
            "pedosphere.nutrient_cycling.q10_value": default_params["q10_value"],
            "pedosphere.nutrient_cycling.reference_temperature": default_params["reference_temperature"],
            "pedosphere.nutrient_cycling.water_stress_enabled": default_params["water_stress_enabled"],
            "pedosphere.nutrient_cycling.demand_regulation_enabled": default_params["demand_regulation_enabled"],
            "pedosphere.nutrient_cycling.max_storage_n": default_params["max_storage_n"],
            "pedosphere.nutrient_cycling.max_storage_p": default_params["max_storage_p"],
        }
        original_values = {
            key: _normalize_param_value(param_manager.get(key)) for key in overrides
        }
        for key, value in overrides.items():
            _set_param_value(param_manager, key, value)
        yield
        for key, value in original_values.items():
            _set_param_value(param_manager, key, value)
    
    @pytest.fixture
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for testing."""
        return {
            'vmax_n': 1.0e-6,     # mol N m-2 s-1
            'vmax_p': 1.0e-7,     # mol P m-2 s-1
            'km_n': 0.1,          # mol N m-3
            'km_p': 0.01,         # mol P m-3
            'root_decay_coeff': 2.0,  # m-1
            'q10_value': 2.0,     # dimensionless
            'reference_temperature': 298.15,  # K
            'water_stress_enabled': True,
            'demand_regulation_enabled': True,
            'max_storage_n': 10.0,  # mol N m-2
            'max_storage_p': 1.0,   # mol P m-2
        }
    
    @pytest.fixture
    def process(self, default_params) -> RootNutrientUptake:
        """Create test instance."""
        return RootNutrientUptake()
    
    @pytest.fixture
    def basic_state(self) -> StateDict:
        """Basic state for testing."""
        return {
            'no3_nitrogen': torch.tensor([0.140067, 0.3361608, 0.420201]),  # g N m-2
            'nh4_nitrogen': torch.tensor([0.0700335, 0.1680804, 0.1680804]),  # g N m-2
            'available_phosphorus': torch.tensor(1.0),               # g P m-2
            'soil_water_content': torch.tensor([0.25, 0.22, 0.18]),  # m3 m-3
            'soil_temperature': torch.tensor(298.15),                # K
            'plant_n_storage': torch.tensor(2.0),                    # mol N m-2
            'plant_p_storage': torch.tensor(0.2),                    # g P m-2
            'soil_depth': torch.tensor([0.1, 0.3, 0.6]),            # m
            'root_density': torch.tensor([1.0, 0.8, 0.5]),          # kg m-3
            'plant_n_demand': torch.tensor(1.5e-6),                 # mol N m-2 s-1
            'plant_p_demand': torch.tensor(1.5e-7),                 # mol P m-2 s-1
            'field_capacity': torch.tensor([0.35, 0.33, 0.30]),     # m3 m-3
            'wilting_point': torch.tensor([0.15, 0.12, 0.10]),      # m3 m-3
        }
    
    # Test 1: Instantiation and Parameter Validation
    def test_instantiation_success(self, default_params):
        """Test successful instantiation with valid parameters."""
        process = RootNutrientUptake()
        
        assert process.process_name == "root_nutrient_uptake"
        assert _normalize_param_value(process.vmax_n) == pytest.approx(default_params["vmax_n"])
        assert _normalize_param_value(process.vmax_p) == pytest.approx(default_params["vmax_p"])
        assert _normalize_param_value(process.km_n) == pytest.approx(default_params["km_n"])
        assert _normalize_param_value(process.km_p) == pytest.approx(default_params["km_p"])
        assert _normalize_param_value(process.root_decay_coeff) == pytest.approx(default_params["root_decay_coeff"])
        assert process.q10_value == pytest.approx(default_params["q10_value"])
        assert process.reference_temperature == pytest.approx(default_params["reference_temperature"])
        assert process.water_stress_enabled is True
        assert process.demand_regulation_enabled is True

    def test_parameter_access_has_no_compatibility_helpers(self):
        """The Functor uses the canonical parameter entrypoint exclusively."""
        source = inspect.getsource(RootNutrientUptake)

        assert "def _param(" not in source
        assert "def _scalar_param(" not in source
        assert "self.require_scalar(" not in source
        assert "parameter_overrides" not in source

    def test_phosphorus_atomic_weight_uses_physical_constants_ssot(self):
        """Elemental P molar mass is owned only by the physical constants SSOT."""
        process = RootNutrientUptake()
        source = inspect.getsource(RootNutrientUptake)

        assert "pedosphere.physical.atomic_weight_phosphorus" not in source
        assert process.atomic_weight_phosphorus == pytest.approx(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.phosphorus"
            )
        )

    def test_nitrogen_atomic_weight_uses_physical_constants_ssot(self):
        """Elemental N molar mass is owned only by the physical constants SSOT."""
        process = RootNutrientUptake()
        source = inspect.getsource(RootNutrientUptake)

        assert "pedosphere.physical.atomic_weight_nitrogen" not in source
        assert process.atomic_weight_nitrogen == pytest.approx(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.nitrogen"
            )
        )

    def test_q10_interval_uses_physical_constants_ssot(self):
        """The Q10 temperature interval must not use the Functor parameter path."""
        process = RootNutrientUptake()
        source = inspect.getsource(RootNutrientUptake)

        assert 'namespace="relic.physical.constants.temperature"' not in source
        assert process.q10_interval == pytest.approx(
            get_physical_constant(
                "relic.physical.constants.temperature.q10_interval"
            )
        )

    def test_metadata_declares_exact_canonical_inputs(self, process):
        """Every runtime driver is a declared canonical input with no aliases."""
        assert process.metadata.input_variables == [
            "no3_nitrogen",
            "nh4_nitrogen",
            "available_phosphorus",
            "soil_water_content",
            "soil_temperature",
            "soil_depth",
            "plant_n_storage",
            "plant_p_storage",
            "root_density",
            "plant_n_demand",
            "plant_p_demand",
            "field_capacity",
            "wilting_point",
        ]
        assert process.optional_input_variables == []
    
    def test_instantiation_invalid_vmax(self, default_params):
        """Test instantiation fails with invalid vmax values."""
        param_manager = get_global_manager()
        vmax_n_key = "pedosphere.nutrient_cycling.vmax_n"
        vmax_p_key = "pedosphere.nutrient_cycling.vmax_p"

        _set_param_value(param_manager, vmax_n_key, 0.0)
        with pytest.raises(ValueError, match="Maximum uptake rates must be positive"):
            RootNutrientUptake()

        _set_param_value(param_manager, vmax_n_key, default_params["vmax_n"])
        _set_param_value(param_manager, vmax_p_key, 0.0)
        with pytest.raises(ValueError, match="Maximum uptake rates must be positive"):
            RootNutrientUptake()
    
    def test_instantiation_invalid_km(self, default_params):
        """Test instantiation fails with invalid Km values."""
        param_manager = get_global_manager()
        km_n_key = "pedosphere.nutrient_cycling.km_n"
        km_p_key = "pedosphere.nutrient_cycling.km_p"

        _set_param_value(param_manager, km_n_key, 0.0)
        with pytest.raises(ValueError, match="Michaelis constants must be positive"):
            RootNutrientUptake()

        _set_param_value(param_manager, km_n_key, default_params["km_n"])
        _set_param_value(param_manager, km_p_key, 0.0)
        with pytest.raises(ValueError, match="Michaelis constants must be positive"):
            RootNutrientUptake()
    
    def test_instantiation_invalid_root_decay(self):
        """Test instantiation fails with invalid root decay coefficient."""
        param_manager = get_global_manager()
        decay_key = "pedosphere.nutrient_cycling.root_decay_coeff"
        _set_param_value(param_manager, decay_key, 0.0, validate=False)
        with pytest.raises(ValueError, match="Root decay coefficient must be positive"):
            RootNutrientUptake()
    
    def test_instantiation_invalid_q10(self):
        """Test instantiation fails with invalid Q10 value."""
        param_manager = get_global_manager()
        q10_key = "pedosphere.nutrient_cycling.q10_value"
        _set_param_value(param_manager, q10_key, 1.0)
        with pytest.raises(ValueError, match="Q10 value must be greater than 1"):
            RootNutrientUptake()
    
    # Test 2: Basic Functionality
    def test_compute_rate_basic(self, process, basic_state):
        """Test basic compute_rate functionality."""
        result = call_forward(process, basic_state)
        
        # Check return type and keys
        assert isinstance(result, dict)
        expected_keys = {
            "no3_nitrogen",
            "nh4_nitrogen",
            "available_phosphorus",
            "plant_n_storage",
            "plant_p_storage",
            "nitrogen_uptake_rate",
            "phosphorus_uptake_rate",
            "no3_uptake_by_layer",
            "nh4_uptake_by_layer",
            "po4_uptake_by_layer",
        }
        assert set(result) == expected_keys
        
        # Check tensor shapes and non-negative values
        assert isinstance(result["nitrogen_uptake_rate"], torch.Tensor)
        assert isinstance(result["phosphorus_uptake_rate"], torch.Tensor)
        assert result["nitrogen_uptake_rate"] >= 0
        assert result["phosphorus_uptake_rate"] >= 0
        
        # Check layer uptakes are non-negative
        assert torch.all(result["no3_uptake_by_layer"] >= 0)
        assert torch.all(result["nh4_uptake_by_layer"] >= 0)
        assert torch.all(result["po4_uptake_by_layer"] >= 0)
        
        # Check layer uptake shapes match soil layers
        assert result["no3_uptake_by_layer"].shape == torch.Size([3])
        assert result["nh4_uptake_by_layer"].shape == torch.Size([3])
        assert result["po4_uptake_by_layer"].shape == torch.Size([3])
    
    def test_forward_emits_explicit_update_semantics(self, process, basic_state):
        """All retained outputs have explicit state or observation semantics."""
        updates = process.forward(basic_state, time=0.0, dt=3600.0)

        state_specs = {
            "no3_nitrogen": ("g N m-2", "N"),
            "nh4_nitrogen": ("g N m-2", "N"),
            "available_phosphorus": ("g P m-2", "P"),
            "plant_n_storage": ("mol N m-2", "N"),
            "plant_p_storage": ("g P m-2", "P"),
        }
        for key, (unit, element) in state_specs.items():
            item = updates[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == unit
            assert item.element == element
            assert item.source_functor == process.name

        observation_specs = {
            "nitrogen_uptake_rate": "mol N m-2 s-1",
            "phosphorus_uptake_rate": "g P m-2 s-1",
            "no3_uptake_by_layer": "mol N m-2 s-1",
            "nh4_uptake_by_layer": "mol N m-2 s-1",
            "po4_uptake_by_layer": "g P m-2 s-1",
        }
        for key, unit in observation_specs.items():
            item = updates[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == unit
            assert item.element is None
            assert item.source_functor == process.name

    def test_hidden_concentration_aliases_do_not_change_canonical_result(
        self, process, basic_state
    ):
        """Solute concentrations cannot replace canonical areal N pools."""
        baseline = call_forward(process, basic_state)
        state_with_hidden_aliases = dict(basic_state)
        state_with_hidden_aliases.update(
            {
                "no3_concentration": torch.full((3,), 1.0e9),
                "nh4_concentration": torch.full((3,), 1.0e9),
            }
        )
        observed = call_forward(process, state_with_hidden_aliases)

        for key in baseline:
            assert torch.allclose(observed[key], baseline[key], rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "missing_key",
        [
            "no3_nitrogen",
            "nh4_nitrogen",
            "available_phosphorus",
            "soil_water_content",
            "soil_temperature",
            "soil_depth",
            "plant_n_storage",
            "plant_p_storage",
            "root_density",
            "plant_n_demand",
            "plant_p_demand",
            "field_capacity",
            "wilting_point",
        ],
    )
    def test_missing_canonical_input_fails_fast(
        self, process, basic_state, missing_key
    ):
        """Parameter defaults and aliases cannot replace runtime state inputs."""
        incomplete_state = dict(basic_state)
        incomplete_state.pop(missing_key)

        with pytest.raises(ValueError, match=missing_key):
            process.forward(incomplete_state, time=0.0, dt=3600.0)

    @pytest.mark.parametrize(
        "invalid_key",
        [
            "no3_nitrogen",
            "nh4_nitrogen",
            "available_phosphorus",
            "soil_water_content",
            "soil_temperature",
            "soil_depth",
            "plant_n_storage",
            "plant_p_storage",
            "root_density",
            "plant_n_demand",
            "plant_p_demand",
            "field_capacity",
            "wilting_point",
        ],
    )
    def test_nonfinite_canonical_input_fails_fast(
        self, process, basic_state, invalid_key
    ):
        """Non-finite scientific inputs are rejected before calculation."""
        invalid_state = dict(basic_state)
        invalid_state[invalid_key] = torch.full_like(
            torch.as_tensor(invalid_state[invalid_key]),
            float("nan"),
        )

        with pytest.raises(ValueError, match=invalid_key):
            process.forward(invalid_state, time=0.0, dt=3600.0)
    
    # Test 3: Michaelis-Menten Kinetics
    def test_michaelis_menten_zero_concentration(self, process):
        """Test Michaelis-Menten kinetics with zero concentration."""
        vmax = torch.tensor(1.0e-6)
        concentration = torch.tensor(0.0)
        km = torch.tensor(0.1)
        
        result = process._michaelis_menten_uptake(vmax, concentration, km)
        
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-10)
    
    def test_michaelis_menten_high_concentration(self, process):
        """Test Michaelis-Menten kinetics approaches Vmax at high concentration."""
        vmax = torch.tensor(1.0e-6)
        concentration = torch.tensor(100.0)  # >> Km
        km = torch.tensor(0.1)
        
        result = process._michaelis_menten_uptake(vmax, concentration, km)
        
        # Should approach Vmax
        assert torch.allclose(result, vmax, rtol=1e-3)
    
    def test_michaelis_menten_km_concentration(self, process):
        """Test Michaelis-Menten kinetics at Km concentration gives Vmax/2."""
        vmax = torch.tensor(1.0e-6)
        concentration = torch.tensor(0.1)  # = Km
        km = torch.tensor(0.1)
        
        result = process._michaelis_menten_uptake(vmax, concentration, km)
        
        # Should be Vmax/2
        assert torch.allclose(result, vmax / 2.0, rtol=1e-6)
    
    # Test 4: Temperature Response
    def test_temperature_response_reference(self, process):
        """Test temperature response at reference temperature."""
        temp = torch.tensor(298.15)  # Reference temperature
        
        result = process._compute_temperature_response(temp)
        
        assert torch.allclose(result, torch.tensor(1.0), rtol=1e-6)
    
    def test_temperature_response_high(self, process):
        """Test temperature response at high temperature."""
        temp = torch.tensor(308.15)  # 10K above reference
        
        result = process._compute_temperature_response(temp)
        
        # Should be Q10 = 2.0
        expected = process.q10_value ** 1.0  # (308.15 - 298.15)/10
        assert torch.allclose(result, torch.tensor(expected), rtol=1e-6)
    
    def test_temperature_response_low(self, process):
        """Test temperature response at low temperature."""
        temp = torch.tensor(288.15)  # 10K below reference
        
        result = process._compute_temperature_response(temp)
        
        # Should be 1/Q10 = 0.5
        expected = process.q10_value ** (-1.0)  # (288.15 - 298.15)/10
        assert torch.allclose(result, torch.tensor(expected), rtol=1e-6)
    
    @pytest.mark.parametrize("temperature", [200.0, 400.0])
    def test_temperature_response_rejects_outside_model_domain(
        self, process, temperature
    ):
        """Extreme temperatures fail instead of entering a hidden clipping path."""
        with pytest.raises(ValueError, match="temperature"):
            process._compute_temperature_response(torch.tensor(temperature))
    
    # Test 5: Water Stress Response
    def test_water_stress_field_capacity(self, process):
        """Test water stress response at field capacity."""
        soil_water = torch.tensor([0.35])
        field_capacity = torch.tensor([0.35])
        wilting_point = torch.tensor([0.15])
        
        result = process._compute_water_stress_response(soil_water, field_capacity, wilting_point)
        
        assert torch.allclose(result, torch.tensor([1.0]), rtol=1e-6)
    
    def test_water_stress_wilting_point(self, process):
        """Test water stress response at wilting point."""
        soil_water = torch.tensor([0.15])
        field_capacity = torch.tensor([0.35])
        wilting_point = torch.tensor([0.15])
        
        result = process._compute_water_stress_response(soil_water, field_capacity, wilting_point)
        
        assert torch.allclose(result, torch.tensor([0.0]), rtol=1e-6)
    
    def test_water_stress_midpoint(self, process):
        """Test water stress response at midpoint between FC and WP."""
        soil_water = torch.tensor([0.25])    # Midpoint
        field_capacity = torch.tensor([0.35])
        wilting_point = torch.tensor([0.15])
        
        result = process._compute_water_stress_response(soil_water, field_capacity, wilting_point)
        
        assert torch.allclose(result, torch.tensor([0.5]), rtol=1e-6)
    
    def test_water_stress_disabled(self):
        """Test water stress response when disabled."""
        param_manager = get_global_manager()
        _set_param_value(
            param_manager,
            "pedosphere.nutrient_cycling.water_stress_enabled",
            False,
        )
        process = RootNutrientUptake()
        
        soil_water = torch.tensor([0.1])     # Below wilting point
        field_capacity = torch.tensor([0.35])
        wilting_point = torch.tensor([0.15])
        
        result = process._compute_water_stress_response(soil_water, field_capacity, wilting_point)
        
        assert torch.allclose(result, torch.tensor([1.0]), rtol=1e-6)
    
    # Test 6: Root Weight Distribution
    def test_root_weight_distribution_surface(self, process):
        """Test root weight distribution is highest at surface."""
        soil_depth = torch.tensor([0.0, 0.5, 1.0])
        root_density = torch.tensor([1.0, 1.0, 1.0])
        
        result = process._compute_root_weight_distribution(soil_depth, root_density)
        
        # Surface should have highest weight
        assert result[0] > result[1] > result[2]
        # Weights should sum to 1
        assert torch.allclose(torch.sum(result), torch.tensor(1.0), rtol=1e-6)
    
    def test_root_weight_distribution_normalization(self, process):
        """Test root weight distribution is normalized."""
        soil_depth = torch.tensor([0.1, 0.3, 0.6])
        root_density = torch.tensor([2.0, 1.5, 1.0])  # Different densities
        
        result = process._compute_root_weight_distribution(soil_depth, root_density)
        
        # Should sum to 1
        assert torch.allclose(torch.sum(result), torch.tensor(1.0), rtol=1e-6)
        # All weights should be positive
        assert torch.all(result > 0)
    
    # Test 7: Plant Demand Regulation
    def test_demand_regulation_unlimited(self, basic_state):
        """Uptake is limited by the kinetic potential when the state demand exceeds it
        and by the demand when the state demand lies below it."""
        # High plant demand (declared state inputs), low storage
        state_low_storage = basic_state.copy()
        state_low_storage['plant_n_storage'] = torch.tensor(0.1)  # Low storage
        state_low_storage['plant_p_storage'] = torch.tensor(0.01)  # g P m-2
        state_low_storage['plant_n_demand'] = torch.tensor(1.0e-3)  # Very high
        state_low_storage['plant_p_demand'] = torch.tensor(1.0e-4)  # Very high
        
        param_manager = get_global_manager()
        _set_param_value(
            param_manager,
            "pedosphere.nutrient_cycling.demand_regulation_enabled",
            True,
        )
        process = RootNutrientUptake()
        result = call_forward(process, state_low_storage)
        
        # Uptake should be limited by potential, not demand
        assert result["nitrogen_uptake_rate"] > 0
        assert result["phosphorus_uptake_rate"] > 0
        potential_n = float(result["nitrogen_uptake_rate"])
        assert potential_n < 1.0e-3
        # Discriminating case: a state demand below the potential binds the uptake.
        binding_demand = 0.5 * potential_n
        state_binding = dict(state_low_storage)
        state_binding['plant_n_demand'] = torch.tensor(binding_demand)
        binding = call_forward(process, state_binding)
        assert float(binding["nitrogen_uptake_rate"]) == pytest.approx(binding_demand, rel=1.0e-6)

    def test_demand_regulation_storage_limited(self, basic_state):
        """Test uptake when limited by storage capacity."""
        # High storage near maximum
        param_manager = get_global_manager()
        _set_param_value(
            param_manager,
            "pedosphere.nutrient_cycling.demand_regulation_enabled",
            True,
        )
        process = RootNutrientUptake()
        state_high_storage = basic_state.copy()
        state_high_storage['plant_n_storage'] = torch.tensor(9.9)  # Near max (10.0)
        remaining_capacity_mol = 1.0e-4
        state_high_storage['plant_p_storage'] = torch.tensor(
            (process.max_storage_p - remaining_capacity_mol)
            * process.atomic_weight_phosphorus
        )
        dt = 86400.0
        result = call_forward(
            process,
            state_high_storage,
            dt=dt,
        )

        # Uptake should be very limited due to storage capacity
        assert result["nitrogen_uptake_rate"] < 5.0e-7  # Relaxed threshold
        expected_capacity_rate = (
            remaining_capacity_mol * process.atomic_weight_phosphorus / dt
        )
        assert result["phosphorus_uptake_rate"].item() == pytest.approx(
            expected_capacity_rate, rel=1.0e-4
        )
    
    def test_demand_regulation_disabled(self, basic_state):
        """Test uptake when demand regulation is disabled."""
        param_manager = get_global_manager()
        _set_param_value(
            param_manager,
            "pedosphere.nutrient_cycling.demand_regulation_enabled",
            False,
        )
        process = RootNutrientUptake()
        result = call_forward(process, basic_state)
        
        # Should only be limited by kinetics, not demand or storage
        assert result["nitrogen_uptake_rate"] > 0
        assert result["phosphorus_uptake_rate"] > 0
    
    # Test 8: Layer Partitioning
    def test_no3_nh4_partitioning(self, process, basic_state):
        """Test N uptake is correctly partitioned between NO3 and NH4."""
        result = call_forward(process, basic_state)
        
        # Total N uptake by layer should equal sum of NO3 and NH4 uptake
        total_n_by_layer = result["no3_uptake_by_layer"] + result["nh4_uptake_by_layer"]
        
        # Compare with concentrations derived from canonical areal mass pools.
        molar_mass = get_physical_constant(
            "relic.physical.constants.molecular_weights.nitrogen"
        )
        no3_conc = basic_state["no3_nitrogen"] / molar_mass / basic_state["soil_depth"]
        nh4_conc = basic_state["nh4_nitrogen"] / molar_mass / basic_state["soil_depth"]
        total_n_conc = no3_conc + nh4_conc
        
        # Check partitioning ratios are consistent with concentrations
        for i in range(len(no3_conc)):
            if total_n_conc[i] > 1e-12:  # Avoid division by zero
                expected_no3_fraction = no3_conc[i] / total_n_conc[i]
                actual_no3_fraction = result["no3_uptake_by_layer"][i] / total_n_by_layer[i]
                assert torch.allclose(actual_no3_fraction, expected_no3_fraction, rtol=1e-5)
    
    # Test 9: Conservation and Mass Balance
    def test_mass_balance_check(self, process, basic_state):
        """Test that layer uptakes sum correctly to total uptake."""
        result = call_forward(process, basic_state)
        
        # Sum of layer uptakes should equal total uptakes (approximately)
        total_no3_uptake = torch.sum(result["no3_uptake_by_layer"])
        total_nh4_uptake = torch.sum(result["nh4_uptake_by_layer"])
        total_n_from_layers = total_no3_uptake + total_nh4_uptake
        
        total_p_from_layers = torch.sum(result["po4_uptake_by_layer"])
        
        # Should match within numerical precision
        # (Note: May not be exact due to demand regulation scaling)
        assert torch.allclose(total_n_from_layers, result["nitrogen_uptake_rate"], rtol=1e-5)
        assert torch.allclose(total_p_from_layers, result["phosphorus_uptake_rate"], rtol=1e-5)

    def test_large_step_is_pool_limited_and_closes_nitrogen_mass(
        self, process, basic_state
    ):
        """A large step cannot overdraft N pools or overfill plant storage."""
        state = dict(basic_state)
        state["no3_nitrogen"] = torch.tensor([1.0e-3, 0.0, 0.0])
        state["nh4_nitrogen"] = torch.tensor([2.0e-3, 0.0, 0.0])
        state["plant_n_storage"] = torch.tensor(process.max_storage_n - 1.0e-6)
        state["plant_n_demand"] = torch.tensor(1.0)
        dt = 86400.0

        updates = process.forward(state, time=0.0, dt=dt)
        no3_delta = updates["no3_nitrogen"].value
        nh4_delta = updates["nh4_nitrogen"].value
        plant_delta = updates["plant_n_storage"].value
        molar_mass = get_physical_constant(
            "relic.physical.constants.molecular_weights.nitrogen"
        )

        # Layered donor pools return per-layer increments (D51 contract); each layer stays non-negative.
        assert tuple(no3_delta.shape) == tuple(state["no3_nitrogen"].shape)
        assert tuple(nh4_delta.shape) == tuple(state["nh4_nitrogen"].shape)
        assert torch.all(state["no3_nitrogen"] + no3_delta >= -1.0e-12)
        assert torch.all(state["nh4_nitrogen"] + nh4_delta >= -1.0e-12)
        assert float(plant_delta.detach().cpu().item()) <= 1.0e-6 + 1.0e-10
        residual_mol = no3_delta.sum() / molar_mass + nh4_delta.sum() / molar_mass + plant_delta
        assert torch.allclose(
            residual_mol,
            torch.zeros_like(residual_mol),
            rtol=0.0,
            atol=1.0e-12,
        )
        layer_total = (
            updates["no3_uptake_by_layer"].value
            + updates["nh4_uptake_by_layer"].value
        ).sum(dim=-1)
        assert torch.allclose(
            layer_total,
            updates["nitrogen_uptake_rate"].value,
            rtol=1.0e-6,
            atol=1.0e-15,
        )
    
    # Test 10: Edge Cases and Robustness
    def test_zero_concentrations(self, process):
        """Test behavior with zero nutrient concentrations."""
        state_zero_nutrients = {
            'no3_nitrogen': torch.tensor([0.0, 0.0, 0.0]),
            'nh4_nitrogen': torch.tensor([0.0, 0.0, 0.0]),
            'available_phosphorus': torch.tensor(0.0),
            'soil_water_content': torch.tensor([0.25, 0.22, 0.18]),
            'soil_temperature': torch.tensor(298.15),
            'plant_n_storage': torch.tensor(2.0),
            'plant_p_storage': torch.tensor(0.2),
            'soil_depth': torch.tensor([0.1, 0.3, 0.6]),
            'root_density': torch.tensor([1.0, 0.8, 0.5]),
            'plant_n_demand': torch.tensor(1.5e-6),
            'plant_p_demand': torch.tensor(1.5e-7),
            'field_capacity': torch.tensor([0.35, 0.33, 0.30]),
            'wilting_point': torch.tensor([0.15, 0.12, 0.10]),
        }
        
        result = call_forward(process, state_zero_nutrients)
        
        # Should return zero uptakes
        assert torch.allclose(result["nitrogen_uptake_rate"], torch.tensor(0.0), atol=1e-12)
        assert torch.allclose(result["phosphorus_uptake_rate"], torch.tensor(0.0), atol=1e-12)
        assert torch.allclose(result["no3_uptake_by_layer"], torch.tensor([0.0, 0.0, 0.0]), atol=1e-12)
        assert torch.allclose(result["nh4_uptake_by_layer"], torch.tensor([0.0, 0.0, 0.0]), atol=1e-12)
        assert torch.allclose(result["po4_uptake_by_layer"], torch.tensor([0.0, 0.0, 0.0]), atol=1e-12)
    
    def test_zero_root_density(self, process, basic_state):
        """Test behavior with zero root density."""
        state_no_roots = basic_state.copy()
        state_no_roots['root_density'] = torch.tensor([0.0, 0.0, 0.0])
        
        result = call_forward(process, state_no_roots)
        
        # Should return zero uptakes due to no roots
        assert torch.allclose(result["nitrogen_uptake_rate"], torch.tensor(0.0), atol=1e-12)
        assert torch.allclose(result["phosphorus_uptake_rate"], torch.tensor(0.0), atol=1e-12)
    
    def test_extreme_dry_conditions(self, process, basic_state):
        """Test behavior under extreme dry conditions."""
        state_dry = basic_state.copy()
        state_dry['soil_water_content'] = torch.tensor([0.05, 0.03, 0.02])  # Below wilting point
        
        result = call_forward(process, state_dry)
        
        # Uptake should be severely limited by water stress
        if process.water_stress_enabled:
            assert result["nitrogen_uptake_rate"] < 1e-8
            assert result["phosphorus_uptake_rate"] < 1e-9
    
    # Test 11: Diagnostic Information
    def test_diagnostic_info(self, process, basic_state):
        """Test diagnostic information retrieval."""
        # Run compute_rate to populate diagnostics
        result = call_forward(process, basic_state)
        
        diagnostics = process.get_diagnostic_info()
        
        # Check diagnostic keys
        expected_diagnostic_keys = {
            "last_n_uptake_mol_m2_s",
            "last_p_uptake_mol_m2_s", 
            "last_water_stress_factor",
            "last_temperature_response",
            "vmax_n_mol_m2_s",
            "vmax_p_mol_m2_s",
            "km_n_mol_m3",
            "km_p_mol_m3",
            "root_decay_coeff_m",
        }
        assert set(diagnostics.keys()) == expected_diagnostic_keys
        
        # Check diagnostic values are reasonable
        assert isinstance(diagnostics["last_n_uptake_mol_m2_s"], float)
        assert isinstance(diagnostics["last_p_uptake_mol_m2_s"], float)
        assert diagnostics["last_water_stress_factor"] >= 0.0
        assert diagnostics["last_water_stress_factor"] <= 1.0
        assert diagnostics["last_temperature_response"] > 0.0
    
    # Test 12: Parameter Sensitivity. vmax_n and km_n are ParameterManager
    # learnable bindings held by reference (item 74): parameter_sensitivity
    # refuses them, and their derivative is taken on the manager calibration
    # path (autograd through the governed call with respect to the trainable
    # binding).
    @staticmethod
    def _trainable_binding(process, name):
        binding = getattr(process, name)
        key = process.manager_binding_key(binding)
        assert key is not None, name
        assert get_global_manager().get_trainable_parameters()[key] is binding, name
        return binding

    def test_parameter_sensitivity_vmax_n(self, process, basic_state):
        """vmax_n is a held manager binding, so parameter_sensitivity refuses it;
        d(uptake)/d(vmax_n) through the governed call on the manager calibration
        path is U / vmax_n (uptake is linear in vmax_n while the demand does not
        bind)."""
        assert "parameter_sensitivity" not in RootNutrientUptake.__dict__
        with pytest.raises(TypeError, match="ParameterManager learnable binding"):
            process.parameter_sensitivity(
                "vmax_n", basic_state, time=0.0, dt=1.0, target_output="nitrogen_uptake_rate"
            )
        vmax_n = self._trainable_binding(process, "vmax_n")
        uptake = process(dict(basic_state), time=0.0, dt=1.0)["nitrogen_uptake_rate"].value
        (derivative,) = torch.autograd.grad(uptake.sum(), vmax_n)
        assert float(derivative) > 0
        expected = float(uptake.detach().sum()) / float(vmax_n.detach().reshape(-1)[0])
        assert float(derivative) == pytest.approx(expected, rel=1.0e-5)

    def test_parameter_sensitivity_km_n(self, process, basic_state):
        """km_n is a held manager binding, so parameter_sensitivity refuses it; a
        higher Km reduces uptake, so the derivative on the manager calibration
        path is negative."""
        with pytest.raises(TypeError, match="ParameterManager learnable binding"):
            process.parameter_sensitivity(
                "km_n", basic_state, time=0.0, dt=1.0, target_output="nitrogen_uptake_rate"
            )
        km_n = self._trainable_binding(process, "km_n")
        uptake = process(dict(basic_state), time=0.0, dt=1.0)["nitrogen_uptake_rate"].value
        (derivative,) = torch.autograd.grad(uptake.sum(), km_n)
        assert float(derivative) < 0
    
    def test_parameter_sensitivity_invalid_parameter(self, process, basic_state):
        """Test parameter sensitivity with invalid parameter name."""
        with pytest.raises(KeyError, match="Parameter .* not found"):
            process.parameter_sensitivity(
                "invalid_param", basic_state, time=0.0, dt=1.0
            )

    def test_parameter_sensitivity_rejects_a_float_member(self, process, basic_state):
        """Only tensor members (nn.Parameter, buffer, tensor attribute) are perturbed;
        q10_value is a float member and is not supported."""
        with pytest.raises(TypeError, match="tensor member"):
            process.parameter_sensitivity(
                "q10_value", basic_state, time=0.0, dt=1.0,
                target_output="nitrogen_uptake_rate",
            )

    def test_parameter_sensitivity_failure_is_not_replaced_by_forward_difference(
        self, process, basic_state, monkeypatch
    ):
        """No substitute scheme: the refusal of a held manager binding comes before
        any evaluation, so no finite-difference (central or one-sided) evaluation of
        the kernel runs in its place."""
        original_core = process._compute_rate_core
        evaluations = []

        def counting_core(state):
            evaluations.append(1)
            return original_core(state)

        monkeypatch.setattr(process, "_compute_rate_core", counting_core)
        with pytest.raises(TypeError, match="ParameterManager learnable binding"):
            process.parameter_sensitivity(
                "vmax_n", basic_state, time=0.0, dt=1.0,
                target_output="nitrogen_uptake_rate",
            )
        assert evaluations == []


    def test_rate_core_reads_the_state_only(self, process, basic_state):
        """The one rate core takes the state only; the forward validates the declared
        state inputs and passes no environment mapping."""
        assert list(inspect.signature(RootNutrientUptake._compute_rate_core).parameters) == [
            "self",
            "state",
        ]
        rates = process._compute_rate_core(dict(basic_state))
        assert float(rates["nitrogen_uptake_rate"]) > 0.0


class TestRootNutrientUptakeIntegration:
    """Integration tests with other modules."""
    
    def test_units_consistency(self):
        """Test that all units are dimensionally consistent."""
        # This is more of a documentation test to ensure units are correct
        process = RootNutrientUptake()
        
        # Check parameter units match documentation
        assert isinstance(process.vmax_n, torch.nn.Parameter)  # Should be learnable
        assert isinstance(process.vmax_p, torch.nn.Parameter)  # Should be learnable
        assert isinstance(process.km_n, torch.nn.Parameter)    # Should be learnable
        assert isinstance(process.km_p, torch.nn.Parameter)    # Should be learnable
        
        # Check diagnostic info has expected units in names
        diagnostics = process.get_diagnostic_info()
        assert "mol_m2_s" in str(diagnostics.keys())  # Rate units
        assert "mol_m3" in str(diagnostics.keys())    # Concentration units
        assert "m" in str(diagnostics.keys())         # Length units
    
    def test_torch_device_consistency(self):
        """Test consistency with different torch devices."""
        # Test CPU device (default)
        process_cpu = RootNutrientUptake(device=torch.device("cpu"))
        assert process_cpu.device == torch.device("cpu")
        
        # Test parameters are on correct device
        assert process_cpu.vmax_n.device == torch.device("cpu")
        assert process_cpu.vmax_p.device == torch.device("cpu")
    
    def test_pytorch_module_inheritance(self):
        """Test that the class properly inherits from nn.Module."""
        process = RootNutrientUptake()
        
        # Should be instance of nn.Module
        assert isinstance(process, torch.nn.Module)
        
        # The learnable parameters are ParameterManager bindings held by
        # reference (item 74): the manager trainable interface owns them and
        # the module registers none of them.
        expected_params = {"vmax_n", "vmax_p", "km_n", "km_p"}
        trainable = get_global_manager().get_trainable_parameters()
        module_parameters = dict(process.named_parameters())
        for name in expected_params:
            binding = getattr(process, name)
            key = process.manager_binding_key(binding)
            assert key is not None, name
            assert trainable[key] is binding, name
            assert binding.requires_grad is True, name
            assert name not in module_parameters, name


def test_forward_kernel_takes_time_and_dt_without_defaults():
    """Round 47: the forward-only kernel signature is (state, time, dt)."""
    parameters = inspect.signature(RootNutrientUptake.forward).parameters
    assert parameters["time"].default is inspect.Parameter.empty
    assert parameters["dt"].default is inspect.Parameter.empty


def test_no_enzyme_kinetics_label_is_passed():
    """Round 48b: no algorithm reads the enzyme_kinetics label; the Functor passes none."""
    assert "enzyme_kinetics" not in inspect.getsource(RootNutrientUptake.__init__)


def test_no_temperature_response_or_ph_dependence_label_is_passed():
    """Round 50.6: no algorithm reads the bridge temperature_response and
    ph_dependence labels; the Functor passes neither."""
    source = inspect.getsource(RootNutrientUptake.__init__)
    assert "temperature_response=" not in source
    assert "ph_dependence=" not in source
