"""
Unit Tests for Bucket Soil Moisture Model

Tests the BucketSoilMoisture functor for:
- Initialization
- Forward method
- Water balance
- Physical constraints
"""

import pytest
import torch

from pathlib import Path

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import ExecutionError, IPHSDECFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.hydrosphere.soil_moisture import BucketSoilMoisture


EXPECTED_PARAMETER_KEYS = (
    "hydrosphere.soil_moisture.soil_depth",
    "hydrosphere.soil_moisture.field_capacity",
    "hydrosphere.soil_moisture.wilting_point",
    "hydrosphere.soil_moisture.porosity",
    "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
    "hydrosphere.soil_moisture.drainage_factor",
    "hydrosphere.soil_moisture.drainage_exponent",
    "hydrosphere.soil_moisture.water_input_source",
)
EXPECTED_PHYSICAL_CONSTANT_KEYS = (
    "relic.physical.constants.water.density_liquid",
)
EXPECTED_OUTPUT_UNITS = {
    "soil_moisture": "m3 m-3",
    "surface_runoff_flux": "kg m-2 s-1",
    "drainage_flux": "kg m-2 s-1",
}


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


class TestBucketSoilMoistureInitialization:
    """Test initialization of Bucket soil moisture model."""
    
    def test_basic_initialization(self):
        """Test that BucketSoilMoisture can be initialized."""
        functor = BucketSoilMoisture()
        assert functor.name == "bucket_soil_moisture"
        assert functor is not None
    
    def test_custom_name(self):
        """Test initialization with custom name."""
        functor = BucketSoilMoisture(name="custom_bucket")
        assert functor.name == "custom_bucket"

    def test_parameter_and_physical_constant_owners_are_disjoint(self):
        """Test exact ParameterManager and physical-constant declarations."""
        functor = BucketSoilMoisture()

        assert functor.required_parameters() == EXPECTED_PARAMETER_KEYS
        assert functor.physical_constant_keys() == EXPECTED_PHYSICAL_CONSTANT_KEYS
        assert set(functor.required_parameters()).isdisjoint(
            functor.physical_constant_keys()
        )
        # metadata.parameters declares the full ParameterManager keys the bucket reads.
        assert set(functor.metadata.parameters) == set(EXPECTED_PARAMETER_KEYS)

    def test_drainage_factor_has_explicit_bounds(self):
        """The drainage factor f_d of K_eff = f_d K_sat is a formal parameter, not a
        default."""
        functor = BucketSoilMoisture()
        manager = functor.parameter_manager
        spec = manager.get_parameter("hydrosphere.soil_moisture.drainage_factor")

        assert spec.bounds == (0.0, 1.0)
        assert spec.references
        assert manager.get_parameter("hydrosphere.soil_moisture.drainage_factor_default") is None

    def test_transport_label_and_no_unused_transport_reads(self):
        """Gravity drainage is vertical percolation without a diffusion term; the bucket
        reads no diffusivity or tortuosity, and no minimum drainage rate (below field
        capacity the drainage is bounded by zero, so that branch never acted)."""
        functor = BucketSoilMoisture()
        assert functor.transport_type == "vertical_percolation"
        assert functor.diffusivity is None
        for key in (
            "hydrosphere.transport.soil_moisture_diffusivity",
            "hydrosphere.transport.soil_tortuosity",
            "hydrosphere.soil_moisture.minimum_drainage",
        ):
            assert key not in functor.required_parameters()

    def test_no_hidden_boundary_condition_state(self):
        """Test the bucket has no mutable non-checkpoint drainage branch."""
        functor = BucketSoilMoisture()

        assert "_drainage_factor" not in functor.__dict__
        assert "set_boundary_conditions" not in BucketSoilMoisture.__dict__

    def test_dependencies_are_canonical_and_not_mutable_fallbacks(self):
        """ET scheduling stays lagged; instance declarations are merged, never dropped.

        An instance-declared provider (add_dependency, config, GraphExecutor
        links) becomes a required edge; naming a lagged ET provider again does
        not turn the lagged read into a same-step edge.
        """
        functor = BucketSoilMoisture()
        expected = {
            "required": [],
            "optional": [],
            "lagged": ["penman_monteith_et", "priestley_taylor_et"],
        }

        assert functor.get_dependencies() == expected
        functor.add_dependency("declared_runtime_provider")
        functor.add_dependency("penman_monteith_et")
        assert functor.get_dependencies() == {
            "required": ["declared_runtime_provider"],
            "optional": [],
            "lagged": ["penman_monteith_et", "priestley_taylor_et"],
        }


class TestBucketSoilMoistureForward:
    """Test forward method computation."""
    
    def test_forward_basic(self, soil_moisture_state, time_params):
        """Test forward method with valid inputs."""
        functor = BucketSoilMoisture()
        result = functor.forward(
            soil_moisture_state,
            time=time_params["time"],
            dt=time_params["dt"]
        )
        
        # Check primary output exists
        assert "soil_moisture" in result
        soil_moisture = result["soil_moisture"]
        assert isinstance(soil_moisture, UpdateItem)
        assert soil_moisture.mode == UpdateMode.REPLACE
        assert isinstance(soil_moisture.value, torch.Tensor)
        assert set(result) == set(functor.metadata.output_variables)

    def test_output_contract_cannot_be_mutated_by_private_test_hook(
        self, soil_moisture_state, time_params
    ):
        """Test runtime always returns the declared canonical output set."""
        functor = BucketSoilMoisture()
        functor._tutorial_drop_outputs = {"drainage_flux"}

        result = functor.forward(
            soil_moisture_state,
            time=time_params["time"],
            dt=time_params["dt"],
        )

        assert set(result) == set(functor.metadata.output_variables)
    
    def test_forward_drainage_output(self, soil_moisture_state, time_params):
        """Test that drainage flux is produced."""
        functor = BucketSoilMoisture()
        result = functor.forward(
            soil_moisture_state,
            time=time_params["time"],
            dt=time_params["dt"]
        )
        
        assert "drainage_flux" in result

    def test_forward_outputs_replace_mode(self, soil_moisture_state, time_params):
        """Test that rate and flux outputs use REPLACE mode."""
        functor = BucketSoilMoisture()
        result = functor.forward(
            soil_moisture_state,
            time=time_params["time"],
            dt=time_params["dt"]
        )

        for key, unit in EXPECTED_OUTPUT_UNITS.items():
            assert isinstance(result[key], UpdateItem)
            assert result[key].mode == UpdateMode.REPLACE
            assert result[key].per_timestep is False
            assert result[key].unit == unit
            assert result[key].source_functor == functor.name

    @pytest.mark.parametrize(
        "missing_key",
        ["precipitation_flux", "evapotranspiration_flux"],
    )
    def test_missing_required_flux_fails_fast(
        self, soil_moisture_state, time_params, missing_key
    ):
        """Test required boundary fluxes are never filled by hidden defaults."""
        functor = BucketSoilMoisture()
        state = dict(soil_moisture_state)
        state.pop(missing_key)

        with pytest.raises(ValueError, match=missing_key):
            functor.forward(state, time=time_params["time"], dt=time_params["dt"])

    @pytest.mark.parametrize(
        "state_key",
        ["soil_moisture", "precipitation_flux", "evapotranspiration_flux"],
    )
    def test_non_tensor_state_fails_fast(
        self, soil_moisture_state, time_params, state_key
    ):
        """Test canonical StateManager inputs remain tensors."""
        functor = BucketSoilMoisture()
        state = dict(soil_moisture_state)
        state[state_key] = float(state[state_key])

        with pytest.raises(TypeError, match=f"{state_key} must be a torch.Tensor"):
            functor.forward(state, time=time_params["time"], dt=time_params["dt"])

    @pytest.mark.parametrize(
        "state_key",
        ["soil_moisture", "precipitation_flux", "evapotranspiration_flux"],
    )
    def test_non_finite_state_fails_fast(
        self, soil_moisture_state, time_params, state_key
    ):
        """Test NaN boundary and storage inputs are rejected."""
        functor = BucketSoilMoisture()
        state = dict(soil_moisture_state)
        state[state_key] = torch.tensor(float("nan"))

        with pytest.raises(ValueError, match=f"{state_key} must be finite"):
            functor.forward(state, time=time_params["time"], dt=time_params["dt"])

    @pytest.mark.parametrize("invalid_dt", [float("nan"), float("inf")])
    def test_non_finite_dt_fails_fast(
        self, soil_moisture_state, invalid_dt
    ):
        """Test the explicit integration interval is finite and positive."""
        functor = BucketSoilMoisture()

        with pytest.raises(ValueError, match="Time step must be finite and positive"):
            functor.forward(soil_moisture_state, time=0.0, dt=invalid_dt)


class TestBucketSoilMoisturePhysicalConstraints:
    """Test physical constraints."""
    
    def test_soil_moisture_bounds(self, soil_moisture_state, time_params):
        """Test that soil moisture stays within physical bounds."""
        functor = BucketSoilMoisture()
        result = functor.forward(
            soil_moisture_state,
            time=time_params["time"],
            dt=time_params["dt"]
        )
        
        sm = result["soil_moisture"]
        sm_value = _unwrap_update(sm)
        assert torch.all(sm_value >= 0), "Soil moisture must be non-negative"
        assert torch.all(sm_value <= 1.0), "Soil moisture must not exceed saturation"
    
    def test_drainage_non_negative(self, soil_moisture_state, time_params):
        """Test that drainage flux is non-negative."""
        functor = BucketSoilMoisture()
        result = functor.forward(
            soil_moisture_state,
            time=time_params["time"],
            dt=time_params["dt"]
        )
        
        if "drainage_flux" in result:
            drainage_flux = _unwrap_update(result["drainage_flux"])
            assert torch.all(drainage_flux >= 0)


class TestBucketSoilMoistureWaterBalance:
    """Test water balance conservation."""
    
    def test_water_balance_simple(self, time_params):
        """Test simple water balance: input - output = change in storage."""
        functor = BucketSoilMoisture()
        
        # Start with known state
        initial_sm = 0.3
        precip = 0.001  # kg m-2 s-1
        et = 0.0001  # kg m-2 s-1
        dt = time_params["dt"]
        
        state = {
            "soil_moisture": torch.tensor(initial_sm),
            "precipitation_flux": torch.tensor(precip),
            "evapotranspiration_flux": torch.tensor(et),
            "air_temperature": torch.tensor(293.15),
        }
        
        result = functor.forward(state, time=0.0, dt=dt)
        final_sm = _unwrap_update(result["soil_moisture"])
        
        # Net input should increase soil moisture (if below field capacity)
        # or generate drainage (if at/above field capacity)
        net_input = precip - et
        if net_input > 0:
            # Soil moisture should increase or drainage should occur
            drainage_flux = _unwrap_update(result["drainage_flux"])
            has_drainage = ("drainage_flux" in result and torch.any(drainage_flux > 0))
            if not has_drainage:
                assert final_sm >= initial_sm, "Soil moisture should increase with net positive input"

    def test_noncanonical_soil_moisture_unit_branch_is_rejected(self, time_params):
        """Test canonical soil moisture accepts only volumetric input."""
        functor = BucketSoilMoisture()
        dt = time_params["dt"]
        base_state = {
            "soil_moisture": torch.tensor(0.35),
            "precipitation_flux": torch.tensor(0.001),
            "evapotranspiration_flux": torch.tensor(0.0001),
            "air_temperature": torch.tensor(293.15),
        }
        soil_moisture_mass = (
            base_state["soil_moisture"]
            * torch.as_tensor(functor.water_density, dtype=base_state["soil_moisture"].dtype)
            * torch.as_tensor(functor.soil_depth, dtype=base_state["soil_moisture"].dtype)
        )
        units_state = {
            **{**base_state, "soil_moisture": soil_moisture_mass},
            "output_units": {"soil_moisture": "kg m-2"},
        }

        with pytest.raises(ValueError, match="Soil moisture content must be in range"):
            functor.forward(units_state, time=0.0, dt=dt)


RAIN_AND_MELT = {"water_input_source": "rain_flux_plus_snow_melt_flux"}


def _bucket_storage(functor, soil_moisture):
    """Soil water storage W = theta * Z * rho_w [kg m-2] of one bucket state."""
    depth = torch.as_tensor(functor.soil_depth, dtype=soil_moisture.dtype)
    density = torch.as_tensor(functor.water_density, dtype=soil_moisture.dtype)
    return soil_moisture * depth * density


class TestBucketWaterInputSource:
    """The declared water input selects exactly one representation of the water
    reaching the soil store.

    Below a snowpack the soil-surface input is the rain reaching the ground plus the
    snowmelt (E3SM ELM SnowHydrologyMod, column without snow layers:
    qflx_top_soil = qflx_rain_grnd + qflx_snomelt); without a snowpack it is the total
    precipitation. The source is a declaration (YAML SSOT, per-instance config), never
    inferred from the variables present in the state."""

    def test_default_source_keeps_the_precipitation_interface(self):
        functor = BucketSoilMoisture()
        assert functor.water_input_source == "precipitation_flux"
        assert functor.metadata.input_variables == [
            "soil_moisture", "precipitation_flux", "evapotranspiration_flux",
        ]
        assert "hydrosphere.soil_moisture.water_input_source" in functor.required_parameters()

    def test_rain_and_melt_source_declares_the_snowpack_outflow(self):
        from relic.contracts import load_contract_registry

        registry = load_contract_registry()
        owner = registry.get_owner("rain_flux")
        assert owner == registry.get_owner("snow_melt_flux") == "snow_pack"
        functor = BucketSoilMoisture(config=dict(RAIN_AND_MELT))
        assert functor.water_input_source == "rain_flux_plus_snow_melt_flux"
        assert functor.metadata.input_variables == [
            "soil_moisture", "evapotranspiration_flux", "rain_flux", "snow_melt_flux",
        ]
        # The snowpack outflow is consumed as a committed value: a lagged provider,
        # which also acknowledges the hold read when the snowpack runs at a slower rate.
        assert functor.get_dependencies() == {
            "required": [],
            "optional": [],
            "lagged": ["penman_monteith_et", "priestley_taylor_et", owner],
        }

    def test_unknown_source_fails_fast(self):
        with pytest.raises(ValueError, match="water_input_source"):
            BucketSoilMoisture(config={"water_input_source": "throughfall"})

    @pytest.mark.parametrize(
        "config, expected_input",
        [({}, 2.0e-4), (RAIN_AND_MELT, 5.0e-5 + 1.0e-4)],
    )
    def test_only_the_declared_representation_is_read(self, config, expected_input):
        """No double counting: with every representation present, the storage change
        closes on the declared input only: rho Z dtheta = (P_in - ET - Q - D) dt."""
        dt = 3600.0
        state = {
            "soil_moisture": torch.tensor(0.2),
            "precipitation_flux": torch.tensor(2.0e-4),
            "rain_flux": torch.tensor(5.0e-5),
            "snow_melt_flux": torch.tensor(1.0e-4),
            "evapotranspiration_flux": torch.tensor(1.0e-5),
        }
        functor = BucketSoilMoisture(config=dict(config))
        result = functor.forward(state, time=0.0, dt=dt)
        storage_change = _bucket_storage(
            functor, _unwrap_update(result["soil_moisture"])
        ) - _bucket_storage(functor, state["soil_moisture"])
        outflow = (
            _unwrap_update(result["surface_runoff_flux"])
            + _unwrap_update(result["drainage_flux"])
            + result["diagnostics.soil_moisture_bucket"]["adjusted_evapotranspiration"]
        )
        assert torch.isclose(
            storage_change,
            (torch.tensor(expected_input) - outflow) * dt,
            rtol=1e-9,
            atol=1e-12,
        )

    def test_missing_declared_representation_fails_fast(self):
        with pytest.raises(ValueError, match="snow_melt_flux"):
            BucketSoilMoisture(config=dict(RAIN_AND_MELT)).forward(
                {
                    "soil_moisture": torch.tensor(0.2),
                    "rain_flux": torch.tensor(1.0e-4),
                    "evapotranspiration_flux": torch.tensor(0.0),
                },
                time=0.0,
                dt=3600.0,
            )
        with pytest.raises(ValueError, match="precipitation_flux"):
            BucketSoilMoisture().forward(
                {
                    "soil_moisture": torch.tensor(0.2),
                    "rain_flux": torch.tensor(1.0e-4),
                    "snow_melt_flux": torch.tensor(0.0),
                    "evapotranspiration_flux": torch.tensor(0.0),
                },
                time=0.0,
                dt=3600.0,
            )

    def test_fast_bucket_holds_the_outflow_of_a_medium_snowpack(self):
        """Cross-rate exchange in one governed run: the snowpack runs in the medium
        group (H = 3 h), the bucket in the fast group (h = 1 h). With
        strict_cross_group_dependencies the publication is admitted only because the
        hold read is acknowledged. Each fast step reads the rain and melt committed by
        the last snowpack call (zero-order hold, one medium interval late), the bucket
        storage closes per step on that held input, and the water handed over equals
        the water released by the snowpack calls whose interval has been consumed."""
        from relic.core.simulator import RELICSimulator
        from relic.scientific.hydrosphere.snow.snowpack import SnowPack

        h, ratio = 3600.0, 3
        simulator = RELICSimulator({
            "time_step": h,
            "max_time": 12.0 * h,
            "device": "cpu",
            "medium_interval_seconds": ratio * h,
            "strict_cross_group_dependencies": True,
            "state_units": {"sublimation_flux": "kg m-2 s-1"},
        })
        simulator.initialize_state({
            "precipitation_flux": torch.tensor([3.0e-5]),
            "air_temperature": torch.tensor([278.15]),
            "snow_water_equivalent": torch.tensor([10.0]),
            "sublimation_flux": torch.tensor([0.0]),
            "soil_moisture": torch.tensor([0.25]),
            "evapotranspiration_flux": torch.tensor([0.0]),
            "rain_flux": torch.tensor([0.0]),
            "snow_melt_flux": torch.tensor([0.0]),
        })
        simulator.register_functor(SnowPack(), execution_group="medium")
        bucket = BucketSoilMoisture(config=dict(RAIN_AND_MELT))
        simulator.register_functor(bucket, execution_group="fast")

        held_inputs, released = [], []
        for _ in range(2 * ratio):
            before = simulator.state_manager.get_state()
            held = (before["rain_flux"] + before["snow_melt_flux"]).reshape(-1).detach().clone()
            theta_before = before["soil_moisture"].reshape(-1).detach().clone()
            simulator.step(dt=h)
            after = simulator.state_manager.get_state()
            storage_change = _bucket_storage(
                bucket, after["soil_moisture"].reshape(-1).detach()
            ) - _bucket_storage(bucket, theta_before)
            outflow = (after["surface_runoff_flux"] + after["drainage_flux"]).reshape(-1).detach()
            assert torch.allclose(storage_change, (held - outflow) * h, rtol=1e-6, atol=1e-9)
            held_inputs.append(held)
            released.append((after["rain_flux"] + after["snow_melt_flux"]).reshape(-1).detach().clone())

        # Zero-order hold: the first medium interval reads the declared initial
        # outflow, the second the value committed by the first snowpack call.
        first_call = released[ratio - 1]
        assert torch.all(first_call > 0.0)
        for k in range(ratio):
            assert torch.equal(held_inputs[k], torch.zeros_like(first_call))
            assert torch.equal(held_inputs[ratio + k], first_call)
        handed_over = sum(value * h for value in held_inputs)
        assert torch.allclose(handed_over, first_call * ratio * h, rtol=1e-12, atol=0.0)


class TestBucketSoilMoisturePhase8:
    """Test Phase 8 IPHS contract."""

    def test_phase8_iphs_protocol(self):
        """Test structural IPHS hooks."""
        functor = BucketSoilMoisture()

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure({})
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.soil_moisture.soil_depth" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "soil_moisture",
            "precipitation_flux",
            "evapotranspiration_flux",
            "surface_runoff_flux",
            "drainage_flux",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {
            "bucket_evapotranspiration_flux",
            "bucket_surface_runoff_flux",
            "bucket_drainage_flux",
        }

    def test_phase8_certificate_is_valid(self):
        """Test Phase 8 certificate completeness."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_soil_moisture_bucket_bucket_soil_moisture.yaml")
        )
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == (
            "relic.scientific.hydrosphere.soil_moisture.bucket.BucketSoilMoisture"
        )
        assert certificate.canonical_role == "transport"
        functor = BucketSoilMoisture()
        # The class interface holds both declared water input representations; an
        # instance declares the one its water_input_source selects.
        assert tuple(certificate.input_variables) == tuple(
            functor.metadata.input_variables
        ) + ("rain_flux", "snow_melt_flux")
        assert tuple(certificate.output_variables) == tuple(
            functor.metadata.output_variables
        )
        assert tuple(certificate.parameter_keys) == functor.required_parameters()
        physical_key = EXPECTED_PHYSICAL_CONSTANT_KEYS[0]
        assert physical_key not in certificate.parameter_map
        assert certificate.parameter_management_map[physical_key]["source"] == (
            "physical constants SSOT via get_physical_constant"
        )

    def test_residual_diagnostics_are_routed(self, soil_moisture_state, time_params):
        """Test projection residual diagnostics are exposed."""
        functor = BucketSoilMoisture()
        result = functor.forward(
            soil_moisture_state,
            time=time_params["time"],
            dt=time_params["dt"],
        )

        diagnostics = result["diagnostics.soil_moisture_bucket"]
        assert "clipped_mass_residual" in diagnostics
        assert "drainage_limiter_residual" in diagnostics
        assert "soil_moisture_change_rate" in diagnostics
        assert "soil_water_storage" in diagnostics
        assert torch.isfinite(diagnostics["clipped_mass_residual"]).all()
        assert torch.isfinite(diagnostics["drainage_limiter_residual"]).all()
        assert torch.isfinite(diagnostics["soil_moisture_change_rate"]).all()
        assert torch.isfinite(diagnostics["soil_water_storage"]).all()
        assert "soil_moisture_change" not in result
        assert "soil_water_storage" not in result

    def test_conservation_uses_explicit_water_ledger(
        self, soil_moisture_state, time_params
    ):
        """Test mass validation reads the routed mass-flux residual only."""
        functor = BucketSoilMoisture()
        result = functor(
            soil_moisture_state,
            time=time_params["time"],
            dt=time_params["dt"],
        )
        residual = result["diagnostics.soil_moisture_bucket"][
            "clipped_mass_residual"
        ]
        expected = float(torch.max(torch.abs(residual)).detach().item())

        assert functor.check_conservation_laws(soil_moisture_state, result) == {
            "mass": expected
        }


# References: Balsamo, G. et al. (2008), ECMWF Technical Memorandum 563, Table 2.
# Mathematical expression: theta_pwp < theta_cap < theta_sat for every texture class.
# Parameter meanings: theta_sat volumetric water content at saturation (porosity),
#   theta_cap at field capacity (matric potential -0.10 bar), theta_pwp at the
#   permanent wilting point (-15 bar); tuple order (theta_sat, theta_cap, theta_pwp).
# Units: m3 m-3.
# Domain/constraints: the six van Genuchten texture classes of the source table.
# Sign convention: not applicable.
ECMWF_TM563_TABLE_2_SOIL_CLASSES = {
    "coarse": (0.403, 0.244, 0.059),
    "medium": (0.439, 0.347, 0.151),
    "medium_fine": (0.430, 0.383, 0.133),
    "fine": (0.520, 0.448, 0.279),
    "very_fine": (0.614, 0.541, 0.335),
    "organic": (0.766, 0.663, 0.267),
}


@pytest.mark.parametrize("soil_class", sorted(ECMWF_TM563_TABLE_2_SOIL_CLASSES))
def test_owner_bounds_admit_the_published_soil_texture_classes(
    soil_class, restore_global_parameter_manager,
):
    """The owner bounds of porosity, field capacity and wilting point admit every soil
    texture class of the published table: a class value is stored as requested, not
    moved to a bound, and the bucket is built with it."""
    manager = restore_global_parameter_manager
    names = ("porosity", "field_capacity", "wilting_point")
    values = dict(zip(names, ECMWF_TM563_TABLE_2_SOIL_CLASSES[soil_class]))
    for name, value in values.items():
        key = "hydrosphere.soil_moisture." + name
        lower, upper = manager.get_parameter(key).bounds
        assert lower <= value <= upper, (key, value, (lower, upper))
        manager.set_value(key, value, units="m3 m-3", validate=True)
        assert float(manager.get(key)) == pytest.approx(value, rel=1.0e-6)
    functor = BucketSoilMoisture()
    assert float(functor.saturation) == pytest.approx(values["porosity"], rel=1.0e-6)
    assert float(functor.field_capacity) == pytest.approx(values["field_capacity"], rel=1.0e-6)
    assert float(functor.wilting_point) == pytest.approx(values["wilting_point"], rel=1.0e-6)


# References: Manabe (1969), Mon. Wea. Rev. 97:739-774; Balsamo, G. et al. (2008),
#   ECMWF Technical Memorandum 563, Table 2 (theta_pwp up to 0.335 and theta_cap
#   from 0.244 over the six texture classes, so the owner intervals of the wilting
#   point and of the field capacity overlap).
# Mathematical expression: beta = clamp((theta - theta_wp) / (theta_fc - theta_wp),
#   0, 1), defined only for theta_wp < theta_fc.
# Parameter meanings: theta_wp wilting point, theta_fc field capacity, beta the
#   water stress factor; the parameter is the excess theta_wp - theta_fc written
#   after construction.
# Units: theta, theta_wp, theta_fc and the excess [m3 m-3]; beta [1].
# Domain/constraints: the written wilting point lies inside its own owner bounds
#   while the pair violates theta_wp < theta_fc.
# Sign convention: not applicable.
@pytest.mark.parametrize("excess", [0.035, 0.0], ids=["above", "equal"])
def test_call_rejects_a_wilting_point_not_below_the_field_capacity(
    excess, restore_global_parameter_manager, time_params,
):
    """A wilting point written at or above the field capacity after construction is
    inside its owner bounds and reaches the built bucket; the next call rejects the
    pair instead of evaluating the water balance with a non-positive
    (field capacity - wilting point)."""
    manager = restore_global_parameter_manager
    functor = BucketSoilMoisture()
    key = "hydrosphere.soil_moisture.wilting_point"
    field_capacity = float(functor.field_capacity.detach())
    wilting_point = field_capacity + excess
    lower, upper = manager.get_parameter(key).bounds
    assert lower <= wilting_point <= upper
    manager.set_value(key, wilting_point, units="m3 m-3", validate=True)
    assert float(functor.wilting_point.detach()) == pytest.approx(
        wilting_point, rel=1.0e-6
    )
    state = {
        "soil_moisture": torch.tensor(field_capacity),
        "precipitation_flux": torch.tensor(0.0),
        "evapotranspiration_flux": torch.tensor(0.0),
    }
    with pytest.raises(ValueError, match="must be less than field capacity"):
        functor.forward(state, time=time_params["time"], dt=time_params["dt"])
    with pytest.raises(ExecutionError, match="must be less than field capacity"):
        functor(state, time=time_params["time"], dt=time_params["dt"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

