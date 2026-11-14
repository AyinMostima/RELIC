"""
L1 Unit Tests for Groundwater Dynamics Module

Tests the groundwater storage dynamics implementation including
water table tracking, porosity profiles, and coupling interfaces.
"""

import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.parameters import get_global_manager
import relic.scientific.hydrosphere.groundwater.dynamics as groundwater_module
from relic.scientific.hydrosphere.groundwater.dynamics import GroundwaterDynamics
from relic.core.state import UpdateItem, UpdateMode


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _to_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


def _canonical_groundwater_state(**overrides):
    """Build the explicit canonical groundwater state contract."""
    state = {
        "groundwater_storage": torch.tensor(200.0, dtype=torch.float64),
        "water_table_depth": torch.tensor(1.5, dtype=torch.float64),
        "drainage_flux": torch.tensor(2.0e-3, dtype=torch.float64),
        "capillary_rise_flux": torch.tensor(2.0e-6, dtype=torch.float64),
        "lateral_inflow": torch.tensor(3.0e-6, dtype=torch.float64),
        "groundwater_pumping": torch.tensor(1.0e-6, dtype=torch.float64),
    }
    state.update(overrides)
    return state


class TestGroundwaterDynamics:
    """Test suite for groundwater dynamics."""

    @pytest.fixture
    def groundwater_params(self):
        """Load groundwater parameters from the ParameterManager."""
        manager = get_global_manager()

        def _get_param(key: str, units: str) -> float:
            return _to_scalar(
                manager.get(
                    key,
                    namespace="hydrosphere.groundwater",
                    units=units,
                )
            )

        return {
            "surface_porosity": _get_param("surface_porosity", "dimensionless"),
            "porosity_decay_length": _get_param("porosity_decay_length", "m"),
            "max_water_table_depth": _get_param("max_water_table_depth", "m"),
            "bedrock_depth": _get_param("bedrock_depth", "m"),
        }

    @pytest.fixture
    def groundwater_model(self):
        """Create groundwater dynamics model instance."""
        return GroundwaterDynamics(name="test_groundwater")
    
    def test_model_initialization(self, groundwater_model, groundwater_params):
        """Test model initializes correctly."""
        assert _to_scalar(groundwater_model.surface_porosity) == pytest.approx(
            groundwater_params["surface_porosity"]
        )
        assert _to_scalar(groundwater_model.porosity_decay_length) == pytest.approx(
            groundwater_params["porosity_decay_length"]
        )
        assert _to_scalar(groundwater_model.max_water_table_depth) == pytest.approx(
            groundwater_params["max_water_table_depth"]
        )
        assert _to_scalar(groundwater_model.bedrock_depth) == pytest.approx(
            groundwater_params["bedrock_depth"]
        )
        # Point model: field identity and units come from the declared output
        # and its state contract, never from instance attributes.
        assert type(groundwater_model).SPATIAL_DIMS is None
        assert "groundwater_storage" in groundwater_model.metadata.output_variables
        for removed in ("field_name", "units", "default_value", "spatial_dims", "temporal_dependence"):
            assert removed not in vars(groundwater_model), removed
        for obsolete_attribute in (
            "average_porosity_factor",
            "initial_water_table_depth",
            "initial_water_table_depth_factor",
            "saturated_conductivity",
            "default_storage",
        ):
            assert not hasattr(groundwater_model, obsolete_attribute)

    def test_water_density_uses_physical_constant_ssot(self, monkeypatch):
        """Resolve water density only through the physical-constant SSOT API."""
        calls = []

        def fake_get_physical_constant(name):
            calls.append(name)
            return 997.0

        monkeypatch.setattr(
            groundwater_module,
            "get_physical_constant",
            fake_get_physical_constant,
            raising=False,
        )

        model = GroundwaterDynamics(name="groundwater_density_ssot")
        hodge = model.build_hodge_contract()

        assert calls == ["water_density"]
        assert _to_scalar(model.water_density_reference) == pytest.approx(997.0)
        assert (
            "relic.physical.constants.water.density_liquid"
            not in model.required_parameters()
        )
        assert (
            "relic.physical.constants.water.density_liquid"
            not in hodge.parameter_keys
        )
        expected_mobility = 997.0 * _to_scalar(model.specific_yield)
        assert _to_scalar(hodge.matrix[0, 0]) == pytest.approx(expected_mobility)

    def test_canonical_interface_is_exact(self, groundwater_model):
        """Expose one explicit mass-flux interface without timestep aliases."""
        assert set(groundwater_model.metadata.input_variables) == {
            "groundwater_storage",
            "water_table_depth",
            "drainage_flux",
            "capillary_rise_flux",
            "lateral_inflow",
            "groundwater_pumping",
        }
        assert set(groundwater_model.metadata.output_variables) == {
            "groundwater_storage",
            "water_table_depth",
            "baseflow_flux",
            "diagnostics.groundwater",
        }
        assert set(groundwater_model.metadata.parameters) == set(
            groundwater_model.required_parameters()
        )
        assert set(groundwater_model.metadata.conservation_laws) == {"mass"}
        assert "soil_drainage" not in groundwater_model.metadata.input_variables
        assert not hasattr(groundwater_model, "_compute_capillary_availability")
        assert not hasattr(groundwater_model, "_storage_to_water_table_depth")
        
    def test_porosity_profile(self, groundwater_model, groundwater_params):
        """Test exponential porosity decay profile."""
        surface_porosity = groundwater_params["surface_porosity"]
        decay_length = groundwater_params["porosity_decay_length"]
        # Test surface porosity
        n_surface = groundwater_model._compute_porosity_at_depth(0.0)
        assert abs(_to_scalar(n_surface) - surface_porosity) < 1e-6, "Surface porosity should match input"
        
        # Test decay with depth
        depths = [0.5, 1.0, 2.0, 4.0]
        porosities = [_to_scalar(groundwater_model._compute_porosity_at_depth(d)) for d in depths]
        
        # Should decrease monotonically
        for i in range(len(porosities) - 1):
            assert porosities[i] > porosities[i+1], "Porosity should decrease with depth"
        
        # Should all be positive and less than surface value
        for n in porosities:
            assert 0 < n <= surface_porosity, "Porosity should be positive and <= surface value"
        
        # Test exponential relationship
        depth_at_decay = _to_scalar(groundwater_model._compute_porosity_at_depth(decay_length))
        expected = surface_porosity * np.exp(-1.0)  # Should be n0 * e^(-1)
        assert abs(depth_at_decay - expected) < 1e-6, "Exponential decay not correct"
        
    def test_storage_integration(self, groundwater_model):
        """Test storage integration between depths."""
        # Test zero interval
        zero_integral = _to_scalar(groundwater_model._integrate_storage(1.0, 1.0))
        assert zero_integral == 0.0, "Zero interval should give zero integral"
        
        # Test positive integral (surface to depth)
        integral_1m = _to_scalar(groundwater_model._integrate_storage(0.0, 1.0))
        assert integral_1m > 0, "Integral should be positive"
        
        # Test additivity: integral(0,2) = integral(0,1) + integral(1,2)
        integral_2m = _to_scalar(groundwater_model._integrate_storage(0.0, 2.0))
        integral_1to2 = _to_scalar(groundwater_model._integrate_storage(1.0, 2.0))
        sum_parts = integral_1m + integral_1to2
        
        assert abs(integral_2m - sum_parts) < 1e-6, "Integral additivity not satisfied"
        
        # Test larger depth gives larger integral
        integral_3m = _to_scalar(groundwater_model._integrate_storage(0.0, 3.0))
        assert integral_3m > integral_2m, "Deeper integral should be larger"
        
    def test_water_table_solver(self, groundwater_model):
        """Test water table depth solver."""
        initial_depth = 2.0  # m
        
        # Test positive input (water table rises)
        net_input_positive = 50.0  # mm
        new_depth_up = _to_scalar(groundwater_model._solve_water_table_change(initial_depth, net_input_positive))
        assert new_depth_up < initial_depth, "Water table should rise with positive input"
        assert new_depth_up >= 0.1, "Water table should not go above minimum depth"
        
        # Test negative input (water table falls)
        net_input_negative = -50.0  # mm
        new_depth_down = _to_scalar(groundwater_model._solve_water_table_change(initial_depth, net_input_negative))
        assert new_depth_down > initial_depth, "Water table should fall with negative input"
        assert new_depth_down <= _to_scalar(groundwater_model.bedrock_depth), "Water table should not go below bedrock"
        
        # Test zero input (no change)
        new_depth_zero = _to_scalar(groundwater_model._solve_water_table_change(initial_depth, 0.0))
        assert abs(new_depth_zero - initial_depth) < 1e-6, "No change for zero input"
        
        # Test conservation: integral should match input
        computed_change = _to_scalar(
            groundwater_model._integrate_storage(initial_depth, new_depth_up) * 1000.0
        )
        assert abs(computed_change - net_input_positive) < 1.0, f"Conservation error: {abs(computed_change - net_input_positive):.2f} mm"
        
    def test_basic_forward_computation(self, groundwater_model):
        """Test basic forward computation."""
        state = _canonical_groundwater_state()
        
        time = 0.0
        dt = 3600.0  # 1 hour
        
        result = groundwater_model.forward(state, time, dt)
        
        # Check required outputs
        assert set(result) == {
            "groundwater_storage",
            "water_table_depth",
            "baseflow_flux",
            "diagnostics.groundwater",
        }
        
        # Check physical constraints
        storage_item = result["groundwater_storage"]
        depth_item = result["water_table_depth"]
        for key, expected_unit in {
            "groundwater_storage": "mm",
            "water_table_depth": "m",
            "baseflow_flux": "kg m-2 s-1",
        }.items():
            item = result[key]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == expected_unit
            assert item.source_functor == groundwater_model.name

        storage_value = _to_scalar(_unwrap_update(storage_item))
        depth_value = _to_scalar(_unwrap_update(depth_item))
        assert storage_value >= 0, "Storage should be non-negative"
        assert depth_value >= 0.1, "Water table depth should be >= minimum"
        assert depth_value <= _to_scalar(groundwater_model.bedrock_depth), "Water table should be above bedrock"
        assert _to_scalar(_unwrap_update(result["baseflow_flux"])) >= 0, "Baseflow flux should be non-negative"

        diagnostics = result["diagnostics.groundwater"]
        assert isinstance(diagnostics, dict)
        for key in [
            "effective_porosity",
            "groundwater_storage_increment",
            "groundwater_storage_change_rate",
            "water_table_change_rate",
            "net_groundwater_flux",
            "storage_deficit",
            "saturation_fraction",
            "baseflow_depth_rate",
        ]:
            assert key in diagnostics, f"Missing diagnostic output: {key}"
        assert 0 <= _to_scalar(diagnostics["saturation_fraction"]) <= 1
        
        # Test storage increase with positive input
        assert storage_value > _to_scalar(state["groundwater_storage"]), "Storage should increase with net positive input"
        
    def test_storage_deficit_computation(self, groundwater_model):
        """Test storage deficit calculation."""
        # Shallow water table (small deficit)
        shallow_depth = torch.tensor(0.5)
        shallow_deficit = _to_scalar(groundwater_model._compute_storage_deficit(shallow_depth))
        
        # Deep water table (large deficit)  
        deep_depth = torch.tensor(3.0)
        deep_deficit = _to_scalar(groundwater_model._compute_storage_deficit(deep_depth))
        
        assert deep_deficit > shallow_deficit, "Deeper water table should have larger deficit"
        assert shallow_deficit >= 0, "Deficit should be non-negative"
        assert deep_deficit >= 0, "Deficit should be non-negative"
        
    def test_saturation_fraction(self, groundwater_model):
        """Test saturation fraction calculation."""
        # High storage, low deficit
        high_storage = torch.tensor(500.0)
        low_deficit = torch.tensor(100.0)
        high_sat = _to_scalar(groundwater_model._compute_saturation_fraction(high_storage, low_deficit))
        
        # Low storage, high deficit
        low_storage = torch.tensor(100.0)
        high_deficit = torch.tensor(500.0)
        low_sat = _to_scalar(groundwater_model._compute_saturation_fraction(low_storage, high_deficit))
        
        assert high_sat > low_sat, "Higher storage should give higher saturation fraction"
        assert 0 <= high_sat <= 1, "Saturation fraction should be in [0,1]"
        assert 0 <= low_sat <= 1, "Saturation fraction should be in [0,1]"
        
    def test_baseflow_computation(self, groundwater_model):
        """Test baseflow calculation."""
        # High storage, shallow water table
        high_storage = torch.tensor(400.0)
        shallow_wt = torch.tensor(1.0)
        high_baseflow = groundwater_model._compute_baseflow(high_storage, shallow_wt)
        
        # Low storage, deep water table
        low_storage = torch.tensor(100.0)
        deep_wt = torch.tensor(4.0)
        low_baseflow = groundwater_model._compute_baseflow(low_storage, deep_wt)
        
        assert high_baseflow > low_baseflow, "Higher storage and shallower WT should give higher baseflow"
        assert high_baseflow >= 0, "Baseflow should be non-negative"
        assert low_baseflow >= 0, "Baseflow should be non-negative"
        
    def test_capillary_rise_is_debited_from_groundwater_storage(self, groundwater_model):
        """Debit the exact capillary transfer that enters the soil column."""
        dt = 3600.0
        capillary_flux = torch.tensor(2.0e-6, dtype=torch.float64)
        with_capillary = groundwater_model.forward(
            _canonical_groundwater_state(capillary_rise_flux=capillary_flux),
            0.0,
            dt,
        )
        without_capillary = groundwater_model.forward(
            _canonical_groundwater_state(
                capillary_rise_flux=torch.tensor(0.0, dtype=torch.float64)
            ),
            0.0,
            dt,
        )

        storage_difference = (
            _unwrap_update(without_capillary["groundwater_storage"])
            - _unwrap_update(with_capillary["groundwater_storage"])
        )
        expected_depth_mm = (
            capillary_flux * 1000.0 / groundwater_model.water_density_reference * dt
        )
        assert storage_difference == pytest.approx(expected_depth_mm, rel=1.0e-5, abs=1.0e-7)
        assert _unwrap_update(with_capillary["baseflow_flux"]) == pytest.approx(
            _unwrap_update(without_capillary["baseflow_flux"]), rel=1.0e-6
        )

    @pytest.mark.parametrize(
        "missing_key",
        [
            "groundwater_storage",
            "water_table_depth",
            "drainage_flux",
            "capillary_rise_flux",
            "lateral_inflow",
            "groundwater_pumping",
        ],
    )
    def test_missing_canonical_input_fails_fast(self, groundwater_model, missing_key):
        """Reject missing state or boundary ports instead of injecting defaults."""
        state = _canonical_groundwater_state()
        state.pop(missing_key)
        if missing_key == "drainage_flux":
            state["soil_drainage"] = torch.tensor(10.0)

        with pytest.raises(KeyError, match=missing_key):
            groundwater_model.forward(state=state, time=0.0, dt=3600.0)

    @pytest.mark.parametrize(
        ("key", "value", "error_type", "message"),
        [
            ("drainage_flux", 1.0e-5, TypeError, "drainage_flux"),
            ("lateral_inflow", torch.tensor(float("nan")), ValueError, "lateral_inflow"),
            ("groundwater_pumping", torch.tensor(float("inf")), ValueError, "groundwater_pumping"),
            ("capillary_rise_flux", torch.tensor(-1.0e-6), ValueError, "capillary_rise_flux"),
            ("groundwater_storage", torch.tensor(-1.0), ValueError, "groundwater_storage"),
            ("water_table_depth", torch.tensor(-0.1), ValueError, "water_table_depth"),
        ],
    )
    def test_invalid_canonical_input_fails_fast(
        self,
        groundwater_model,
        key,
        value,
        error_type,
        message,
    ):
        """Reject invalid state and flux values without clamping or coercion."""
        state = _canonical_groundwater_state(**{key: value})

        with pytest.raises(error_type, match=message):
            groundwater_model.forward(state=state, time=0.0, dt=3600.0)
        
    def test_single_dynamic_mode_and_invalid_dt_rejection(self):
        """Test the sole water-table dynamics path and fail-closed timestep contract."""
        model = GroundwaterDynamics("dynamic")
        state = _canonical_groundwater_state(
            groundwater_storage=torch.tensor(150.0),
            water_table_depth=torch.tensor(2.0),
        )

        result = model.forward(state, 0.0, 3600.0)

        assert _unwrap_update(result["groundwater_storage"]) > 0
        assert torch.isfinite(_unwrap_update(result["water_table_depth"]))
        assert not hasattr(model, "enable_storage_dynamics")
        with pytest.raises(ValueError, match="dt must be positive"):
            model.forward(state, 0.0, 0.0)
        
    def test_physical_field_interface(self, groundwater_model):
        """Reject stateless field queries instead of returning parameter defaults."""
        coords = torch.tensor([[0.0], [1.0], [2.0]])  # Dummy coordinates
        with pytest.raises(NotImplementedError, match="point model.*StateManager"):
            groundwater_model.get_value_at(coords)

        # A point model has no coordinate axis, so no gradient is fabricated.
        with pytest.raises(NotImplementedError, match="point model"):
            groundwater_model.compute_gradient(coords)

        source_coords = torch.tensor([[0.0], [1.0]])
        target_coords = torch.tensor([[0.5], [1.5], [2.5]])
        with pytest.raises(NotImplementedError, match="point model.*StateManager"):
            groundwater_model.interpolate(source_coords, target_coords)
        
    def test_mass_conservation(self, groundwater_model):
        """Close drainage, lateral, pumping, capillary, and baseflow exactly."""
        dt = 3600.0
        initial_storage = 200.0
        state = _canonical_groundwater_state(
            groundwater_storage=torch.tensor(initial_storage, dtype=torch.float64),
            water_table_depth=torch.tensor(2.0, dtype=torch.float64),
            drainage_flux=torch.tensor(1.0e-5, dtype=torch.float64),
            lateral_inflow=torch.tensor(4.0e-6, dtype=torch.float64),
            groundwater_pumping=torch.tensor(1.0e-6, dtype=torch.float64),
            capillary_rise_flux=torch.tensor(2.0e-6, dtype=torch.float64),
        )

        result = groundwater_model.forward(state, 0.0, dt)

        baseflow_flux = _unwrap_update(result["baseflow_flux"])
        net_mass_flux = (
            state["drainage_flux"]
            + state["lateral_inflow"]
            - state["groundwater_pumping"]
            - state["capillary_rise_flux"]
            - baseflow_flux
        )
        expected_delta_mm = (
            net_mass_flux * 1000.0 / groundwater_model.water_density_reference * dt
        )
        actual_delta_mm = (
            _unwrap_update(result["groundwater_storage"]) - initial_storage
        )

        assert actual_delta_mm == pytest.approx(expected_delta_mm, rel=1.0e-5, abs=1.0e-6)
        assert result["diagnostics.groundwater"]["storage_balance_residual"] == pytest.approx(
            torch.zeros_like(actual_delta_mm), abs=1.0e-6
        )

    def test_float32_storage_roundoff_is_explicitly_bounded(
        self,
        groundwater_model,
    ):
        """Machine roundoff must remain visible but not become a mass warning."""
        state = _canonical_groundwater_state(
            groundwater_storage=torch.tensor(100.0, dtype=torch.float32),
            water_table_depth=torch.tensor(2.0, dtype=torch.float32),
            drainage_flux=torch.tensor(
                0.008981130085885525,
                dtype=torch.float32,
            ),
            lateral_inflow=torch.tensor(0.0, dtype=torch.float32),
            groundwater_pumping=torch.tensor(0.0, dtype=torch.float32),
            capillary_rise_flux=torch.tensor(0.0, dtype=torch.float32),
        )

        result = groundwater_model.forward(state, time=0.0, dt=3600.0)
        diagnostics = result["diagnostics.groundwater"]
        residual = torch.abs(diagnostics["storage_balance_residual"])
        roundoff_bound = diagnostics["storage_balance_roundoff_bound"]

        assert residual > 0.0
        assert roundoff_bound > 0.0
        assert residual <= roundoff_bound
        assert groundwater_model.check_conservation_laws(
            state,
            result,
        )["mass"] == 0.0

    def test_open_five_port_mass_conservation_checker(self, groundwater_model):
        """Validate the existing open-control-volume storage residual."""
        state = _canonical_groundwater_state()
        result = groundwater_model(state, time=0.0, dt=3600.0)
        expected = torch.max(
            torch.abs(
                result["diagnostics.groundwater"]["storage_balance_residual"]
            )
        ).item()

        assert groundwater_model.check_conservation_laws(state, result)["mass"] == pytest.approx(
            expected
        )

        perturbed = dict(result)
        perturbed["diagnostics.groundwater"] = dict(
            result["diagnostics.groundwater"]
        )
        perturbed["diagnostics.groundwater"]["storage_balance_residual"] = (
            torch.tensor([-0.125, 0.25])
        )
        assert groundwater_model.check_conservation_laws(state, perturbed)["mass"] == pytest.approx(
            0.25
        )

    def test_open_five_port_mass_conservation_checker_fails_fast(
        self,
        groundwater_model,
    ):
        """Reject missing or malformed storage residual diagnostics."""
        state = _canonical_groundwater_state()
        result = groundwater_model.forward(state, time=0.0, dt=3600.0)

        missing_diagnostics = dict(result)
        missing_diagnostics.pop("diagnostics.groundwater")
        with pytest.raises(KeyError, match="diagnostics.groundwater"):
            groundwater_model.check_conservation_laws(state, missing_diagnostics)

        invalid_diagnostics = dict(result)
        invalid_diagnostics["diagnostics.groundwater"] = []
        with pytest.raises(TypeError, match="mapping"):
            groundwater_model.check_conservation_laws(state, invalid_diagnostics)

        for invalid_residual, error_type in (
            ("invalid", TypeError),
            (torch.empty(0), ValueError),
            (torch.tensor(float("nan")), ValueError),
            (torch.tensor(float("inf")), ValueError),
        ):
            invalid = dict(result)
            invalid["diagnostics.groundwater"] = dict(
                result["diagnostics.groundwater"]
            )
            invalid["diagnostics.groundwater"]["storage_balance_residual"] = (
                invalid_residual
            )
            with pytest.raises(error_type):
                groundwater_model.check_conservation_laws(state, invalid)

        missing_residual = dict(result)
        missing_residual["diagnostics.groundwater"] = dict(
            result["diagnostics.groundwater"]
        )
        missing_residual["diagnostics.groundwater"].pop(
            "storage_balance_residual"
        )
        with pytest.raises(KeyError, match="storage_balance_residual"):
            groundwater_model.check_conservation_laws(state, missing_residual)
        
    def test_boundary_conditions(self, groundwater_model):
        """Test boundary condition handling."""
        # Test very shallow water table (near surface)
        shallow_state = {
            **_canonical_groundwater_state(),
            "groundwater_storage": torch.tensor(1000.0),
            "water_table_depth": torch.tensor(0.1),
            "drainage_flux": torch.tensor(5.0e-5),
        }
        
        shallow_result = groundwater_model.forward(shallow_state, 0.0, 3600.0)
        
        # Should handle shallow water table gracefully
        assert _unwrap_update(shallow_result["water_table_depth"]) >= 0.1, "Should respect minimum depth"
        assert torch.isfinite(_unwrap_update(shallow_result["groundwater_storage"])), "Storage should be finite"
        
        # Test deep water table (near bedrock)
        deep_state = {
            **_canonical_groundwater_state(),
            "groundwater_storage": torch.tensor(50.0),
            "water_table_depth": torch.tensor(4.8),
            "drainage_flux": torch.tensor(0.0),
            "groundwater_pumping": torch.tensor(5.0e-5),
        }
        
        deep_result = groundwater_model.forward(deep_state, 0.0, 3600.0)
        
        # Should handle deep water table gracefully
        assert _unwrap_update(deep_result["water_table_depth"]) <= groundwater_model.bedrock_depth, "Should respect bedrock limit"
        assert _unwrap_update(deep_result["groundwater_storage"]) >= 0, "Storage should remain non-negative"


class TestGroundwaterDynamicsPhase8:
    """Test Phase 8 IPHS contract."""

    def test_phase8_iphs_protocol(self):
        """Test structural IPHS hooks."""
        functor = GroundwaterDynamics()

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure({})
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.groundwater.surface_porosity" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "groundwater_storage",
            "water_table_depth",
            "baseflow_flux",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {
            "groundwater_recharge_flux",
            "groundwater_lateral_inflow_flux",
            "groundwater_pumping_flux",
            "groundwater_capillary_rise_flux",
            "groundwater_baseflow_flux",
        }
        ports_by_name = {port.name: port for port in ports}
        assert ports_by_name["groundwater_recharge_flux"].flow_var == "drainage_flux"
        assert ports_by_name["groundwater_recharge_flux"].positive_power_convention == "into_system"
        assert ports_by_name["groundwater_lateral_inflow_flux"].flow_var == "lateral_inflow"
        assert ports_by_name["groundwater_lateral_inflow_flux"].positive_power_convention == "into_system"
        assert ports_by_name["groundwater_pumping_flux"].flow_var == "groundwater_pumping"
        assert ports_by_name["groundwater_pumping_flux"].positive_power_convention == "out_of_system"
        assert ports_by_name["groundwater_capillary_rise_flux"].flow_var == "capillary_rise_flux"
        assert ports_by_name["groundwater_capillary_rise_flux"].positive_power_convention == "out_of_system"
        assert ports_by_name["groundwater_baseflow_flux"].positive_power_convention == "out_of_system"

        dependencies = functor.get_dependencies()
        assert set(dependencies["optional"]) == {
            "bucket_soil_moisture",
            "richards_soil_moisture",
            "vertical_percolation_transport",
            "capillary_rise_transport",
        }

    def test_phase8_certificate_is_valid(self):
        """Test Phase 8 certificate completeness."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_groundwater_dynamics_groundwater_dynamics.yaml")
        )
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == (
            "relic.scientific.hydrosphere.groundwater.dynamics.GroundwaterDynamics"
        )
        assert certificate.canonical_role == "field"
        assert certificate.iphs_level == 3
        assert certificate.iphs_level_evidence["level_3_limit"]
        assert audit["runtime_evidence_status"] == "descriptive_only"

    def test_runtime_audits_are_exposed(self):
        """Test storage, effort, dissipation, and port-power diagnostics."""
        functor = GroundwaterDynamics()
        state = {
            **_canonical_groundwater_state(),
        }

        energy_terms = functor.energy_terms(state)
        entropy_terms = functor.entropy_terms(state)
        result = functor.forward(state, time=0.0, dt=3600.0)
        diagnostics = result["diagnostics.groundwater"]

        assert torch.is_tensor(energy_terms["storage_functional"])
        assert torch.is_tensor(energy_terms["effort"])
        assert torch.is_tensor(entropy_terms["dissipation_rate"])
        assert entropy_terms["dissipation_rate"] >= 0.0
        assert "energy_storage" in diagnostics
        assert "dissipation_rate" in diagnostics
        assert "dH_dt_audit" in diagnostics
        assert "port_power_residual" in diagnostics

    def test_port_effort_uses_g_transpose_hodge_mapping(self):
        """Map state effort to the conjugate hydraulic-head port effort.

        References:
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
          - Pasumarthy et al. (2012), port-Hamiltonian open-channel flow.
        Mathematical expression:
          y = G.T e, e = rho_w S_y z, G = -1 / (rho_w S_y), so y = -z.
        Parameter meanings:
          z is positive-downward water-table depth [m], rho_w is water
          density [kg m-3], and S_y is specific yield [1].
        Units:
          y [m] times mass flux u [kg m-2 s-1] gives
          head-weighted port power [kg m-1 s-1].
        """
        functor = GroundwaterDynamics()
        depth = torch.tensor(2.0, dtype=torch.float64)
        mass_flux = torch.tensor(2.0e-3, dtype=torch.float64)

        state_effort = functor._groundwater_effort(depth)
        port_effort = functor._groundwater_port_effort(depth)
        expected_input_map = -1.0 / (
            functor.water_density_reference * _to_scalar(functor.specific_yield)
        )

        assert port_effort == pytest.approx(-depth)
        assert port_effort == pytest.approx(expected_input_map * state_effort)
        assert torch.sum(port_effort * mass_flux) == pytest.approx(-4.0e-3)

    def test_port_mass_conversion_reuses_canonical_unit_converter(
        self,
        monkeypatch,
    ):
        """Use the sole UnitConverter for baseflow and port-audit mass flux."""
        functor = GroundwaterDynamics()
        calls = []
        original = groundwater_module.unit_converter.convert

        def tracked_conversion(data, variable_name, source_units, target_units, *args):
            calls.append((variable_name, source_units, target_units))
            return original(data, variable_name, source_units, target_units, *args)

        monkeypatch.setattr(
            groundwater_module.unit_converter,
            "convert",
            tracked_conversion,
        )

        functor.forward(_canonical_groundwater_state(), time=0.0, dt=3600.0)

        # Four boundary fluxes to depth rates, then baseflow and the port audit back
        # to mass flux, all through the single liquid-water-equivalent rule.
        water_calls = [call for call in calls if "kg m-2 s-1" in call[1:]]
        assert water_calls == [
            ("drainage_flux", "kg m-2 s-1", "mm s-1"),
            ("capillary_rise_flux", "kg m-2 s-1", "mm s-1"),
            ("lateral_inflow", "kg m-2 s-1", "mm s-1"),
            ("groundwater_pumping", "kg m-2 s-1", "mm s-1"),
            ("baseflow_flux", "mm s-1", "kg m-2 s-1"),
            ("water_flux", "mm s-1", "kg m-2 s-1"),
        ]
        assert not hasattr(functor, "_depth_rate_mm_to_mass_flux")

    def test_residual_diagnostics_are_routed(self):
        """Test projection residual diagnostics are exposed."""
        functor = GroundwaterDynamics()
        result = functor.forward(
            {
                **_canonical_groundwater_state(),
            },
            time=0.0,
            dt=3600.0,
        )

        diagnostics = result["diagnostics.groundwater"]
        assert "storage_balance_residual" in diagnostics
        assert "storage_balance_roundoff_bound" in diagnostics
        assert "water_table_bounds_residual" in diagnostics
        assert "groundwater_storage_negative_residual" in diagnostics
        assert torch.isfinite(diagnostics["storage_balance_residual"]).all()
        assert torch.isfinite(
            diagnostics["storage_balance_roundoff_bound"]
        ).all()
        assert torch.all(
            diagnostics["storage_balance_roundoff_bound"] >= 0.0
        )
        assert torch.all(diagnostics["water_table_bounds_residual"] >= 0.0)
        assert torch.all(diagnostics["groundwater_storage_negative_residual"] >= 0.0)


if __name__ == "__main__":
    # Run basic smoke test
    config = {
        "parameters": {
            "surface_porosity": {"value": 0.4, "units": "dimensionless"},
            "porosity_decay_length": {"value": 1.5, "units": "m"},
            "bedrock_depth": {"value": 3.0, "units": "m"}
        }
    }
    
    model = GroundwaterDynamics("test", config)
    
    state = {
        **_canonical_groundwater_state(),
        "groundwater_storage": torch.tensor(180.0),
        "water_table_depth": torch.tensor(1.8),
    }
    
    try:
        result = model.forward(state, 0.0, 3600.0)
        print("Groundwater dynamics test passed!")
        storage = _to_scalar(_unwrap_update(result["groundwater_storage"]))
        depth = _to_scalar(_unwrap_update(result["water_table_depth"]))
        baseflow = _to_scalar(_unwrap_update(result["baseflow_flux"]))
        print(f"Storage: {storage:.1f} mm (change: {storage - 180.0:.1f} mm)")
        print(f"Water table depth: {depth:.2f} m")
        print(f"Baseflow: {baseflow:.2e} kg m-2 s-1")
        
        # Test porosity profile
        depths = [0.0, 0.5, 1.0, 2.0, 3.0]
        porosities = [model._compute_porosity_at_depth(d) for d in depths]
        print(f"Porosity profile: {[f'{p:.3f}' for p in porosities]}")
        
    except Exception as e:
        print(f"Groundwater dynamics test failed: {e}")
        import traceback
        traceback.print_exc()

