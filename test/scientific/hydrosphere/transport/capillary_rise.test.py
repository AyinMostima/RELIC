"""
L1 Unit Tests for Capillary Rise Transport Module

Tests the Gardner-Eagleson capillary rise implementation including
hydraulic parameterizations, flux calculations, and depth profiles.
"""

from pathlib import Path
import pytest
import torch
import sys
import os

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.device_policy import is_cuda_available
from relic.core.functor import IPHSDECFunctor
from relic.core.simulator import RELICSimulator
from relic.parameters import get_global_manager
from relic.parameters.ops.nn_params import relocate_binding
from relic.scientific.hydrosphere.transport.capillary_rise import CapillaryRiseTransport
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _set_parameter(module, name, value):
    param = getattr(module, name)
    tensor_value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if isinstance(param, torch.nn.Parameter):
        param.data = tensor_value.to(param.data.device, dtype=param.data.dtype)
    else:
        setattr(module, name, tensor_value)


def _use_gardner_coefficients(monkeypatch, module, gardner_a, gardner_b, gardner_B):
    """Make one capillary instance compute with the given Gardner coefficients.

    The coefficients are read-only values derived from the held parameters at
    every use, so a test that needs coefficients no parameter set produces
    replaces the derivation of this instance.
    """
    monkeypatch.setattr(
        module,
        "_compute_gardner_coefficients",
        lambda reference=None: (gardner_a, gardner_b, gardner_B),
    )


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


class TestCapillaryRiseTransport:
    """Test suite for capillary rise transport mechanism."""
    
    @pytest.fixture
    def capillary_model(self):
        """Create capillary rise model instance."""
        return CapillaryRiseTransport(name="test_capillary")

    @pytest.fixture(autouse=True)
    def restore_k_saturated(self):
        """Restore shared ParameterManager state after each test."""
        manager = get_global_manager()
        key = "hydrosphere.soil_moisture.saturated_hydraulic_conductivity"
        spec = manager.get_parameter(key)
        if spec is None:
            yield
            return
        original_value = spec.value
        original_units = spec.units
        yield
        manager.set_value(key, original_value, units=original_units, validate=True, notify=False)
    
    def test_model_initialization(self, capillary_model):
        """Test model initializes correctly."""
        manager = get_global_manager()
        expected_max_height = manager.get(
            "max_capillary_height",
            namespace="hydrosphere.transport",
            units="m",
        )
        expected_pore_size = manager.get(
            "brooks_corey_pore_size_index",
            namespace="hydrosphere.transport",
            units="dimensionless",
        )
        expected_air_entry = manager.get(
            "air_entry_pressure",
            namespace="hydrosphere.transport",
            units="m",
        )
        expected_surface_porosity = manager.get(
            "surface_porosity",
            namespace="hydrosphere.groundwater",
            units="dimensionless",
        )

        assert _as_tensor(capillary_model.max_capillary_height).item() == pytest.approx(
            _as_tensor(expected_max_height).item()
        )
        assert _as_tensor(capillary_model.pore_size_index).item() == pytest.approx(
            _as_tensor(expected_pore_size).item()
        )
        air_entry = _as_tensor(capillary_model.air_entry_pressure).detach().cpu().item()
        assert air_entry == pytest.approx(_as_tensor(expected_air_entry).item())
        assert _as_tensor(capillary_model.surface_porosity).item() == pytest.approx(
            _as_tensor(expected_surface_porosity).item()
        )
        assert capillary_model.transport_type == "capillary_rise"
        
    def test_gardner_coefficients(self, capillary_model):
        """Test Gardner coefficient calculation."""
        # Check that coefficients are computed
        assert hasattr(capillary_model, 'gardner_a')
        assert hasattr(capillary_model, 'gardner_b') 
        assert hasattr(capillary_model, 'gardner_B')
        
        # Gardner coefficients should be positive and finite
        assert _as_tensor(capillary_model.gardner_a).item() > 0
        assert _as_tensor(capillary_model.gardner_b).item() > 0
        assert _as_tensor(capillary_model.gardner_B).item() > 0
        assert torch.isfinite(_as_tensor(capillary_model.gardner_a))
        assert torch.isfinite(_as_tensor(capillary_model.gardner_b))
        assert torch.isfinite(_as_tensor(capillary_model.gardner_B))

    def test_brooks_corey_exponent_matches_reference(self, capillary_model):
        """Brooks-Corey exponent should match reference formulation."""
        capillary_model.hydraulic_method = "brooks_corey"
        capillary_model._compute_gardner_coefficients()
        lambda_value = _as_tensor(capillary_model.pore_size_index).item()
        expected_b = 2.0 + 3.0 * lambda_value
        expected_B = 1.0 + 1.5 / (expected_b - 1.0)
        assert _as_tensor(capillary_model.gardner_b).item() == pytest.approx(expected_b)
        assert _as_tensor(capillary_model.gardner_B).item() == pytest.approx(expected_B)
        
    def test_gardner_flux_calculation(self, capillary_model, monkeypatch):
        """Test Gardner flux calculation with different depths."""
        _set_parameter(capillary_model, "k_saturated", 1.0)
        _, gardner_b, gardner_B = capillary_model._compute_gardner_coefficients()
        _use_gardner_coefficients(
            monkeypatch, capillary_model, torch.tensor(1e-6), gardner_b, gardner_B
        )
        # Test shallow water table (high flux)
        shallow_depth = 0.5  # m
        shallow_flux = capillary_model._compute_gardner_flux(shallow_depth)
        assert shallow_flux > 0, "Should have positive flux for shallow water table"
        
        # Test moderate depth
        moderate_depth = 1.0  # m
        moderate_flux = capillary_model._compute_gardner_flux(moderate_depth)
        assert moderate_flux > 0, "Should have positive flux for moderate depth"
        assert moderate_flux < shallow_flux, "Deeper water table should have lower flux"
        
        # Test deep water table (beyond capillary height)
        deep_depth = 5.0  # m (beyond max_capillary_height = 2.0)
        deep_flux = capillary_model._compute_gardner_flux(deep_depth)
        assert deep_flux == 0.0, "No flux beyond maximum capillary height"
        
        # Test zero/negative depth
        zero_flux = capillary_model._compute_gardner_flux(0.0)
        assert zero_flux == 0.0, "No flux for zero depth"
        
        negative_flux = capillary_model._compute_gardner_flux(-1.0)
        assert negative_flux == 0.0, "No flux for negative depth"
        
    def test_flux_profile_computation(self, capillary_model):
        """Test depth-dependent flux profile calculation."""
        gw_depth = 1.5  # m
        layer_depths = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])  # m
        
        flux_profile = capillary_model._compute_flux_profile(gw_depth, layer_depths)
        
        assert len(flux_profile) == len(layer_depths)
        assert torch.all(flux_profile >= 0), "All fluxes should be non-negative"
        assert torch.all(torch.isfinite(flux_profile)), "All fluxes should be finite"
        
        # Fluxes should generally decrease with depth (closer to water table)
        # But this depends on the porosity profile
        assert torch.sum(flux_profile) > 0, "Total flux should be positive"
        
    def test_saturation_constraints(self, capillary_model):
        """Test saturation constraint application."""
        flux_profile = torch.tensor([1e-6, 2e-6, 1.5e-6, 0.5e-6, 0.1e-6])  # m/s
        
        # Dry soil (no constraint)
        dry_moisture = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3])
        constrained_dry = capillary_model._apply_saturation_constraints(flux_profile, dry_moisture)
        assert torch.allclose(constrained_dry, flux_profile), "No constraint for dry soil"
        
        # Wet soil (significant constraint)
        wet_moisture = torch.tensor([0.42, 0.41, 0.40, 0.35, 0.30])  # Near saturation
        constrained_wet = capillary_model._apply_saturation_constraints(flux_profile, wet_moisture)
        assert torch.all(constrained_wet <= flux_profile), "Flux should be reduced for wet soil"
        assert torch.all(constrained_wet >= 0), "Constrained flux should be non-negative"
        
    def test_full_forward_computation(self, capillary_model):
        """Test complete forward computation."""
        # Test with shallow groundwater
        state = {
            "water_table_depth": torch.tensor(1.0),
            "soil_moisture_profile": torch.tensor([0.2, 0.25, 0.3, 0.35, 0.4])
        }
        
        time = 0.0
        dt = 3600.0
        
        result = capillary_model.forward(state, time, dt, 
                                       soil_layers=5, layer_thickness=0.2)
        
        expected_outputs = {
            "capillary_rise_flux",
            "diagnostics.capillary_rise",
            "diagnostics.capillary_rise.conductivity_clipping_residual",
            "diagnostics.capillary_rise.saturation_constraint_residual",
        }
        assert set(result) == expected_outputs
        assert set(capillary_model.metadata.output_variables) == expected_outputs

        diagnostics = result["diagnostics.capillary_rise"]
        required_diag = [
            "capillary_rise_profile",
            "effective_capillary_height",
            "capillary_availability",
        ]
        for key in required_diag:
            assert key in diagnostics, f"Missing diagnostic output: {key}"
        # One output path per quantity: the limiter residuals are the declared flat
        # outputs, not repeated in the nested mapping.
        assert "conductivity_clipping_residual" not in diagnostics
        assert "saturation_constraint_residual" not in diagnostics
        assert "total_capillary_input" not in diagnostics
        assert "water_table_depth" not in diagnostics

        # Check physical constraints
        surface_flux = _unwrap_update(result["capillary_rise_flux"])
        assert surface_flux >= 0, "Surface flux should be non-negative"
        assert torch.all(diagnostics["capillary_rise_profile"] >= 0), "Profile fluxes should be non-negative"
        assert diagnostics["effective_capillary_height"] > 0, "Should have effective capillary height"
        assert 0 <= diagnostics["capillary_availability"] <= 1, "Availability should be in [0,1]"
        
        # Test with deep groundwater (no capillary rise)
        deep_state = {
            "water_table_depth": torch.tensor(5.0)  # Beyond max height
        }
        
        deep_result = capillary_model.forward(deep_state, time, dt)
        deep_flux = _unwrap_update(deep_result["capillary_rise_flux"])
        deep_diag = deep_result["diagnostics.capillary_rise"]
        assert deep_flux == 0.0, "No flux for deep groundwater"
        assert deep_diag["capillary_availability"] == 0.0, "No availability for deep groundwater"

    @pytest.mark.parametrize("water_table_depth", [-0.1, float("nan"), float("inf")])
    def test_forward_rejects_invalid_water_table_depth(
        self,
        capillary_model,
        water_table_depth,
    ):
        """Reject negative or non-finite groundwater boundary depths."""
        with pytest.raises(ValueError, match="Water table depth"):
            capillary_model.forward(
                {"water_table_depth": torch.tensor(water_table_depth)},
                time=0.0,
                dt=3600.0,
            )

    @pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
    def test_forward_rejects_nonfinite_soil_moisture_profile(
        self,
        capillary_model,
        invalid_value,
    ):
        """Reject non-finite optional saturation-limiter inputs."""
        with pytest.raises(ValueError, match="soil_moisture_profile"):
            capillary_model.forward(
                {
                    "water_table_depth": torch.tensor(1.0),
                    "soil_moisture_profile": torch.tensor(
                        [0.2, 0.25, invalid_value, 0.35, 0.4]
                    ),
                },
                time=0.0,
                dt=3600.0,
                soil_layers=5,
                layer_thickness=0.2,
            )
        
    def test_van_genuchten_parameterization(self):
        """Test Van Genuchten hydraulic parameterization."""
        manager = get_global_manager()
        key = "hydrosphere.transport.hydraulic_parameterization_method"
        spec = manager.get_parameter(key)
        assert spec is not None, f"Missing parameter spec for {key}"
        original_value = spec.value
        original_units = spec.units
        manager.set_value(
            key,
            "van_genuchten",
            units=original_units,
            validate=True,
            notify=False,
        )
        try:
            vg_model = CapillaryRiseTransport("test_vg")
            assert vg_model.hydraulic_method == "van_genuchten"
        finally:
            manager.set_value(
                key,
                original_value,
                units=original_units,
                validate=True,
                notify=False,
            )
        
        # Should have computed Gardner coefficients
        assert vg_model.gardner_a > 0
        assert vg_model.gardner_b > 0
        assert vg_model.gardner_B > 0
        
        # Test flux calculation
        flux = vg_model._compute_gardner_flux(1.0)
        assert flux > 0, "VG parameterization should give positive flux"
        
    def test_boundary_conditions_integration(self, capillary_model):
        """Test integration with boundary conditions."""
        state = {"water_table_depth": torch.tensor(0.8)}
        
        result = capillary_model.forward(state, 0.0, 3600.0)
        
        # Check that flux can serve as boundary condition
        surface_flux = _unwrap_update(result["capillary_rise_flux"])  # kg m-2 s-1
        assert surface_flux > 0, "Should provide positive upward flux"
        
        # Convert to volumetric flux for Richards equation
        volumetric_flux = surface_flux / capillary_model.water_density  # m s-1
        assert volumetric_flux > 0, "Volumetric flux should be positive"

    @pytest.mark.parametrize(
        ("state", "kwargs"),
        [
            ({}, {}),
            ({"groundwater_depth": torch.tensor(1.0)}, {}),
            ({}, {"water_table_depth": torch.tensor(1.0)}),
        ],
    )
    def test_water_table_depth_is_canonical_and_required(
        self,
        capillary_model,
        state,
        kwargs,
    ):
        """Reject missing or compatibility-only groundwater inputs."""
        with pytest.raises(ValueError, match="water_table_depth"):
            capillary_model.forward(state=state, time=0.0, dt=3600.0, **kwargs)

    def test_water_density_uses_physical_constant_ssot(self, capillary_model):
        """Keep water density outside the ordinary functor parameter inventory."""
        expected = float(get_physical_constant("water_density"))
        assert float(capillary_model.water_density) == pytest.approx(expected)
        assert (
            "relic.physical.constants.water.density_liquid"
            not in capillary_model.required_parameters()
        )
        assert capillary_model.build_hodge_contract().parameter_keys == (
            "relic.physical.constants.water.density_liquid",
        )
        
    def test_compute_flux_interface(self, capillary_model):
        """Test the canonical state-driven capillary flux interface."""
        result = capillary_model.forward(
            {"water_table_depth": torch.tensor(1.2)},
            time=0.0,
            dt=3600.0,
        )

        flux = _unwrap_update(result["capillary_rise_flux"])
        assert isinstance(flux, torch.Tensor)
        assert torch.isfinite(flux).all()
        assert torch.all(flux >= 0)
        
    def test_parameter_sensitivity(self, capillary_model):
        """Test sensitivity to key parameters."""
        _set_parameter(capillary_model, "k_saturated", 1.0)
        capillary_model._compute_gardner_coefficients()
        base_state = {"water_table_depth": torch.tensor(1.0)}
        base_result = capillary_model.forward(base_state, 0.0, 3600.0)
        base_flux = _unwrap_update(base_result["capillary_rise_flux"])
        
        # Test sensitivity to groundwater depth
        shallow_state = {"water_table_depth": torch.tensor(0.5)}
        shallow_result = capillary_model.forward(shallow_state, 0.0, 3600.0)    
        shallow_flux = _unwrap_update(shallow_result["capillary_rise_flux"])
        
        assert shallow_flux > base_flux, "Shallower water table should give higher flux"
        
        # Test sensitivity to saturated conductivity
        original_ks = _as_tensor(capillary_model.k_saturated).detach().clone()
        _set_parameter(capillary_model, "k_saturated", original_ks * 2.0)
        capillary_model._compute_gardner_coefficients()  # Recompute
        
        high_k_result = capillary_model.forward(base_state, 0.0, 3600.0)        
        high_k_flux = _unwrap_update(high_k_result["capillary_rise_flux"])
        
        assert high_k_flux > base_flux, "Higher K_sat should give higher flux"
        
        # Restore original value
        _set_parameter(capillary_model, "k_saturated", original_ks)
        capillary_model._compute_gardner_coefficients()
        
    def test_conductivity_limiter_ledger(self, capillary_model, monkeypatch):
        """Account for the conductivity cap without mixing profile measures."""
        _set_parameter(capillary_model, "k_saturated", 1.0e-9)
        _use_gardner_coefficients(
            monkeypatch,
            capillary_model,
            torch.tensor(1.0),
            torch.tensor(2.0),
            torch.tensor(1.5),
        )
        water_table_depth = torch.tensor(0.5)

        result = capillary_model.forward(
            {"water_table_depth": water_table_depth},
            0.0,
            3600.0,
        )
        retained = result["capillary_rise_flux"].value
        clipping = result[
            "diagnostics.capillary_rise.conductivity_clipping_residual"
        ].value
        raw_candidate = (
            capillary_model.gardner_B
            * capillary_model.gardner_a
            / water_table_depth.pow(capillary_model.gardner_b)
            * capillary_model.water_density
        )

        torch.testing.assert_close(retained + clipping, raw_candidate)

    def test_open_port_limiter_conservation_checker(self, capillary_model):
        """Validate the retained-flux and limiter-residual boundary ledger."""
        state = {"water_table_depth": torch.tensor(0.5)}
        result = capillary_model.forward(state, 0.0, 3600.0)

        assert set(capillary_model.metadata.conservation_laws) == {"mass"}
        assert capillary_model.check_conservation_laws(state, result) == {
            "mass": pytest.approx(0.0)
        }

        perturbed = dict(result)
        perturbed["capillary_rise_flux"] = (
            result["capillary_rise_flux"].value + torch.tensor(1.0e-5)
        )
        assert capillary_model.check_conservation_laws(state, perturbed) == {
            "mass": pytest.approx(1.0e-5)
        }

        for missing_key in (
            "capillary_rise_flux",
            "diagnostics.capillary_rise.conductivity_clipping_residual",
        ):
            missing = dict(result)
            missing.pop(missing_key)
            with pytest.raises(KeyError, match="Missing conservation output"):
                capillary_model.check_conservation_laws(state, missing)

        missing_nested = dict(result)
        nested_diagnostics = dict(result["diagnostics.capillary_rise"])
        nested_diagnostics.pop("negative_flux_projection_residual")
        missing_nested["diagnostics.capillary_rise"] = nested_diagnostics
        with pytest.raises(KeyError, match="negative_flux_projection_residual"):
            capillary_model.check_conservation_laws(state, missing_nested)

        nonfinite = dict(result)
        nonfinite["capillary_rise_flux"] = torch.tensor(float("nan"))
        with pytest.raises(ValueError, match="must be finite"):
            capillary_model.check_conservation_laws(state, nonfinite)

    def test_real_simulator_open_port_provenance(self):
        """Route the open boundary flux without a false local-conservation warning."""
        model = CapillaryRiseTransport(name="capillary_rise_transport")
        state = {
            "water_table_depth": torch.tensor(0.5),
            "soil_moisture_profile": torch.tensor(
                [0.42, 0.41, 0.40, 0.39, 0.38]
            ),
        }
        simulator = RELICSimulator(
            {
                "max_time": 3600.0,
                "time_step": 3600.0,
                "conservation_checks": True,
                "strict_dependencies": False,
                "device": "cpu",
            }
        )
        simulator.initialize_state(state)
        simulator.register_functor(model, execution_group="fast")
        direct = model.forward(state, time=0.0, dt=3600.0)

        simulator.step(dt=3600.0)

        actual = simulator.state_manager.get_state("capillary_rise_flux")
        expected = direct["capillary_rise_flux"].value.to(
            device=actual.device,
            dtype=actual.dtype,
        )
        torch.testing.assert_close(actual, expected)
        assert set(model.metadata.conservation_laws) == {"mass"}
        assert simulator.get_effective_device(model.name).type == "cpu"
        provenance = simulator.get_variable_provenance("capillary_rise_flux")
        assert {entry["functor"] for entry in provenance} == {model.name}
        assert {entry["role"] for entry in provenance} == {"writer", "final_writer"}

    def test_phase8_iphs_protocol(self, capillary_model):
        """Test Phase 8 IPHS structural protocol for capillary rise."""
        forms = capillary_model.declared_forms()
        hodge_terms = capillary_model.build_hodge_terms()
        dissipative = capillary_model.dissipative_structure({})
        ports = capillary_model.ports()

        assert isinstance(capillary_model, IPHSDECFunctor)
        assert capillary_model.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.transport.max_capillary_height" in capillary_model.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "water_table_depth",
            "capillary_volumetric_flux",
            "capillary_mass_flux",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {"capillary_groundwater_boundary_flux"}
        producer_port = ports[0]
        assert producer_port.flow_var == "capillary_rise_flux"
        assert producer_port.flow_space.units == "kg m-2 s-1"
        assert producer_port.positive_power_convention == "out_of_system"
        dependencies = capillary_model.get_dependencies()
        assert dependencies["optional"] == []
        assert dependencies["lagged"] == ["groundwater_dynamics"]

    def test_phase8_certificate_is_valid(self):
        """Test Phase 8 IPHS equivalence certificate completeness."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path(
                "hydrosphere_transport_capillary_rise_capillary_rise_transport.yaml"
            )
        )
        audit = certificate.validate()

        assert audit["passed"] is True
        assert (
            certificate.process_id
            == "relic.scientific.hydrosphere.transport.capillary_rise.CapillaryRiseTransport"
        )
        assert certificate.canonical_role == "transport"

    def test_phase8_residual_diagnostics_are_routed(self, capillary_model, monkeypatch):
        """Test capillary limiting residuals are exposed in diagnostics."""
        _set_parameter(capillary_model, "k_saturated", 1.0e-9)
        _use_gardner_coefficients(
            monkeypatch,
            capillary_model,
            torch.tensor(1.0),
            torch.tensor(2.0),
            torch.tensor(1.5),
        )

        result = capillary_model.forward(
            {
                "water_table_depth": torch.tensor(0.5),
                "soil_moisture_profile": torch.tensor([0.42, 0.41, 0.40, 0.39, 0.38]),
            },
            0.0,
            3600.0,
            soil_layers=5,
            layer_thickness=0.2,
        )
        diagnostics = result["diagnostics.capillary_rise"]

        conductivity_update = result[
            "diagnostics.capillary_rise.conductivity_clipping_residual"
        ]
        saturation_update = result[
            "diagnostics.capillary_rise.saturation_constraint_residual"
        ]
        for update in (conductivity_update, saturation_update):
            assert isinstance(update, UpdateItem)
            assert update.mode is UpdateMode.REPLACE
            assert update.per_timestep is False
            assert update.unit == "kg m-2 s-1"
            assert update.source_functor == capillary_model.name

        assert torch.all(conductivity_update.value > 0.0)
        assert torch.all(diagnostics["negative_flux_projection_residual"] == 0.0)
        assert torch.all(saturation_update.value >= 0.0)
        # One output path per quantity: the nested mapping does not repeat them.
        assert "conductivity_clipping_residual" not in diagnostics
        assert "saturation_constraint_residual" not in diagnostics
        expected_boundary_flux = (
            _as_tensor(capillary_model.k_saturated)
            * _as_tensor(capillary_model.water_density)
        )
        torch.testing.assert_close(
            result["capillary_rise_flux"].value,
            expected_boundary_flux,
        )

    def test_diagnostics_are_numeric_and_profiles_are_plain_tensors(self, capillary_model):
        """Every diagnostic is a real number or a plain torch Tensor (no wrapper type);
        the layer profiles keep their shape in the active and the inactive case."""
        active = capillary_model.forward(
            {
                "water_table_depth": torch.tensor(1.0),
                "soil_moisture_profile": torch.tensor([0.2, 0.25, 0.3, 0.35, 0.4]),
            },
            0.0,
            3600.0,
            soil_layers=5,
            layer_thickness=0.2,
        )
        inactive = capillary_model.forward({"water_table_depth": torch.tensor(5.0)}, 0.0, 3600.0)
        profile_shape = (int(capillary_model.capillary_profile_layer_count),)
        for result in (active, inactive):
            diagnostics = result["diagnostics.capillary_rise"]
            for key, value in diagnostics.items():
                assert isinstance(value, (torch.Tensor, bool, int, float)), key
            assert type(diagnostics["capillary_rise_profile"]) is torch.Tensor
            assert diagnostics["capillary_rise_profile"].shape == profile_shape
            update = result["diagnostics.capillary_rise.saturation_constraint_residual"]
            assert type(update.value) is torch.Tensor
            assert update.value.shape == profile_shape
            assert "saturation_constraint_residual" not in diagnostics

    def test_hodge_contract_is_the_volumetric_to_mass_conversion(self, capillary_model):
        """The structural Hodge maps m s-1 to kg m-2 s-1 with rho_w only; Ks is the runtime limiter."""
        from cf_units import Unit

        hodge = capillary_model.build_hodge_contract()
        rho_w = float(get_physical_constant("water_density"))
        n = hodge.matrix.shape[0]
        torch.testing.assert_close(
            hodge.matrix.to(torch.float64),
            torch.eye(n, dtype=torch.float64) * rho_w,
        )
        assert hodge.operator_units == "kg m-3"
        assert hodge.domain.units == "m s-1"
        assert hodge.codomain.units == "kg m-2 s-1"
        mapped = Unit(hodge.operator_units) * Unit(hodge.domain.units)
        assert mapped.is_convertible(Unit(hodge.codomain.units))
        assert hodge.parameter_keys == ("relic.physical.constants.water.density_liquid",)
        assert (
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity"
            not in hodge.parameter_keys
        )

    def test_hodge_contract_reproduces_the_forward_mass_conversion(self, capillary_model):
        """Applying the Hodge to the volumetric Gardner flux must equal the produced mass flux."""
        depth = torch.tensor(1.2)
        result = capillary_model.forward({"water_table_depth": depth}, time=0.0, dt=3600.0)
        mass_flux = _unwrap_update(result["capillary_rise_flux"])
        volumetric_flux, _ = capillary_model._compute_gardner_flux_with_residuals(depth)
        hodge = capillary_model.build_hodge_contract(n_dofs=1)
        converted = hodge.matrix.to(volumetric_flux.dtype) @ volumetric_flux.reshape(1)
        assert float(volumetric_flux) > 0.0
        torch.testing.assert_close(converted.reshape(()), mass_flux.to(converted.dtype))
        # Ks bounds the volumetric flux inside the runtime law, not inside the Hodge.
        original_ks = _as_tensor(capillary_model.k_saturated).detach().clone()
        _set_parameter(capillary_model, "k_saturated", original_ks * 2.0)
        try:
            doubled = capillary_model.build_hodge_contract(n_dofs=1)
            torch.testing.assert_close(doubled.matrix, hodge.matrix)
        finally:
            _set_parameter(capillary_model, "k_saturated", original_ks)


if __name__ == "__main__":
    # Run basic smoke test
    model = CapillaryRiseTransport("test")
    
    state = {
        "water_table_depth": torch.tensor(0.8),
        "soil_moisture_profile": torch.tensor([0.15, 0.20, 0.25, 0.30, 0.35])
    }
    
    try:
        result = model.forward(state, 0.0, 3600.0, soil_layers=5, layer_thickness=0.2)
        diagnostics = result.get("diagnostics.capillary_rise", {})
        surface_flux = _unwrap_update(result["capillary_rise_flux"])
        print("Capillary rise test passed!")
        print(f"Surface flux: {surface_flux:.2e} kg m-2 s-1")
        print(f"Effective height: {diagnostics.get('effective_capillary_height', 0.0):.2f} m")
        print(f"Availability: {diagnostics.get('capillary_availability', 0.0):.2f}")
        print(f"Gardner coefficients: a={model.gardner_a:.2e}, b={model.gardner_b:.2f}, B={model.gardner_B:.2f}")
    except Exception as e:
        print(f"Capillary rise test failed: {e}")
        import traceback
        traceback.print_exc()



def test_capillary_rise_reads_no_unused_transport_or_limiting_default():
    """The Gardner steady flux has no scalar diffusivity (None) and uses no tortuosity or
    porosity copy; B = 1 + 1.5 / (b - 1) is defined for b > 1, which every valid pore-size
    index or van Genuchten n gives, so no limiting-case default B is read."""
    import inspect

    functor = CapillaryRiseTransport(name="capillary_contract")
    assert functor.transport_type == "capillary_rise"
    assert functor.diffusivity is None
    for key in (
        "hydrosphere.transport.capillary_rise.diffusivity",
        "hydrosphere.transport.soil_tortuosity",
        "hydrosphere.numerical.gardner_B_default",
        "hydrosphere.numerical.gardner_coefficient_epsilon",
    ):
        assert key not in functor.required_parameters()
        assert key not in functor.metadata.parameters
    assert "self.porosity = self.surface_porosity" not in inspect.getsource(CapillaryRiseTransport)


def test_gardner_exponent_at_or_below_one_is_rejected():
    functor = CapillaryRiseTransport(name="capillary_b_domain")
    functor.hydraulic_method = "brooks_corey"
    # A fresh Parameter object: the registered one is the ParameterManager's live value
    # and must not be mutated by this test.
    functor.pore_size_index = torch.nn.Parameter(torch.tensor(-1.0 / 3.0), requires_grad=False)
    with pytest.raises(ValueError, match="Gardner exponent"):
        functor._compute_gardner_coefficients()


def test_soil_moisture_profile_of_another_size_is_rejected():
    functor = CapillaryRiseTransport(name="capillary_profile_size")
    state = {
        "water_table_depth": torch.tensor(1.0),
        "soil_moisture_profile": torch.tensor([0.2, 0.25, 0.3]),
    }
    with pytest.raises(ValueError, match="soil_moisture_profile"):
        functor.forward(state, 0.0, 3600.0, soil_layers=5, layer_thickness=0.2)


def _capillary_tensors(result) -> dict:
    """Every tensor of a capillary update, keyed by output and diagnostic name."""
    tensors = {}
    for name, update in result.items():
        value = update.value if isinstance(update, UpdateItem) else update
        members = value.items() if isinstance(value, dict) else [(None, value)]
        for member, item in members:
            if isinstance(item, torch.Tensor):
                tensors[name if member is None else f"{name}:{member}"] = item
    return tensors


def _capillary_state() -> dict:
    """A water table inside the capillary fringe and below the conductivity cap."""
    return {"water_table_depth": torch.tensor(1.0, dtype=torch.float32)}


_HYDRAULIC_METHOD_KEY = "hydrosphere.transport.hydraulic_parameterization_method"
_PORE_SIZE_INDEX_KEY = "hydrosphere.transport.brooks_corey_pore_size_index"
_AIR_ENTRY_PRESSURE_KEY = "hydrosphere.transport.air_entry_pressure"
_K_SATURATED_KEY = "hydrosphere.soil_moisture.saturated_hydraulic_conductivity"
_N_VG_KEY = "hydrosphere.soil_moisture.n_vg"
_ALPHA_VG_KEY = "hydrosphere.soil_moisture.alpha_vg"


def _select_hydraulic_method(manager, method: str) -> None:
    spec = manager.get_parameter(_HYDRAULIC_METHOD_KEY)
    manager.set_value(
        _HYDRAULIC_METHOD_KEY, method, units=spec.units, validate=True, notify=False
    )


@pytest.mark.parametrize(
    "method, key, value",
    [
        ("brooks_corey", _PORE_SIZE_INDEX_KEY, 0.55),
        ("brooks_corey", _AIR_ENTRY_PRESSURE_KEY, -0.1122),
        ("brooks_corey", _K_SATURATED_KEY, 1.1e-5),
        ("van_genuchten", _N_VG_KEY, 2.079),
        ("van_genuchten", _ALPHA_VG_KEY, 8.25),
        ("van_genuchten", _K_SATURATED_KEY, 1.1e-5),
    ],
    ids=[
        "brooks_corey-pore_size_index",
        "brooks_corey-air_entry_pressure",
        "brooks_corey-k_saturated",
        "van_genuchten-n_vg",
        "van_genuchten-alpha_vg",
        "van_genuchten-k_saturated",
    ],
)
def test_capillary_gardner_coefficients_follow_a_changed_binding(
    restore_global_parameter_manager,
    method,
    key,
    value,
):
    """The Gardner coefficients are derived from the held bindings at use, so an
    instance built before a binding changed computes with the same consistent
    coefficients as one built afterwards."""
    # References: Gardner (1958), Soil Science 85(4), 228-232; Eagleson (1978),
    #   Water Resour. Res. 14(5), 722-730; Brooks and Corey (1964); van
    #   Genuchten (1980), Soil Sci. Soc. Am. J. 44, 892-898.
    # Mathematical expression: Brooks-Corey b = 2 + 3 lambda_p,
    #   a = Ks |h_entry|^b; van Genuchten approximation m = 1 - 1/n, b = 2/m,
    #   a = Ks (1/alpha)^b; B = 1 + 1.5/(b - 1); q = min(B a / Z^b, Ks).
    # Parameter meanings: lambda_p pore-size index, h_entry air-entry head, n
    #   and alpha van Genuchten shape and inverse-length parameters, Ks
    #   saturated conductivity, Z water-table depth, q capillary flux.
    # Units: h_entry, Z [m]; alpha [m-1]; Ks, q [m s-1]; a [m^(b+1) s-1];
    #   lambda_p, n, m, b, B [1].
    # Domain/constraints: lambda_p > 0 and n > 1 give b > 1; the new values lie
    #   inside the ParameterManager bounds, so no value is clamped; Z = 1 m is
    #   inside the capillary fringe and q stays below Ks.
    # Sign convention: q is an upward nonnegative magnitude.
    manager = restore_global_parameter_manager
    _select_hydraulic_method(manager, method)
    model = CapillaryRiseTransport(name="capillary_built_before_the_change")
    manager.set_value(key, value, units=manager.get_parameter(key).units, validate=True)
    fresh = CapillaryRiseTransport(name="capillary_built_after_the_change")

    assert float(model.held_manager_parameters()[key].detach()) == pytest.approx(value)

    k_saturated = model.k_saturated.detach()
    if method == "brooks_corey":
        expected_b = 2.0 + 3.0 * model.pore_size_index.detach()
        expected_a = k_saturated * (
            abs(model.air_entry_pressure.detach()) ** expected_b
        )
    else:
        expected_m = 1.0 - 1.0 / model.n_vg.detach()
        torch.testing.assert_close(
            model.m_vg.detach(), expected_m, rtol=0.0, atol=0.0
        )
        expected_b = 2.0 / expected_m
        expected_a = k_saturated * ((1.0 / model.alpha_vg.detach()) ** expected_b)
    expected_B = 1.0 + 1.5 / (expected_b - 1.0)
    torch.testing.assert_close(
        model.gardner_b.detach(), expected_b, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        model.gardner_a.detach(), expected_a, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        model.gardner_B.detach(), expected_B, rtol=0.0, atol=0.0
    )

    expected = _capillary_tensors(fresh(_capillary_state(), time=0.0, dt=3600.0))
    actual = _capillary_tensors(model(_capillary_state(), time=0.0, dt=3600.0))
    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        torch.testing.assert_close(
            tensor, expected[name], rtol=0.0, atol=0.0, msg=name
        )


@pytest.mark.parametrize("method", ["brooks_corey", "van_genuchten"])
def test_capillary_supports_repeated_backward_through_one_instance(
    restore_global_parameter_manager,
    method,
):
    """No autograd graph is carried from one call into the next: the instance
    stores no tensor derived from a binding, a second backward through it
    works and it gives the gradients of the first."""
    manager = restore_global_parameter_manager
    _select_hydraulic_method(manager, method)
    model = CapillaryRiseTransport(name="capillary_repeated_backward")
    stored_graphs = sorted(
        name
        for name, attribute in vars(model).items()
        if isinstance(attribute, torch.Tensor) and attribute.grad_fn is not None
    )
    assert stored_graphs == []

    held = model.held_manager_parameters()
    differentiated = {
        "brooks_corey": (_PORE_SIZE_INDEX_KEY, _AIR_ENTRY_PRESSURE_KEY, _K_SATURATED_KEY),
        "van_genuchten": (_N_VG_KEY, _ALPHA_VG_KEY, _K_SATURATED_KEY),
    }[method]
    gradients = []
    try:
        for _ in range(2):
            for binding in held.values():
                binding.grad = None
            update = model(_capillary_state(), time=0.0, dt=3600.0)
            update["capillary_rise_flux"].value.sum().backward()
            gradients.append(
                {key: held[key].grad.detach().cpu().clone() for key in differentiated}
            )
    finally:
        for binding in held.values():
            binding.grad = None

    for key in differentiated:
        assert torch.isfinite(gradients[0][key]).all(), key
        assert bool((gradients[0][key] != 0).all()), key
        torch.testing.assert_close(
            gradients[1][key], gradients[0][key], rtol=0.0, atol=0.0, msg=key
        )


@pytest.mark.skipif(
    not is_cuda_available(), reason="Held-binding relocation requires a CUDA device"
)
@pytest.mark.parametrize("method", ["brooks_corey", "van_genuchten"])
def test_capillary_computes_on_the_state_device_wherever_its_bindings_sit(
    restore_global_parameter_manager,
    method,
):
    """A held ParameterManager binding has one device, that of the latest
    registration of any Functor holding it. The coefficients are derived on the
    device of the state: with the bindings on CUDA a CPU state gives the
    unchanged update with every tensor on the CPU, for an instance built before
    the relocation and for one built after it."""
    manager = restore_global_parameter_manager
    _select_hydraulic_method(manager, method)
    bindings = manager.get_trainable_parameters()
    keys = tuple(
        CapillaryRiseTransport(name="capillary_binding_keys").held_manager_parameters()
    )
    assert keys
    devices = {key: bindings[key].device for key in keys}
    state = {"water_table_depth": torch.tensor([0.5, 1.0, 1.7], dtype=torch.float32)}
    try:
        for key in keys:
            relocate_binding(manager, key, "cpu")
        reference = CapillaryRiseTransport(name="capillary_reference_binding_device")
        expected = _capillary_tensors(reference(dict(state), time=0.0, dt=3600.0))
        early = CapillaryRiseTransport(name="capillary_built_before_relocation")

        for key in keys:
            relocate_binding(manager, key, "cuda")
        late = CapillaryRiseTransport(name="capillary_built_after_relocation")
        results = {
            "before_relocation": _capillary_tensors(
                early(dict(state), time=0.0, dt=3600.0)
            ),
            "after_relocation": _capillary_tensors(
                late(dict(state), time=0.0, dt=3600.0)
            ),
        }
    finally:
        for key, device in devices.items():
            relocate_binding(manager, key, device)

    for built, actual in results.items():
        assert set(actual) == set(expected), built
        for name, tensor in actual.items():
            assert tensor.device.type == "cpu", (built, name)
            torch.testing.assert_close(
                tensor, expected[name], rtol=0.0, atol=0.0, msg=f"{built}: {name}"
            )
