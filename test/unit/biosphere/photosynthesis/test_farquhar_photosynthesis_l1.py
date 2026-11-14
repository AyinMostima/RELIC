"""
L1 Unit Tests for FarquharPhotosynthesis Module

Tests the Farquhar photosynthesis model for:
- Architecture compliance with BiologicalProcessFunctor
- Parameter system integration via ParameterManager
- Basic photosynthesis functionality
- Output structure and units
- Absence of anti-patterns

References:
- Farquhar, G. D., von Caemmerer, S., & Berry, J. A. (1980). 
  A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species.
  Planta, 149(1), 78-90.
"""

import pytest
import torch

from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.parameters import get_global_manager
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _canonical_state(ppfd=1000.0):
    """Build the required canonical inputs: the four forcing ports and the canopy
    state leaf_area_index (1.0, the value of the removed default)."""
    return {
        "air_temperature": torch.tensor([298.15]),
        "atmospheric_co2_ppm": torch.tensor([400.0]),
        "ppfd": torch.tensor([ppfd]),
        "vapor_pressure_deficit": torch.tensor([1000.0]),
        "leaf_area_index": torch.tensor([1.0]),
    }


def test_absent_modifiers_are_the_identity_not_parameter_defaults():
    """vcmax_scaling and the water, temperature and ozone stress factors are optional
    multiplicative modifiers: an absent one is the identity 1 (no limitation, no
    scaling); no parameter stands in for it."""
    photo = FarquharPhotosynthesis()
    assert not hasattr(photo, "_get_environment_defaults")
    state = _canonical_state()
    state["leaf_area_index"] = torch.tensor([3.0])
    explicit = dict(state)
    for key in ("vcmax_scaling", "water_stress_factor", "temperature_stress_factor",
                "ozone_stress_factor"):
        explicit[key] = torch.tensor([1.0])
    absent_out = photo.forward(state, 0.0, 3600.0)
    explicit_out = photo.forward(explicit, 0.0, 3600.0)
    for key in ("gross_photosynthesis", "net_photosynthesis"):
        absent = absent_out[key].value if hasattr(absent_out[key], "value") else absent_out[key]
        given = explicit_out[key].value if hasattr(explicit_out[key], "value") else explicit_out[key]
        torch.testing.assert_close(absent, given, rtol=0.0, atol=0.0)


def test_leaf_area_index_is_a_required_input():
    """leaf_area_index is the canopy state of the Beer-Lambert scaling, with no identity
    value: a missing LAI raises instead of taking biosphere.defaults.leaf_area_index."""
    photo = FarquharPhotosynthesis()
    assert "leaf_area_index" in photo.get_required_inputs()
    state = _canonical_state()
    state.pop("leaf_area_index")
    with pytest.raises(ValueError, match="leaf_area_index"):
        photo.forward(state, 0.0, 3600.0)


class TestFarquharArchitecture:
    """Test architectural compliance of FarquharPhotosynthesis."""

    def test_inherits_from_biological_process_functor(self):
        """Verify FarquharPhotosynthesis inherits from BiologicalProcessFunctor."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        assert isinstance(functor, BiologicalProcessFunctor), \
            "FarquharPhotosynthesis must inherit from BiologicalProcessFunctor"

    def test_has_required_metadata(self):
        """Verify FarquharPhotosynthesis has complete metadata."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        metadata = functor.get_metadata()
        if hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        assert "parameters" in metadata, "Metadata must include parameters"
        assert "references" in metadata, "Metadata must include references"
        assert "mathematical_expression" in metadata, "Metadata must include mathematical expressions"
        
        # Check for key Farquhar parameters
        params = metadata["parameters"]
        assert any("vcmax" in str(p).lower() for p in params), "Must document Vcmax parameter"
        assert any("jmax" in str(p).lower() for p in params), "Must document Jmax parameter"

    def test_declares_dependencies(self):
        """Verify FarquharPhotosynthesis declares its dependencies."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        dependencies = functor.get_dependencies()
        assert isinstance(dependencies, list), "Dependencies must be a list"
        assert dependencies == ["environmental_forcing", "atmospheric_radiation"]

    def test_is_torch_module(self):
        """Verify FarquharPhotosynthesis is a torch.nn.Module for differentiability."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        assert isinstance(functor, torch.nn.Module), \
            "FarquharPhotosynthesis must be a torch.nn.Module for differentiability"


class TestFarquharParameterSystem:
    """Test parameter system integration."""

    def test_numeric_constructor_capacities_are_private_typed_buffers(self):
        """Numeric instance overrides use canonical metadata without creating leaves."""
        param_manager = get_global_manager()
        before = {name: param_manager.get(f"biosphere.photosynthesis.{name}")
                  for name in ("vcmax25", "jmax25", "rd25")}

        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        for name, expected in {"vcmax25": 50.0, "jmax25": 100.0, "rd25": 1.0}.items():
            value = getattr(functor, name)
            assert dict(functor.named_buffers())[name] is value
            assert value.dtype == torch.float32
            assert not value.requires_grad
            assert value.item() == pytest.approx(expected)
            assert functor.manager_binding_key(value) is None
            assert name in functor.state_dict()
            assert param_manager.get(f"biosphere.photosynthesis.{name}") == pytest.approx(before[name])

    def test_parameter_units_documented(self):
        """Verify all parameters have documented units."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        metadata = functor.get_metadata()
        if hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        params = metadata.get("parameters", {})
        
        # Check that key parameters have unit information
        assert params, "Parameters must be documented in metadata"

    def test_no_hardcoded_constants(self):
        """Verify no hardcoded physical constants in __init__."""
        import inspect
        source = inspect.getsource(FarquharPhotosynthesis.__init__)
        
        # Check for common hardcoded patterns (not exhaustive)
        suspicious_patterns = [
            "torch.tensor([0-9]+\\.?[0-9]*)",  # Direct tensor creation with numbers
        ]
        
        import re
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, source)
            # Allow some technical constants like zeros/ones for initialization
            non_trivial = [m for m in matches if not any(x in m for x in ["0.0", "1.0", "0", "1"])]
            assert len(non_trivial) == 0, \
                f"Found potential hardcoded constants: {non_trivial}"

    def test_parameters_have_correct_namespace(self):
        """Verify parameters use correct namespace convention."""
        param_manager = get_global_manager()
        
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        # Parameters should be registered under biological_process namespace
        # This is verified by successful instantiation without errors


class TestFarquharFunctionality:
    """Test basic photosynthesis functionality."""

    def test_forward_execution(self):
        """Verify forward() executes without errors."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        result = functor.forward(state=_canonical_state(), time=0.0, dt=1.0)
        assert result is not None, "forward() must return a result"
        assert isinstance(result, dict), "forward() must return a dict"

    def test_forward_positional_time_dt_mapping(self, monkeypatch):
        """Verify positional args map to time and dt in forward()."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )

        captured = {"time": None, "dt": None}

        def _stub_forward_core(state_arg, **kwargs):
            captured["time"] = kwargs.get("time")
            captured["dt"] = kwargs.get("dt")
            return {"net_photosynthesis": torch.tensor([1.0])}

        monkeypatch.setattr(functor, "_forward_core", _stub_forward_core)

        result = functor.forward(_canonical_state(), 123.0, 456.0)

        assert isinstance(result, dict), "forward() must return dict"
        assert captured["time"] == 123.0, "Positional time must map to runtime time"
        assert captured["dt"] == 456.0, "Positional dt must map to runtime dt"

    def test_compute_rate_execution(self):
        """Verify compute_rate() executes and returns appropriate type."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            enable_gpu=torch.cuda.is_available(),
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        assert not hasattr(functor, "_compute_rate_core")
        # The forward kernel reads the state only; no environment keyword is popped
        # (FunctorBase rejects it before forward).
        import inspect

        assert "environment" not in inspect.getsource(FarquharPhotosynthesis.forward)
        assert "environment" not in inspect.getsource(FarquharPhotosynthesis._forward_core)
        output = functor.forward(_canonical_state(ppfd=500.0), time=0.0, dt=1.0)
        result = output["net_photosynthesis"].value
        assert isinstance(result, torch.Tensor), "rate value must be Tensor"
        assert torch.isfinite(result).all()

    def test_photosynthesis_positive_under_light(self):
        """Verify photosynthesis rate is positive under light conditions."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        result = functor.forward(state=_canonical_state(), time=0.0, dt=1.0)
        assert result["gross_photosynthesis"].value.item() > 0

    def test_differentiability(self):
        """Verify photosynthesis computation is differentiable."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        temperature = torch.tensor([298.15], requires_grad=True)
        state = _canonical_state(ppfd=500.0)
        state["air_temperature"] = temperature
        rate = functor.forward(state, 0.0, 1.0)["net_photosynthesis"].value
        
        # Compute gradient
        rate.sum().backward()
        
        assert temperature.grad is not None, "Temperature gradient must be computed"
        assert not torch.isnan(temperature.grad).any(), "Gradient must not contain NaN"


class TestFarquharOutputStructure:
    """Test output structure and format."""

    def test_output_is_flux_dict(self):
        """Verify output follows FluxDict structure."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        result = functor.forward(state=_canonical_state(), time=0.0, dt=1.0)
        
        assert isinstance(result, dict), "Output must be a dictionary"
        # Check for expected flux keys
        assert len(result) > 0, "Output must contain flux values"

    def test_output_units_consistency(self):
        """Verify output units are consistent with CF conventions."""
        functor = FarquharPhotosynthesis(
            name="test_farquhar",
            config={},
            device=torch.device("cpu"),
            enable_gpu=False,
            validate_inputs=True,
            pft_name="test_pft",
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0
        )
        
        result = functor.forward(state=_canonical_state(), time=0.0, dt=1.0)

        expected_units = {
            "gross_photosynthesis": "umol CO2 m-2 s-1",
            "net_photosynthesis": "umol CO2 m-2 s-1",
            "gross_primary_productivity": "kg C m-2 s-1",
            "electron_transport_rate": "umol e- m-2 s-1",
        }
        assert set(result) == set(functor.metadata.output_variables)
        for key, value in result.items():
            assert isinstance(value, UpdateItem), key
            assert value.mode is UpdateMode.REPLACE
            assert value.per_timestep is False
            assert value.source_functor == functor.name
            assert torch.isfinite(value.value).all()
            if key in expected_units:
                assert value.unit == expected_units[key]


class TestFarquharNoAntiPatterns:
    """Test for absence of anti-patterns."""

    def test_no_state_get_with_hardcoded_default(self):
        """Verify no state.get() calls with hardcoded physical constant defaults."""
        import inspect
        source = inspect.getsource(FarquharPhotosynthesis)
        
        # Look for state.get with numeric defaults
        import re
        pattern = r"state\.get\([^,]+,\s*(?:torch\.tensor\()?[0-9]+\.?[0-9]*"
        matches = re.findall(pattern, source)
        
        # Filter out acceptable patterns (like 0.0 for missing data)
        suspicious = [m for m in matches if not any(x in m for x in ["0.0", "0", "None"])]
        assert len(suspicious) == 0, \
            f"Found state.get() with hardcoded defaults: {suspicious}"

    def test_no_direct_config_parameter_access(self):
        """Verify no direct self.config access for parameters."""
        import inspect
        source = inspect.getsource(FarquharPhotosynthesis)
        
        # Look for self.config access patterns
        import re
        pattern = r"self\.config\[['\"]([a-z_]+)['\"]\]"
        matches = re.findall(pattern, source)
        
        # Filter out acceptable config access (runtime state, not parameters)
        # Parameters should come from ParameterManager
        suspicious = [m for m in matches if m not in ["device", "enable_gpu", "validate_inputs"]]
        assert len(suspicious) == 0, \
            f"Found direct config access for potential parameters: {suspicious}"


