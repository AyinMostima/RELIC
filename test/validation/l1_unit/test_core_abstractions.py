"""
L1 Unit Tests - Core Abstractions

Tests the four fundamental abstractions of RELIC:
- PhysicalField
- BiologicalProcess  
- TransportMechanism
- ScaleCoupling

These tests verify the basic functionality and contracts of each abstraction
in isolation, without dependencies on other modules.
"""

import inspect
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
import torch

# Import core abstractions
from relic.core.abstractions import (
    BiologicalProcess,
    HierarchicalScalingBase,
    PhysicalField,
    ScaleCoupling,
    SpatialScalingBase,
    TemporalScalingBase,
    TransportMechanism,
)
from test.utils.forward_call import call_forward, call_transport_forward
from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorMetadata,
    FunctorSubstreamMixin,
    ProcessCategory,
)
from relic.core.state import ExternalIntent, UpdateItem


def _sensitivity_metadata(name, outputs=("rate",)):
    return FunctorMetadata(
        name=name,
        category=ProcessCategory.BIOLOGICAL_PROCESS,
        description="Algebraic fixture of the base finite-difference sensitivity",
        references=["Press et al. (2007). Numerical Recipes, 3rd ed., section 5.7."],
        input_variables=["x"],
        output_variables=list(outputs),
    )


class _SensitivityProcess(BiologicalProcessFunctor):
    """Algebraic fixture for the base finite-difference contract (r = k*x).

    k is a registered nn.Parameter of the Functor (``k``) or of a submodule
    (``kinetics.k``); the base replaces it through torch.func.functional_call.
    """

    def __init__(self, *, nested=True, wrapped=False, failure=None):
        super().__init__(
            name="sensitivity_fixture", config={}, metadata=_sensitivity_metadata("sensitivity_fixture"),
        )
        self.nested = nested
        self.wrapped = wrapped
        self.failure = failure
        self.calls = 0
        self.forward_times = []
        value = torch.nn.Parameter(torch.tensor(2.125, dtype=torch.float64))
        if nested:
            self.kinetics = torch.nn.Module()
            self.kinetics.k = value
        else:
            self.k = value

    def forward(self, state, time, dt, **kwargs):
        # References: Press et al. (2007), Numerical Recipes, finite differences.
        # Mathematical expression: r = k*x, dr/dk = x.
        # Parameter meanings: k is a scalar coefficient and x is the input.
        # Units: dimensionless algebraic fixture.
        # Domain/constraints: k > 0 for the smooth L2 case.
        # Sign convention: positive k increases r.
        assert "target_output" not in state
        assert kwargs == {}, kwargs
        self.forward_times.append((time, dt))
        self.calls += 1
        if self.failure is not None:
            raise self.failure("injected sensitivity failure")
        value = self.kinetics.k if self.nested else self.k
        rate = value * state["x"]
        return {
            "rate": UpdateItem(rate) if self.wrapped else rate,
            "diagnostics": {"duplicate": rate * 100.0},
            "request": ExternalIntent("checkpoint", "unused-sensitivity.pt", "sensitivity"),
        }


class _CubicSensitivityProcess(BiologicalProcessFunctor):
    """Analytic fixture: r = k**3 * x + b * x, so dr/dk = 3 k**2 x and dr/db = x.

    k is an nn.Parameter of a submodule and b a registered buffer.
    """

    def __init__(self, dtype=torch.float64, k=1.5):
        super().__init__(
            name="cubic_sensitivity_fixture", config={},
            metadata=_sensitivity_metadata("cubic_sensitivity_fixture"),
        )
        self.kinetics = torch.nn.Module()
        self.kinetics.k = torch.nn.Parameter(torch.tensor(k, dtype=dtype))
        self.register_buffer("b", torch.tensor(0.25, dtype=dtype))
        self.scale = 4.0

    def forward(self, state, time, dt, **kwargs):
        # References: Press et al. (2007), Numerical Recipes, section 5.7.
        # Mathematical expression: r = k^3 x + b x; dr/dk = 3 k^2 x; dr/db = x.
        # Parameter meanings: k is a cubic coefficient, b a linear coefficient,
        #   x the state input.
        # Units: dimensionless algebraic fixture.
        # Domain/constraints: any finite k, b and x.
        # Sign convention: positive dr/dk increases r.
        return {"rate": self.kinetics.k ** 3 * state["x"] + self.b * state["x"]}


class _LazySensitivityProcess(BiologicalProcessFunctor):
    """A tensor provided lazily by __getattr__ (not in __dict__, not a
    parameter or buffer): it is not a real member of the module."""

    def __init__(self):
        super().__init__(
            name="lazy_sensitivity_fixture", config={},
            metadata=_sensitivity_metadata("lazy_sensitivity_fixture"),
        )

    def __getattr__(self, name):
        if name == "lazy_k":
            return torch.tensor(2.0, dtype=torch.float64)
        return super().__getattr__(name)

    def forward(self, state, time, dt, **kwargs):
        # References: Press et al. (2007), Numerical Recipes, section 5.7.
        # Mathematical expression: r = k x.
        # Parameter meanings: k a lazily provided coefficient, x the state input.
        # Units: dimensionless algebraic fixture.
        # Domain/constraints: any finite k and x.
        # Sign convention: positive k increases r.
        return {"rate": self.lazy_k * state["x"]}


class _MutatingSensitivityProcess(BiologicalProcessFunctor):
    """Violates I1: forward advances a buffer in place (like an accumulator)."""

    def __init__(self):
        super().__init__(
            name="mutating_sensitivity_fixture", config={},
            metadata=_sensitivity_metadata("mutating_sensitivity_fixture"),
        )
        self.k = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        self.register_buffer("cumulative_rate", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, state, time, dt, **kwargs):
        # References: Press et al. (2007), Numerical Recipes, section 5.7.
        # Mathematical expression: r = k x; cumulative_rate += r (the I1 violation).
        # Parameter meanings: k a coefficient, x the state input.
        # Units: dimensionless algebraic fixture.
        # Domain/constraints: any finite k and x.
        # Sign convention: positive k increases r.
        rate = self.k * state["x"]
        self.cumulative_rate += rate.detach()
        return {"rate": rate}


class _StochasticSensitivityProcess(FunctorSubstreamMixin, BiologicalProcessFunctor):
    """Stochastic fixture: r = k * (x + z), z ~ N(0, 1) from the Functor substream.

    With common random numbers every evaluation uses the same z, so the finite
    difference is exactly dr/dk = x + z(omega_0), z(omega_0) being the first
    draw from the substream position before the sensitivity call.
    """

    EXTRA_STATE_FORMAT = "relic.test.sensitivity_substream"
    SUBSTREAM_LABEL = "SensitivityFixture"

    def __init__(self, seed=20260926):
        super().__init__(
            name="stochastic_sensitivity_fixture", config={},
            metadata=_sensitivity_metadata("stochastic_sensitivity_fixture"),
        )
        self._initialize_substream(seed)
        self.k = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float64))

    def forward(self, state, time, dt, **kwargs):
        # References: Glasserman (2004), Monte Carlo Methods in Financial
        #   Engineering, section 7.1 (finite differences, common random numbers).
        # Mathematical expression: r = k (x + z), z ~ N(0, 1); dr/dk = x + z.
        # Parameter meanings: k a coefficient, x the state input, z one standard
        #   normal draw of the Functor substream.
        # Units: dimensionless algebraic fixture.
        # Domain/constraints: one draw per evaluation (draw_count counts them).
        # Sign convention: positive dr/dk increases r.
        z = torch.randn((), dtype=torch.float64, generator=self.generator)
        self.draw_count += 1
        return {"rate": self.k * (state["x"] + z)}


def _declared_field_class(dims, declare=True):
    namespace = {
        "get_value_at": lambda self, coordinates: coordinates[..., 0],
        "compute_gradient": lambda self, coordinates: torch.zeros_like(coordinates),
        "interpolate": lambda self, source, target: target[..., 0],
        "update_field": lambda self, time_step, source_terms, boundary_conditions=None: None,
    }
    if declare:
        namespace["SPATIAL_DIMS"] = dims
    return type("DeclaredField", (PhysicalField,), namespace)


class TestPhysicalFieldContract:
    """SPATIAL_DIMS is the one declared field property; identity and units come
    from metadata and contracts, so the abstraction fabricates none."""

    def test_the_constructor_takes_only_the_device(self):
        import inspect

        assert list(inspect.signature(PhysicalField.__init__).parameters) == ["self", "device"]
        field = _declared_field_class(2)()
        for removed in ("field_name", "units", "default_value", "temporal_dependence", "spatial_dims"):
            assert not hasattr(field, removed), removed
        assert not hasattr(PhysicalField, "get_field_metadata")

    def test_an_undeclared_spatial_dims_raises(self):
        with pytest.raises(TypeError, match="SPATIAL_DIMS"):
            _declared_field_class(None, declare=False)()

    @pytest.mark.parametrize("dims", [0, 4, 2.0, "2", True])
    def test_an_invalid_spatial_dims_raises(self, dims):
        with pytest.raises((TypeError, ValueError), match="SPATIAL_DIMS"):
            _declared_field_class(dims)()

    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_coordinates_are_checked_against_spatial_dims(self, dims):
        field = _declared_field_class(dims)()
        assert field.validate_coordinates(torch.zeros(4, dims))
        assert not field.validate_coordinates(torch.zeros(4, dims + 1))

    def test_interpolate_takes_no_method_argument(self):
        """No implementation selects an interpolation method: the abstract
        contract, the bridge default and the relic fields take only the source
        and target coordinates."""
        import inspect

        from relic.core.functor import PhysicalFieldFunctor
        from relic.scientific.atmosphere.radiation import RadiationField
        from relic.scientific.hydrosphere.groundwater.dynamics import GroundwaterDynamics
        from relic.scientific.hydrosphere.snow.albedo import SnowAlbedo

        for cls in (PhysicalField, PhysicalFieldFunctor, RadiationField, GroundwaterDynamics, SnowAlbedo):
            parameters = list(inspect.signature(cls.interpolate).parameters)
            assert parameters == ["self", "source_coordinates", "target_coordinates"], cls.__name__

    def test_a_point_model_has_no_coordinate_query(self):
        field = _declared_field_class(None)()
        with pytest.raises(RuntimeError, match="point model"):
            field.validate_coordinates(torch.zeros(4, 2))


class TestPhysicalField:
    """Test suite for PhysicalField abstraction."""
    
    def test_field_creation(self):
        """Test creation of basic physical field."""
        
        class TemperatureField(PhysicalField):
            SPATIAL_DIMS = 2

            def __init__(self, base_temp: float = 298.15):
                super().__init__()
                self.base_temp = base_temp
                
            def get_value_at(self, coordinates: torch.Tensor) -> torch.Tensor:
                # Simple linear gradient
                return torch.full_like(coordinates[..., 0], self.base_temp)
            
            def compute_gradient(self, coordinates: torch.Tensor) -> torch.Tensor:
                # Constant gradient for testing
                grad = torch.zeros_like(coordinates)
                grad[..., 0] = 0.01  # 0.01 K/m gradient in x
                return grad
            
            def interpolate(self, source_coordinates: torch.Tensor,
                          target_coordinates: torch.Tensor) -> torch.Tensor:
                # Simple interpolation - just return values at target coords
                return self.get_value_at(target_coordinates)
            
            def update_field(self, time_step: float, 
                           source_terms: Dict, 
                           boundary_conditions: Optional[Dict] = None) -> None:
                # Simple update - no-op for testing
                pass
        
        field = TemperatureField(base_temp=300.0)
        coords = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        
        values = field.get_value_at(coords)
        assert values.shape == (3,), "Value shape mismatch"
        assert torch.allclose(values, torch.tensor([300.0, 300.0, 300.0])), "Incorrect field values"
        
        gradients = field.compute_gradient(coords)
        assert gradients.shape == (3, 2), "Gradient shape mismatch"
        assert torch.allclose(gradients[:, 0], torch.tensor([0.01, 0.01, 0.01])), "Incorrect gradients"
    
    def test_field_interpolation(self):
        """Test field interpolation methods."""
        
        class GridField(PhysicalField):
            SPATIAL_DIMS = 2

            def __init__(self, grid_data: torch.Tensor):
                super().__init__()
                self.grid_data = grid_data
                self.nx, self.ny = grid_data.shape
                
            def get_value_at(self, coordinates: torch.Tensor) -> torch.Tensor:
                # Bilinear interpolation
                x = coordinates[..., 0].clamp(0, self.nx - 1)
                y = coordinates[..., 1].clamp(0, self.ny - 1)
                
                x0 = torch.floor(x).long()
                x1 = torch.ceil(x).long()
                y0 = torch.floor(y).long()
                y1 = torch.ceil(y).long()
                
                # Bilinear weights
                wx = x - x0.float()
                wy = y - y0.float()
                
                v00 = self.grid_data[x0, y0]
                v10 = self.grid_data[x1, y0]
                v01 = self.grid_data[x0, y1]
                v11 = self.grid_data[x1, y1]
                
                return (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 + \
                       (1 - wx) * wy * v01 + wx * wy * v11
            
            def compute_gradient(self, coordinates: torch.Tensor) -> torch.Tensor:
                # Finite difference gradient
                eps = 0.01
                x_plus = coordinates.clone()
                x_plus[..., 0] += eps
                x_minus = coordinates.clone()
                x_minus[..., 0] -= eps
                
                y_plus = coordinates.clone()
                y_plus[..., 1] += eps
                y_minus = coordinates.clone()
                y_minus[..., 1] -= eps
                
                grad_x = (self.get_value_at(x_plus) - self.get_value_at(x_minus)) / (2 * eps)
                grad_y = (self.get_value_at(y_plus) - self.get_value_at(y_minus)) / (2 * eps)
                
                return torch.stack([grad_x, grad_y], dim=-1)
            
            def interpolate(self, source_coordinates: torch.Tensor,
                          target_coordinates: torch.Tensor) -> torch.Tensor:
                # Use bilinear interpolation
                return self.get_value_at(target_coordinates)
            
            def update_field(self, time_step: float,
                           source_terms: Dict,
                           boundary_conditions: Optional[Dict] = None) -> None:
                # No-op for testing
                pass
        
        # Create test grid
        grid = torch.tensor([[1.0, 2.0, 3.0],
                            [2.0, 3.0, 4.0],
                            [3.0, 4.0, 5.0]])
        
        field = GridField(grid)
        
        # Test interpolation at grid points
        coords = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        values = field.get_value_at(coords)
        expected = torch.tensor([1.0, 3.0, 5.0])
        assert torch.allclose(values, expected, atol=1e-5), "Grid point interpolation failed"
        
        # Test interpolation between grid points
        coords = torch.tensor([[0.5, 0.5]])
        value = field.get_value_at(coords)
        # Bilinear interpolation at (0.5, 0.5) with corners 1,2,2,3
        # value = 0.25*1 + 0.25*2 + 0.25*2 + 0.25*3 = 2.0
        expected = torch.tensor([2.0])
        assert torch.allclose(value, expected, atol=1e-5), "Midpoint interpolation failed"


class TestBiologicalProcess:
    """Test suite for BiologicalProcess abstraction."""

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("wrapped", [False, True])
    @pytest.mark.parametrize("aggregation", ["target", "sum", "L2"])
    def test_base_sensitivity_handles_output_channels_and_restores_parameter(self, nested, wrapped, aggregation):
        process = _SensitivityProcess(nested=nested, wrapped=wrapped)
        owner = process.kinetics if nested else process
        original = owner.k
        target = {"target_output": "rate"} if aggregation == "target" else {}
        sensitivity = process.parameter_sensitivity(
            "kinetics.k" if nested else "k", {"x": torch.tensor(3.0, dtype=torch.float64)},
            time=5.0, dt=60.0, absolute_eps=0.01, relative_eps=0.01,
            aggregation_method=aggregation, **target,
        )
        assert torch.allclose(sensitivity, torch.tensor(3.0, dtype=torch.float64), atol=1e-10, rtol=1e-10)
        assert owner.k is original
        assert isinstance(owner.k, torch.nn.Parameter)
        assert owner.k.item() == 2.125
        assert set(process.forward_times) == {(5.0, 60.0)}

    def test_base_sensitivity_takes_inputs_from_the_state_only(self):
        """No environment argument; time and dt are required keywords."""
        import inspect

        parameters = inspect.signature(BiologicalProcess.parameter_sensitivity).parameters
        assert "environment" not in parameters
        # The finite-difference step is eps = max(absolute_eps, relative_eps*|theta|);
        # the former 'perturbation' argument never entered it. Stochastic Functors
        # are differenced with common random numbers, so no median of repeated
        # evaluations exists.
        for removed in ("perturbation", "enable_noise_mitigation", "num_evaluations_for_median"):
            assert removed not in parameters, removed
        # No literal step: the default follows the perturbed tensor's dtype.
        assert parameters["absolute_eps"].default is None
        assert parameters["relative_eps"].default is None
        for name in ("time", "dt", "target_output"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameters["time"].default is inspect.Parameter.empty
        assert parameters["dt"].default is inspect.Parameter.empty
        process = _SensitivityProcess()
        with pytest.raises(TypeError):
            process.parameter_sensitivity("kinetics.k", {"x": torch.tensor(3.0)}, {})

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
    def test_base_sensitivity_failure_restores_original_parameter_object(self, nested, failure):
        from relic.core.functor.functor_exceptions import ExecutionError

        process = _SensitivityProcess(nested=nested, failure=failure)
        owner = process.kinetics if nested else process
        original = owner.k
        # The governed call reports a forward exception as ExecutionError; an
        # interrupt propagates unchanged. Either way the member is restored.
        expected = ExecutionError if failure is RuntimeError else failure
        with pytest.raises(expected, match="injected sensitivity failure"):
            process.parameter_sensitivity(
                "kinetics.k" if nested else "k", {"x": torch.tensor(3.0, dtype=torch.float64)},
                time=0.0, dt=1.0,
            )
        assert owner.k is original
        assert isinstance(owner.k, torch.nn.Parameter)
        assert process.calls == 1

    @pytest.mark.parametrize("central", [True, False])
    def test_base_sensitivity_matches_the_analytic_derivative_without_side_effects(self, central):
        process = _CubicSensitivityProcess()
        state = {"x": torch.tensor(2.0, dtype=torch.float64)}
        before = {key: value.clone() for key, value in process.state_dict().items()}
        parameter = process.kinetics.k
        k, x, rel = 1.5, 2.0, 1e-4
        sensitivity = process.parameter_sensitivity(
            "kinetics.k", state, time=0.0, dt=1.0, target_output="rate",
            absolute_eps=1e-12, relative_eps=rel, use_central_difference=central,
        )
        eps = rel * k
        if central:
            # (f(k+e) - f(k-e)) / 2e = 3 k^2 x + e^2 x for f = k^3 x.
            expected = 3.0 * k**2 * x + eps**2 * x
        else:
            # (f(k+e) - f(k)) / e = 3 k^2 x + 3 k e x + e^2 x.
            expected = 3.0 * k**2 * x + 3.0 * k * eps * x + eps**2 * x
        assert sensitivity.item() == pytest.approx(expected, rel=1e-9)
        assert sensitivity.item() == pytest.approx(3.0 * k**2 * x, rel=1e-3 if not central else 1e-7)
        buffer_sensitivity = process.parameter_sensitivity(
            "b", state, time=0.0, dt=1.0, target_output="rate", absolute_eps=1e-6, relative_eps=1e-6,
        )
        assert buffer_sensitivity.item() == pytest.approx(x, rel=1e-8)
        # Side-effect free: the same Parameter object, the same values.
        assert process.kinetics.k is parameter
        after = process.state_dict()
        assert set(after) == set(before)
        for key, value in before.items():
            assert torch.equal(after[key], value), key

    @pytest.mark.parametrize("central", [True, False])
    def test_base_sensitivity_uses_common_random_numbers_without_advancing_the_stream(self, central):
        process = _StochasticSensitivityProcess()
        process.forward({"x": torch.tensor(0.0, dtype=torch.float64)}, time=0.0, dt=1.0)
        before = process.get_extra_state()
        probe = torch.Generator()
        probe.set_state(before["generator_state"].clone())
        z0 = torch.randn((), dtype=torch.float64, generator=probe).item()
        state = {"x": torch.tensor(2.0, dtype=torch.float64)}
        kwargs = dict(
            time=0.0, dt=1.0, target_output="rate", absolute_eps=1e-6, relative_eps=1e-6,
            use_central_difference=central,
        )
        first = process.parameter_sensitivity("k", state, **kwargs)
        second = process.parameter_sensitivity("k", state, **kwargs)
        # Same random numbers for every evaluation: dr/dk = x + z(omega_0).
        assert first.item() == pytest.approx(2.0 + z0, rel=1e-9)
        assert torch.equal(first, second)
        after = process.get_extra_state()
        assert after["draw_count"] == before["draw_count"]
        assert torch.equal(after["generator_state"], before["generator_state"])
        assert after["wiener_pending"] == before["wiener_pending"]

    @pytest.mark.parametrize("central", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_default_step_follows_the_parameter_dtype(self, dtype, central):
        """h = eps_m^(1/3) |k| (central) or eps_m^(1/2) |k| (forward) on the analytic cubic."""
        process = _CubicSensitivityProcess(dtype=dtype)
        state = {"x": torch.tensor(2.0, dtype=dtype)}
        sensitivity = process.parameter_sensitivity(
            "kinetics.k", state, time=0.0, dt=1.0, target_output="rate", use_central_difference=central,
        )
        assert sensitivity.dtype == dtype
        tolerance = {
            (torch.float32, True): 1e-3, (torch.float32, False): 3e-3,
            (torch.float64, True): 1e-9, (torch.float64, False): 1e-6,
        }[(dtype, central)]
        assert sensitivity.item() == pytest.approx(3.0 * 1.5**2 * 2.0, rel=tolerance)

    @pytest.mark.parametrize("central", [True, False])
    def test_a_zero_parameter_is_stepped_from_the_yaml_scale_floor(self, central):
        """At k = 0 the curvature scale is the declared floor x_floor, so the
        difference quotient of r = k^3 x is h^2 x with h = eps_m^(1/p) x_floor."""
        from relic.parameters import get_global_manager

        floor = float(get_global_manager().get(
            "optimization.sensitivity.finite_difference.zero_parameter_scale", required=True,
        ))
        process = _CubicSensitivityProcess(dtype=torch.float64, k=0.0)
        state = {"x": torch.tensor(2.0, dtype=torch.float64)}
        sensitivity = process.parameter_sensitivity(
            "kinetics.k", state, time=0.0, dt=1.0, target_output="rate", use_central_difference=central,
        )
        order = 1.0 / 3.0 if central else 0.5
        step = torch.finfo(torch.float64).eps ** order * floor
        # b x cancels in the difference; k^3 x gives h^2 x for both schemes.
        assert sensitivity.item() == pytest.approx(step**2 * 2.0, rel=1e-6)

    @pytest.mark.parametrize(
        "absolute_eps, relative_eps",
        [(1e-3, None), (None, 1e-3), (1e-3, 1e-4), (1e-5, 1e-3)],
    )
    def test_an_explicit_step_is_the_largest_given_bound(self, absolute_eps, relative_eps):
        """At k = 0 (x_c = x_floor) the central quotient of k^3 x is h^2 x with
        h = max of the given bounds {absolute_eps, relative_eps * x_floor}."""
        from relic.parameters import get_global_manager

        floor = float(get_global_manager().get(
            "optimization.sensitivity.finite_difference.zero_parameter_scale", required=True,
        ))
        process = _CubicSensitivityProcess(dtype=torch.float64, k=0.0)
        sensitivity = process.parameter_sensitivity(
            "kinetics.k", {"x": torch.tensor(2.0, dtype=torch.float64)}, time=0.0, dt=1.0,
            target_output="rate", absolute_eps=absolute_eps, relative_eps=relative_eps,
        )
        bounds = [value for value in (absolute_eps, None if relative_eps is None else relative_eps * floor) if value is not None]
        assert sensitivity.item() == pytest.approx(max(bounds) ** 2 * 2.0, rel=1e-9)

    @pytest.mark.parametrize(
        "parameter_name, ppfd, reference",
        [("vcmax25", 2000.0, 0.147178), ("jmax25", 1000.0, 0.047613), ("rd25", 1000.0, -0.786939)],
    )
    def test_float32_farquhar_default_step_matches_the_float64_reference(
        self, parameter_name, ppfd, reference, restore_global_parameter_manager,
    ):
        """Float32 FarquharPhotosynthesis with the default step reproduces the
        float64 autograd derivatives (calibrated set 75.3 / 148.6 / 1.2)."""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            model = FarquharPhotosynthesis(vcmax25=75.3, jmax25=148.6, rd25=1.2)
            # Constructor values enter through config and are persistent module buffers.
            assert dict(model.named_buffers())[parameter_name].dtype == torch.float32
            state = {
                "air_temperature": torch.tensor(298.15),
                "atmospheric_co2_ppm": torch.tensor(400.0),
                "ppfd": torch.tensor(ppfd),
                "leaf_area_index": torch.tensor(1.0),
                "vapor_pressure_deficit": torch.tensor(1000.0),
                "vcmax_scaling": torch.tensor(1.0),
            }
            sensitivity = BiologicalProcess.parameter_sensitivity(
                model, parameter_name, state, time=0.0, dt=1.0, target_output="net_photosynthesis",
            )
        finally:
            torch.set_default_dtype(previous)
        assert sensitivity.item() == pytest.approx(reference, rel=5e-3)

    def test_an_in_place_change_of_a_member_violates_i1(self):
        """A forward that changes a parameter or buffer in place is reported
        (member named, I1 cited); the change is not undone."""
        process = _MutatingSensitivityProcess()
        with pytest.raises(RuntimeError, match=r"cumulative_rate.*I1"):
            process.parameter_sensitivity(
                "k", {"x": torch.tensor(2.0, dtype=torch.float64)}, time=0.0, dt=1.0,
                target_output="rate",
            )
        assert process.cumulative_rate.item() != 0.0

    def test_a_forward_without_side_effects_passes_the_i1_check(self):
        process = _CubicSensitivityProcess()
        versions = {name: tensor._version for name, tensor in process.state_dict(keep_vars=True).items()}
        process.parameter_sensitivity(
            "kinetics.k", {"x": torch.tensor(2.0, dtype=torch.float64)}, time=0.0, dt=1.0,
            target_output="rate",
        )
        assert {name: tensor._version for name, tensor in process.state_dict(keep_vars=True).items()} == versions

    def test_a_tensor_provided_by_getattr_is_not_a_member(self):
        """A lazily provided tensor passes isinstance(Tensor) but is no real
        member; it is rejected before functional_call with a clear TypeError."""
        process = _LazySensitivityProcess()
        assert isinstance(process.lazy_k, torch.Tensor)
        with pytest.raises(TypeError, match="lazy_k.*not a real tensor member"):
            process.parameter_sensitivity(
                "lazy_k", {"x": torch.tensor(3.0, dtype=torch.float64)}, time=0.0, dt=1.0,
                target_output="rate",
            )

    def test_float32_isoprene_default_step_is_exact_for_the_linear_molecular_weight(self, restore_global_parameter_manager):
        """The mass emission is linear in the float32 buffer isoprene_mw, F_mass =
        F_umol MW, so dF_mass/dMW = F_mass / MW. In float32 the dtype-aware default
        step meets the relative tolerance 1e-3 and the former 1e-6 step does not. (V_max
        is a ParameterManager binding held by reference and no member to perturb.)"""
        from relic.scientific.biosphere.bvoc.synthesis.isoprene import IsopreneSynthesis
        from relic.utils.physical_constants import get_physical_constant

        target = "diagnostics.bvoc.isoprene.emission_rate_mass"
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            model = IsopreneSynthesis()
            assert model.isoprene_mw.dtype == torch.float32
            state = {
                "air_temperature": torch.tensor(298.15),
                "gross_photosynthesis": torch.tensor(20.0),
                "electron_transport_rate": torch.tensor(150.0),
                "atmospheric_co2_ppm": torch.tensor(300.0),
                "leaf_area_index": torch.tensor(1.0),
            }
            hour = get_physical_constant("seconds_per_hour")
            mass_emission = model(dict(state), time=0.0, dt=hour)[target].value
            expected = (mass_emission / model.isoprene_mw).item()
            default = model.parameter_sensitivity(
                "isoprene_mw", state, time=0.0, dt=hour, target_output=target,
            ).item()
            former = model.parameter_sensitivity(
                "isoprene_mw", state, time=0.0, dt=hour, target_output=target,
                absolute_eps=1e-6, relative_eps=1e-6,
            ).item()
            with pytest.raises(TypeError, match="v_max_is.*not a real tensor member"):
                model.parameter_sensitivity(
                    "v_max_is", state, time=0.0, dt=hour, target_output="isoprene_emission",
                )
        finally:
            torch.set_default_dtype(previous)
        assert default == pytest.approx(expected, rel=1e-3)
        assert abs(former / expected - 1.0) > 1e-3

    def test_base_sensitivity_requires_a_tensor_member_of_a_module(self):
        process = _CubicSensitivityProcess()
        with pytest.raises(TypeError, match="tensor"):
            process.parameter_sensitivity("scale", {"x": torch.tensor(2.0)}, time=0.0, dt=1.0)
        with pytest.raises(KeyError, match="absent"):
            process.parameter_sensitivity("kinetics.absent", {"x": torch.tensor(2.0)}, time=0.0, dt=1.0)
        with pytest.raises(KeyError, match="missing_module"):
            process.parameter_sensitivity("missing_module.k", {"x": torch.tensor(2.0)}, time=0.0, dt=1.0)

        class _PlainProcess(BiologicalProcess):
            def __init__(self):
                super().__init__(process_name="plain")
                self.k = torch.tensor(1.0)

            def forward(self, state, time, dt, **kwargs):
                return {"rate": self.k * state["x"]}

        with pytest.raises(TypeError, match="nn.Module"):
            _PlainProcess().parameter_sensitivity("k", {"x": torch.tensor(2.0)}, time=0.0, dt=1.0)

    @pytest.mark.parametrize(
        "parameter_name, ppfd, reference",
        [("vcmax25", 2000.0, 0.147), ("jmax25", 1000.0, 0.0476), ("rd25", 1000.0, -0.787)],
    )
    def test_base_sensitivity_reproduces_the_farquhar_reference_derivatives(
        self, parameter_name, ppfd, reference, restore_global_parameter_manager,
    ):
        """dA/dtheta of FarquharPhotosynthesis at 25 C, 400 ppm, VPD 1 kPa.

        The references are the autograd derivatives of the governed call at the
        parameter set vcmax25 = 75.3, jmax25 = 148.6, rd25 = 1.2 (the calibrated
        overlay; the test pins it, since the test session skips the overlay).
        The central difference of the base must reproduce them and the
        derivative computed here by autograd through the same functional_call.
        """
        from torch.func import functional_call

        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        model = FarquharPhotosynthesis(vcmax25=75.3, jmax25=148.6, rd25=1.2)
        state = {
            "air_temperature": torch.as_tensor(298.15),
            "atmospheric_co2_ppm": torch.as_tensor(400.0),
            "ppfd": torch.as_tensor(ppfd),
            "leaf_area_index": torch.tensor(1.0),
            "vapor_pressure_deficit": torch.as_tensor(1000.0),
            "vcmax_scaling": torch.as_tensor(1.0),
        }
        # Constructor values enter through config and are persistent module buffers.
        parameter = dict(model.named_buffers())[parameter_name]
        before = parameter.detach().clone()
        sensitivity = BiologicalProcess.parameter_sensitivity(
            model, parameter_name, state, time=0.0, dt=1.0, target_output="net_photosynthesis",
            absolute_eps=1e-3, relative_eps=1e-3,
        )
        theta = parameter.detach().clone().requires_grad_(True)
        output = functional_call(model, {parameter_name: theta}, args=(dict(state),), kwargs={"time": 0.0, "dt": 1.0})
        (autograd_derivative,) = torch.autograd.grad(output["net_photosynthesis"].value.sum(), theta)
        assert sensitivity.item() == pytest.approx(autograd_derivative.item(), rel=1e-3)
        assert sensitivity.item() == pytest.approx(reference, rel=2e-3)
        assert dict(model.named_buffers())[parameter_name] is parameter
        assert torch.equal(parameter.detach(), before)

    def test_sensitivity_refuses_a_manager_held_binding(self):
        """A ParameterManager learnable binding held by reference has one owner, the
        manager; its sensitivity goes through the manager calibration path
        (get_trainable_parameters), so parameter_sensitivity refuses it."""
        from relic.core.functor import BiologicalProcessFunctor, FunctorMetadata

        class _Manager:
            def __init__(self):
                self.nn_parameters = {"test.theta": torch.nn.Parameter(torch.tensor(2.0))}

            def get_parameter(self, key):
                return None

            def get(self, key, **kwargs):
                return {"validation.numerical.response.ph_optimum": 7.0}.get(key)

        class _Holder(BiologicalProcessFunctor):
            def __init__(self, manager):
                super().__init__(
                    name="sensitivity_holder",
                    config={},
                    metadata=FunctorMetadata(
                        name="sensitivity_holder",
                        description="holder of a manager binding",
                        references=["unit test"],
                        input_variables=["x"],
                        output_variables=["rate"],
                    ),
                    parameter_manager=manager,
                )
                self.theta = manager.nn_parameters["test.theta"]

            def forward(self, state, time, dt, **kwargs):
                return {"rate": self.theta * state["x"]}

            def get_dependencies(self):
                return []

        manager = _Manager()
        holder = _Holder(manager)
        before = manager.nn_parameters["test.theta"].detach().clone()
        with pytest.raises(TypeError, match="ParameterManager"):
            BiologicalProcess.parameter_sensitivity(
                holder, "theta", {"x": torch.tensor(1.0)}, time=0.0, dt=1.0,
            )
        assert torch.equal(manager.nn_parameters["test.theta"].detach(), before)

    @pytest.mark.parametrize("target", ["request", "missing", 17])
    def test_base_sensitivity_rejects_invalid_target(self, target):
        process = _SensitivityProcess()
        original = process.kinetics.k
        with pytest.raises((TypeError, ValueError, RuntimeError, KeyError), match="target|numeric"):
            process.parameter_sensitivity(
                "kinetics.k", {"x": torch.tensor(3.0)}, time=0.0, dt=1.0, target_output=target,
            )
        assert process.kinetics.k is original
    
    def test_process_rate_computation(self):
        """Test biological process rate computation."""
        
        class PhotosynthesisProcess(BiologicalProcess):
            def __init__(self):
                super().__init__(process_name="photosynthesis")
                
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                # Simplified Farquhar model; the drivers are state inputs
                temperature = state['temperature']
                co2 = state['co2']
                light = state['light']
                
                # Temperature response
                temp_opt = 298.15
                temp_factor = torch.exp(-((temperature - temp_opt) / 10) ** 2)
                
                # Light response (Michaelis-Menten)
                light_half = 500.0
                light_factor = light / (light + light_half)
                
                # CO2 response
                co2_factor = co2 / 400.0
                
                # Maximum rate
                vmax = 50.0  # 碌mol CO2 m-2 s-1
                
                rate = vmax * temp_factor * light_factor * co2_factor
                return {"co2_assimilation": rate}
        
        process = PhotosynthesisProcess()
        
        # Test at optimal conditions
        state = {
            'temperature': torch.tensor(298.15),
            'co2': torch.tensor(400.0),
            'light': torch.tensor(1000.0)
        }

        rates = call_forward(process, state)
        rate = rates["co2_assimilation"]
        expected = 50.0 * 1.0 * (1000.0 / 1500.0) * 1.0  # ~33.33
        assert torch.allclose(rate, torch.tensor(expected), rtol=1e-4), "Rate computation failed"
        
        # Test temperature response
        state['temperature'] = torch.tensor(308.15)  # 10K above optimal
        rates = call_forward(process, state)
        rate = rates["co2_assimilation"]
        assert rate < expected, "Temperature response incorrect"

    def test_generic_temperature_response_runtime_fails_fast(self):
        """The abstract contract must not own a second response formula."""

        class TestProcess(BiologicalProcess):
            def __init__(self):
                super().__init__(process_name="test_process")

            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {}

        process = TestProcess()

        with pytest.raises(RuntimeError, match="concrete BiologicalProcessFunctor"):
            process.temperature_function(torch.tensor([298.15]))


class TestBiologicalProcessConstruction:
    """The abstract contract resolves no process parameter and masks no constant error."""

    @staticmethod
    def _process_class():
        class PlainProcess(BiologicalProcess):
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {}

        return PlainProcess

    def test_reference_temperature_constant_failure_propagates(self, monkeypatch):
        import relic.utils.physical_constants as physical_constants

        def _missing(name):
            raise KeyError(f"physical constant '{name}' is not defined")

        monkeypatch.setattr(physical_constants, "get_physical_constant", _missing)
        with pytest.raises(KeyError, match="reference_temperature"):
            self._process_class()(process_name="plain")

    def test_process_parameters_are_left_to_the_executable_bridge(self, monkeypatch):
        import relic.core.abstractions as abstractions

        def _forbidden(*_args, **_kwargs):
            raise AssertionError("the abstract contract resolves no process parameter")

        monkeypatch.setattr(abstractions, "_resolve_parameter_with_manager", _forbidden)
        process = self._process_class()(process_name="plain")
        assert process.q10_value is None and process.ph_optimum is None
        assert process.temperature_optimum is not None


class TestTransportMechanism:
    """Test suite for TransportMechanism abstraction."""

    @staticmethod
    def _label_only_transport(label: str = "diffusion"):
        class LabelOnlyTransport(TransportMechanism):
            def __init__(self):
                super().__init__(transport_type=label)

            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {}

        return LabelOnlyTransport()

    def test_diffusivity_is_absent_until_the_validated_setter_sets_it(self):
        """No constructor argument and no default: labels are descriptive only."""
        import inspect

        assert "diffusivity" not in inspect.signature(TransportMechanism.__init__).parameters
        transport = self._label_only_transport("diffusion")
        assert transport.diffusivity is None

        transport.set_scalar_diffusivity(2.5)
        assert transport.diffusivity == 2.5
        live = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        transport.set_scalar_diffusivity(live)
        assert transport.diffusivity is live

        for invalid in (0.0, -1.0, float("inf"), float("nan"), torch.tensor(-1.0), torch.tensor([1.0, 2.0])):
            with pytest.raises(ValueError, match="diffusivity"):
                transport.set_scalar_diffusivity(invalid)
        for invalid in (None, True, "1e-5", torch.tensor(1)):
            with pytest.raises(TypeError, match="diffusivity"):
                transport.set_scalar_diffusivity(invalid)
        assert transport.diffusivity is live

    def test_transport_classes_set_the_scalar_diffusivity_only_through_the_setter(self):
        """No relic transport class assigns the attribute; set_scalar_diffusivity does."""
        import ast
        import importlib
        import inspect
        import pkgutil
        import textwrap

        import relic.scientific

        for module in pkgutil.walk_packages(relic.scientific.__path__, "relic.scientific."):
            if not module.name.endswith("_cli"):
                importlib.import_module(module.name)

        def subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from subclasses(sub)

        checked = 0
        offenders = []
        for cls in set(subclasses(TransportMechanism)):
            if not cls.__module__.startswith("relic."):
                continue
            checked += 1
            tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    targets = [node.target]
                else:
                    continue
                # Resetting to None marks D absent (a field-valued law); any
                # other value must go through set_scalar_diffusivity.
                resets_to_none = (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is None
                )
                if resets_to_none:
                    continue
                if any(isinstance(t, ast.Attribute) and t.attr == "diffusivity" for t in targets):
                    offenders.append(f"{cls.__module__}.{cls.__qualname__}:{node.lineno}")
        assert checked > 20
        assert offenders == []

    def test_advection_diffusion_is_a_declared_transport_type(self):
        assert "advection_diffusion" in TransportMechanism.TRANSPORT_TYPES
        assert self._label_only_transport("advection_diffusion").transport_type == "advection_diffusion"

    def test_managed_application_is_an_explicit_boundary_transport(self):
        """Managed nutrient application must use the transport contract."""

        class ManagedApplicationTransport(TransportMechanism):
            def __init__(self):
                super().__init__(transport_type="managed_application")

            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {"managed_boundary_flux": state["applied_mass"] / dt}

        transport = ManagedApplicationTransport()
        output = transport.forward(
            {"applied_mass": torch.tensor(0.72)},
            time=0.0,
            dt=3600.0,
        )

        assert transport.transport_type == "managed_application"
        assert transport.diffusivity is None
        field = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(RuntimeError, match="no diffusivity"):
            transport._resolve_diffusivity_tensor(field)
        diffusivity = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        transport.set_scalar_diffusivity(diffusivity)
        flux = -transport._resolve_diffusivity_tensor(field) * field
        assert torch.equal(flux, -2.0 * field)
        assert torch.autograd.grad(flux.sum(), diffusivity)[0].item() == -3.0
        assert torch.allclose(
            output["managed_boundary_flux"],
            torch.tensor(2.0e-4),
        )

    def test_boundary_conditions_belong_to_the_concrete_process(self):
        """The base holds no generic boundary law (compute_diffusivity_tensor precedent).

        Every boundary type, a missing boundary value and an unknown type raise
        instead of returning a zero flux or a law with a hardcoded coefficient.
        """

        class DiffusiveBoundaryTransport(TransportMechanism):
            def __init__(self):
                super().__init__(transport_type="diffusion")
                self.set_scalar_diffusivity(2.0)

            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {}

        transport = DiffusiveBoundaryTransport()
        field = torch.tensor([1.0, 2.0], dtype=torch.float64)
        normal = torch.tensor([[-1.0], [-1.0]], dtype=torch.float64)
        for boundary_type, boundary_value in (
            (None, None),
            ("dirichlet", 1.0),
            ("dirichlet", None),
            ("neumann", None),
            ("robin", 0.5),
            ("periodic", None),
            ("open", None),
            ("no_such_type", 0.0),
        ):
            with pytest.raises(RuntimeError, match="concrete transport"):
                transport.boundary_conditions("x_min", field, normal, boundary_type, boundary_value)
        for removed in (
            "_compute_default_boundary_flux",
            "_compute_manager_boundary_flux",
            "_boundary_flux_from_manager_if_available",
            "_initialize_boundary_manager",
            "_apply_boundary_manager_override",
            "_normalize_boundary_type",
            "_map_boundary_face",
            "_ensure_boundary_flux_shape",
        ):
            assert not hasattr(TransportMechanism, removed), removed
        import relic.core.abstractions as abstractions

        for removed in (
            "_resolve_boundary_condition_manager",
            "BoundaryConditionManager",
            "BoundaryType",
            "HAS_BC_MANAGER",
        ):
            assert not hasattr(abstractions, removed), removed

    def test_base_declares_no_generic_medium_properties(self):
        """Tortuosity and porosity belong to (and are exported by) the concrete process."""

        class AdvectiveTransport(TransportMechanism):
            def __init__(self):
                super().__init__(transport_type="advection")

            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {}

        transport = AdvectiveTransport()
        assert not hasattr(transport, "tortuosity")
        assert not hasattr(transport, "porosity")
        assert set(transport.get_transport_metadata()) == {
            "transport_type",
            "diffusivity",
            "peclet_number",
            "has_velocity_field",
        }

    def test_unknown_transport_type_still_fails_fast(self):
        """The transport contract must reject undeclared mechanism types."""

        class UnknownTransport(TransportMechanism):
            def __init__(self):
                super().__init__(transport_type="unknown_mechanism")

            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {}

        with pytest.raises(ValueError, match="Invalid transport type"):
            UnknownTransport()
    
    def test_diffusion_transport(self):
        """Test diffusive transport mechanism."""
        
        class DiffusionTransport(TransportMechanism):
            def __init__(self, diffusivity: float = 1e-5):
                super().__init__(transport_type="diffusion")
                self.set_scalar_diffusivity(diffusivity)
                
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                source = state["source"]
                # Simple diffusion flux: -D * gradient
                coords = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
                
                source_grad = source.compute_gradient(coords)
                # Flux in x-direction
                flux = -self.diffusivity * source_grad[:, 0]
                
                return {"diffusive_flux": flux.mean()}  # Return flux dict
        
        # Create concentration field with gradient
        class ConcentrationField(PhysicalField):
            SPATIAL_DIMS = 2

            def __init__(self):
                super().__init__()
                
            def get_value_at(self, coordinates: torch.Tensor) -> torch.Tensor:
                # Linear concentration gradient
                return 100.0 - 10.0 * coordinates[..., 0]
            
            def compute_gradient(self, coordinates: torch.Tensor) -> torch.Tensor:
                grad = torch.zeros_like(coordinates)
                grad[..., 0] = -10.0  # Constant gradient
                return grad
            
            def interpolate(self, source_coordinates: torch.Tensor,
                          target_coordinates: torch.Tensor) -> torch.Tensor:
                return self.get_value_at(target_coordinates)
            
            def update_field(self, time_step: float,
                           source_terms: Dict,
                           boundary_conditions: Optional[Dict] = None) -> None:
                pass
        
        source = ConcentrationField()
        sink = ConcentrationField()  # Not used in this simple example
        
        transport = DiffusionTransport(diffusivity=1e-5)
        flux_dict = call_transport_forward(transport, source, sink)
        flux = flux_dict["diffusive_flux"]
        
        expected = -1e-5 * (-10.0)  # -D * gradient
        assert torch.allclose(flux, torch.tensor(expected)), "Diffusion flux incorrect"
    
    def test_advection_transport(self):
        """Test advective transport mechanism."""
        
        class AdvectionTransport(TransportMechanism):
            def __init__(self, velocity: torch.Tensor):
                super().__init__(transport_type="advection")
                self.velocity = velocity
                
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                source = state["source"]
                # Advective flux: v * C
                coords = torch.tensor([[0.0, 0.0]])
                concentration = source.get_value_at(coords)
                
                # Flux = velocity * concentration
                flux = self.velocity[0] * concentration
                
                return {"advective_flux": flux}
        
        class ConstantField(PhysicalField):
            SPATIAL_DIMS = 2

            def __init__(self, value: float):
                super().__init__()
                self.value = value
                
            def get_value_at(self, coordinates: torch.Tensor) -> torch.Tensor:
                return torch.full((coordinates.shape[0],), self.value)
            
            def compute_gradient(self, coordinates: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(coordinates)
            
            def interpolate(self, source_coordinates: torch.Tensor,
                          target_coordinates: torch.Tensor) -> torch.Tensor:
                return self.get_value_at(target_coordinates)
            
            def update_field(self, time_step: float,
                           source_terms: Dict,
                           boundary_conditions: Optional[Dict] = None) -> None:
                pass
        
        source = ConstantField(value=50.0)
        sink = ConstantField(value=0.0)
        
        velocity = torch.tensor([2.0, 0.0])  # 2 m/s in x-direction
        transport = AdvectionTransport(velocity)
        
        flux_dict = call_transport_forward(transport, source, sink)
        flux = flux_dict["advective_flux"]
        expected = 2.0 * 50.0  # v * C
        assert torch.allclose(flux, torch.tensor(expected)), "Advection flux incorrect"


class TestScaleCoupling:
    """Test suite for ScaleCoupling abstraction."""

    def test_abstract_scale_transforms_have_no_identity_implementation(self):
        """The scientific contract must fail when no scale operator is supplied."""

        class MissingScaleOperator(ScaleCoupling):
            def __init__(self):
                super().__init__(source_scale="fine", target_scale="coarse", coupling_type="spatial")

            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                return super().upscale(fine_data)

            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                return super().downscale(coarse_data)

        scaler = MissingScaleOperator()
        with pytest.raises(NotImplementedError, match="upscale operator"):
            scaler.upscale(torch.tensor([1.0]))
        with pytest.raises(NotImplementedError, match="downscale operator"):
            scaler.downscale(torch.tensor([1.0]))
    
    def test_upscaling(self):
        """Test upscaling from fine to coarse resolution."""
        
        class AveragingUpscaler(ScaleCoupling):
            def __init__(self):
                super().__init__(source_scale="fine", target_scale="coarse", coupling_type="spatial")
                
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                # Simple 2x2 averaging
                batch, height, width = fine_data.shape
                
                # Ensure even dimensions
                if height % 2 != 0 or width % 2 != 0:
                    raise ValueError("Dimensions must be even for 2x2 averaging")
                
                # Reshape and average
                coarse = fine_data.reshape(batch, height // 2, 2, width // 2, 2)
                coarse = coarse.mean(dim=(2, 4))
                
                return coarse
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                # Not implemented for this test
                raise NotImplementedError
            
            def resolution_converter(self, data: torch.Tensor,
                                   source_resolution: tuple,
                                   target_resolution: tuple) -> torch.Tensor:
                # Simple implementation for testing
                if source_resolution[0] < target_resolution[0]:
                    return self.upscale(data)
                else:
                    return self.downscale(data)
        
        scaler = AveragingUpscaler()
        
        # Create fine resolution data (4x4)
        fine_data = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                  [2.0, 3.0, 4.0, 5.0],
                                  [3.0, 4.0, 5.0, 6.0],
                                  [4.0, 5.0, 6.0, 7.0]]])
        
        coarse_data = scaler.upscale(fine_data)
        
        # Expected: 2x2 average
        expected = torch.tensor([[[2.0, 4.0],
                                 [4.0, 6.0]]])
        
        assert coarse_data.shape == (1, 2, 2), "Upscaling shape incorrect"
        assert torch.allclose(coarse_data, expected), "Upscaling values incorrect"
    
    def test_downscaling(self):
        """Test downscaling from coarse to fine resolution."""
        
        class BilinearDownscaler(ScaleCoupling):
            def __init__(self):
                super().__init__(source_scale="coarse", target_scale="fine", coupling_type="spatial")
                
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                # Not implemented for this test
                raise NotImplementedError
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                # Simple bilinear interpolation (2x upsampling)
                batch, height, width = coarse_data.shape
                
                # Use torch interpolate
                coarse_4d = coarse_data.unsqueeze(1)  # Add channel dim
                fine_4d = torch.nn.functional.interpolate(
                    coarse_4d, 
                    scale_factor=2, 
                    mode='bilinear',
                    align_corners=True
                )
                fine_data = fine_4d.squeeze(1)  # Remove channel dim
                
                return fine_data
            
            def resolution_converter(self, data: torch.Tensor,
                                   source_resolution: tuple,
                                   target_resolution: tuple) -> torch.Tensor:
                # Simple implementation for testing
                if source_resolution[0] > target_resolution[0]:
                    return self.downscale(data)
                else:
                    return self.upscale(data)
        
        scaler = BilinearDownscaler()
        
        # Create coarse resolution data (2x2)
        coarse_data = torch.tensor([[[1.0, 2.0],
                                    [3.0, 4.0]]])
        
        fine_data = scaler.downscale(coarse_data)
        
        assert fine_data.shape == (1, 4, 4), "Downscaling shape incorrect"
        # Check corners match original values
        assert torch.allclose(fine_data[0, 0, 0], torch.tensor(1.0)), "Corner value incorrect"
        assert torch.allclose(fine_data[0, 0, -1], torch.tensor(2.0)), "Corner value incorrect"
        assert torch.allclose(fine_data[0, -1, 0], torch.tensor(3.0)), "Corner value incorrect"
        assert torch.allclose(fine_data[0, -1, -1], torch.tensor(4.0)), "Corner value incorrect"
    
    def test_conservation(self):
        """Test that upscaling conserves quantities."""
        
        class ConservativeUpscaler(ScaleCoupling):
            def __init__(self):
                super().__init__(source_scale="fine", target_scale="coarse", coupling_type="spatial")
                
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                # Sum-preserving upscaling
                batch, height, width = fine_data.shape
                
                if height % 2 != 0 or width % 2 != 0:
                    raise ValueError("Dimensions must be even")
                
                # Reshape and sum
                coarse = fine_data.reshape(batch, height // 2, 2, width // 2, 2)
                coarse = coarse.sum(dim=(2, 4))
                
                return coarse
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                # Equal distribution downscaling
                batch, height, width = coarse_data.shape
                
                # Each coarse cell distributes equally to 4 fine cells
                fine = coarse_data.repeat_interleave(2, dim=1)
                fine = fine.repeat_interleave(2, dim=2)
                fine = fine / 4.0  # Distribute value
                
                return fine
            
            def resolution_converter(self, data: torch.Tensor,
                                   source_resolution: tuple,
                                   target_resolution: tuple) -> torch.Tensor:
                # Simple implementation for testing
                if source_resolution[0] < target_resolution[0]:
                    return self.upscale(data)
                else:
                    return self.downscale(data)
        
        scaler = ConservativeUpscaler()
        
        # Create fine data
        fine_data = torch.ones(1, 4, 4) * 10.0
        
        # Upscale
        coarse_data = scaler.upscale(fine_data)
        
        # Check conservation
        fine_total = fine_data.sum()
        coarse_total = coarse_data.sum()
        assert torch.allclose(fine_total, coarse_total), "Upscaling not conservative"
        
        # Downscale back
        fine_again = scaler.downscale(coarse_data)
        
        # Check conservation
        fine_again_total = fine_again.sum()
        assert torch.allclose(coarse_total, fine_again_total), "Downscaling not conservative"


class TestSpatialScalingBase:
    """Test suite for SpatialScalingBase abstraction."""
    
    def test_spatial_regrid_implementation(self):
        """Test spatial regridding with conservative method."""
        
        class ConservativeSpatialScaler(SpatialScalingBase):
            def __init__(self):
                super().__init__(
                    source_scale="fine_grid",
                    target_scale="coarse_grid",
                    method="conservative",
                    preserve="mass"
                )
            
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                # Simple averaging for upscaling
                if fine_data.dim() == 1:
                    # 1D case: average pairs
                    n = fine_data.shape[0]
                    if n % 2 != 0:
                        fine_data = fine_data[:-1]  # Truncate to even
                    coarse_data = fine_data.reshape(-1, 2).mean(dim=1)
                else:
                    # 2D case: 2x2 averaging
                    batch, height, width = fine_data.shape
                    coarse_h, coarse_w = height // 2, width // 2
                    coarse_data = fine_data[:, :coarse_h*2, :coarse_w*2]
                    coarse_data = coarse_data.reshape(batch, coarse_h, 2, coarse_w, 2)
                    coarse_data = coarse_data.mean(dim=(2, 4))
                return coarse_data
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                # Simple interpolation for downscaling
                if coarse_data.dim() == 1:
                    # 1D case: repeat each value
                    fine_data = coarse_data.repeat_interleave(2)
                else:
                    # 2D case: bilinear interpolation
                    coarse_4d = coarse_data.unsqueeze(1)
                    fine_4d = torch.nn.functional.interpolate(
                        coarse_4d, scale_factor=2, mode='bilinear', align_corners=True
                    )
                    fine_data = fine_4d.squeeze(1)
                return fine_data
            
            def regrid(
                self,
                data: torch.Tensor,
                source_grid: Any,
                target_grid: Any,
                method: str = "conservative",
                preserve: str = "mass"
            ) -> torch.Tensor:
                # Mock grid objects for testing
                if not hasattr(source_grid, 'n_cells'):
                    source_grid.n_cells = data.numel()
                if not hasattr(target_grid, 'n_cells'):
                    target_grid.n_cells = source_grid.n_cells // 4
                
                if target_grid.n_cells < source_grid.n_cells:
                    return self.upscale(data)
                else:
                    return self.downscale(data)
        
        scaler = ConservativeSpatialScaler()
        
        # Test 1D regridding
        fine_data_1d = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        class MockGrid:
            pass
        
        source_grid = MockGrid()
        target_grid = MockGrid()
        source_grid.n_cells = 4
        target_grid.n_cells = 2
        
        coarse_data_1d = scaler.regrid(fine_data_1d, source_grid, target_grid)
        expected_1d = torch.tensor([1.5, 3.5])
        assert torch.allclose(coarse_data_1d, expected_1d), "1D regridding failed"
        
        # Test 2D regridding
        fine_data_2d = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                     [2.0, 3.0, 4.0, 5.0],
                                     [3.0, 4.0, 5.0, 6.0],
                                     [4.0, 5.0, 6.0, 7.0]]])
        
        source_grid.n_cells = 16
        target_grid.n_cells = 4
        
        coarse_data_2d = scaler.regrid(fine_data_2d, source_grid, target_grid)
        assert coarse_data_2d.shape == (1, 2, 2), "2D regridding shape incorrect"
        assert coarse_data_2d[0, 0, 0] > 0, "2D regridding failed"
    
    def test_different_regrid_methods(self):
        """Test different regridding methods."""
        
        class MultiMethodSpatialScaler(SpatialScalingBase):
            def __init__(self):
                super().__init__(
                    source_scale="source",
                    target_scale="target",
                    method="conservative",
                    preserve="mass"
                )
            
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                return fine_data[::2]  # Simple decimation
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                return coarse_data.repeat_interleave(2)
            
            def regrid(
                self,
                data: torch.Tensor,
                source_grid: Any,
                target_grid: Any,
                method: str = "conservative",
                preserve: str = "mass"
            ) -> torch.Tensor:
                if method == "conservative":
                    # Conservative regridding preserves totals
                    total = data.sum()
                    if hasattr(target_grid, 'n_cells'):
                        n_target = target_grid.n_cells
                    else:
                        n_target = len(data) // 2
                    result = torch.full((n_target,), total / n_target)
                    return result
                elif method == "bilinear":
                    # Bilinear interpolation
                    return self.upscale(data)
                elif method == "nearest":
                    # Nearest neighbor
                    return data[::2]
                else:
                    return self.upscale(data)
        
        scaler = MultiMethodSpatialScaler()
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        class MockGrid:
            n_cells = 2
        
        source_grid = MockGrid()
        target_grid = MockGrid()
        
        # Test conservative method
        result_conservative = scaler.regrid(data, source_grid, target_grid, method="conservative")
        expected_total = data.sum()
        assert torch.allclose(result_conservative.sum(), expected_total), "Conservative regridding not conserving"
        
        # Test bilinear method
        result_bilinear = scaler.regrid(data, source_grid, target_grid, method="bilinear")
        expected_bilinear = torch.tensor([1.0, 3.0])
        assert torch.allclose(result_bilinear, expected_bilinear), "Bilinear regridding incorrect"
        
        # Test nearest method
        result_nearest = scaler.regrid(data, source_grid, target_grid, method="nearest")
        expected_nearest = torch.tensor([1.0, 3.0])
        assert torch.allclose(result_nearest, expected_nearest), "Nearest regridding incorrect"


class TestTemporalScalingBase:
    """Test suite for TemporalScalingBase abstraction."""

    def test_temporal_contract_uses_duration_overlap_not_count_formulas(self):
        class_contract = (TemporalScalingBase.__doc__ or "").lower()
        method_contract = (TemporalScalingBase.resample.__doc__ or "").lower()
        combined = f"{class_contract}\n{method_contract}"

        for stale in (
            "1/n",
            "sum_i",
            "n * mean(pattern)",
            "sumw(t_i) = n",
        ):
            assert stale not in combined
        for required in (
            "o_ij",
            "delta_i",
            "delta_j",
            "half-open",
            "explicit bounds",
            "interval_mean",
            "rate",
            "interval_accumulation",
            "state_at_boundary",
            "piecewise-constant",
            "mapping residual",
            "gap",
            "q_ref",
        ):
            assert required in combined

    def test_temporal_contract_revision_preserves_abstract_signature(self):
        signature = inspect.signature(TemporalScalingBase.resample)

        assert list(signature.parameters) == [
            "self",
            "data",
            "source_dt",
            "target_dt",
            "method",
            "pattern",
        ]
        assert TemporalScalingBase.__abstractmethods__ == {
            "upscale",
            "downscale",
            "resample",
        }
    
    def test_temporal_resampling(self):
        """Test temporal resampling with different methods."""
        
        class TimeSeriesResampler(TemporalScalingBase):
            def __init__(self):
                super().__init__(
                    source_scale="hourly",
                    target_scale="daily",
                    method="mean",
                    pattern_type="constant"
                )
            
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                # Aggregate hourly to daily (24:1 ratio)
                if fine_data.shape[0] % 24 != 0:
                    # Pad or truncate to multiple of 24
                    n_complete_days = fine_data.shape[0] // 24
                    fine_data = fine_data[:n_complete_days * 24]
                
                daily_data = fine_data.reshape(-1, 24).mean(dim=1)
                return daily_data
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                # Disaggregate daily to hourly (1:24 ratio)
                hourly_data = coarse_data.repeat_interleave(24)
                return hourly_data
            
            def resample(
                self,
                data: torch.Tensor,
                source_dt: float,
                target_dt: float,
                method: str = "mean",
                pattern: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                ratio = target_dt / source_dt
                
                if ratio > 1.0:
                    # Aggregation needed
                    window_size = int(ratio)
                    if data.shape[0] % window_size != 0:
                        # Truncate to complete windows
                        n_complete = (data.shape[0] // window_size) * window_size
                        data = data[:n_complete]
                    
                    reshaped = data.reshape(-1, window_size)
                    if method == "mean":
                        return reshaped.mean(dim=1)
                    elif method == "sum":
                        return reshaped.sum(dim=1)
                    elif method == "max":
                        return reshaped.max(dim=1)[0]
                    elif method == "min":
                        return reshaped.min(dim=1)[0]
                    else:
                        return reshaped.mean(dim=1)
                
                elif ratio < 1.0:
                    # Disaggregation needed
                    repeat_factor = int(1.0 / ratio)
                    if pattern is not None and len(pattern) == repeat_factor:
                        # Use provided pattern - assume it's already properly normalized
                        # Pattern should sum to 1 for proportional distribution
                        result = []
                        for value in data:
                            # Distribute each value according to the pattern
                            distributed = value * pattern
                            result.append(distributed)
                        return torch.cat(result)
                    else:
                        # Uniform disaggregation
                        return data.repeat_interleave(repeat_factor)
                else:
                    # Same resolution
                    return data.clone()
        
        resampler = TimeSeriesResampler()
        
        # Test hourly to daily aggregation
        hourly_data = torch.arange(48.0)  # 48 hours of data
        daily_data = resampler.resample(hourly_data, source_dt=3600, target_dt=86400, method="mean")
        assert daily_data.shape[0] == 2, "Daily aggregation shape incorrect"
        assert torch.allclose(daily_data[0], torch.tensor(11.5)), "Daily mean incorrect"  # Mean of 0-23
        
        # Test daily to hourly disaggregation
        daily_values = torch.tensor([10.0, 20.0])
        hourly_disagg = resampler.resample(daily_values, source_dt=86400, target_dt=3600)
        assert hourly_disagg.shape[0] == 48, "Hourly disaggregation shape incorrect"
        assert torch.allclose(hourly_disagg[:24], torch.tensor(10.0)), "Hourly disaggregation values incorrect"
        
        # Test with pattern-based disaggregation
        diurnal_pattern = torch.sin(torch.linspace(0, 2*3.14159, 24)) + 1.0  # Positive sine wave
        # Normalize pattern so that sum equals 1 (pattern represents proportional distribution)
        diurnal_pattern = diurnal_pattern / diurnal_pattern.sum()
        
        daily_val = torch.tensor([24.0])  # Daily total of 24
        hourly_with_pattern = resampler.resample(
            daily_val, source_dt=86400, target_dt=3600, pattern=diurnal_pattern
        )
        assert hourly_with_pattern.shape[0] == 24, "Pattern disaggregation shape incorrect"
        assert torch.allclose(hourly_with_pattern.sum(), daily_val), "Pattern disaggregation not conserving"
    
    def test_aggregation_methods(self):
        """Test different temporal aggregation methods."""
        
        class AggregationTester(TemporalScalingBase):
            def __init__(self):
                super().__init__(
                    source_scale="fine",
                    target_scale="coarse",
                    method="mean",
                    pattern_type="constant"
                )
            
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                return fine_data.mean()
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                return coarse_data.repeat(4)
            
            def resample(
                self,
                data: torch.Tensor,
                source_dt: float,
                target_dt: float,
                method: str = "mean",
                pattern: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                window_size = int(target_dt / source_dt)
                if window_size <= 1:
                    return data
                
                n_windows = data.shape[0] // window_size
                windowed = data[:n_windows * window_size].reshape(n_windows, window_size)
                
                if method == "sum":
                    return windowed.sum(dim=1)
                elif method == "mean":
                    return windowed.mean(dim=1)
                elif method == "max":
                    return windowed.max(dim=1)[0]
                elif method == "min":
                    return windowed.min(dim=1)[0]
                else:
                    return windowed.mean(dim=1)
        
        tester = AggregationTester()
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Test sum aggregation
        result_sum = tester.resample(data, source_dt=1.0, target_dt=4.0, method="sum")
        expected_sum = torch.tensor([10.0, 26.0])  # [1+2+3+4, 5+6+7+8]
        assert torch.allclose(result_sum, expected_sum), "Sum aggregation incorrect"
        
        # Test mean aggregation
        result_mean = tester.resample(data, source_dt=1.0, target_dt=4.0, method="mean")
        expected_mean = torch.tensor([2.5, 6.5])  # [mean(1,2,3,4), mean(5,6,7,8)]
        assert torch.allclose(result_mean, expected_mean), "Mean aggregation incorrect"
        
        # Test max aggregation
        result_max = tester.resample(data, source_dt=1.0, target_dt=4.0, method="max")
        expected_max = torch.tensor([4.0, 8.0])
        assert torch.allclose(result_max, expected_max), "Max aggregation incorrect"
        
        # Test min aggregation
        result_min = tester.resample(data, source_dt=1.0, target_dt=4.0, method="min")
        expected_min = torch.tensor([1.0, 5.0])
        assert torch.allclose(result_min, expected_min), "Min aggregation incorrect"


class TestHierarchicalScalingBase:
    """Test suite for HierarchicalScalingBase abstraction."""
    
    def test_hierarchy_aggregation(self):
        """Test hierarchical aggregation across levels."""
        
        class EcosystemHierarchy(HierarchicalScalingBase):
            def __init__(self):
                super().__init__(
                    source_scale="leaf",
                    target_scale="canopy",
                    hierarchy_levels=["leaf", "branch", "tree", "canopy"],
                    weight_type="area",
                    rule="conservative"
                )
            
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                # Simple aggregation by summing
                return fine_data.sum(dim=0, keepdim=True)
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                # Simple disaggregation by equal distribution
                n_fine = 4  # Assume 4 fine units per coarse
                return coarse_data.repeat(n_fine) / n_fine
            
            def aggregate_hierarchy(
                self,
                levels: List[str],
                data_by_level: Dict[str, torch.Tensor],
                weights: Optional[Dict[str, torch.Tensor]] = None,
                rule: str = "area_weighted"
            ) -> torch.Tensor:
                if not levels:
                    return torch.tensor(0.0)
                
                # Start with finest level
                current_data = data_by_level[levels[0]]
                
                # Aggregate through levels
                for i in range(len(levels) - 1):
                    if rule == "area_weighted":
                        # Area-weighted aggregation (sum for extensive quantities)
                        if weights and levels[i] in weights:
                            current_data = (current_data * weights[levels[i]]).sum()
                        else:
                            current_data = current_data.sum()
                    elif rule == "mass_weighted":
                        # Mass-weighted mean (for intensive quantities)
                        current_data = current_data.mean()
                    else:
                        # Conservative sum
                        current_data = current_data.sum()
                    
                    # Make it a tensor for next iteration
                    current_data = current_data.unsqueeze(0) if current_data.dim() == 0 else current_data
                
                return current_data
            
            def disaggregate_hierarchy(
                self,
                levels: List[str],
                data: torch.Tensor,
                target_level: str,
                weights: Optional[Dict[str, torch.Tensor]] = None,
                rule: str = "area_proportional"
            ) -> Dict[str, torch.Tensor]:
                result = {}
                current_data = data.clone()
                
                target_idx = levels.index(target_level) if target_level in levels else len(levels) - 1
                
                for i in range(target_idx + 1):
                    level = levels[i]
                    
                    if i == 0:
                        # Coarsest level
                        result[level] = current_data.clone()
                    else:
                        # Disaggregate from previous level
                        if rule == "area_proportional":
                            # Each unit gets proportional share
                            n_units = 2 ** i  # Exponential branching
                            current_data = current_data.repeat(n_units) / n_units
                        elif rule == "uniform":
                            # Equal distribution
                            n_units = 2 ** i
                            current_data = current_data.repeat(n_units) / n_units
                        else:
                            # Default: proportional
                            n_units = 2 ** i
                            current_data = current_data.repeat(n_units) / n_units
                        
                        result[level] = current_data.clone()
                
                return result
        
        hierarchy = EcosystemHierarchy()
        levels = ["leaf", "branch", "tree", "canopy"]
        
        # Test aggregation
        leaf_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        branch_data = torch.tensor([5.0, 15.0])
        tree_data = torch.tensor([10.0])
        canopy_data = torch.tensor([20.0])
        
        data_by_level = {
            "leaf": leaf_data,
            "branch": branch_data,
            "tree": tree_data,
            "canopy": canopy_data
        }
        
        # Test area-weighted aggregation
        aggregated = hierarchy.aggregate_hierarchy(
            levels=["leaf", "branch"], 
            data_by_level=data_by_level,
            rule="area_weighted"
        )
        expected = leaf_data.sum()  # Sum of leaf values
        assert torch.allclose(aggregated, expected), "Hierarchy aggregation failed"
        
        # Test mass-weighted aggregation
        aggregated_mean = hierarchy.aggregate_hierarchy(
            levels=["leaf", "branch"],
            data_by_level=data_by_level,
            rule="mass_weighted"
        )
        expected_mean = leaf_data.mean()
        assert torch.allclose(aggregated_mean, expected_mean), "Mass-weighted aggregation failed"
    
    def test_hierarchy_disaggregation(self):
        """Test hierarchical disaggregation."""
        
        class SimpleHierarchy(HierarchicalScalingBase):
            def __init__(self):
                super().__init__(
                    source_scale="ecosystem",
                    target_scale="plant",
                    rule="conservative",
                    hierarchy_levels=["ecosystem", "stand", "plant"],
                    weight_type="area"
                )
            
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                return fine_data.sum()
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                return coarse_data.repeat(4) / 4
            
            def aggregate_hierarchy(
                self,
                levels: List[str],
                data_by_level: Dict[str, torch.Tensor],
                weights: Optional[Dict[str, torch.Tensor]] = None,
                rule: str = "area_weighted"
            ) -> torch.Tensor:
                if not levels:
                    return torch.tensor(0.0)
                return data_by_level[levels[0]].sum()
            
            def disaggregate_hierarchy(
                self,
                levels: List[str],
                data: torch.Tensor,
                target_level: str,
                weights: Optional[Dict[str, torch.Tensor]] = None,
                rule: str = "area_proportional"
            ) -> Dict[str, torch.Tensor]:
                result = {}
                
                # Simple branching: each level has 2x more units than previous
                level_multipliers = {"ecosystem": 1, "stand": 2, "plant": 4}
                
                for level in levels:
                    if level == levels[0]:  # Coarsest level
                        result[level] = data.clone()
                    else:
                        # Disaggregate proportionally
                        n_units = level_multipliers.get(level, 1)
                        base_units = level_multipliers.get(levels[0], 1)
                        factor = n_units / base_units
                        result[level] = data.repeat(int(factor)) / factor
                
                return result
        
        hierarchy = SimpleHierarchy()
        ecosystem_data = torch.tensor([100.0])
        levels = ["ecosystem", "stand", "plant"]
        
        disaggregated = hierarchy.disaggregate_hierarchy(
            levels=levels,
            data=ecosystem_data,
            target_level="plant"
        )
        
        # Check conservation
        total_plant = disaggregated["plant"].sum()
        assert torch.allclose(total_plant, ecosystem_data), "Disaggregation not conserving"
        
        # Check structure
        assert disaggregated["ecosystem"].shape[0] == 1, "Ecosystem level shape incorrect"
        assert disaggregated["stand"].shape[0] == 2, "Stand level shape incorrect"  
        assert disaggregated["plant"].shape[0] == 4, "Plant level shape incorrect"
        
        # Check values
        expected_stand = ecosystem_data / 2
        expected_plant = ecosystem_data / 4
        assert torch.allclose(disaggregated["stand"], expected_stand), "Stand disaggregation incorrect"
        assert torch.allclose(disaggregated["plant"], expected_plant), "Plant disaggregation incorrect"
    
    def test_weighted_hierarchy_operations(self):
        """Test hierarchy operations with custom weights."""
        
        class WeightedHierarchy(HierarchicalScalingBase):
            def __init__(self):
                super().__init__(
                    source_scale="pixel",
                    target_scale="region",
                    rule="conservative",
                    hierarchy_levels=["pixel", "patch", "region"],
                    weight_type="area"
                )
            
            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                return fine_data.mean()
            
            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                return coarse_data.repeat(4)
            
            def aggregate_hierarchy(
                self,
                levels: List[str],
                data_by_level: Dict[str, torch.Tensor],
                weights: Optional[Dict[str, torch.Tensor]] = None,
                rule: str = "area_weighted"
            ) -> torch.Tensor:
                if not levels or len(levels) < 2:
                    return data_by_level[levels[0]] if levels else torch.tensor(0.0)
                
                # Use provided weights
                data = data_by_level[levels[0]]
                if weights and levels[0] in weights:
                    weight_tensor = weights[levels[0]]
                    # Weighted average
                    weighted_sum = (data * weight_tensor).sum()
                    weight_sum = weight_tensor.sum()
                    return weighted_sum / weight_sum if weight_sum > 0 else data.mean()
                else:
                    return data.mean()
            
            def disaggregate_hierarchy(
                self,
                levels: List[str],
                data: torch.Tensor,
                target_level: str,
                weights: Optional[Dict[str, torch.Tensor]] = None,
                rule: str = "area_proportional"
            ) -> Dict[str, torch.Tensor]:
                result = {levels[0]: data.clone()}
                
                if target_level in levels:
                    target_idx = levels.index(target_level)
                    for i in range(1, target_idx + 1):
                        level = levels[i]
                        
                        if weights and level in weights:
                            # Use provided weights for disaggregation
                            weight_tensor = weights[level]
                            n_units = weight_tensor.shape[0]
                            # Distribute proportionally by weights
                            total_data = result[levels[0]]
                            distributed = total_data * weight_tensor / weight_tensor.sum()
                            result[level] = distributed
                        else:
                            # Uniform distribution
                            n_units = 2 ** i  # Default: exponential branching
                            result[level] = data.repeat(n_units) / n_units
                
                return result
        
        hierarchy = WeightedHierarchy()
        
        # Test weighted aggregation
        pixel_data = torch.tensor([10.0, 20.0, 30.0, 40.0])
        pixel_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Different area weights
        
        data_by_level = {"pixel": pixel_data}
        weights = {"pixel": pixel_weights}
        
        aggregated = hierarchy.aggregate_hierarchy(
            levels=["pixel", "patch"],
            data_by_level=data_by_level,
            weights=weights,
            rule="area_weighted"
        )
        
        # Expected weighted average: (10*0.1 + 20*0.2 + 30*0.3 + 40*0.4) / (0.1+0.2+0.3+0.4) = 30
        expected_weighted = torch.tensor(30.0)
        assert torch.allclose(aggregated, expected_weighted), "Weighted aggregation failed"
        
        # Test weighted disaggregation
        region_data = torch.tensor([100.0])
        patch_weights = torch.tensor([0.3, 0.7])  # Unequal patch weights
        
        levels = ["region", "patch"]
        weights = {"patch": patch_weights}
        
        disaggregated = hierarchy.disaggregate_hierarchy(
            levels=levels,
            data=region_data,
            target_level="patch",
            weights=weights
        )
        
        expected_patch = torch.tensor([30.0, 70.0])  # 100 * [0.3, 0.7]
        assert torch.allclose(disaggregated["patch"], expected_patch), "Weighted disaggregation failed"
        assert torch.allclose(disaggregated["patch"].sum(), region_data), "Weighted disaggregation not conserving"


def test_abstraction_composition():
    """Test that abstractions can be composed correctly."""
    
    # Create a temperature field
    class SimpleField(PhysicalField):
        SPATIAL_DIMS = 2

        def __init__(self, value: float):
            super().__init__()
            self.value = value
            
        def get_value_at(self, coordinates: torch.Tensor) -> torch.Tensor:
            return torch.full((coordinates.shape[0],), self.value)
        
        def compute_gradient(self, coordinates: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(coordinates)
        
        def interpolate(self, source_coordinates: torch.Tensor,
                       target_coordinates: torch.Tensor) -> torch.Tensor:
            return self.get_value_at(target_coordinates)
        
        def update_field(self, time_step: float,
                        source_terms: Dict,
                        boundary_conditions: Optional[Dict] = None) -> None:
            pass
    
    # Create a biological process that depends on the field
    class TemperatureDependentProcess(BiologicalProcess):
        def __init__(self):
            super().__init__(process_name="temp_dependent")
            
        def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
            temp_field = state['temperature_field']
            
            coords = torch.tensor([[0.0, 0.0]])
            temp = temp_field.get_value_at(coords)
            
            # Q10 = 2 temperature response
            q10 = 2.0
            ref_temp = 298.15
            rate = torch.pow(q10, (temp - ref_temp) / 10.0)
            
            return {"rate": rate}
    
    # Create transport that moves products of the biological process
    class SimpleTransport(TransportMechanism):
        def __init__(self):
            super().__init__(transport_type="diffusion")
            
        def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
            source = state["source"]
            coords = torch.tensor([[0.0, 0.0]])
            flux = source.get_value_at(coords) * 0.1  # 10% transport rate
            return {"flux": flux}
    
    # Test composition
    temp_field = SimpleField(value=308.15)  # 35掳C
    
    process = TemperatureDependentProcess()
    rates = call_forward(process, {'temperature_field': temp_field})
    rate = rates["rate"]
    expected_rate = 2.0  # Q10=2 for 10K increase
    assert torch.allclose(rate, torch.tensor(expected_rate)), "Process-field composition failed"
    
    # Create product field based on process rate
    product_field = SimpleField(value=rate.item() * 100.0)  # Scale up for concentration
    sink_field = SimpleField(value=0.0)
    
    transport = SimpleTransport()
    flux_dict = call_transport_forward(transport, product_field, sink_field)
    flux = flux_dict["flux"]
    expected_flux = rate.item() * 100.0 * 0.1
    assert torch.allclose(flux, torch.tensor(expected_flux)), "Transport-process composition failed"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

