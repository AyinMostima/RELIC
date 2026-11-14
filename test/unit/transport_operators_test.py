"""
L1 Unit Tests for Transport Operators Integration

This test suite validates that transport modules correctly use the public
operator library interface through the OperatorFactory pattern.

Test Coverage:
- Advection module uses OperatorFactory
- Diffusion module uses OperatorFactory
- Advection-diffusion module uses OperatorFactory
- Operator registration and retrieval
- Mathematical correctness of operations

Mathematical Expressions:
    Advection: 鈭侰/鈭倀 + v路鈭嘋 = 0
    Diffusion: 鈭侰/鈭倀 = D鈭嚶睠
    Advection-Diffusion: 鈭侰/鈭倀 + v路鈭嘋 = D鈭嚶睠

References:
    - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of
      Time-Dependent Advection-Diffusion-Reaction Equations. Springer.
    - LeVeque, R.J. (2002). Finite Volume Methods for Hyperbolic Problems.
      Cambridge University Press.
"""

import pytest
import torch

import relic.computation.operators as computation_operators
import relic.computation.operators.functional as functional_operators
import relic.scientific.transport as scientific_transport
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.computation.operators import OperatorFactory


def _tracer_increment(result):
    """Return the canonical integrated tracer increment."""
    update = result["tracer_concentration"]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return update.value


# (scheme, time integration method) pairs of the AdvectionOperator: the
# allowed_values of computation.operators.transport.scheme times those of
# transport.numerical.time_integration_method. None marks a pair that advances
# a step; a string is the reason the pair is rejected at construction
# (central FTCS is unconditionally unstable for pure advection; the fully
# discrete schemes already contain their own time update).
_ADVECTION_SCHEME_METHOD_PAIRS = {
    ("upwind", "euler"): None,
    ("upwind", "implicit_midpoint"): None,
    ("upwind", "crank_nicolson"): None,
    ("lax_wendroff", "euler"): None,
    ("lax_wendroff", "implicit_midpoint"): "fully discrete",
    ("lax_wendroff", "crank_nicolson"): "fully discrete",
    ("superbee", "euler"): None,
    ("superbee", "implicit_midpoint"): "fully discrete",
    ("superbee", "crank_nicolson"): "fully discrete",
    ("van_leer", "euler"): None,
    ("van_leer", "implicit_midpoint"): "fully discrete",
    ("van_leer", "crank_nicolson"): "fully discrete",
    ("central", "euler"): "unconditionally unstable",
    ("central", "implicit_midpoint"): None,
    ("central", "crank_nicolson"): None,
}
_ACCEPTED_ADVECTION_PAIRS = sorted(
    pair for pair, reason in _ADVECTION_SCHEME_METHOD_PAIRS.items() if reason is None
)
_REJECTED_ADVECTION_PAIRS = sorted(
    (scheme, method, reason)
    for (scheme, method), reason in _ADVECTION_SCHEME_METHOD_PAIRS.items()
    if reason is not None
)


def _advection_operator(scheme, method=None):
    config = {
        "spatial_dims": 1,
        "scheme": scheme,
        "boundary_method": "periodic",
        "check_cfl": True,
    }
    if method is not None:
        config["time_integration_method"] = method
    return OperatorFactory().get_operator(
        "transport.advection", config=config, dtype=torch.float64
    )


def test_advection_pair_table_covers_the_yaml_choices():
    from itertools import product

    from relic.parameters import get_global_manager

    manager = get_global_manager()
    schemes = manager.get_parameter("computation.operators.transport.scheme").allowed_values
    methods = manager.get_parameter(
        "transport.numerical.time_integration_method"
    ).allowed_values
    assert set(product(schemes, methods)) == set(_ADVECTION_SCHEME_METHOD_PAIRS)


@pytest.mark.parametrize(("scheme", "method"), _ACCEPTED_ADVECTION_PAIRS)
def test_accepted_advection_pair_advances_with_the_configured_method(scheme, method):
    """The operator advances one step with its configured method, like the kernel."""
    cells = torch.arange(32, dtype=torch.float64)
    concentration = torch.exp(-(((cells - 16.0) / 3.0) ** 2)).reshape(1, -1)
    velocity = torch.ones_like(concentration)
    operator = _advection_operator(scheme, method)

    actual = operator.apply(concentration, velocity, 1.0, 0.1)
    expected = functional_operators.apply_advection(
        concentration, velocity, 1.0, 1, scheme, "periodic", 0.1, True,
        time_integration_method=method,
    )

    assert operator.time_integration_method == method
    torch.testing.assert_close(
        actual["concentration"], expected["concentration"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        torch.sum(actual["concentration"]), torch.sum(concentration), rtol=0.0, atol=1.0e-12
    )


@pytest.mark.parametrize(("scheme", "method", "reason"), _REJECTED_ADVECTION_PAIRS)
def test_rejected_advection_pair_fails_at_construction(scheme, method, reason):
    with pytest.raises(ValueError, match=reason):
        _advection_operator(scheme, method)


def test_advection_operator_method_defaults_to_the_transport_database_value():
    from relic.parameters import get_global_manager

    configured = get_global_manager().get(
        "transport.numerical.time_integration_method", required=True
    )
    assert _advection_operator("upwind").time_integration_method == configured


# Splitting rule of the AdvectionDiffusionOperator: combined splitting is one
# unsplit explicit Euler step of the summed rates, so it needs euler. The
# advection sub-step obeys the (scheme, method) table above.
_SPLITTING_METHOD_REASONS = {
    ("combined", "implicit_midpoint"): "combined splitting requires euler",
    ("combined", "crank_nicolson"): "combined splitting requires euler",
}
_SPLITTINGS = ("sequential", "symmetric", "combined")
_ADVECTION_DIFFUSION_CASES = sorted(
    (scheme, method, splitting, advection_reason or _SPLITTING_METHOD_REASONS.get((splitting, method)))
    for (scheme, method), advection_reason in _ADVECTION_SCHEME_METHOD_PAIRS.items()
    for splitting in _SPLITTINGS
)


def _advection_diffusion_operator(scheme, splitting, method=None):
    config = {
        "diffusivity": 0.005,
        "spatial_dims": 1,
        "splitting_method": splitting,
        "advection_scheme": scheme,
        "boundary_method": "periodic",
        "check_stability": True,
        "enforce_non_negative": False,
    }
    if method is not None:
        config["time_integration_method"] = method
    return OperatorFactory().get_operator(
        "transport.advection_diffusion", config=config, dtype=torch.float64
    )


def test_advection_diffusion_case_table_covers_the_yaml_choices():
    from itertools import product

    from relic.parameters import get_global_manager

    manager = get_global_manager()
    schemes = manager.get_parameter(
        "computation.operators.transport.advection_scheme"
    ).allowed_values
    methods = manager.get_parameter(
        "transport.numerical.time_integration_method"
    ).allowed_values
    splittings = manager.get_parameter(
        "computation.operators.transport.splitting_method"
    ).allowed_values
    assert set(product(schemes, methods, splittings)) == {
        case[:3] for case in _ADVECTION_DIFFUSION_CASES
    }


@pytest.mark.parametrize(
    ("scheme", "method", "splitting"),
    [case[:3] for case in _ADVECTION_DIFFUSION_CASES if case[3] is None],
)
def test_accepted_advection_diffusion_case_advances_with_the_configured_method(
    scheme, method, splitting
):
    cells = torch.arange(32, dtype=torch.float64)
    concentration = torch.exp(-(((cells - 16.0) / 3.0) ** 2)).reshape(1, -1)
    velocity = torch.full_like(concentration, 0.5)
    operator = _advection_diffusion_operator(scheme, splitting, method)

    actual = operator.apply(concentration, velocity, 1.0, 0.1)
    expected = functional_operators.advance_advection_diffusion(
        concentration, velocity, 0.005, 1.0, 0.1, 1, splitting, scheme, "periodic",
        True, False, time_integration_method=method,
    )

    assert operator.time_integration_method == method
    torch.testing.assert_close(
        actual["concentration"], expected["concentration"], rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    ("scheme", "method", "splitting", "reason"),
    [case for case in _ADVECTION_DIFFUSION_CASES if case[3] is not None],
)
def test_rejected_advection_diffusion_case_fails_at_construction(
    scheme, method, splitting, reason
):
    with pytest.raises(ValueError, match=reason):
        _advection_diffusion_operator(scheme, splitting, method)


def test_advection_diffusion_method_defaults_to_the_transport_database_value():
    from relic.parameters import get_global_manager

    configured = get_global_manager().get(
        "transport.numerical.time_integration_method", required=True
    )
    operator = _advection_diffusion_operator("upwind", "sequential")
    assert operator.time_integration_method == configured


class TestTransportOperatorIntegration:
    """Test suite for transport operators integration with public operator library."""

    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing."""
        concentration = torch.ones((1, 10, 10), dtype=torch.float32)
        concentration[:, 4:6, 4:6] = 2.0
        velocity_x = torch.full_like(concentration, 0.1)
        velocity_y = torch.full_like(concentration, 0.05)
        return {
            "tracer_concentration": concentration,
            "source_term": torch.zeros_like(concentration),
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "velocity_field": torch.stack((velocity_x, velocity_y), dim=-1),
            "diffusivity": torch.full_like(concentration, 0.01),
        }

    @pytest.fixture
    def operator_factory(self):
        """Create operator factory instance."""
        return OperatorFactory()

    def test_advection_uses_operator_factory(self, sample_state):
        """Test that Advection module uses OperatorFactory."""
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(1.0, 1.0),
            boundary_method="periodic",
            non_negative=False,
            track_conservation=False,
        )

        # Verify operator factory is initialized
        assert hasattr(advection, 'operator_factory')
        assert isinstance(advection.operator_factory, OperatorFactory)

        # Test operator retrieval
        advection_op = advection.operator_factory.get_operator("transport.advection")
        assert advection_op is not None

        result = advection.forward(
            {
                "tracer_concentration": sample_state["tracer_concentration"],
                "source_term": sample_state["source_term"],
                "velocity_field": sample_state["velocity_field"],
            },
            time=0.0,
            dt=0.1,
        )
        result_tensor = _tracer_increment(result)

        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.shape == sample_state["tracer_concentration"].shape
        assert result_tensor.dtype == torch.float32
        assert torch.isfinite(result_tensor).all()

    def test_advection_operator_forwards_open_boundary_state(self):
        """The registry facade must preserve the functional boundary ledger."""
        concentration = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        velocity = torch.ones_like(concentration)
        boundary_state = torch.zeros_like(concentration)
        operator = OperatorFactory().get_operator(
            "transport.advection",
            config={
                "spatial_dims": 1,
                "scheme": "upwind",
                "boundary_method": "open",
                "check_cfl": True,
            },
            dtype=torch.float32,
        )

        expected = functional_operators.apply_advection(
            concentration=concentration,
            velocity=velocity,
            grid_spacing=1.0,
            spatial_dims=1,
            scheme="upwind",
            boundary_method="open",
            boundary_state=boundary_state,
            time_step=0.1,
            check_cfl=True,
        )
        actual = operator.apply(
            concentration=concentration,
            velocity=velocity,
            grid_spacing=1.0,
            boundary_method="open",
            boundary_state=boundary_state,
            time_step=0.1,
            check_cfl=True,
        )

        assert set(actual["boundary_fluxes"]) == {"x_min", "x_max"}
        for face, expected_flux in expected["boundary_fluxes"].items():
            torch.testing.assert_close(
                actual["boundary_fluxes"][face],
                expected_flux,
            )

    def test_diffusion_uses_operator_factory(self, sample_state):
        """Test that Diffusion module uses OperatorFactory."""
        diffusion = FickianDiffusion(
            spatial_dims=2,
            diffusivity=sample_state["diffusivity"], diffusivity_layout="scalar_field",
            grid_spacing=(1.0, 1.0),
            non_negative=False,
            track_conservation=False,
        )

        # Verify operator factory is initialized
        assert hasattr(diffusion, 'operator_factory')
        assert isinstance(diffusion.operator_factory, OperatorFactory)

        # Test operator retrieval
        diffusion_op = diffusion.operator_factory.get_operator("transport.diffusion")
        assert diffusion_op is not None

        result = diffusion.forward(
            {
                "tracer_concentration": sample_state["tracer_concentration"],
                "source_term": sample_state["source_term"],
            },
            time=0.0,
            dt=0.1,
        )
        result_tensor = _tracer_increment(result)

        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.shape == sample_state["tracer_concentration"].shape
        assert result_tensor.dtype == torch.float32
        assert torch.isfinite(result_tensor).all()

    def test_diffusion_operator_forwards_dirichlet_boundary_ledger(self):
        """The registry facade must preserve the canonical diffusion port ledger."""
        concentration = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        boundary_state = torch.zeros_like(concentration)
        source_term = torch.full_like(concentration, 0.05)
        operator = OperatorFactory().get_operator(
            "transport.diffusion",
            config={
                "diffusivity": 0.01,
                "spatial_dims": 1,
                "boundary_method": "dirichlet",
                "check_stability": True,
                "enforce_non_negative": True,
            },
            dtype=torch.float32,
        )

        expected = functional_operators.apply_diffusion(
            concentration=concentration,
            diffusivity=0.01,
            grid_spacing=1.0,
            spatial_dims=1,
            boundary_method="dirichlet",
            boundary_state=boundary_state,
            source_term=source_term,
            time_step=0.1,
            run_conservation_check=True,
            time_integration_method="euler", check_stability=True, enforce_non_negative=True,
        )
        actual = operator.apply(
            concentration=concentration,
            grid_spacing=1.0,
            boundary_state=boundary_state,
            source_term=source_term,
            time_step=0.1,
            run_conservation_check=True,
            time_integration_method="euler",
        )

        assert actual["concentration"].dtype == concentration.dtype
        torch.testing.assert_close(actual["concentration"], expected["concentration"])
        torch.testing.assert_close(
            actual["conservation_error"],
            expected["conservation_error"],
        )
        assert set(actual["boundary_fluxes"]) == {"x_min", "x_max"}
        for face, expected_flux in expected["boundary_fluxes"].items():
            torch.testing.assert_close(actual["boundary_fluxes"][face], expected_flux)

    def test_diffusion_operator_preserves_dtype_and_autograd(self):
        """Runtime tensors own dtype and gradient history at the operator boundary."""
        concentration = torch.tensor(
            [[1.0, 2.0, 4.0, 8.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        operator = OperatorFactory().get_operator(
            "transport.diffusion",
            config={
                "diffusivity": 0.01,
                "spatial_dims": 1,
                "boundary_method": "periodic",
            },
            dtype=torch.float32,
        )

        result = operator.apply(
            concentration=concentration,
            grid_spacing=1.0,
            time_step=0.1,
            run_conservation_check=False,
        )
        assert result["concentration"].dtype == concentration.dtype
        result["concentration"].sum().backward()
        assert concentration.grad is not None
        assert concentration.grad.dtype == concentration.dtype

    def test_advection_operator_preserves_dtype_and_autograd(self):
        """The advection facade must not narrow a float64 state to its own dtype."""
        concentration = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64, requires_grad=True,
        )
        velocity = torch.full_like(concentration, 0.2)
        operator = OperatorFactory().get_operator(
            "transport.advection",
            config={
                "spatial_dims": 1,
                "scheme": "upwind",
                "boundary_method": "periodic",
                "check_cfl": True,
            },
            dtype=torch.float32,
        )
        result = operator.apply(
            concentration=concentration,
            velocity=velocity,
            grid_spacing=1.0,
            time_step=0.5,
        )
        # References: LeVeque (2002), periodic first-order upwind finite volumes.
        # Mathematical expression: c_new = c - (v*dt/dx)*(c - roll(c, 1)) for v > 0.
        # Parameter meanings: v = 0.2 m s-1, dt = 0.5 s, dx = 1 m, CFL = 0.1.
        # Units: c [kg m-3]; v [m s-1]; dt [s]; dx [m].
        # Domain/constraints: periodic grid; float64 arithmetic with no float32 round trip.
        # Sign convention: positive velocity transports toward increasing index.
        detached = concentration.detach()
        expected = detached - 0.1 * (detached - torch.roll(detached, 1, dims=-1))
        assert result["concentration"].dtype == torch.float64
        torch.testing.assert_close(result["concentration"], expected, rtol=1.0e-12, atol=0.0)
        gradient = torch.autograd.grad(result["concentration"].sum(), concentration)[0]
        assert gradient.dtype == torch.float64
        torch.testing.assert_close(gradient, torch.ones_like(detached), rtol=1.0e-12, atol=0.0)

    def test_advection_operator_aligns_velocity_to_the_state_dtype(self):
        """Velocity dtype follows the state; the state dtype is never promoted or narrowed."""
        operator = OperatorFactory().get_operator(
            "transport.advection",
            config={"spatial_dims": 1, "scheme": "upwind", "boundary_method": "periodic"},
            dtype=torch.float32,
        )
        state64 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        result64 = operator.apply(
            concentration=state64,
            velocity=torch.full((1, 4), 0.2, dtype=torch.float32),
            grid_spacing=1.0,
            time_step=0.5,
        )
        assert result64["concentration"].dtype == torch.float64
        # The float32 velocity is aligned to float64 exactly, so the oracle uses its widened value.
        courant = float(torch.tensor(0.2, dtype=torch.float32).double()) * 0.5 / 1.0
        expected64 = state64 - courant * (state64 - torch.roll(state64, 1, dims=-1))
        torch.testing.assert_close(result64["concentration"], expected64, rtol=1.0e-12, atol=0.0)

        state32 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        result32 = operator.apply(
            concentration=state32,
            velocity=torch.full((1, 4), 0.2, dtype=torch.float64),
            grid_spacing=1.0,
            time_step=0.5,
        )
        assert result32["concentration"].dtype == torch.float32

        with pytest.raises(TypeError, match="real floating-point"):
            operator.apply(
                concentration=state64,
                velocity=torch.full((1, 4), 0.2, dtype=torch.complex64),
                grid_spacing=1.0,
                time_step=0.5,
            )

    def test_advection_diffusion_operator_preserves_dtype_and_autograd(self):
        """The combined facade keeps float64 state, aligns tensor K, and keeps the graph."""
        concentration = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64, requires_grad=True,
        )
        velocity = torch.full((1, 4), 0.2, dtype=torch.float32)
        diffusivity = torch.full((1, 4), 0.05, dtype=torch.float32)
        operator = OperatorFactory().get_operator(
            "transport.advection_diffusion",
            config={
                "spatial_dims": 1,
                "splitting_method": "combined",
                "advection_scheme": "upwind",
                "boundary_method": "periodic",
                "check_stability": True,
                "enforce_non_negative": True,
            },
            dtype=torch.float32,
        )
        result = operator.apply(
            concentration=concentration,
            velocity=velocity,
            diffusivity=diffusivity,
            diffusivity_layout="scalar_field",
            grid_spacing=1.0,
            time_step=0.5,
            spatial_dims=1,
        )
        # References: Hundsdorfer and Verwer (2003), unsplit explicit transport.
        # Mathematical expression: dc = dt*(-v*(c-c_left)/dx + K*(c_right-2c+c_left)/dx**2).
        # Parameter meanings: v = 0.2 m s-1, K = 0.05 m2 s-1, dt = 0.5 s, dx = 1 m.
        # Units: c [kg m-3]; K [m2 s-1]; v [m s-1]; dt [s]; dx [m].
        # Domain/constraints: periodic grid; CFL 0.1 + 2*0.025 = 0.15 < 1; positive result.
        # Sign convention: flux divergence removes storage; periodic mass is conserved.
        detached = concentration.detach()
        k_value = float(diffusivity.double()[0, 0])
        v_value = float(velocity.double()[0, 0])
        expected = detached + 0.5 * (
            -v_value * (detached - torch.roll(detached, 1, dims=-1))
            + k_value * (torch.roll(detached, -1, dims=-1) - 2.0 * detached + torch.roll(detached, 1, dims=-1))
        )
        assert result["concentration"].dtype == torch.float64
        torch.testing.assert_close(result["concentration"], expected, rtol=1.0e-12, atol=0.0)
        assert float(result["conservation_error"]) < 1.0e-12
        gradient = torch.autograd.grad(result["concentration"].sum(), concentration)[0]
        assert gradient.dtype == torch.float64
        torch.testing.assert_close(gradient, torch.ones_like(detached), rtol=1.0e-12, atol=0.0)

    def test_advection_diffusion_operator_forwards_boundary_states(self):
        """The registry facade must preserve both finite-volume boundary ledgers."""
        concentration = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        velocity = torch.zeros_like(concentration)
        boundary_state = torch.zeros_like(concentration)
        operator = OperatorFactory().get_operator(
            "transport.advection_diffusion",
            config={
                "diffusivity": 0.01,
                "spatial_dims": 1,
                "splitting_method": "combined",
                "boundary_method": "dirichlet",
                "check_stability": True,
                "enforce_non_negative": True,
            },
            dtype=torch.float32,
        )

        expected = functional_operators.advance_advection_diffusion(
            concentration=concentration,
            velocity=velocity,
            diffusivity=0.01,
            grid_spacing=1.0,
            time_step=0.1,
            spatial_dims=1,
            splitting_method="combined",
            boundary_method="dirichlet",
            advection_boundary_state=boundary_state,
            diffusion_boundary_state=boundary_state, advection_scheme="upwind", check_stability=True, enforce_non_negative=True,
        )
        actual = operator.apply(
            concentration=concentration,
            velocity=velocity,
            grid_spacing=1.0,
            time_step=0.1,
            boundary_method="dirichlet",
            advection_boundary_state=boundary_state,
            diffusion_boundary_state=boundary_state,
        )

        torch.testing.assert_close(actual["concentration"], expected["concentration"])
        assert set(actual["boundary_fluxes"]) == {"x_min", "x_max"}
        for face, expected_flux in expected["boundary_fluxes"].items():
            torch.testing.assert_close(actual["boundary_fluxes"][face], expected_flux)

    def test_advection_diffusion_uses_operator_factory(self, sample_state):
        """Test that AdvectionDiffusion module uses OperatorFactory."""
        advection_diffusion = AdvectionDiffusion(
            spatial_dims=2,
            diffusivity=0.01,
            grid_spacing=(1.0, 1.0),
            splitting_method="combined",
            boundary_method="periodic",
            non_negative=False,
        )
        advection_diffusion.track_conservation = False

        # Verify operator factory is initialized
        assert hasattr(advection_diffusion, 'operator_factory')
        assert isinstance(advection_diffusion.operator_factory, OperatorFactory)

        # Test operator retrieval
        advection_op = advection_diffusion.operator_factory.get_operator("transport.advection")
        diffusion_op = advection_diffusion.operator_factory.get_operator("transport.diffusion")
        combined_op = advection_diffusion.operator_factory.get_operator(
            "transport.advection_diffusion"
        )
        assert advection_op is not None
        assert diffusion_op is not None
        assert combined_op is not None

        result = advection_diffusion.forward(
            {
                "tracer_concentration": sample_state["tracer_concentration"],
                "source_term": sample_state["source_term"],
                "velocity_field": sample_state["velocity_field"],
            },
            time=0.0,
            dt=0.1,
        )
        result_tensor = _tracer_increment(result)

        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.shape == sample_state["tracer_concentration"].shape
        assert result_tensor.dtype == torch.float32
        assert torch.isfinite(result_tensor).all()

    def test_operator_factory_registration(self):
        """Test that required operators are registered in factory."""
        available_operators = set(OperatorFactory.list_available_operators())
        expected = {
            "transport.advection",
            "transport.diffusion",
            "transport.advection_diffusion",
        }
        assert expected <= available_operators
        factory = OperatorFactory()
        for operator_name in sorted(expected):
            assert factory.get_operator(operator_name) is not None

    def test_no_parallel_transport_operator_facade(self):
        """The registry-backed transport operators are the sole public classes."""
        assert not hasattr(functional_operators, "TransportOperator")
        assert not hasattr(functional_operators, "compute_u_star")
        assert not hasattr(functional_operators, "compute_L")

    def test_functional_module_has_no_parameter_accessor_wrapper(self):
        """Non-Functor infrastructure uses the global manager directly."""
        assert not hasattr(functional_operators, "get_static_parameter_value")

    def test_scientific_transport_does_not_reexport_ledger_losing_diffusion(self):
        """Scientific packages must not expose a diffusion path without its ledger."""
        assert not hasattr(scientific_transport, "apply_diffusion_conservative")

    def test_operator_package_does_not_reexport_ledger_losing_diffusion(self):
        """The computation package must expose only the ledger-aware diffusion API."""
        assert not hasattr(computation_operators, "apply_diffusion_conservative")

    def test_functional_module_has_no_ledger_losing_diffusion_path(self):
        """The functional module must retain boundary ledgers for diffusion."""
        assert not hasattr(functional_operators, "apply_diffusion_conservative")

    def test_mathematical_conservation(self, sample_state):
        """Test that transport operations preserve mathematical properties."""
        positive_conc = sample_state["tracer_concentration"]
        diffusion = FickianDiffusion(
            spatial_dims=2,
            diffusivity=sample_state["diffusivity"], diffusivity_layout="scalar_field",
            grid_spacing=(1.0, 1.0),
            non_negative=False,
            track_conservation=False,
        )

        result = diffusion.forward(
            {
                "tracer_concentration": positive_conc,
                "source_term": torch.zeros_like(positive_conc),
            },
            time=0.0,
            dt=0.01,
        )
        result_tensor = _tracer_increment(result)

        new_conc = positive_conc + result_tensor
        assert torch.all(new_conc >= 0), "Diffusion should preserve positivity"
        mass_roundoff_bound = float(
            torch.finfo(positive_conc.dtype).eps
            * positive_conc.numel()
            * torch.max(torch.abs(positive_conc))
        )
        torch.testing.assert_close(
            torch.sum(new_conc),
            torch.sum(positive_conc),
            rtol=0.0,
            atol=mass_roundoff_bound,
        )

    def test_operator_error_handling(self, sample_state):
        """Test error handling in transport operators."""
        wrong_velocity = torch.full((1, 5, 5), 0.1, dtype=torch.float32)
        advection = LinearAdvection(
            spatial_dims=2,
            velocity_field=(wrong_velocity, sample_state["velocity_y"]),
            grid_spacing=(1.0, 1.0),
            boundary_method="periodic",
            non_negative=False,
            track_conservation=False,
        )

        with pytest.raises(ValueError, match="shape|broadcast"):
            advection.forward(
                {
                    "tracer_concentration": sample_state["tracer_concentration"],
                    "source_term": sample_state["source_term"],
                },
                time=0.0,
                dt=0.1,
            )

    def test_tensor_properties_preserved(self, sample_state):
        """Test that tensor properties (device, dtype) are preserved."""
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(1.0, 1.0),
            boundary_method="periodic",
            non_negative=False,
            track_conservation=False,
        )
        result_cpu = advection.forward(
            {
                "tracer_concentration": sample_state["tracer_concentration"],
                "source_term": sample_state["source_term"],
                "velocity_field": sample_state["velocity_field"],
            },
            time=0.0,
            dt=0.1,
        )
        result_tensor = _tracer_increment(result_cpu)

        assert result_tensor.device == sample_state["tracer_concentration"].device
        assert result_tensor.dtype == sample_state["tracer_concentration"].dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

