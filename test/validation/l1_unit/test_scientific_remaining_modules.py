"""
L1 validation tests for remaining scientific modules (isotopes, anthroposphere
management, and generic transport functors).
"""

from test.utils.functor_minimal_construction import construct_minimal
import torch
import pytest

import relic.scientific.isotopes.functor as isotope_functor_module
from relic.scientific.isotopes.functor import (
    IsotopeHeartbeatError,
    IsotopeHeartbeatFunctor,
)
from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator
from relic.scientific.anthroposphere.management.fire_management import FireManagement
from relic.scientific.anthroposphere.management.harvest import HarvestManagement
from relic.scientific.anthroposphere.land_use.transitions import LandUseTransitions
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.core.functor import FunctorBase
from relic.core.state import UpdateItem, UpdateMode


class TestIsotopesFunctors:
    """Fail-closed checks for the runtime-unbound isotope functor."""

    def test_isotope_inheritance(self):
        assert issubclass(IsotopeHeartbeatFunctor, FunctorBase)

    def test_isotope_constructor_rejects_before_bootstrap_side_effects(
        self,
        monkeypatch,
    ):
        effects = []
        monkeypatch.setattr(
            isotope_functor_module,
            "get_default_isotope_config",
            lambda: effects.append("config"),
            raising=False,
        )

        with pytest.raises(IsotopeHeartbeatError, match=r"\[N\].*runtime_unbound"):
            IsotopeHeartbeatFunctor()

        assert effects == []

    def test_isotope_forward_is_fail_closed_even_for_unsafe_instance(self):
        isotope = object.__new__(IsotopeHeartbeatFunctor)
        with pytest.raises(IsotopeHeartbeatError, match=r"\[N\].*runtime_unbound"):
            isotope.forward({}, time=43200.0, dt=86400.0)

    def test_isotope_module_has_no_service_or_writer_runtime_surface(self):
        forbidden = {
            "create_isotope_heartbeat_functor",
            "ServiceConnectionError",
            "OutputCoordinationError",
        }
        assert all(
            not hasattr(isotope_functor_module, name)
            for name in forbidden
        )


class TestAnthroposphereManagement:
    """L1 checks for anthroposphere management functors."""

    @pytest.mark.parametrize(
        "cls",
        [ManagementCoordinator, FireManagement, HarvestManagement, LandUseTransitions],
    )
    def test_management_inheritance(self, cls):
        instance = construct_minimal(f"{cls.__module__}:{cls.__name__}")
        assert isinstance(instance, FunctorBase)
        assert hasattr(instance, "forward")


class TestTransportFunctors:
    """L1 checks for basic transport functors."""

    @pytest.mark.parametrize(
        "cls", [LinearAdvection, FickianDiffusion, AdvectionDiffusion]
    )
    def test_transport_forward(self, cls):
        model = construct_minimal(f"{cls.__module__}:{cls.__name__}")
        concentration = torch.tensor([1.0, 2.0, 3.0])
        state = {"tracer_concentration": concentration}
        if cls in (LinearAdvection, AdvectionDiffusion):
            state["velocity_field"] = torch.zeros_like(concentration)
        time = torch.tensor(0.0)
        dt = torch.tensor(0.01)
        result = model.forward(state, time=time, dt=dt)
        assert isinstance(result, dict)
        tracer_update = result["tracer_concentration"]
        assert isinstance(tracer_update, UpdateItem)
        assert tracer_update.mode == UpdateMode.ACCUMULATE
        assert tracer_update.per_timestep is True
        assert isinstance(tracer_update.value, torch.Tensor)
        for key, value in result.items():
            if key == "tracer_concentration":
                continue
            if key.startswith("diagnostics."):
                assert isinstance(value, torch.Tensor)
                continue
            assert key.endswith("_boundary_flux")
            assert isinstance(value, UpdateItem)
            assert value.mode == UpdateMode.REPLACE
            assert value.per_timestep is False
            assert isinstance(value.value, torch.Tensor)
