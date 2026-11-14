# -*- coding: utf-8 -*-
"""
Test eddy diffusion functor reads CFL safety factor from ParameterManager.

- References: CFL condition; LeVeque (2007)
- Expression: dt_cfl ~ dz^2/(2*K) scaled by safety factor
- Parameters: cfl_safety_factor (dimensionless)
"""

from relic.parameters import get_global_manager
import pytest


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


def test_cfl_factor_loaded_from_parameters():
    pm = get_global_manager()
    pm.set_value("transport.numerical.cfl_safety_factor", 0.75)

    from relic.scientific.atmosphere.transport.eddy_diffusion_blm import EddyDiffusionBLMFunctor
    functor = EddyDiffusionBLMFunctor(name="eddydiff_test")
    snapshot = functor.get_transport_metadata()["parameter_snapshot"]
    assert abs(snapshot["cfl_factor"] - 0.75) < 1e-8

