"""
Failure-mode contract tests for RichardsSoilMoisture flux core.
"""

from __future__ import annotations

import pytest
import torch

from relic.parameters import ParameterSpec, get_global_manager
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture


@pytest.fixture
def parameter_manager_snapshot():
    manager = get_global_manager()
    snapshot = manager.snapshot()
    yield manager
    manager.restore(snapshot)


def _set_obsolete_flag(manager, key: str, value: bool) -> None:
    """Put a removed flag into the manager the way a stale configuration would."""
    manager.register_parameter(ParameterSpec(
        key=key,
        value=value,
        units="dimensionless",
        dtype="bool",
        description="Removed Richards flag supplied by a stale configuration", bounds=(-float("inf"), float("inf")), category="model_parameter",
    ))


def test_richards_forward_rejects_out_of_range_moisture(
    parameter_manager_snapshot,
) -> None:
    """
    References:
    - Celia et al. (1990), Water Resources Research 26, 1483-1496.
    - Richards (1931), Journal of Applied Physics 1, 318-333.

    Mathematical expression:
    - q = -K(theta) * (dh/dz - 1)
    - theta must remain within the physical volumetric-water-content domain.

    Parameter meanings:
    - theta is volumetric water content [m3 m-3].
    - obsolete debug flags must not change the canonical fail-fast runtime.
    """
    manager = parameter_manager_snapshot
    _set_obsolete_flag(manager, "hydrosphere.soil_moisture.richards_flux_fail_fast", True)
    _set_obsolete_flag(
        manager,
        "hydrosphere.soil_moisture.richards_zero_flux_fallback_debug",
        False,
    )

    model = RichardsSoilMoisture(name="richards_fail_fast_test")
    assert not hasattr(model, "_compute_flux_core")
    assert not hasattr(model, "richards_flux_fail_fast")
    assert not hasattr(model, "richards_zero_flux_fallback_debug")
    with pytest.raises(ValueError, match="must be in range"):
        model.forward(
            {
                "soil_moisture": torch.full(
                    (model.soil_layers,), -0.1, dtype=torch.float32
                ),
                "infiltration_flux": torch.tensor(0.0, dtype=torch.float32),
                "evapotranspiration_flux": torch.tensor(
                    0.0, dtype=torch.float32
                ),
            },
            time=0.0,
            dt=60.0,
        )


def test_richards_forward_rejects_non_finite_moisture_without_fallback(
    parameter_manager_snapshot,
) -> None:
    """
    References:
    - Celia et al. (1990), Water Resources Research 26, 1483-1496.
    - Richards (1931), Journal of Applied Physics 1, 318-333.

    Mathematical expression:
    - Non-finite theta must reject the step instead of substituting q = 0.

    Parameter meanings:
    - theta is volumetric water content [m3 m-3].
    - obsolete debug flags are ignored by the canonical IPHS runtime.
    """
    manager = parameter_manager_snapshot
    _set_obsolete_flag(manager, "hydrosphere.soil_moisture.richards_flux_fail_fast", True)
    _set_obsolete_flag(
        manager,
        "hydrosphere.soil_moisture.richards_zero_flux_fallback_debug",
        True,
    )

    model = RichardsSoilMoisture(name="richards_debug_fallback_test")
    assert not hasattr(model, "_compute_flux_core")
    with pytest.raises(ValueError, match="must be finite"):
        model.forward(
            {
                "soil_moisture": torch.full(
                    (model.soil_layers,), float("nan"), dtype=torch.float32
                ),
                "infiltration_flux": torch.tensor(0.0, dtype=torch.float32),
                "evapotranspiration_flux": torch.tensor(
                    0.0, dtype=torch.float32
                ),
            },
            time=0.0,
            dt=60.0,
        )
