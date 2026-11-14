"""L1 tests for canonical upwind boundary validation."""

import pytest
import torch

from relic.computation.operators.functional import apply_advection


def test_upwind_advection_rejects_unknown_boundary_method():
    """Unknown boundary methods must raise an explicit error."""
    concentration = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    velocity = torch.tensor(1.0, dtype=torch.float32)

    with pytest.raises(ValueError, match="Unknown advection boundary method"):
        apply_advection(
            concentration,
            velocity,
            1.0,
            boundary_method="invalid_mode",
            check_cfl=False, spatial_dims=1, scheme="upwind", time_step=None,
        )


def test_upwind_advection_rejects_removed_reflect_alias():
    """The removed reflect alias must not re-enter the production API."""
    concentration = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    velocity = torch.tensor(-1.0, dtype=torch.float32)

    with pytest.raises(ValueError, match="Unknown advection boundary method"):
        apply_advection(
            concentration,
            velocity,
            1.0,
            boundary_method="reflect",
            check_cfl=False, spatial_dims=1, scheme="upwind", time_step=None,
        )
