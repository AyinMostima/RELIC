import pytest
import torch

from relic.coupling.scales.temporal import (
    ConservationLevel,
    DisaggregationConfig,
    DisaggregationMethod,
    ScalingDirection,
    TemporalConfig,
    TemporalDisaggregator,
)
from relic.coupling.scales.types import SpatialQuantityBasis
from relic.parameters import get_global_manager


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def test_disaggregate_uniform_respects_conservation_tolerance(monkeypatch):
    param_manager = get_global_manager()
    tolerance = _to_float(param_manager.get("validation.conservation.conservation_tolerance"))

    config = TemporalConfig(
        source_frequency="daily",
        target_frequency="hourly",
        direction=ScalingDirection.DOWNSCALE,
        spatial_basis=SpatialQuantityBasis.CELL_TOTAL,
        conservation_level=ConservationLevel.STRICT,
        device=torch.device("cpu"),
    )
    disagg = TemporalDisaggregator(
        config,
        DisaggregationConfig(method=DisaggregationMethod.UNIFORM, time_dim=0),
    )
    q_ref_parameter = "tests.temporal.integration_q_ref"
    original_lookup = disagg.get_parameter_value

    def lookup(name, **kwargs):
        if name == q_ref_parameter:
            return 1.0
        return original_lookup(name, **kwargs)

    monkeypatch.setattr(disagg, "get_parameter_value", lookup)

    coarse = torch.tensor([8.0, 4.0], dtype=torch.float64)
    coarse_bounds = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    fine_bounds = torch.tensor(
        [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]],
        dtype=torch.float64,
    )

    result = disagg.disaggregate(
        coarse,
        torch.mean(coarse_bounds, dim=1),
        torch.mean(fine_bounds, dim=1),
        standard_name="precipitation_flux",
        units="kg",
        calendar="noleap",
        time_units="days since 2000-01-01",
        coarse_bounds=coarse_bounds,
        fine_bounds=fine_bounds,
        temporal_semantics="interval_accumulation",
        target_units="kg",
        measure_units="kg",
        q_ref_parameter=q_ref_parameter,
    )

    assert result.conservation_error <= tolerance
    torch.testing.assert_close(
        result.data,
        torch.tensor([4.0, 4.0, 2.0, 2.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert result.diagnostics["conservation_checked"] is True
    assert result.diagnostics["conservation_passed"] is True


def test_physical_constraints_reject_negative_values_without_clamp():
    config = TemporalConfig(
        source_frequency="daily",
        target_frequency="hourly",
        direction=ScalingDirection.DOWNSCALE,
        spatial_basis=SpatialQuantityBasis.CELL_TOTAL,
        conservation_level=ConservationLevel.NONE,
        device=torch.device("cpu"),
    )
    disagg = TemporalDisaggregator(
        config,
        DisaggregationConfig(
            method=DisaggregationMethod.UNIFORM,
            allow_negative=False,
            use_physical_constraints=True,
            time_dim=0,
        ),
    )

    coarse = torch.tensor([-1.0], dtype=torch.float64)
    coarse_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    fine_bounds = torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="nonnegative"):
        disagg.disaggregate(
            coarse,
            torch.mean(coarse_bounds, dim=1),
            torch.mean(fine_bounds, dim=1),
            standard_name="precipitation_flux",
            units="kg",
            calendar="noleap",
            time_units="days since 2000-01-01",
            coarse_bounds=coarse_bounds,
            fine_bounds=fine_bounds,
            temporal_semantics="interval_accumulation",
            target_units="kg",
            measure_units="kg",
        )

