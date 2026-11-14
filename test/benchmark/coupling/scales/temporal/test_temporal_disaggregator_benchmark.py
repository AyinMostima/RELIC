
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


def test_disaggregation_conservation_benchmark(monkeypatch):
    manager = get_global_manager()
    tolerance = _to_float(manager.get("validation.conservation.conservation_tolerance"))

    config = TemporalConfig(
        source_frequency="weekly",
        target_frequency="daily",
        direction=ScalingDirection.DOWNSCALE,
        spatial_basis=SpatialQuantityBasis.CELL_TOTAL,
        conservation_level=ConservationLevel.STRICT,
        device=torch.device("cpu"),
    )
    disagg = TemporalDisaggregator(
        config,
        DisaggregationConfig(
            method=DisaggregationMethod.UNIFORM,
            use_physical_constraints=False,
            time_dim=0,
        ),
    )
    q_ref_parameter = "tests.temporal.benchmark_q_ref"
    original_lookup = disagg.get_parameter_value

    def lookup(name, **kwargs):
        if name == q_ref_parameter:
            return 1.0
        return original_lookup(name, **kwargs)

    monkeypatch.setattr(disagg, "get_parameter_value", lookup)

    coarse = torch.tensor([7.0, 14.0, 21.0], dtype=torch.float64)
    coarse_bounds = torch.tensor(
        [[0.0, 7.0], [7.0, 14.0], [14.0, 21.0]], dtype=torch.float64
    )
    fine_bounds = torch.stack(
        (
            torch.arange(21, dtype=torch.float64),
            torch.arange(1, 22, dtype=torch.float64),
        ),
        dim=1,
    )

    result = disagg.disaggregate(
        coarse,
        torch.mean(coarse_bounds, dim=1),
        torch.mean(fine_bounds, dim=1),
        variable_name="benchmark_precip",
        standard_name="precipitation_flux",
        units="kg",
        cell_methods="time: sum",
        calendar="noleap",
        time_units="days since 2000-01-01",
        coarse_bounds=coarse_bounds,
        fine_bounds=fine_bounds,
        temporal_semantics="interval_accumulation",
        target_units="kg",
        measure_units="kg",
        q_ref_parameter=q_ref_parameter,
    )

    expected = torch.repeat_interleave(
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        7,
    )
    torch.testing.assert_close(result.data, expected, rtol=0.0, atol=0.0)
    assert result.conservation_error <= tolerance
    assert result.diagnostics["conservation_checked"] is True
    assert result.diagnostics["conservation_passed"] is True
    torch.testing.assert_close(
        result.diagnostics["mapping_residual"],
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0.0,
        atol=torch.finfo(torch.float64).eps * 64.0,
    )
    assert "history" in result.metadata
    assert result.metadata["cell_methods"] == "time: sum"

