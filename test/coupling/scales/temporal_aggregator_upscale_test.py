import torch

from relic.coupling.scales.temporal import (
    TemporalAggregator,
    TemporalConfig,
    AggregationConfig,
    AggregationResult,
    QualityFlags,
    ScalingDirection,
)
from relic.parameters import get_global_manager


def test_upscale_delegates_to_aggregate(monkeypatch):
    """
    Ensure TemporalAggregator.upscale uses aggregate when time coordinates are provided
    instead of raising or returning identity data.
    """

    pm = get_global_manager()
    config = TemporalConfig(
        source_frequency="hourly",
        target_frequency="daily",
        direction=ScalingDirection.UPSCALE,
    )
    agg_config = AggregationConfig()
    aggregator = TemporalAggregator(config, aggregation_config=agg_config, parameter_manager=pm)

    expected = torch.tensor([1.23], dtype=torch.float32)
    called = {}

    def fake_aggregate(**kwargs):
        called["invoked"] = True
        return AggregationResult(
            data=expected,
            metadata={"mock": True},
            quality_flags=QualityFlags(
                valid_fraction=torch.tensor([1.0]),
                num_samples=torch.tensor([1]),
                missing_fraction=torch.tensor([0.0]),
                insufficient_coverage_mask=torch.tensor([False]),
            ),
            diagnostics={"path": "fake"},
        )

    monkeypatch.setattr(aggregator, "aggregate", fake_aggregate, raising=True)

    source_time = torch.tensor([0.0, 3600.0, 7200.0, 10800.0])
    target_time = torch.tensor([0.0, 14400.0])
    data = torch.ones_like(source_time)

    result = aggregator.upscale(
        data,
        source_time=source_time,
        target_time=target_time,
        variable_name="test_var",
        standard_name="air_temperature",
        units="K",
    )

    assert called.get("invoked") is True
    assert torch.allclose(result, expected)
