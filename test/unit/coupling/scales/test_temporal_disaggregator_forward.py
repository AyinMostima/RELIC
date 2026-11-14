import torch

from relic.coupling.scales.temporal import (
    ConservationLevel,
    ScalingDirection,
    TemporalConfig,
)
from relic.coupling.scales.types import SpatialQuantityBasis
from relic.coupling.scales.temporal.disaggregator import TemporalDisaggregator


def test_temporal_disaggregator_forward_outputs_metadata_and_error(monkeypatch):
    config = TemporalConfig(
        source_frequency="daily",
        target_frequency="hourly",
        direction=ScalingDirection.DOWNSCALE,
        spatial_basis=SpatialQuantityBasis.AREA_DENSITY,
        conservation_level=ConservationLevel.MODERATE,
        device=torch.device("cpu"),
    )
    disaggregator = TemporalDisaggregator(config)
    q_ref_parameter = "tests.temporal.forward_q_ref"
    original_lookup = disaggregator.get_parameter_value

    def lookup(name, **kwargs):
        if name == q_ref_parameter:
            return 1.0
        return original_lookup(name, **kwargs)

    monkeypatch.setattr(disaggregator, "get_parameter_value", lookup)

    coarse_bounds = torch.tensor(
        [[0.0, 86400.0], [86400.0, 172800.0]], dtype=torch.float64
    )
    fine_bounds = torch.tensor(
        [
            [0.0, 43200.0],
            [43200.0, 86400.0],
            [86400.0, 129600.0],
            [129600.0, 172800.0],
        ],
        dtype=torch.float64,
    )
    coarse_time = torch.mean(coarse_bounds, dim=1)
    fine_time = torch.mean(fine_bounds, dim=1)
    data = torch.tensor([2.0, 4.0], dtype=torch.float64)

    output = disaggregator.forward(
        {},
        time=0.0,
        dt=3600.0,
        data=data,
        coarse_time=coarse_time,
        fine_time=fine_time,
        variable_name="precipitation",
        standard_name="precipitation_flux",
        units="kg m-2 s-1",
        calendar="noleap",
        time_units="seconds since 2000-01-01",
        coarse_bounds=coarse_bounds,
        fine_bounds=fine_bounds,
        temporal_semantics="rate",
        target_units="kg m-2 s-1",
        measure_units="kg m-2",
        integration_time_unit="s",
        q_ref_parameter=q_ref_parameter,
    )

    assert "precipitation" in output
    torch.testing.assert_close(
        output["precipitation"],
        torch.tensor([2.0, 2.0, 4.0, 4.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert "_temporal_metadata" in output
    assert "_conservation_error" in output
    assert output["_temporal_metadata"]["cell_methods"] == "time: mean"
    assert output["_temporal_diagnostics"]["conservation_checked"] is True
    assert output["_temporal_diagnostics"]["conservation_passed"] is True
    assert output["_conservation_error"].item() == 0.0



def _metadata_contract_case(monkeypatch):
    """One disaggregator with the complete forward keyword set of the case above."""
    config = TemporalConfig(
        source_frequency="daily",
        target_frequency="hourly",
        direction=ScalingDirection.DOWNSCALE,
        spatial_basis=SpatialQuantityBasis.AREA_DENSITY,
        conservation_level=ConservationLevel.MODERATE,
        device=torch.device("cpu"),
    )
    disaggregator = TemporalDisaggregator(config)
    q_ref_parameter = "tests.temporal.forward_q_ref"
    original_lookup = disaggregator.get_parameter_value

    def lookup(name, **kwargs):
        if name == q_ref_parameter:
            return 1.0
        return original_lookup(name, **kwargs)

    monkeypatch.setattr(disaggregator, "get_parameter_value", lookup)
    coarse_bounds = torch.tensor([[0.0, 86400.0], [86400.0, 172800.0]], dtype=torch.float64)
    fine_bounds = torch.tensor(
        [[0.0, 43200.0], [43200.0, 86400.0], [86400.0, 129600.0], [129600.0, 172800.0]],
        dtype=torch.float64,
    )
    kwargs = dict(
        data=torch.tensor([2.0, 4.0], dtype=torch.float64),
        coarse_time=torch.mean(coarse_bounds, dim=1),
        fine_time=torch.mean(fine_bounds, dim=1),
        variable_name="precipitation",
        standard_name="precipitation_flux",
        units="kg m-2 s-1",
        calendar="noleap",
        time_units="seconds since 2000-01-01",
        coarse_bounds=coarse_bounds,
        fine_bounds=fine_bounds,
        temporal_semantics="rate",
        target_units="kg m-2 s-1",
        measure_units="kg m-2",
        integration_time_unit="s",
        q_ref_parameter=q_ref_parameter,
    )
    return disaggregator, kwargs


def test_forward_requires_the_variable_name_and_units(monkeypatch):
    """The output key and the value units are required; no empty default stands in for them."""
    import pytest

    for missing in ("variable_name", "units"):
        disaggregator, kwargs = _metadata_contract_case(monkeypatch)
        kwargs.pop(missing)
        with pytest.raises(ValueError, match=f"requires a non-empty {missing}"):
            disaggregator.forward({}, time=0.0, dt=3600.0, **kwargs)
        disaggregator, kwargs = _metadata_contract_case(monkeypatch)
        kwargs[missing] = ""
        with pytest.raises(ValueError, match=f"requires a non-empty {missing}"):
            disaggregator.forward({}, time=0.0, dt=3600.0, **kwargs)


def test_absent_cf_identity_and_cell_methods_have_a_declared_meaning(monkeypatch):
    """standard_name and cell_methods are optional: absent means no CF identity and no
    earlier cell method; neither is published as an empty string, and an empty string
    is not an accepted value."""
    import pytest

    disaggregator, kwargs = _metadata_contract_case(monkeypatch)
    kwargs.pop("standard_name")
    output = disaggregator.forward({}, time=0.0, dt=3600.0, **kwargs)
    metadata = output["_temporal_metadata"]
    assert "standard_name" not in metadata
    assert metadata["cell_methods"] == "time: mean"
    assert metadata["variable_name"] == "precipitation"

    for name in ("standard_name", "cell_methods"):
        disaggregator, kwargs = _metadata_contract_case(monkeypatch)
        kwargs[name] = ""
        with pytest.raises(ValueError, match=name):
            disaggregator.forward({}, time=0.0, dt=3600.0, **kwargs)
