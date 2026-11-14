"""Advanced numerical and parameter tests for SectorAllocation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from relic.core.state import UpdateItem
from relic.scientific.anthroposphere.emissions.sector_allocation import (
    SectorAllocation,
)


def _state(
    *,
    batch: int = 2,
    height: int = 3,
    width: int = 4,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    shape = (batch, height, width)
    grid = torch.arange(1, batch * height * width + 1, dtype=dtype).reshape(shape)
    state = {
        key: grid * ((index + 1) * 1.0e-7)
        for index, key in enumerate(SectorAllocation.CANONICAL_INPUTS)
    }
    state["population_density"] = grid + 1.0
    state["gdp_density"] = torch.flip(grid, dims=(-2,)) + 2.0
    state["road_density"] = torch.flip(grid, dims=(-1,)) + 3.0
    state["land_use_fractions"] = torch.stack(
        (
            torch.full_like(grid, 0.15),
            torch.full_like(grid, 0.25),
            torch.full_like(grid, 0.35),
            torch.full_like(grid, 0.25),
        ),
        dim=1,
    )
    return state


def _forward(
    allocator: SectorAllocation,
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return allocator.forward(state, time=12345.0, dt=300.0)


def _value(outputs: dict[str, Any]) -> torch.Tensor:
    item = outputs["sector_emission_flux"]
    assert isinstance(item, UpdateItem)
    return item.value


def _patch_parameter(
    monkeypatch: pytest.MonkeyPatch,
    allocator: SectorAllocation,
    predicate: Callable[[str], bool],
    replacement: Any,
) -> None:
    original = allocator.get_parameter_value

    def patched(name: str, **kwargs: Any) -> Any:
        if predicate(name):
            return replacement
        return original(name, **kwargs)

    monkeypatch.setattr(allocator, "get_parameter_value", patched)


def test_float64_path_preserves_dtype() -> None:
    allocator = SectorAllocation()
    outputs = _forward(allocator, _state(dtype=torch.float64))
    assert _value(outputs).dtype == torch.float64
    assert outputs["diagnostics.sector_allocation.allocation_weights"].dtype == torch.float64


def test_batch_proxies_are_normalized_independently() -> None:
    allocator = SectorAllocation()
    state = _state()
    outputs = _forward(allocator, state)
    proxies = outputs["diagnostics.sector_allocation.spatial_distribution"]
    assert torch.allclose(
        proxies.mean(dim=(-2, -1)),
        torch.ones_like(proxies.mean(dim=(-2, -1))),
    )


def test_modifying_first_batch_does_not_change_second_batch() -> None:
    allocator = SectorAllocation()
    state = _state()
    baseline = _value(_forward(allocator, state))
    modified = {key: value.clone() for key, value in state.items()}
    for key in ("population_density", "gdp_density", "road_density"):
        modified[key][0] *= 100.0
    modified["land_use_fractions"][0, 2] *= 100.0
    changed = _value(_forward(allocator, modified))
    assert torch.equal(baseline[..., 1, :, :], changed[..., 1, :, :])


def test_proxy_scale_invariance() -> None:
    allocator = SectorAllocation()
    state = _state()
    baseline = _value(_forward(allocator, state))
    scaled = {key: value.clone() for key, value in state.items()}
    scaled["population_density"] *= 3.0
    scaled["gdp_density"] *= 5.0
    scaled["road_density"] *= 7.0
    scaled["land_use_fractions"] *= 11.0
    result = _value(_forward(allocator, scaled))
    assert torch.allclose(baseline, result, rtol=1.0e-6, atol=1.0e-12)


def test_very_small_positive_flux_is_conserved() -> None:
    allocator = SectorAllocation()
    state = _state()
    for key in allocator.CANONICAL_INPUTS:
        state[key].fill_(1.0e-30)
    output = _value(_forward(allocator, state))
    expected = torch.stack([state[key] for key in allocator.CANONICAL_INPUTS])
    assert torch.allclose(output.sum(dim=0), expected, rtol=1.0e-6, atol=1.0e-35)


def test_large_finite_flux_is_conserved() -> None:
    allocator = SectorAllocation()
    state = _state(dtype=torch.float64)
    for key in allocator.CANONICAL_INPUTS:
        state[key].mul_(1.0e12)
    output = _value(_forward(allocator, state))
    expected = torch.stack([state[key] for key in allocator.CANONICAL_INPUTS])
    assert torch.allclose(output.sum(dim=0), expected, rtol=1.0e-12)


def test_negative_sector_parameter_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = SectorAllocation()
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name == "sector_fractions.power",
        -0.1,
    )
    with pytest.raises(ValueError, match="non-negative"):
        _forward(allocator, _state())


def test_zero_sector_parameter_support_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = SectorAllocation()
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name.startswith("sector_fractions."),
        0.0,
    )
    with pytest.raises(ValueError, match="positive support"):
        _forward(allocator, _state())


def test_negative_species_parameter_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = SectorAllocation()
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name == "transport_sector.species_fractions.co",
        -1.0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        _forward(allocator, _state())


def test_missing_species_support_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = SectorAllocation()
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name == "industry_sector.species_fractions.nmvoc",
        0.0,
    )
    with pytest.raises(ValueError, match="voc"):
        _forward(allocator, _state())


def test_nonpositive_proxy_floor_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = SectorAllocation()
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name == "sector_allocation.proxy_floor",
        0.0,
    )
    with pytest.raises(ValueError, match="must be positive"):
        _forward(allocator, _state())


def test_fractional_cropland_index_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = SectorAllocation()
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name == "sector_allocation.cropland_class_index",
        1.5,
    )
    with pytest.raises(ValueError, match="exact integer"):
        _forward(allocator, _state())


def test_out_of_range_cropland_index_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = SectorAllocation()
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name == "sector_allocation.cropland_class_index",
        99,
    )
    with pytest.raises(ValueError, match="outside"):
        _forward(allocator, _state())


def test_zero_cropland_support_fails() -> None:
    allocator = SectorAllocation()
    state = _state()
    state["land_use_fractions"][:, 2].zero_()
    with pytest.raises(ValueError, match="positive support"):
        _forward(allocator, state)


def test_integer_flux_is_rejected() -> None:
    allocator = SectorAllocation()
    state = _state()
    state["anthropogenic_nox_flux"] = torch.ones((2, 3, 4), dtype=torch.int64)
    with pytest.raises(TypeError, match="float32 or float64"):
        _forward(allocator, state)


def test_scalar_proxy_is_rejected() -> None:
    allocator = SectorAllocation()
    state = _state()
    state["gdp_density"] = torch.tensor(1.0)
    with pytest.raises(ValueError, match="must match"):
        _forward(allocator, state)


def test_functor_call_path_validates_and_runs() -> None:
    allocator = SectorAllocation()
    state = _state()
    outputs = allocator(state, 0.0, 300.0)
    assert isinstance(outputs["sector_emission_flux"], UpdateItem)


def test_output_is_differentiable_with_respect_to_proxy() -> None:
    allocator = SectorAllocation()
    state = _state()
    state["population_density"].requires_grad_()
    output = _value(_forward(allocator, state))
    loss = output[3, 0, 0, 0, 0]
    loss.backward()
    assert state["population_density"].grad is not None
    assert torch.isfinite(state["population_density"].grad).all()


def test_mass_residual_matches_direct_reconstruction() -> None:
    allocator = SectorAllocation()
    state = _state()
    outputs = _forward(allocator, state)
    direct = torch.stack([state[key] for key in allocator.CANONICAL_INPUTS])
    direct = direct - _value(outputs).sum(dim=0)
    assert torch.equal(
        outputs["diagnostics.sector_allocation.mass_residual"],
        direct,
    )


def test_sector_and_species_order_are_stable() -> None:
    allocator = SectorAllocation()
    assert allocator.STRUCTURAL_SECTORS == (
        "power",
        "transport",
        "industry",
        "buildings",
        "agriculture",
    )
    assert allocator.CANONICAL_SPECIES == (
        "nox",
        "co",
        "so2",
        "voc",
        "nh3",
        "pm25",
    )


def test_required_parameter_declaration_is_unique_and_complete() -> None:
    allocator = SectorAllocation()
    keys = allocator.required_parameters()
    assert len(keys) == len(set(keys)) == 20
    assert set(keys) >= {
        "anthroposphere.emissions.sector_fractions.power",
        "anthroposphere.emissions.industry_sector.species_fractions.nmvoc",
        "anthroposphere.emissions.sector_allocation.proxy_floor",
        "anthroposphere.emissions.sector_allocation.cropland_class_index",
        "anthroposphere.emissions.conservation_tolerance",
    }


def test_runtime_weights_are_spatially_proxy_dependent_not_static() -> None:
    allocator = SectorAllocation()
    weights = _forward(allocator, _state())[
        "diagnostics.sector_allocation.allocation_weights"
    ]
    assert torch.max(torch.abs(weights[..., 1:, :] - weights[..., :-1, :])) > 0.0


def test_runtime_outputs_are_finite_and_nonnegative() -> None:
    outputs = _forward(SectorAllocation(), _state())
    for key in (
        "sector_emission_flux",
        "diagnostics.sector_allocation.allocation_weights",
        "diagnostics.sector_allocation.spatial_distribution",
    ):
        value = outputs[key].value if isinstance(outputs[key], UpdateItem) else outputs[key]
        assert torch.isfinite(value).all()
        assert torch.all(value >= 0.0)
    assert torch.isfinite(outputs["diagnostics.sector_allocation.mass_residual"]).all()


def test_same_inputs_and_parameters_are_bitwise_deterministic() -> None:
    allocator = SectorAllocation()
    state = _state()
    first = _forward(allocator, state)
    second = _forward(allocator, state)
    assert set(first) == set(second)
    for key in first:
        first_value = first[key].value if isinstance(first[key], UpdateItem) else first[key]
        second_value = second[key].value if isinstance(second[key], UpdateItem) else second[key]
        assert torch.equal(first_value, second_value)


def test_non_parameter_config_is_allowed_but_not_used() -> None:
    allocator = SectorAllocation(config={"enabled": True})
    assert allocator.config.get("enabled") is True
    outputs = _forward(allocator, _state())
    assert _value(outputs).shape[:2] == (5, 6)


def test_module_defines_no_parallel_parameter_accessor_chain() -> None:
    assert "get_parameter_value" not in SectorAllocation.__dict__
    assert "_get_param" not in SectorAllocation.__dict__
    assert "_require_param" not in SectorAllocation.__dict__


def test_module_source_and_target_scales_are_explicit() -> None:
    allocator = SectorAllocation()
    assert allocator.source_scale == "inventory_grid"
    assert allocator.target_scale == "sector_species_grid"
