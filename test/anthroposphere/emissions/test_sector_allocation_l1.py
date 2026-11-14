"""L1 tests for the governed SectorAllocation runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from relic.core.functor import ScaleCouplingFunctor
from relic.core.state import StateWriteMode, UpdateItem, UpdateMode
from relic.scientific.anthroposphere.emissions import sector_allocation as module
from relic.scientific.anthroposphere.emissions.sector_allocation import (
    SectorAllocation,
)


def _state(
    shape: tuple[int, ...] = (2, 3),
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    grid = torch.arange(
        1,
        1 + int(torch.tensor(shape).prod().item()),
        dtype=dtype,
    ).reshape(shape)
    state: dict[str, torch.Tensor] = {}
    for index, key in enumerate(SectorAllocation.CANONICAL_INPUTS, start=1):
        state[key] = grid * (index * 1.0e-6)

    state["population_density"] = grid + 10.0
    state["gdp_density"] = torch.flip(grid, dims=(-1,)) + 20.0
    state["road_density"] = grid * 0.5 + 1.0
    class_axis = len(shape) - 2
    state["land_use_fractions"] = torch.stack(
        (
            torch.full_like(grid, 0.1),
            torch.full_like(grid, 0.2),
            torch.full_like(grid, 0.4),
            torch.full_like(grid, 0.3),
        ),
        dim=class_axis,
    )
    return state


def _value(outputs: dict[str, Any]) -> torch.Tensor:
    item = outputs["sector_emission_flux"]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.per_timestep is False
    return item.value


def _inputs(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([state[key] for key in SectorAllocation.CANONICAL_INPUTS])


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


@pytest.fixture
def allocator() -> SectorAllocation:
    return SectorAllocation()


def test_instantiation_uses_scale_coupling_bridge(allocator: SectorAllocation) -> None:
    assert isinstance(allocator, ScaleCouplingFunctor)
    assert isinstance(allocator, torch.nn.Module)
    assert allocator.name == "sector_allocation"


def test_metadata_declares_exact_inputs_and_outputs(allocator: SectorAllocation) -> None:
    assert allocator.metadata.input_variables == [
        *allocator.CANONICAL_INPUTS,
        *allocator.PROXY_INPUTS,
    ]
    assert allocator.metadata.output_variables == [
        "sector_emission_flux",
        "diagnostics.sector_allocation.allocation_weights",
        "diagnostics.sector_allocation.spatial_distribution",
        "diagnostics.sector_allocation.temporal_factors",
        "diagnostics.sector_allocation.mass_residual",
    ]


def test_forward_returns_exact_declared_keys(allocator: SectorAllocation) -> None:
    outputs = allocator.forward(_state(), time=0.0, dt=3600.0)
    assert set(outputs) == set(allocator.metadata.output_variables)


def test_forward_uses_fixed_sector_species_prefix(allocator: SectorAllocation) -> None:
    outputs = allocator.forward(_state((2, 3)), time=0.0, dt=3600.0)
    assert _value(outputs).shape == (5, 6, 2, 3)


def test_forward_preserves_batch_shape(allocator: SectorAllocation) -> None:
    outputs = allocator.forward(_state((2, 3, 4)), time=0.0, dt=3600.0)
    assert _value(outputs).shape == (5, 6, 2, 3, 4)


def test_forward_preserves_arbitrary_leading_shape(allocator: SectorAllocation) -> None:
    outputs = allocator.forward(_state((2, 3, 4, 5)), time=0.0, dt=3600.0)
    assert _value(outputs).shape == (5, 6, 2, 3, 4, 5)


def test_pointwise_species_mass_is_conserved(allocator: SectorAllocation) -> None:
    state = _state()
    outputs = allocator.forward(state, time=0.0, dt=3600.0)
    assert torch.allclose(_value(outputs).sum(dim=0), _inputs(state), rtol=1.0e-6)


def test_voc_is_included_in_mass_ledger(allocator: SectorAllocation) -> None:
    state = _state()
    outputs = allocator.forward(state, time=0.0, dt=3600.0)
    voc_index = allocator.CANONICAL_SPECIES.index("voc")
    assert torch.allclose(
        _value(outputs)[:, voc_index].sum(dim=0),
        state["anthropogenic_voc_flux"],
        rtol=1.0e-6,
    )


def test_zero_flux_remains_zero(allocator: SectorAllocation) -> None:
    state = _state()
    for key in allocator.CANONICAL_INPUTS:
        state[key].zero_()
    outputs = allocator.forward(state, time=0.0, dt=60.0)
    assert torch.count_nonzero(_value(outputs)) == 0
    assert torch.count_nonzero(
        outputs["diagnostics.sector_allocation.mass_residual"]
    ) == 0


def test_rate_output_is_dt_invariant(allocator: SectorAllocation) -> None:
    state = _state()
    short = _value(allocator.forward(state, time=0.0, dt=60.0))
    long = _value(allocator.forward(state, time=0.0, dt=7200.0))
    assert torch.equal(short, long)


def test_temperature_and_policy_do_not_rescale_flux(allocator: SectorAllocation) -> None:
    state = _state()
    baseline = _value(allocator.forward(state, time=0.0, dt=3600.0))
    modified = {key: value.clone() for key, value in state.items()}
    modified["temperature"] = torch.full_like(state["population_density"], 320.0)
    modified["emission_tax"] = torch.tensor(1000.0)
    modified["subsidy_rate"] = torch.tensor(1000.0)
    modified["conservation_policy"] = torch.tensor(1.0)
    result = _value(allocator.forward(modified, time=0.0, dt=3600.0))
    assert torch.equal(baseline, result)


def test_extra_state_does_not_create_dynamic_outputs(allocator: SectorAllocation) -> None:
    state = _state()
    state["sector_emissions"] = torch.ones_like(state["population_density"])
    outputs = allocator.forward(state, time=0.0, dt=3600.0)
    assert set(outputs) == set(allocator.metadata.output_variables)


def test_allocation_weights_sum_to_one(allocator: SectorAllocation) -> None:
    outputs = allocator.forward(_state(), time=0.0, dt=3600.0)
    weights = outputs["diagnostics.sector_allocation.allocation_weights"]
    assert torch.allclose(weights.sum(dim=0), torch.ones_like(weights[0]))


def test_spatial_proxies_normalize_per_leading_index(allocator: SectorAllocation) -> None:
    outputs = allocator.forward(_state((2, 3, 4)), time=0.0, dt=3600.0)
    proxies = outputs["diagnostics.sector_allocation.spatial_distribution"]
    means = proxies.mean(dim=(-2, -1))
    assert torch.allclose(means, torch.ones_like(means), rtol=1.0e-6)


def test_temporal_diagnostic_confirms_upstream_ownership(
    allocator: SectorAllocation,
) -> None:
    outputs = allocator.forward(_state(), time=0.0, dt=3600.0)
    factors = outputs["diagnostics.sector_allocation.temporal_factors"]
    assert torch.equal(factors, torch.ones(5))


def test_population_proxy_changes_building_share_without_changing_mass(
    allocator: SectorAllocation,
) -> None:
    state = _state()
    baseline = _value(allocator.forward(state, time=0.0, dt=3600.0))
    concentrated = {key: value.clone() for key, value in state.items()}
    concentrated["population_density"][0, 0] *= 100.0
    changed = _value(allocator.forward(concentrated, time=0.0, dt=3600.0))
    assert changed[3, 0, 0, 0] > baseline[3, 0, 0, 0]
    assert torch.allclose(changed.sum(dim=0), _inputs(state), rtol=1.0e-6)


@pytest.mark.parametrize("missing_key", SectorAllocation.CANONICAL_INPUTS)
def test_missing_canonical_flux_fails(
    allocator: SectorAllocation,
    missing_key: str,
) -> None:
    state = _state()
    state.pop(missing_key)
    with pytest.raises(KeyError, match="Missing canonical inventory fluxes"):
        allocator.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize("missing_key", SectorAllocation.PROXY_INPUTS)
def test_missing_proxy_fails(
    allocator: SectorAllocation,
    missing_key: str,
) -> None:
    state = _state()
    state.pop(missing_key)
    with pytest.raises((TypeError, KeyError), match=missing_key):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_negative_flux_fails(allocator: SectorAllocation) -> None:
    state = _state()
    state["anthropogenic_nox_flux"][0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_nonfinite_flux_fails(allocator: SectorAllocation) -> None:
    state = _state()
    state["anthropogenic_nox_flux"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_nonfinite_proxy_fails(allocator: SectorAllocation) -> None:
    state = _state()
    state["road_density"][0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_zero_single_sector_proxy_is_allowed(allocator: SectorAllocation) -> None:
    state = _state()
    state["population_density"].zero_()
    outputs = allocator.forward(state, time=0.0, dt=3600.0)
    assert torch.allclose(_value(outputs).sum(dim=0), _inputs(state), rtol=1.0e-6)
    assert torch.isfinite(
        outputs["diagnostics.sector_allocation.allocation_weights"]
    ).all()


def test_all_sector_zero_denominator_fails(allocator: SectorAllocation) -> None:
    state = _state()
    for key in ("population_density", "gdp_density", "road_density"):
        state[key].zero_()
    state["land_use_fractions"].zero_()
    with pytest.raises(ValueError, match="Allocation weights lack positive support"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_flux_shape_mismatch_fails(allocator: SectorAllocation) -> None:
    state = _state()
    state["anthropogenic_co_flux"] = torch.ones((3, 3))
    with pytest.raises(ValueError, match="share shape"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_flux_dtype_mismatch_fails(allocator: SectorAllocation) -> None:
    state = _state()
    state["anthropogenic_co_flux"] = state["anthropogenic_co_flux"].double()
    with pytest.raises(ValueError, match="share shape"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_proxy_shape_mismatch_fails(allocator: SectorAllocation) -> None:
    state = _state()
    state["road_density"] = torch.ones((3, 3))
    with pytest.raises(ValueError, match="must match"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_land_use_requires_explicit_class_axis(allocator: SectorAllocation) -> None:
    state = _state()
    state["land_use_fractions"] = torch.ones((2, 3))
    with pytest.raises(ValueError, match="class axis"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_flux_requires_trailing_spatial_axes(allocator: SectorAllocation) -> None:
    state = _state()
    for key in allocator.CANONICAL_INPUTS:
        state[key] = torch.ones(3)
    with pytest.raises(ValueError, match="trailing 2D"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_forward_does_not_mutate_inputs(allocator: SectorAllocation) -> None:
    state = _state()
    before = {key: value.clone() for key, value in state.items()}
    allocator.forward(state, time=0.0, dt=3600.0)
    assert state.keys() == before.keys()
    assert all(torch.equal(state[key], before[key]) for key in state)


def test_flux_gradients_are_preserved(allocator: SectorAllocation) -> None:
    state = _state()
    for key in allocator.CANONICAL_INPUTS:
        state[key].requires_grad_()
    loss = _value(allocator.forward(state, time=0.0, dt=3600.0)).sum()
    loss.backward()
    for key in allocator.CANONICAL_INPUTS:
        assert state[key].grad is not None
        assert torch.allclose(state[key].grad, torch.ones_like(state[key]))


def test_config_cannot_override_parameter_manager_values() -> None:
    with pytest.raises(ValueError, match="ParameterManager YAML"):
        SectorAllocation(
            config={"parameters": {"sector_fractions.power": {"value": 999.0}}}
        )


def test_dictionary_name_compatibility_path_is_rejected() -> None:
    with pytest.raises(TypeError, match="string"):
        SectorAllocation({"name": "old"})  # type: ignore[arg-type]


def test_dependencies_contain_no_state_variable_aliases(
    allocator: SectorAllocation,
) -> None:
    assert allocator.get_dependencies() == []


def test_required_parameters_exclude_secondary_process_laws(
    allocator: SectorAllocation,
) -> None:
    keys = set(allocator.required_parameters())
    assert "anthroposphere.emissions.sector_allocation.proxy_floor" in keys
    assert "anthroposphere.emissions.sector_allocation.cropland_class_index" in keys
    assert not any("temperature_response" in key for key in keys)
    assert not any("policy_response" in key for key in keys)
    assert "anthroposphere.inventory.default_emission_factor" not in keys


def test_upscale_is_exact_sector_sum(allocator: SectorAllocation) -> None:
    state = _state()
    sector_flux = _value(allocator.forward(state, time=0.0, dt=3600.0))
    assert torch.allclose(allocator.upscale(sector_flux), _inputs(state))


def test_upscale_rejects_wrong_sector_axis(allocator: SectorAllocation) -> None:
    with pytest.raises(ValueError, match="canonical sector axis"):
        allocator.upscale(torch.ones((4, 6, 2, 3)))


def test_proxy_free_downscale_fails_fast(allocator: SectorAllocation) -> None:
    with pytest.raises(RuntimeError, match="requires declared proxy state"):
        allocator.downscale(torch.ones((6, 2, 3)))


def test_conservation_hook_uses_current_payload(allocator: SectorAllocation) -> None:
    state = _state()
    outputs = allocator.forward(state, time=0.0, dt=3600.0)
    assert allocator.check_conservation_laws(state, outputs)["mass"] < 1.0e-6

    tampered = dict(outputs)
    tampered["sector_emission_flux"] = replace(
        outputs["sector_emission_flux"],
        value=_value(outputs) * 0.9,
    )
    assert allocator.check_conservation_laws(state, tampered)["mass"] > 0.09


@pytest.mark.parametrize(
    "name",
    [
        "compute",
        "declared_forms",
        "build_hodge_contract",
        "build_hodge_terms",
        "energy_terms",
        "entropy_terms",
        "reversible_structure",
        "dissipative_structure",
        "ports",
        "build_allocation_operator",
        "allocation_matrix",
    ],
)
def test_false_structural_and_second_runtime_apis_are_absent(
    allocator: SectorAllocation,
    name: str,
) -> None:
    assert not hasattr(allocator, name)


def test_false_runtime_claim_attributes_are_absent(allocator: SectorAllocation) -> None:
    assert not hasattr(allocator, "PHASE8_RUNTIME_REPLACEMENT_STATUS")
    assert not hasattr(allocator, "runtime_path")


def test_module_source_has_no_dec_hodge_port_or_entropy_structure() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    for token in (
        "CochainSpaceSpec",
        "HodgeStarSpec",
        "PortSpecIPHS",
        "build_hodge_contract",
        "build_hodge_terms",
        "dissipative_structure",
        "canonical_iphs_runtime",
        "canonical_iphs_sector_allocation_chain_map",
    ):
        assert token not in source


def test_updateitem_has_complete_rate_and_owner_contract(
    allocator: SectorAllocation,
) -> None:
    item = allocator.forward(_state(), time=0.0, dt=3600.0)[
        "sector_emission_flux"
    ]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.write_mode is StateWriteMode.DERIVED
    assert item.unit == "kg m-2 s-1"
    assert item.basis == "per_second"
    assert item.per_timestep is False
    assert item.source_functor == allocator.name


@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.int64, torch.complex64, torch.float16, torch.bfloat16],
)
def test_runtime_flux_rejects_unsupported_dtype(
    allocator: SectorAllocation,
    dtype: torch.dtype,
) -> None:
    state = _state()
    state["anthropogenic_nox_flux"] = torch.ones((2, 3), dtype=dtype)
    with pytest.raises(TypeError, match="float32 or float64"):
        allocator.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize(
    "bad_tensor",
    [
        torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]]),
            torch.ones(2),
            (2, 3),
            check_invariants=False,
        ),
        torch.empty((2, 3), device="meta"),
        torch.quantize_per_tensor(
            torch.ones((2, 3), dtype=torch.float32),
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        ),
    ],
)
def test_runtime_flux_rejects_nonmaterialized_or_nonstrided_tensor(
    allocator: SectorAllocation,
    bad_tensor: torch.Tensor,
) -> None:
    state = _state()
    state["anthropogenic_nox_flux"] = bad_tensor
    with pytest.raises(TypeError, match="dense strided|materialized"):
        allocator.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.int64, torch.complex64, torch.float16, torch.bfloat16],
)
def test_spatial_proxy_rejects_unsupported_dtype_without_cast(
    allocator: SectorAllocation,
    dtype: torch.dtype,
) -> None:
    state = _state()
    state["population_density"] = torch.ones((2, 3), dtype=dtype)
    with pytest.raises(TypeError, match="float32 or float64"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_spatial_proxy_rejects_dtype_mismatch_without_cast(
    allocator: SectorAllocation,
) -> None:
    state = _state()
    state["population_density"] = state["population_density"].double()
    with pytest.raises(ValueError, match="dtype"):
        allocator.forward(state, time=0.0, dt=3600.0)


def test_spatial_proxy_rejects_device_mismatch_without_move(
    allocator: SectorAllocation,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    state = _state()
    state["population_density"] = state["population_density"].cuda()
    with pytest.raises(ValueError, match="device"):
        allocator.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.complex64])
def test_land_use_rejects_unsupported_dtype_without_cast(
    allocator: SectorAllocation,
    dtype: torch.dtype,
) -> None:
    state = _state()
    state["land_use_fractions"] = torch.ones((4, 2, 3), dtype=dtype)
    with pytest.raises(TypeError, match="float32 or float64"):
        allocator.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (2, 3, 0)])
def test_runtime_state_rejects_empty_semantic_axes(
    allocator: SectorAllocation,
    shape: tuple[int, ...],
) -> None:
    state = _state(shape)
    with pytest.raises(ValueError, match="non-empty"):
        allocator.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize(
    ("parameter_name", "parameter_group", "loss_index"),
    [
        ("sector_fractions.buildings", "sector", (3, 0, 0, 0)),
        (
            "buildings_sector.species_fractions.nox",
            "species",
            (3, 0, 0, 0),
        ),
        ("sector_allocation.proxy_floor", "floor", (3, 0, 0, 0)),
    ],
)
def test_continuous_parameter_tensor_preserves_manager_leaf_gradient(
    allocator: SectorAllocation,
    monkeypatch: pytest.MonkeyPatch,
    parameter_name: str,
    parameter_group: str,
    loss_index: tuple[int, ...],
) -> None:
    _ = parameter_group
    leaf = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name == parameter_name,
        leaf,
    )
    output = _value(allocator.forward(_state(), time=0.0, dt=3600.0))
    output[loss_index].backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad)
    assert leaf.grad != 0


def test_parameter_cross_device_projection_preserves_source_leaf_gradient(
    allocator: SectorAllocation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    leaf = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name == "sector_fractions.buildings",
        leaf,
    )
    state = {key: value.cuda() for key, value in _state().items()}
    output = _value(allocator.forward(state, time=0.0, dt=3600.0))
    assert output.device.type == "cuda"
    output[3, 0, 0, 0].backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad)
    assert leaf.grad != 0


@pytest.mark.parametrize("bad_tolerance", [True, float("nan"), float("inf"), -1.0])
def test_conservation_tolerance_rejects_invalid_scalar(
    allocator: SectorAllocation,
    monkeypatch: pytest.MonkeyPatch,
    bad_tolerance: Any,
) -> None:
    _patch_parameter(
        monkeypatch,
        allocator,
        lambda name: name == "conservation_tolerance",
        bad_tolerance,
    )
    with pytest.raises(ValueError, match="conservation_tolerance"):
        allocator.forward(_state(), time=0.0, dt=3600.0)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("mode", UpdateMode.ACCUMULATE),
        ("write_mode", None),
        ("unit", "kg s-1"),
        ("basis", "per_step"),
        ("per_timestep", True),
        ("source_functor", "other_functor"),
    ],
)
def test_conservation_hook_rejects_wrong_updateitem_contract(
    allocator: SectorAllocation,
    field_name: str,
    bad_value: Any,
) -> None:
    state = _state()
    outputs = allocator.forward(state, time=0.0, dt=3600.0)
    outputs["sector_emission_flux"] = replace(
        outputs["sector_emission_flux"],
        **{field_name: bad_value},
    )
    with pytest.raises(ValueError, match="UpdateItem contract"):
        allocator.check_conservation_laws(state, outputs)


def test_conservation_hook_rejects_negative_but_balanced_output(
    allocator: SectorAllocation,
) -> None:
    state = _state()
    outputs = allocator.forward(state, time=0.0, dt=3600.0)
    value = _value(outputs).clone()
    amount = value[0, 0].clone()
    value[0, 0] = -amount
    value[1, 0] = value[1, 0] + 2.0 * amount
    outputs["sector_emission_flux"] = replace(
        outputs["sector_emission_flux"],
        value=value,
    )
    with pytest.raises(ValueError, match="non-negative"):
        allocator.check_conservation_laws(state, outputs)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_conservation_hook_rejects_nonfinite_output(
    allocator: SectorAllocation,
    bad_value: float,
) -> None:
    state = _state()
    outputs = allocator.forward(state, time=0.0, dt=3600.0)
    value = _value(outputs).clone()
    value[0, 0, 0, 0] = bad_value
    outputs["sector_emission_flux"] = replace(
        outputs["sector_emission_flux"],
        value=value,
    )
    with pytest.raises(ValueError, match="finite"):
        allocator.check_conservation_laws(state, outputs)


@pytest.mark.parametrize(
    "bad_tensor",
    [
        torch.ones((5, 6, 2, 3), dtype=torch.int64),
        -torch.ones((5, 6, 2, 3), dtype=torch.float32),
        torch.full((5, 6, 2, 3), float("inf")),
    ],
)
def test_upscale_rejects_invalid_sector_tensor(
    allocator: SectorAllocation,
    bad_tensor: torch.Tensor,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        allocator.upscale(bad_tensor)


def test_noncontiguous_runtime_tensors_preserve_autograd(
    allocator: SectorAllocation,
) -> None:
    state = _state((3, 2))
    for key, value in tuple(state.items()):
        if key == "land_use_fractions":
            state[key] = value.transpose(-1, -2).requires_grad_()
        else:
            state[key] = value.transpose(-1, -2).requires_grad_()
        assert state[key].is_contiguous() is False
    output = _value(allocator.forward(state, time=0.0, dt=3600.0))
    output.sum().backward()
    assert state["anthropogenic_nox_flux"].grad is not None
    assert state["population_density"].grad is not None


def test_removed_legacy_runtime_apis_are_absent(allocator: SectorAllocation) -> None:
    for name in (
        "allocate_emissions",
        "allocate_sector_emissions",
        "allocate_emissions_by_sector",
        "_resolve_global_scale",
        "_collect_total_emissions",
        "get_sector_statistics",
    ):
        assert not hasattr(allocator, name)
