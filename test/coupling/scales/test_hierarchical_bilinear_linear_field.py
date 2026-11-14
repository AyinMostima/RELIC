"""
Hierarchical bilinear interpolation tests.

Validates that hierarchical bilinear aggregation reconstructs a linear scalar
field at an off-center target point, instead of falling back to area mean.
"""

from __future__ import annotations

import torch

from relic.coupling.scales.hierarchical.aggregator import HierarchicalAggregator
from relic.coupling.scales.hierarchical.hierarchy_index import HierarchyIndex


def test_hierarchical_bilinear_reconstructs_linear_field_off_center() -> None:
    """
    Bilinear interpolation should reconstruct linear field values.

    References:
        - Press et al. (2007), Numerical Recipes, bilinear interpolation.

    Mathematical Expression:
        f(x, y) = a * x + b * y + c
        f_target = bilinear(f00, f10, f01, f11; tx, ty)

    Parameter Meanings:
        a, b, c: linear field coefficients
        tx, ty: normalized target coordinates in source cell
        f00..f11: corner values
    """
    source_coordinates = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    target_coordinates = [[0.25, 0.75]]
    topology = {
        "n_source_units": 4,
        "n_target_units": 1,
        "parent_map": {0: 0, 1: 0, 2: 0, 3: 0},
        "source_areas": [1.0, 1.0, 1.0, 1.0],
        "target_areas": [4.0],
        "source_coordinates": source_coordinates,
        "target_coordinates": target_coordinates,
    }

    index = HierarchyIndex(
        source_scale="patch",
        target_scale="hillslope",
        topology=topology,
        device=torch.device("cpu"),
    )
    aggregator = HierarchicalAggregator(index=index, device=torch.device("cpu"), weight_type="area", rule="conservative")

    a, b, c = 2.0, 1.0, 5.0
    source_values = torch.tensor(
        [a * x + b * y + c for x, y in source_coordinates],
        dtype=torch.float64,
    )

    outputs, _audit = aggregator.aggregate(
        data={"air_temperature": source_values},
        variables=["air_temperature"],
        strategy="bilinear",
    )

    expected = a * target_coordinates[0][0] + b * target_coordinates[0][1] + c
    bilinear_value = outputs["air_temperature"][0]
    arithmetic_mean = torch.mean(source_values)

    assert torch.isclose(bilinear_value, torch.tensor(expected, dtype=torch.float64), atol=1.0e-10)
    assert not torch.isclose(bilinear_value, arithmetic_mean, atol=1.0e-10)


def _square_without_coordinates() -> HierarchicalAggregator:
    topology = {
        "n_source_units": 4,
        "n_target_units": 1,
        "parent_map": {0: 0, 1: 0, 2: 0, 3: 0},
        "source_areas": [1.0, 1.0, 1.0, 1.0],
        "target_areas": [4.0],
    }
    index = HierarchyIndex(source_scale="patch", target_scale="hillslope", topology=topology, device=torch.device("cpu"))
    return HierarchicalAggregator(index=index, device=torch.device("cpu"), weight_type="area", rule="conservative")


def test_fast_approx_is_bilinear_on_declared_coordinates_without_fallback() -> None:
    """fast_approx selects bilinear; without declared coordinates it raises (no grid inference, no area mean)."""
    import inspect

    import pytest

    aggregator = _square_without_coordinates()
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    metadata = {"signal": {"spatial_basis": "intensive"}}
    with pytest.raises(RuntimeError, match="coordinates"):
        aggregator.aggregate({"signal": values}, ["signal"], mode="fast_approx", variable_metadata=metadata)
    with pytest.raises(ValueError, match="Unknown aggregation mode"):
        aggregator.aggregate({"signal": values}, ["signal"], mode="approximate", variable_metadata=metadata)
    assert not hasattr(HierarchicalAggregator, "_infer_bilinear_coordinates")
    assert "bilinear_fallback" not in inspect.getsource(HierarchicalAggregator.aggregate)


def test_bilinear_coordinates_come_from_the_declared_topology_keys_only() -> None:
    """source_coordinates/target_coordinates are the declared topology keys; the former aliases
    source_coords/target_coords are rejected, and the aggregator reads no alias attribute."""
    import pytest

    topology = {
        "n_source_units": 4,
        "n_target_units": 1,
        "parent_map": {0: 0, 1: 0, 2: 0, 3: 0},
        "source_areas": [1.0, 1.0, 1.0, 1.0],
        "target_areas": [4.0],
        "source_coords": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "target_coords": [[0.5, 0.5]],
    }
    with pytest.raises(ValueError, match="source_coordinates"):
        HierarchyIndex(source_scale="patch", target_scale="hillslope", topology=topology, device=torch.device("cpu"))

    aggregator = _square_without_coordinates()
    aggregator.index.source_coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    aggregator.index.target_coords = torch.tensor([[0.5, 0.5]])
    with pytest.raises(RuntimeError, match="coordinates"):
        aggregator.aggregate(
            {"signal": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)},
            ["signal"],
            strategy="bilinear",
            variable_metadata={"signal": {"spatial_basis": "intensive"}},
        )


def test_a_missing_bilinear_corner_raises_instead_of_using_the_nearest_node() -> None:
    """Three nodes of a square: the corner (1, 1) around the centre does not exist, so the
    interpolation raises instead of substituting the nearest node."""
    import pytest

    topology = {
        "n_source_units": 3,
        "n_target_units": 1,
        "parent_map": {0: 0, 1: 0, 2: 0},
        "source_areas": [1.0, 1.0, 1.0],
        "target_areas": [3.0],
        "source_coordinates": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "target_coordinates": [[0.5, 0.5]],
    }
    index = HierarchyIndex(source_scale="patch", target_scale="hillslope", topology=topology, device=torch.device("cpu"))
    aggregator = HierarchicalAggregator(index=index, device=torch.device("cpu"), weight_type="area", rule="conservative")
    with pytest.raises(RuntimeError, match="corner"):
        aggregator.aggregate(
            {"signal": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)},
            ["signal"],
            strategy="bilinear",
            variable_metadata={"signal": {"spatial_basis": "intensive"}},
        )


def _unit_square_aggregator(source_coordinates, target_coordinates) -> HierarchicalAggregator:
    n_source = len(source_coordinates)
    topology = {
        "n_source_units": n_source,
        "n_target_units": 1,
        "parent_map": {i: 0 for i in range(n_source)},
        "source_areas": [1.0] * n_source,
        "target_areas": [float(n_source)],
        "source_coordinates": source_coordinates,
        "target_coordinates": target_coordinates,
    }
    index = HierarchyIndex(source_scale="patch", target_scale="hillslope", topology=topology, device=torch.device("cpu"))
    return HierarchicalAggregator(index=index, device=torch.device("cpu"), weight_type="area", rule="conservative")


def test_a_target_outside_the_source_nodes_raises_instead_of_taking_the_boundary_value() -> None:
    """(1.5, 0.5) lies outside the nodes of the unit square: bilinear interpolation does not
    extrapolate and does not substitute the boundary value."""
    import pytest

    aggregator = _unit_square_aggregator([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[1.5, 0.5]])
    with pytest.raises(RuntimeError, match="outside"):
        aggregator.aggregate(
            {"signal": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)},
            ["signal"],
            strategy="bilinear",
            variable_metadata={"signal": {"spatial_basis": "intensive"}},
        )


def test_duplicate_source_coordinates_raise() -> None:
    """Two source nodes at (1, 0) make the corner ambiguous: the interpolation raises instead of
    keeping the first node."""
    import pytest

    aggregator = _unit_square_aggregator(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [[0.5, 0.5]]
    )
    with pytest.raises(RuntimeError, match="share"):
        aggregator.aggregate(
            {"signal": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)},
            ["signal"],
            strategy="bilinear",
            variable_metadata={"signal": {"spatial_basis": "intensive"}},
        )
