"""
Irregular conservative regridder tests.

Validates polygon-overlap conservative weights on unstructured meshes.
"""

from __future__ import annotations

import torch

from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.coupling.scales.spatial.regrid_conservative import ConservativeRegridder


def _make_source_unstructured_grid() -> GridDescriptor:
    # Two adjacent unit squares: [0,1]x[0,1] and [1,2]x[0,1].
    vertex_coords = torch.tensor(
        [
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [2.0, 0.0],  # 2
            [0.0, 1.0],  # 3
            [1.0, 1.0],  # 4
            [2.0, 1.0],  # 5
        ],
        dtype=torch.float64,
    )
    connectivity = torch.tensor(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
        ],
        dtype=torch.long,
    )
    return GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(2,),
        x_coords=torch.tensor([0.5, 1.5], dtype=torch.float64),
        y_coords=torch.tensor([0.5, 0.5], dtype=torch.float64),
        connectivity=connectivity,
        vertex_coords=vertex_coords,
        cell_areas=torch.tensor([1.0, 1.0], dtype=torch.float64),
    )


def _make_target_unstructured_grid() -> GridDescriptor:
    # One rectangle that fully covers both source cells: [0,2]x[0,1].
    vertex_coords = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    connectivity = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    return GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(1,),
        x_coords=torch.tensor([1.0], dtype=torch.float64),
        y_coords=torch.tensor([0.5], dtype=torch.float64),
        connectivity=connectivity,
        vertex_coords=vertex_coords,
        cell_areas=torch.tensor([2.0], dtype=torch.float64),
    )


def test_irregular_conservative_weights_and_mass_mapping() -> None:
    """
    Irregular conservative remapping should compute overlap-based weights.

    References:
        - Jones (1999), first-order conservative remapping.

    Mathematical Expression:
        W_ij = A_overlap(i,j) / A_source(i)
        phi_target(j) = sum_i (phi_source(i) * A_source(i) * W_ij) / A_target(j)

    Parameter Meanings:
        A_overlap: polygon intersection area
        A_source, A_target: source/target cell area
        W_ij: conservative interpolation weights
    """
    source = _make_source_unstructured_grid()
    target = _make_target_unstructured_grid()

    regridder = ConservativeRegridder(
        source_grid=source,
        target_grid=target,
        use_sparse_weights=False,
        device=torch.device("cpu"),
    )

    weights = regridder.weight_matrix
    assert weights.shape == (2, 1)
    assert torch.allclose(weights[:, 0], torch.ones(2, dtype=weights.dtype), atol=1.0e-10)

    # Validate conservative mapping numerically without relying on structured-grid forward path.
    source_data = torch.tensor([10.0, 20.0], dtype=torch.float64)
    source_areas = regridder.source_areas.view(-1).to(dtype=torch.float64)
    target_area = regridder.target_areas.view(-1).to(dtype=torch.float64)
    mapped = torch.matmul((source_data * source_areas).unsqueeze(0), weights).squeeze(0) / target_area

    assert torch.allclose(mapped, torch.tensor([15.0], dtype=torch.float64), atol=1.0e-10)


def test_irregular_partial_coverage_weights_are_not_renormalized() -> None:
    """A source cell half outside the target keeps W = 0.5; its other half is reported unmapped."""
    source = _make_source_unstructured_grid()
    vertex_coords = torch.tensor(
        [[0.0, 0.0], [1.5, 0.0], [1.5, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    target = GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(1,),
        x_coords=torch.tensor([0.75], dtype=torch.float64),
        y_coords=torch.tensor([0.5], dtype=torch.float64),
        connectivity=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        vertex_coords=vertex_coords,
        cell_areas=torch.tensor([1.5], dtype=torch.float64),
    )
    regridder = ConservativeRegridder(
        source_grid=source,
        target_grid=target,
        use_sparse_weights=False,
        device=torch.device("cpu"),
        quantity_basis="area_density",
    )
    assert torch.allclose(regridder.weight_matrix[:, 0], torch.tensor([1.0, 0.5], dtype=torch.float64))
    density = regridder.downscale(torch.tensor([4.0, 8.0], dtype=torch.float64))
    assert torch.allclose(density, torch.tensor([8.0 / 1.5], dtype=torch.float64))
    assert abs(regridder.last_regrid_report["uncovered_source_amount"] - 4.0) < 1.0e-12
