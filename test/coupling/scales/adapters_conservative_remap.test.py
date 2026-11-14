"""Owner tests for the overlap-weighted ConservativeRemap adapter (ledger D64).

The adapter must implement the Jones (1999) first-order conservative remap on
explicit overlap areas A_iJ [m2] for one explicit SpatialQuantityBasis per variable:

    intensive:    y_J = sum_i x_i A_iJ / sum_i A_iJ
    cell_total:   Y_J = sum_i X_i A_iJ / A_i
    area_density: y_J = sum_i x_i A_iJ / A_J

keep the ScalingAudit, report the unmapped source amount, run through the governed
FunctorBase call, and refuse to run without geometry or without a declared basis.
"""
from __future__ import annotations

import pytest
import torch

from relic.coupling.scales.adapters import ConservativeRemap, conservative_remap
from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.coupling.scales.types import ScalingAudit


def _geometry():
    # Two unit-square source cells [0,1]x[0,1] and [1,2]x[0,1]; two target cells
    # J0 = [0,1.5]x[0,1] (area 1.5) and J1 = [1.5,2]x[0,1] (area 0.5).
    overlap = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    source_areas = torch.tensor([1.0, 1.0], dtype=torch.float64)
    target_areas = torch.tensor([1.5, 0.5], dtype=torch.float64)
    return overlap, source_areas, target_areas


CPU = torch.device("cpu")


def _functor(**kwargs) -> ConservativeRemap:
    # The adapter resolves "auto" to the GPU when one exists; the owner test pins the CPU
    # so that the expected tensors and the outputs share one device.
    return ConservativeRemap(preserve_mass=True, preserve_energy=False, device=CPU, **kwargs)


def test_intensive_remap_is_overlap_area_weighted_mean():
    overlap, source_areas, target_areas = _geometry()
    functor = _functor()
    state = {"soil_temperature": torch.tensor([270.0, 280.0], dtype=torch.float64)}
    out = functor.forward(
        state, 0.0, 1.0,
        variables=["soil_temperature"],
        overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
        variable_metadata={"soil_temperature": {"variable_type": "intensive"}},
    )
    y = out["soil_temperature_remapped"]
    expected = torch.tensor([(270.0 * 1.0 + 280.0 * 0.5) / 1.5, 280.0], dtype=torch.float64)
    assert torch.allclose(y, expected)
    # Area-weighted totals are conserved: sum_J y_J A_J = sum_i x_i A_i.
    assert abs(float((y * target_areas).sum()) - float((state["soil_temperature"] * source_areas).sum())) < 1e-9
    assert float(out["diagnostics.conservative_remap.relative_error"]) < 1e-12
    audit = functor.last_scaling_audit
    assert isinstance(audit, ScalingAudit)
    assert audit.conservation_enforced is True
    assert audit.mass_balance_error < 1e-9


def test_extensive_remap_splits_source_totals_by_overlap_fraction():
    overlap, source_areas, target_areas = _geometry()
    functor = _functor()
    state = {"carbon_flux": torch.tensor([4.0, 8.0], dtype=torch.float64)}
    out = functor.forward(
        state, 0.0, 1.0,
        variables=["carbon_flux"],
        overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
        variable_metadata={"carbon_flux": {"spatial_basis": "cell_total"}},
    )
    y = out["carbon_flux_remapped"]
    assert torch.allclose(y, torch.tensor([4.0 * 1.0 + 8.0 * 0.5, 8.0 * 0.5], dtype=torch.float64))
    assert abs(float(y.sum()) - 12.0) < 1e-12
    assert float(out["diagnostics.conservative_remap.mass_balance_error"]) < 1e-12


def test_partial_source_coverage_is_reported_not_hidden():
    # Only half of source cell 1 lies inside the target support.
    overlap = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    source_areas = torch.tensor([1.0, 1.0], dtype=torch.float64)
    functor = _functor()
    state = {"carbon_flux": torch.tensor([4.0, 8.0], dtype=torch.float64)}
    out = functor.forward(
        state, 0.0, 1.0, variables=["carbon_flux"],
        overlap_areas=overlap, source_areas=source_areas,
        variable_metadata={"carbon_flux": {"spatial_basis": "cell_total"}},
    )
    assert torch.allclose(out["carbon_flux_remapped"], torch.tensor([8.0], dtype=torch.float64))
    assert abs(float(out["diagnostics.conservative_remap.covered_source_fraction"]) - 0.75) < 1e-12
    assert functor.last_scaling_audit.metadata["covered_source_fraction"] == pytest.approx(0.75)


def test_missing_geometry_is_rejected():
    functor = _functor()
    with pytest.raises(ValueError, match="overlap_areas"):
        functor.forward({"x": torch.ones(4, dtype=torch.float64)}, 0.0, 1.0, variables=["x"])


def test_overlap_exceeding_source_area_is_rejected():
    overlap = torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=torch.float64)
    source_areas = torch.tensor([1.0, 1.0], dtype=torch.float64)
    functor = _functor()
    with pytest.raises(ValueError, match="exceed"):
        functor.forward(
            {"x": torch.ones(2, dtype=torch.float64)}, 0.0, 1.0, variables=["x"],
            overlap_areas=overlap, source_areas=source_areas,
            variable_metadata={"x": {"variable_type": "intensive"}},
        )


def test_uncovered_target_is_rejected_for_intensive_fields():
    overlap = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    source_areas = torch.tensor([1.0, 1.0], dtype=torch.float64)
    functor = _functor()
    with pytest.raises(ValueError, match="no source overlap"):
        functor.forward(
            {"x": torch.ones(2, dtype=torch.float64)}, 0.0, 1.0, variables=["x"],
            overlap_areas=overlap, source_areas=source_areas,
            variable_metadata={"x": {"variable_type": "intensive"}},
        )


def test_positivity_clipping_residual_is_reported():
    overlap, source_areas, target_areas = _geometry()
    functor = _functor(preserve_positivity=True)
    state = {"carbon_flux": torch.tensor([-4.0, 8.0], dtype=torch.float64)}
    out = functor.forward(
        state, 0.0, 1.0, variables=["carbon_flux"],
        overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
        variable_metadata={"carbon_flux": {"spatial_basis": "cell_total"}},
    )
    y = out["carbon_flux_remapped"]
    # Unclipped totals: Y_0 = -4 + 4 = 0, Y_1 = 4; nothing is negative here, so no residual.
    assert torch.all(y >= 0.0)
    assert float(out["diagnostics.conservative_remap.positivity_clip_residual"]) == pytest.approx(0.0)
    state = {"carbon_flux": torch.tensor([-8.0, 4.0], dtype=torch.float64)}
    out = functor.forward(
        state, 0.0, 1.0, variables=["carbon_flux"],
        overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
        variable_metadata={"carbon_flux": {"spatial_basis": "cell_total"}},
    )
    # Y_0 = -8 + 2 = -6 is raised to 0: the clip adds 6, disclosed with the StateManager
    # clip_loss sign (requested minus clipped): -6.
    assert torch.allclose(out["carbon_flux_remapped"], torch.tensor([0.0, 2.0], dtype=torch.float64))
    assert float(out["diagnostics.conservative_remap.positivity_clip_residual"]) == pytest.approx(-6.0)
    assert any("positivity" in warning for warning in functor.last_scaling_audit.warnings)


def test_downscale_uses_the_transposed_overlap_weights():
    overlap, source_areas, target_areas = _geometry()
    functor = _functor()
    # Extensive: X_i = sum_J Y_J A_iJ / A_J redistributes target totals by overlap fraction.
    extensive = functor.downscale(
        torch.tensor([6.0, 2.0], dtype=torch.float64),
        variables=["carbon_flux"], overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
        variable_metadata={"carbon_flux": {"spatial_basis": "cell_total"}},
    )
    assert torch.allclose(extensive, torch.tensor([6.0 * 1.0 / 1.5, 6.0 * 0.5 / 1.5 + 2.0 * 0.5 / 0.5], dtype=torch.float64))
    assert abs(float(extensive.sum()) - 8.0) < 1e-12
    # Intensive: a constant target field stays constant on the source cells.
    intensive = functor.downscale(
        torch.tensor([275.0, 275.0], dtype=torch.float64),
        variables=["soil_temperature"], overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
        variable_metadata={"soil_temperature": {"variable_type": "intensive"}},
    )
    assert torch.allclose(intensive, torch.full((2,), 275.0, dtype=torch.float64))


def _regular_grid(nx: int, x_edges: list[float], cell_areas: list[float]) -> GridDescriptor:
    x_bounds = torch.tensor([[x_edges[i], x_edges[i + 1]] for i in range(nx)], dtype=torch.float64)
    return GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(1, nx),
        x_coords=x_bounds.mean(dim=1),
        y_coords=torch.tensor([0.5], dtype=torch.float64),
        x_bounds=x_bounds,
        y_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        cell_areas=torch.tensor(cell_areas, dtype=torch.float64).reshape(1, nx),
    )


def test_grid_descriptor_path_delegates_to_the_conservative_regridder():
    source_grid = _regular_grid(2, [0.0, 1.0, 2.0], [1.0, 1.0])
    target_grid = _regular_grid(1, [0.0, 2.0], [2.0])
    functor = _functor()
    state = {"soil_temperature": torch.tensor([270.0, 280.0], dtype=torch.float64)}
    out = functor.forward(
        state, 0.0, 1.0, variables=["soil_temperature"],
        source_grid=source_grid, target_grid=target_grid,
        variable_metadata={"soil_temperature": {"variable_type": "intensive"}},
    )
    assert torch.allclose(out["soil_temperature_remapped"].reshape(-1), torch.tensor([275.0], dtype=torch.float64))
    regridded = functor.regrid(
        state["soil_temperature"], source_grid, target_grid, preserve="none", quantity_basis="intensive",
        method="conservative",
    )
    assert torch.allclose(regridded.reshape(-1), torch.tensor([275.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="quantity_basis"):
        functor.regrid(state["soil_temperature"], source_grid, target_grid, preserve="mass", method="conservative")


def test_grid_geometry_follows_the_descriptor_content_not_its_identity():
    """The overlap geometry is keyed by the descriptor content: the same descriptor object that
    now declares the cell [0, 1] of 1 m2 maps totals 4 and 8 kg to 4 kg, not to the former 12 kg."""
    source_grid = _regular_grid(2, [0.0, 1.0, 2.0], [1.0, 1.0])
    target_grid = _regular_grid(1, [0.0, 2.0], [2.0])
    functor = _functor()
    state = {"carbon_mass": torch.tensor([4.0, 8.0], dtype=torch.float64)}
    request = dict(
        variables=["carbon_mass"],
        variable_metadata={"carbon_mass": {"spatial_basis": "cell_total"}},
        source_grid=source_grid,
        target_grid=target_grid,
    )
    wide = functor.forward(state, 0.0, 1.0, **request)
    assert wide["carbon_mass_remapped"].reshape(-1).tolist() == [12.0]
    target_grid.x_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    target_grid.x_coords = torch.tensor([0.5], dtype=torch.float64)
    target_grid.cell_areas = torch.tensor([[1.0]], dtype=torch.float64)
    narrow = functor.forward(state, 0.0, 1.0, **request)
    assert narrow["carbon_mass_remapped"].reshape(-1).tolist() == [4.0]


def test_governed_call_matches_forward_and_reports_the_unmapped_amount():
    """functor(state, time, dt) runs through FunctorBase.__call__ with the forward numbers."""
    overlap = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    source_areas = torch.tensor([1.0, 1.0], dtype=torch.float64)
    functor = _functor()
    state = {"carbon_mass": torch.tensor([4.0, 8.0], dtype=torch.float64)}
    out = functor(
        state, 0.0, 1.0,
        variables=["carbon_mass"],
        overlap_areas=overlap, source_areas=source_areas,
        variable_metadata={"carbon_mass": {"spatial_basis": "cell_total"}},
    )
    assert torch.allclose(out["carbon_mass_remapped"], torch.tensor([8.0], dtype=torch.float64))
    assert float(out["diagnostics.conservative_remap.covered_source_fraction"]) == pytest.approx(0.75)
    assert float(out["carbon_mass_uncovered_source_amount"]) == pytest.approx(4.0)
    audit = functor.last_scaling_audit
    assert audit.metadata["uncovered_source_amount"]["carbon_mass"] == pytest.approx(4.0)
    assert audit.metadata["quantity_basis"]["carbon_mass"] == "cell_total"


def test_area_density_basis_is_converted_on_the_covered_support():
    """Densities [4, 8] kg m-2 on 1 m2 cells, target 1.5 m2: 8 kg over 1.5 m2, 4 kg unmapped."""
    overlap = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    source_areas = torch.tensor([1.0, 1.0], dtype=torch.float64)
    target_areas = torch.tensor([1.5], dtype=torch.float64)
    functor = _functor()
    out = functor.forward(
        {"soil_carbon": torch.tensor([4.0, 8.0], dtype=torch.float64)}, 0.0, 1.0,
        variables=["soil_carbon"],
        overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
        variable_metadata={"soil_carbon": {"spatial_basis": "area_density"}},
    )
    density = out["soil_carbon_remapped"]
    assert torch.allclose(density, torch.tensor([8.0 / 1.5], dtype=torch.float64))
    assert float((density * target_areas).sum()) == pytest.approx(8.0)
    assert float(out["soil_carbon_uncovered_source_amount"]) == pytest.approx(4.0)
    assert functor.last_scaling_audit.metadata["quantity_basis"]["soil_carbon"] == "area_density"
    with pytest.raises(ValueError, match="target_areas"):
        functor.forward(
            {"soil_carbon": torch.tensor([4.0, 8.0], dtype=torch.float64)}, 0.0, 1.0,
            variables=["soil_carbon"], overlap_areas=overlap, source_areas=source_areas,
            variable_metadata={"soil_carbon": {"spatial_basis": "area_density"}},
        )


def test_bare_extensive_or_undeclared_variables_are_rejected():
    """No name heuristic and no silent representation: the basis must be declared."""
    overlap, source_areas, target_areas = _geometry()
    functor = _functor()
    state = {"mystery_flux": torch.tensor([4.0, 8.0], dtype=torch.float64)}
    with pytest.raises(ValueError, match="ambiguous"):
        functor.forward(
            state, 0.0, 1.0, variables=["mystery_flux"],
            overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
            variable_metadata={"mystery_flux": {"variable_type": "extensive"}},
        )
    with pytest.raises(ValueError, match="declare spatial_basis"):
        functor.forward(
            state, 0.0, 1.0, variables=["mystery_flux"],
            overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
        )


def test_catalog_variables_resolve_their_basis_from_yaml():
    """A catalog variable (precipitation_flux, kg m-2 s-1) is remapped as an area density."""
    overlap, source_areas, target_areas = _geometry()
    functor = _functor()
    out = functor.forward(
        {"precipitation_flux": torch.tensor([2.0, 4.0], dtype=torch.float64)}, 0.0, 1.0,
        variables=["precipitation_flux"],
        overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
    )
    y = out["precipitation_flux_remapped"]
    assert torch.allclose(y, torch.tensor([(2.0 + 4.0 * 0.5) / 1.5, 4.0 * 0.5 / 0.5], dtype=torch.float64))
    assert functor.last_scaling_audit.metadata["quantity_basis"]["precipitation_flux"] == "area_density"


def test_functional_api_passes_geometry_through():
    overlap, source_areas, target_areas = _geometry()
    out = conservative_remap(
        {"carbon_flux": torch.tensor([4.0, 8.0], dtype=torch.float64)},
        variables=["carbon_flux"],
        overlap_areas=overlap, source_areas=source_areas, target_areas=target_areas,
        variable_metadata={"carbon_flux": {"spatial_basis": "cell_total"}},
        preserve_energy=False,
        device=CPU,
    )
    assert abs(float(out["carbon_flux_remapped"].sum()) - 12.0) < 1e-12
