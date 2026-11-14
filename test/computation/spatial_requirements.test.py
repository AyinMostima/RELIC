import pytest

from relic.computation.operators.base import OperatorMetadata, SpatialRequirements
from relic.computation.operators.diagnostics import compute_mesh_compatibility_score
from relic.computation.operators.transport import (
    AdvectionDiffusionOperator,
    AdvectionOperator,
    DiffusionOperator,
)


def test_spatial_requirements_declare_the_stencil() -> None:
    """Round 5: stencil radius, halo depth and locality belong to each operator."""
    with pytest.raises(TypeError):
        SpatialRequirements()
    for missing in ("stencil_radius", "halo_depth", "locality"):
        values = {"stencil_radius": 2, "halo_depth": 2, "locality": "wide"}
        values.pop(missing)
        with pytest.raises(TypeError, match=missing):
            SpatialRequirements(**values)
    requirements = SpatialRequirements(stencil_radius=2, halo_depth=2, locality="wide")
    assert (requirements.stencil_radius, requirements.halo_depth) == (2, 2)
    assert requirements.locality == "wide"
    assert requirements.preferred_layout == "soa"


def test_operator_metadata_declares_units_and_dimensions() -> None:
    """Round 5: no 'dimensionless' / 0 declaration is made for an operator."""
    declared = {"input_units": "kg m-3", "output_units": "kg m-3 s-1", "spatial_dimensions": 2}
    for missing in declared:
        values = dict(declared)
        values.pop(missing)
        with pytest.raises(TypeError, match=missing):
            OperatorMetadata(name="probe", operator_type="transport", **values)


def test_transport_operators_declare_their_documented_units() -> None:
    diffusion = DiffusionOperator(name="diff")
    advection = AdvectionOperator(name="adv")
    advection_diffusion = AdvectionDiffusionOperator(name="adv_diff")
    for operator in (diffusion, advection, advection_diffusion):
        assert operator.metadata.input_units == "kg m-3"
        assert operator.metadata.spatial_dimensions == 0
    assert advection_diffusion.metadata.output_units == "kg m-3"
    assert "kg m-3 s-1" in diffusion.metadata.output_units
    assert "kg m-3 s-1" in advection.metadata.output_units


def test_mesh_compatibility_score() -> None:
    metadata = OperatorMetadata(
        name="test_operator",
        operator_type="transport",
        spatial_requirements=SpatialRequirements(
            stencil_radius=1,
            halo_depth=1,
            locality="compact",
            min_resolution=100.0,
            preferred_layout="soa",
        ), input_units="dimensionless", output_units="dimensionless", spatial_dimensions=0,
    )

    result = compute_mesh_compatibility_score(
        operator_metadata=metadata,
        grid_spacing=100.0,
        partition_size=20,
    )

    assert result["stability_ratio"] == pytest.approx(1.0)
    assert result["halo_overhead"] == pytest.approx(0.1)
    assert result["score"] == pytest.approx(0.9)


def _compact_metadata(min_resolution) -> OperatorMetadata:
    return OperatorMetadata(
        name="score_probe",
        operator_type="transport",
        spatial_requirements=SpatialRequirements(min_resolution=min_resolution, stencil_radius=1, halo_depth=1, locality="compact"), input_units="dimensionless", output_units="dimensionless", spatial_dimensions=0,
    )


@pytest.mark.parametrize("partition_size", (0, -4))
def test_mesh_compatibility_rejects_a_nonpositive_partition_size(partition_size) -> None:
    """A partition without cells is invalid input, not a one-cell partition."""
    with pytest.raises(ValueError, match="partition_size"):
        compute_mesh_compatibility_score(
            operator_metadata=_compact_metadata(100.0),
            grid_spacing=100.0,
            partition_size=partition_size,
        )


@pytest.mark.parametrize("grid_spacing", (0.0, -10.0, float("nan"), float("inf")))
def test_mesh_compatibility_rejects_an_invalid_grid_spacing(grid_spacing) -> None:
    """A non-positive spacing is invalid input, not a zero stability ratio."""
    with pytest.raises(ValueError, match="grid_spacing"):
        compute_mesh_compatibility_score(
            operator_metadata=_compact_metadata(100.0),
            grid_spacing=grid_spacing,
            partition_size=20,
        )


@pytest.mark.parametrize("min_resolution", (0.0, -5.0))
def test_mesh_compatibility_rejects_a_nonpositive_declared_resolution(min_resolution) -> None:
    """A declared non-positive min_resolution is invalid, not 'no requirement'."""
    with pytest.raises(ValueError, match="min_resolution"):
        compute_mesh_compatibility_score(
            operator_metadata=_compact_metadata(min_resolution),
            grid_spacing=100.0,
            partition_size=20,
        )


def test_mesh_compatibility_without_a_resolution_requirement_scores_the_halo_only() -> None:
    result = compute_mesh_compatibility_score(
        operator_metadata=_compact_metadata(None),
        grid_spacing=100.0,
        partition_size=20,
    )

    assert result["stability_ratio"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(0.9)


def test_high_order_stencil() -> None:
    metadata = OperatorMetadata(
        name="high_order_diffusion",
        operator_type="transport",
        spatial_requirements=SpatialRequirements(
            stencil_radius=2,
            halo_depth=2,
            locality="wide",
            min_resolution=50.0,
            preferred_layout="soa",
        ), input_units="dimensionless", output_units="dimensionless", spatial_dimensions=0,
    )
    assert metadata.spatial_requirements is not None
    assert metadata.spatial_requirements.stencil_radius == 2
    assert metadata.spatial_requirements.halo_depth == 2
    assert metadata.spatial_requirements.locality == "wide"


def test_transport_operators_declare_spatial_requirements() -> None:
    diffusion = DiffusionOperator(name="diff")
    advection = AdvectionOperator(name="adv")
    advection_diffusion = AdvectionDiffusionOperator(name="adv_diff")

    assert diffusion.metadata.spatial_requirements is not None
    assert advection.metadata.spatial_requirements is not None
    assert advection_diffusion.metadata.spatial_requirements is not None


def test_advection_operator_accepts_three_spatial_dimensions_and_rejects_four() -> None:
    operator = AdvectionOperator(
        name="adv_3d",
        config={"spatial_dims": 3},
    )

    assert operator.spatial_dims == 3
    assert operator.metadata.parameters["spatial_dims"] == 3

    with pytest.raises(ValueError, match="outside bounds"):
        AdvectionOperator(
            name="adv_4d",
            config={"spatial_dims": 4},
        )
