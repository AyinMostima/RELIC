"""Contract tests for internal coupling functor metadata completeness."""

from relic.core.functor import SpatialScale, VariableKind
from relic.parameters import get_global_manager
from relic.coupling.internal import (
    BiosphereAtmosphereCoupling,
    SoilVegetationCoupling,
    get_available_couplings,
)


def test_internal_coupling_functors_declare_required_metadata_fields() -> None:
    pm = get_global_manager()
    functors = {
        BiosphereAtmosphereCoupling(
            name="metadata_ba", config={}, parameter_manager=pm
        ): SpatialScale.PATCH,
        SoilVegetationCoupling(
            name="metadata_sv", config={}, parameter_manager=pm
        ): SpatialScale.GRID_CELL,
    }

    for functor, expected_scale in functors.items():
        metadata = functor.metadata
        assert metadata.spatial_scale == expected_scale
        assert metadata.variable_kind != VariableKind.UNKNOWN

    boundary = next(
        functor
        for functor in functors
        if isinstance(functor, BiosphereAtmosphereCoupling)
    )
    assert boundary.metadata.variable_kind == VariableKind.DIAGNOSTIC
    assert boundary.metadata.input_variables == [
        "friction_velocity",
        "air_temperature",
        "atmospheric_pressure",
    ]
    assert boundary.metadata.output_variables == ["boundary_layer_conductance"]
    assert set(get_available_couplings()) == {
        "biosphere_atmosphere",
        "soil_vegetation",
    }
