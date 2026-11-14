"""Hydrosphere metadata completeness tests for 2026-02 Functor contract."""

import pytest

from relic.core.functor import SpatialScale, VariableKind
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    PenmanMonteithET,
)
from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import (
    PriestleyTaylorET,
)
from relic.scientific.hydrosphere.groundwater.dynamics import GroundwaterDynamics
from relic.scientific.hydrosphere.runoff.subsurface import SubsurfaceRunoff
from relic.scientific.hydrosphere.runoff.surface import SurfaceRunoff
from relic.scientific.hydrosphere.snow.albedo import SnowAlbedo
from relic.scientific.hydrosphere.snow.snowpack import SnowPack
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.scientific.hydrosphere.transport.capillary_rise import (
    CapillaryRiseTransport,
)
from relic.scientific.hydrosphere.transport.vertical_percolation import (
    VerticalPercolationTransport,
)


def _assert_scale_kind_complete(functor) -> None:
    metadata = functor.metadata
    assert metadata is not None, f"{functor.name} metadata missing"
    assert isinstance(metadata.spatial_scale, SpatialScale), (
        f"{functor.name} spatial_scale must be SpatialScale enum"
    )
    assert metadata.spatial_scale != SpatialScale.UNKNOWN, (
        f"{functor.name} spatial_scale cannot be UNKNOWN"
    )
    assert isinstance(metadata.variable_kind, VariableKind), (
        f"{functor.name} variable_kind must be VariableKind enum"
    )
    assert metadata.variable_kind != VariableKind.UNKNOWN, (
        f"{functor.name} variable_kind cannot be UNKNOWN"
    )


def test_hydrosphere_targeted_functors_have_spatial_scale_and_variable_kind():
    functors = [
        PenmanMonteithET(config={"canopy_resistance_source": "state"}),
        PriestleyTaylorET(),
        BucketSoilMoisture(),
        RichardsSoilMoisture(),
        SurfaceRunoff(),
        SubsurfaceRunoff(),
        CapillaryRiseTransport(),
        GroundwaterDynamics(),
        SnowPack(),
        SnowAlbedo(),
        VerticalPercolationTransport(),
    ]
    for functor in functors:
        _assert_scale_kind_complete(functor)


def test_hydrosphere_diagnostic_outputs_are_declared_in_metadata():
    checks = [
        (BucketSoilMoisture(), "diagnostics.soil_moisture_bucket"),
        (CapillaryRiseTransport(), "diagnostics.capillary_rise"),
        (GroundwaterDynamics(), "diagnostics.groundwater"),
    ]
    for functor, diagnostic_key in checks:
        assert diagnostic_key in functor.metadata.output_variables, (
            f"{functor.name} missing metadata declaration for {diagnostic_key}"
        )


def test_state_dominant_hydrosphere_functors_are_marked_as_state_kind():
    assert RichardsSoilMoisture().metadata.variable_kind == VariableKind.STATE
    assert SubsurfaceRunoff().metadata.variable_kind == VariableKind.STATE


# metadata.parameters lists the ParameterManager keys a Functor reads (its
# required_parameters and certificate parameter_keys), each with its meaning and its
# YAML units in trailing brackets; input variables are declared only as inputs.
_KEY_DECLARING_FUNCTORS = {
    "PenmanMonteithET": lambda: PenmanMonteithET(config={"canopy_resistance_source": "state"}),
    "PriestleyTaylorET": PriestleyTaylorET,
    "SubsurfaceRunoff": SubsurfaceRunoff,
    "SurfaceRunoff": SurfaceRunoff,
    "SnowAlbedo": SnowAlbedo,
    "CapillaryRiseTransport": CapillaryRiseTransport,
    "SnowPack": SnowPack,
    "BucketSoilMoisture": BucketSoilMoisture,
    "VerticalPercolationTransport": VerticalPercolationTransport,
}


def _certificate_parameter_keys(functor):
    from pathlib import Path

    import yaml

    process_id = f"{type(functor).__module__}.{type(functor).__name__}"
    root = Path(__file__).resolve().parents[3] / "docs" / "source" / "theory" / "iphs_certificates"
    for path in sorted(root.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if data.get("process_id") == process_id:
            return list(data.get("parameter_keys") or [])
    raise AssertionError(f"no certificate for {process_id}")


@pytest.mark.parametrize("name", sorted(_KEY_DECLARING_FUNCTORS))
def test_metadata_parameters_are_the_parameter_keys_the_functor_reads(name):
    import re

    from relic.parameters import get_global_manager

    functor = _KEY_DECLARING_FUNCTORS[name]()
    declared = dict(functor.metadata.parameters)
    manager = get_global_manager()

    assert set(declared) == set(functor.required_parameters())
    assert set(declared) == set(_certificate_parameter_keys(functor))
    # Optional inputs are a subset of metadata.input_variables where a Functor declares them.
    inputs = set(functor.metadata.input_variables)
    assert inputs
    assert not set(declared) & inputs
    for key, text in declared.items():
        spec = manager.get_parameter(key)
        assert spec is not None, key
        units = re.search(r"\[([^\[\]]+)\]\s*$", text)
        assert units is not None and units.group(1) == str(spec.units), (key, text, spec.units)
