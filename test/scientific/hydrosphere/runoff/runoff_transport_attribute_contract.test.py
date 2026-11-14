"""Contract tests for runoff transport attributes.

Each runoff Functor reads only the transport parameters that its executed law uses, and
transport_type labels that law. SurfaceRunoff keeps the surface-water diffusivity that its
static Hodge contract uses as mobility; SubsurfaceRunoff (TOPMODEL / linear-reservoir
storage-discharge, a kinematic-wave lateral Darcy flow) has no diffusion term, so its
diffusivity is None. Neither law uses a tortuosity or a porosity.
"""

from __future__ import annotations

from relic.scientific.hydrosphere.runoff.surface import SurfaceRunoff
from relic.scientific.hydrosphere.runoff.subsurface import SubsurfaceRunoff

UNUSED_KEYS = (
    "hydrosphere.transport.soil_tortuosity",
    "hydrosphere.soil_moisture.porosity",
)


def test_surface_runoff_reads_only_the_hodge_diffusivity() -> None:
    model = SurfaceRunoff(name="surface_runoff_contract")
    required = set(model.required_parameters())
    assert "hydrosphere.transport.surface_water_diffusivity" in required
    for key in UNUSED_KEYS:
        assert key not in required
        assert key not in model.metadata.parameters
    assert model.transport_type == "advection"
    assert model.diffusivity is not None
    hodge = model.build_hodge_contract(n_dofs=1)
    assert "hydrosphere.transport.surface_water_diffusivity" in hodge.parameter_keys


def test_subsurface_runoff_has_no_diffusion_term() -> None:
    model = SubsurfaceRunoff(name="subsurface_runoff_contract")
    required = set(model.required_parameters())
    for key in UNUSED_KEYS + ("hydrosphere.transport.subsurface_water_diffusivity",):
        assert key not in required
        assert key not in model.metadata.parameters
    assert model.transport_type == "advection"
    assert model.diffusivity is None
