import pytest

from relic.core.functor import ExecutionError
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    PenmanMonteithET,
)
from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import (
    PriestleyTaylorET,
)
from relic.scientific.hydrosphere.runoff.subsurface import SubsurfaceRunoff
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.scientific.hydrosphere.transport.capillary_rise import CapillaryRiseTransport


@pytest.mark.parametrize(
    "model",
    [
        PenmanMonteithET(config={"canopy_resistance_source": "state"}),
        PriestleyTaylorET(),
        BucketSoilMoisture(),
        RichardsSoilMoisture(),
        SubsurfaceRunoff(),
        CapillaryRiseTransport(),
    ],
    ids=lambda model: type(model).__name__,
)
def test_governed_call_is_the_one_entry_and_requires_explicit_dt(model):
    """Forward-only rule: no secondary _compute_rate_core alias of the governed
    call exists, and the governed call itself rejects a missing dt."""
    assert not hasattr(model, "_compute_rate_core")
    with pytest.raises(ExecutionError, match="dt parameter is required"):
        model({}, time=0.0, dt=None)
