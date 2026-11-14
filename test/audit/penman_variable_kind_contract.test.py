import pytest

from relic.core.functor import VariableKind
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.utils.physical_constants import get_physical_constant


def test_penman_metadata_uses_flux_variable_kind() -> None:
    model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
    assert model.metadata.variable_kind == VariableKind.FLUX


def test_penman_uses_physical_constants_ssot() -> None:
    model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
    assert model.specific_heat_air == pytest.approx(float(get_physical_constant("air_specific_heat_cp")))
    assert model.latent_heat_vaporization == pytest.approx(float(get_physical_constant("latent_heat_vaporization")))
    # Moist-air density is state dependent (compute_moist_air_density), not the 0 degC
    # dry-air constant air_density_standard.
    assert not hasattr(model, "air_density")
