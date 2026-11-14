
import math
from pathlib import Path

import pytest
import yaml

from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.utils.physical_constants import get_physical_constant


def test_coupling_namespace_is_canonical_for_stomatal_parameters():
    pm = get_global_manager()
    pm.database.cache_dirty = True
    pm.database.load_all()
    coupling_ns = resolve_namespace(ParameterNamespace.COUPLING_BIOSPHERE_ATMOSPHERE)

    assert pm.get_parameter(f"{coupling_ns}.g0") is not None
    assert pm.get_parameter(f"{coupling_ns}.g1") is not None
    assert pm.get_parameter("biosphere.photosynthesis.g0") is None
    assert pm.get_parameter("biosphere.photosynthesis.g1") is None


def test_farquhar_skips_biosphere_stomatal_parameters(monkeypatch):
    pm = get_global_manager()
    pm.database.cache_dirty = True
    pm.database.load_all()

    original_get = FarquharPhotosynthesis.get_parameter_value

    def _guard(self, name, *args, **kwargs):
        namespace = kwargs.get("namespace")
        if name in {"g0", "g1"} and (
            namespace is None or "biosphere.photosynthesis" in str(namespace)
        ):
            raise AssertionError("Stomatal parameters must be sourced from coupling namespace")
        return original_get(self, name, *args, **kwargs)

    monkeypatch.setattr(FarquharPhotosynthesis, "get_parameter_value", _guard)
    FarquharPhotosynthesis(name="farquhar_stomata_unified", config={})


def test_medlyn_slope_and_declared_vpd_floor_have_runtime_convertible_units():
    manager = get_global_manager()
    namespace = resolve_namespace(ParameterNamespace.COUPLING_BIOSPHERE_ATMOSPHERE)
    slope = manager.get_parameter(namespace + ".g1")
    assert slope.units == "kPa^0.5"
    assert slope.dimension == "pressure^0.5"
    assert float(manager.get(namespace + ".g1", units="kPa^0.5", required=True)) == pytest.approx(4.0)
    assert float(manager.get(namespace + ".g1", units="Pa^0.5", required=True)) == pytest.approx(4 * math.sqrt(1000), rel=1e-6)
    floor = manager.get_parameter(namespace + ".vpd_min_pa")
    assert tuple(floor.bounds) == (50.0, 50.0)
    assert float(manager.get(namespace + ".vpd_min_pa", units="kPa", required=True)) == pytest.approx(0.05)
    assert float(manager.get(namespace + ".g0", units="mol m-2 s-1", required=True)) == pytest.approx(0.01)
    ball_berry = manager.get_parameter("biosphere.pft.broadleaf_deciduous_temperate.photosynthesis.g1")
    assert ball_berry.units == "dimensionless"
    assert float(manager.get(ball_berry.key, required=True)) == pytest.approx(9.0)


def test_medlyn_parameter_contract_declares_its_closures_and_keeps_the_physical_constant_owner():
    path = Path(__file__).resolve().parents[3] / "relic/parameters/database/relic/coupling/biosphere_atmosphere.yaml"
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract = content["metadata"]["stomatal_model_contract"]
    assert contract["nonpositive_assimilation"] == "explicit_residual_g0_without_replacing_the_input_An"
    assert contract["dark_condition"] == "leaf_ppfd_nonpositive"
    assert contract["vpd_regularization"] == "max_declared_leaf_surface_VPD_and_vpd_min_pa"
    assert contract["full_ctsm_equivalence"] is False
    assert contract["ctsm_maximum_resistance_implemented"] is False
    assert contract["atmospheric_input_substitution"] == "explicit_conditional_leaf_surface_equals_canopy_air_assumption"
    assert contract["h2o_co2_diffusivity_constant"] == "relic.physical.constants.conversions.h2o_co2_diffusivity_ratio"
    assert float(get_physical_constant(contract["h2o_co2_diffusivity_constant"])) == pytest.approx(1.6)
    required = {"value", "units", "dtype", "bounds", "category", "description", "references", "tags", "requires_grad", "provenance"}
    for short in ("g0", "g1", "vpd_min_pa"):
        entry = content["parameters"]["coupling.scales.biosphere_atmosphere." + short]
        assert required <= set(entry)
        assert entry["references"] and entry["provenance"]["last_modified"] == "2026-10-15T00:00:00Z"
    assert not any("diffusivity_ratio" in key for key in content["parameters"])
