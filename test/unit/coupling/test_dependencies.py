from relic.coupling.internal.water_carbon import WaterCarbonCoupling
from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling


def test_water_carbon_declares_soil_dependency():
    # The default soil-vegetation owner is resolved at construction and held as
    # the instance declaration (FunctorBase), not recomputed by a fallback.
    wc = WaterCarbonCoupling(name="water_carbon", config={})
    deps = wc.get_dependencies()
    assert "soil_vegetation" in deps
    wc.add_dependency("probe_dep")
    assert wc.get_dependencies() == ["soil_vegetation", "probe_dep"]


def test_soil_vegetation_has_no_implicit_functor_dependency():
    sv = SoilVegetationCoupling(name="soil_vegetation", config={})
    deps = sv.get_dependencies()
    # Soil profiles and potential transpiration are StateManager ports; no
    # predecessor may be invented when no producer is explicitly configured.
    assert deps == []
