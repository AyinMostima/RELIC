from relic.core.simulator.main import RELICSimulator
from relic.scientific.anthroposphere.management.fertilization import FertilizationManagement
from relic.scientific.anthroposphere.management.fire_management import FireManagement
from relic.scientific.anthroposphere.management.grazing import GrazingManagement
from relic.scientific.anthroposphere.management.harvest import HarvestManagement
from relic.scientific.anthroposphere.management.irrigation import IrrigationManagement
from relic.scientific.anthroposphere.management.tillage import TillageManagement


def test_management_functor_dependencies_are_functor_names():
    functors = [
        GrazingManagement(),
        HarvestManagement(),
        IrrigationManagement(),
        FertilizationManagement(),
        TillageManagement(),
        FireManagement(config={"seed": 0}),
    ]
    forbidden_state_names = {
        "aboveground_biomass",
        "temperature",
        "soil_moisture",
        "evapotranspiration",
        "soil_nitrogen",
        "burn_fraction",
        "fuel_load",
    }

    for functor in functors:
        deps = functor.get_dependencies()
        if isinstance(deps, dict):
            assert "required" in deps
            assert "optional" in deps
            assert isinstance(deps["required"], list)
            assert isinstance(deps["optional"], list)
            declared = deps["required"] + deps["optional"]
        else:
            assert isinstance(deps, list)
            declared = deps
        assert not (set(declared) & forbidden_state_names)

    assert IrrigationManagement().get_dependencies() == []


def test_management_functors_do_not_create_missing_required_dependencies():
    sim = RELICSimulator({"conservation_checks": False, "strict_dependencies": True})
    sim.register_functor(GrazingManagement())
    sim.register_functor(HarvestManagement())
    sim.register_functor(IrrigationManagement())
    sim.register_functor(FertilizationManagement())
    sim.register_functor(TillageManagement())
    sim.register_functor(FireManagement(config={"seed": 0}))

    assert sim._find_missing_required_dependencies() == []
    assert isinstance(sim.execution_order, list)
    assert len(sim.execution_order) == 6
