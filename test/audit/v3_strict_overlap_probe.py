from relic.core.simulator import RELICSimulator
from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrogen import NitrogenCycling
from relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching import NitrogenLeachingProcess
from relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake import RootNutrientUptake


def main():
    sim = RELICSimulator({"strict_dependencies": True, "conservation_checks": False})
    functors = [
        DenitrificationProcess(),
        NitrificationProcess(),
        NitrogenCycling(),
        NitrogenLeachingProcess(),
        RootNutrientUptake(),
    ]
    for functor in functors:
        name = functor.name
        try:
            sim.register_functor(functor, execution_group="fast")
            print("[OK]", name)
        except Exception as exc:
            print("[ERROR]", name, type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
