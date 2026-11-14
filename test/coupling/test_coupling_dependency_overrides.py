"""
Dependency override tests for internal coupling functors.

Coverage:
- L1: Functor-level dependency resolution with explicit config overrides.
- L2: Dependency graph ordering when functor IDs are customized.
- L3: Simulator registration preserves custom dependency chains.
"""


from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.simulator import RELICSimulator
from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.coupling.internal.water_carbon import WaterCarbonCoupling
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor


def test_soil_vegetation_dependency_override_l1():
    """L1: SoilVegetationCoupling should honor explicit upstream override."""
    config = {
        "enabled": True,
        "root_distribution": "exponential",
        "depends_on": "bio_custom",
    }
    coupling = SoilVegetationCoupling(name="sv_custom", config=config)

    assert coupling.get_dependencies() == ["bio_custom"]


def test_water_carbon_dependency_override_l1():
    """L1: WaterCarbonCoupling should honor explicit upstream override."""
    config = {
        "enabled": True,
        "stress_function_type": "logistic",
        "wilting_point": 0.05,
        "field_capacity": 0.35,
        "upstream_soil_vegetation_functor": "sv_custom",
    }
    coupling = WaterCarbonCoupling(name="wc_custom", config=config)

    assert coupling.get_dependencies() == ["sv_custom"]


def test_biosphere_atmosphere_dependency_override_l1():
    """L1: Boundary exchange should honor the unique turbulence ID override."""
    config = {"upstream_turbulence_functor": "turbulence_custom"}
    coupling = BiosphereAtmosphereCoupling(name="ba_custom", config=config)

    assert coupling.get_dependencies() == ["turbulence_custom"]


def test_dependency_resolver_handles_custom_ids_l2():
    """L2: DependencyGraphResolver should resolve custom functor IDs."""
    turbulence = TurbulenceFunctor(name="turbulence_custom", config={})
    bio_atm = BiosphereAtmosphereCoupling(
        name="bio_custom",
        config={"upstream_turbulence_functor": "turbulence_node"},
    )
    soil_veg = SoilVegetationCoupling(
        name="sv_custom",
        config={"depends_on": "bio_node"},
    )
    water_carbon = WaterCarbonCoupling(
        name="wc_custom",
        config={
            "stress_function_type": "logistic",
            "wilting_point": 0.05,
            "field_capacity": 0.35,
            "upstream_soil_vegetation_functor": "sv_node",
        },
    )

    resolver = DependencyGraphResolver()
    resolver.add_functor(turbulence, functor_id="turbulence_node")
    resolver.add_functor(bio_atm, functor_id="bio_node")
    resolver.add_functor(soil_veg, functor_id="sv_node")
    resolver.add_functor(water_carbon, functor_id="wc_node")

    execution_order = resolver.topological_sort()

    assert execution_order.index("bio_node") > execution_order.index("turbulence_node")
    assert execution_order.index("sv_node") > execution_order.index("bio_node")
    assert execution_order.index("wc_node") > execution_order.index("sv_node")


def test_simulator_preserves_custom_dependencies_l3():
    """L3: RELICSimulator should keep custom dependency chains when registering functors."""
    simulator = RELICSimulator({"time_step": 1800.0, "enable_auto_scale": False})

    turbulence = TurbulenceFunctor(name="turbulence_alias", config={})
    bio_atm = BiosphereAtmosphereCoupling(
        name="bio_alias",
        config={"upstream_turbulence_functor": "turbulence_alias"},
    )
    soil_veg = SoilVegetationCoupling(
        name="sv_alias",
        config={"depends_on": "bio_alias"},
    )
    water_carbon = WaterCarbonCoupling(
        name="wc_alias",
        config={
            "stress_function_type": "logistic",
            "wilting_point": 0.05,
            "field_capacity": 0.35,
            "upstream_soil_vegetation_functor": "sv_alias",
        },
    )

    simulator.register_functor(turbulence)
    simulator.register_functor(bio_atm)
    simulator.register_functor(soil_veg)
    simulator.register_functor(water_carbon)

    execution_order = simulator.dependency_graph.topological_sort()

    assert execution_order.index("bio_alias") > execution_order.index("turbulence_alias")
    assert execution_order.index("sv_alias") > execution_order.index("bio_alias")
    assert execution_order.index("wc_alias") > execution_order.index("sv_alias")
