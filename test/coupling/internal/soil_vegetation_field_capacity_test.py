from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace


def test_feddes_thresholds_are_parameter_backed_without_duplicate_retention_curve():
    pm = get_global_manager()
    functor = SoilVegetationCoupling("soil_veg_test", config={}, parameter_manager=pm)
    namespace = ParameterNamespace.COUPLING_SOIL_VEGETATION.value
    expected_wilt = float(pm.get(f"{namespace}.wilting_point_potential"))
    expected_optimal = tuple(float(value) for value in pm.get(f"{namespace}.optimal_range"))
    expected_anaerobic = float(pm.get(f"{namespace}.anaerobic_point"))

    assert functor.stress_params.wilting_point == expected_wilt
    assert functor.stress_params.optimal_range == expected_optimal
    assert functor.stress_params.anaerobic_point == expected_anaerobic
    assert expected_wilt < expected_optimal[0] < expected_optimal[1] < expected_anaerobic
    assert not hasattr(functor.stress_params, "field_capacity")
    assert not hasattr(functor, "_compute_field_capacity_volumetric")
