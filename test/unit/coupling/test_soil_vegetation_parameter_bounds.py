from typing import Dict, Any


from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace


def test_soil_vegetation_uses_yaml_parameter_bounds(monkeypatch):
    pm = get_global_manager()
    soil_ns = ParameterNamespace.COUPLING_SOIL_VEGETATION.value
    expected_root_bounds = tuple(pm.get_parameter(f"{soil_ns}.root_depth_max").bounds)
    expected_shape_bounds = tuple(pm.get_parameter(f"{soil_ns}.shape_parameter").bounds)

    captured: Dict[str, Any] = {}
    original_get = SoilVegetationCoupling.get_parameter_value

    def _wrapped(self, name, *args, **kwargs):
        captured.setdefault(name, []).append(kwargs.get("bounds"))
        return original_get(self, name, *args, **kwargs)

    monkeypatch.setattr(SoilVegetationCoupling, "get_parameter_value", _wrapped)

    coupling = SoilVegetationCoupling(
        name="soil_vegetation_test", config={}, parameter_manager=pm
    )

    # Bounds are owned by ParameterSpec YAML and applied by the canonical
    # get_parameter_value path; the Functor must not reconstruct them.
    assert captured.get("root_depth_max") == [None]
    assert captured.get("shape_parameter") == [None]
    assert expected_root_bounds[0] <= coupling.root_params.root_depth_max <= expected_root_bounds[1]
    assert expected_shape_bounds[0] <= coupling.root_params.shape_parameter <= expected_shape_bounds[1]
