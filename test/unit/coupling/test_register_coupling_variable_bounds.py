import pytest

from relic.coupling.internal import register_coupling_variables
from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace


class _StubValidator:
    def __init__(self):
        self.registered = []

    def register_variable(self, spec):
        self.registered.append(spec)


class _StubParameterManager:
    def __init__(self, missing_key=None, always_missing=False):
        self.missing_key = missing_key
        self.always_missing = always_missing

    def get(self, key, namespace=None, **_kwargs):
        full_key = f"{namespace}.{key}" if namespace and not key.startswith(namespace) else key
        if self.always_missing or (self.missing_key and full_key == self.missing_key):
            raise KeyError(full_key)
        return [0.0, 1.0]


def test_register_coupling_variables_loads_bounds_from_coupling_namespace():
    validator = _StubValidator()
    pm = get_global_manager()
    count = register_coupling_variables(validator, parameter_manager=pm)
    assert count == len(validator.registered)

    expected_bounds = pm.get(
        "boundary_layer_conductance.bounds",
        namespace=resolve_namespace(ParameterNamespace.COUPLING_VARIABLES),
        units="mol m-2 s-1",
        expected_type=list,
        required=True,
    )
    boundary_specs = [
        spec
        for spec in validator.registered
        if spec.name == "boundary_layer_conductance"
    ]
    assert len(boundary_specs) == 1
    assert boundary_specs[0].bounds == tuple(expected_bounds)

    registered_names = {spec.name for spec in validator.registered}
    assert {
        "stomatal_conductance",
        "canopy_conductance",
        "evapotranspiration_flux",
        "transpiration_rate",
        "sensible_heat_flux",
        "latent_heat_flux",
        "canopy_temperature",
        "intercellular_co2",
        "water_use_efficiency",
    }.isdisjoint(registered_names)
    soil_vegetation_units = {
        "root_uptake_profile": "kg m-3 s-1",
        "effective_transpiration": "kg m-2 s-1",
        "diagnostics.soil_vegetation.feddes_stress_factor": "dimensionless",
        "diagnostics.soil_vegetation.root_zone_moisture": "m3 m-3",
        "diagnostics.soil_vegetation.root_distribution": "m-1",
        "diagnostics.soil_vegetation.water_balance_residual": "kg m-2 s-1",
    }
    soil_vegetation_specs = {
        spec.name: spec for spec in validator.registered
        if spec.name in soil_vegetation_units
    }
    assert set(soil_vegetation_specs) == set(soil_vegetation_units)
    assert {
        name: spec.units for name, spec in soil_vegetation_specs.items()
    } == soil_vegetation_units
    assert {
        "water_stress_factor",
        "soil_moisture_change",
        "root_zone_moisture",
        "root_distribution",
    }.isdisjoint(registered_names)
    assert registered_names == {
        "boundary_layer_conductance",
        *soil_vegetation_units,
    }


def test_register_coupling_variables_errors_when_bounds_missing():
    validator = _StubValidator()
    pm = _StubParameterManager(always_missing=True)
    with pytest.raises(KeyError):
        register_coupling_variables(validator, parameter_manager=pm)


def test_coupling_variable_bounds_are_exactly_the_registered_outputs():
    """coupling.variables.<name>.bounds declares the bounds of exactly the variables that
    register_coupling_variables registers (its only reader, which builds the key from the
    registered name); no other name keeps a bounds entry."""
    from pathlib import Path

    import yaml

    validator = _StubValidator()
    register_coupling_variables(validator, parameter_manager=get_global_manager())
    registered = {spec.name for spec in validator.registered}
    path = (
        Path(__file__).resolve().parents[3]
        / "relic/parameters/database/relic/coupling/variable_bounds.yaml"
    )
    declared = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["parameters"]
    prefix, suffix = "coupling.variables.", ".bounds"
    assert all(key.startswith(prefix) and key.endswith(suffix) for key in declared)
    assert {key[len(prefix):-len(suffix)] for key in declared} == registered
