"""Initial-state schema from declarations only (no units or bounds inferred from names).

Sources: the SCM contract of the variable (ContractRegistry), else its SSOT
catalog entry (ParameterManager variable definitions), plus bounds declared
for the variable in validation.numerical.state_variable_bounds, and units
the caller declares with the simulator setting ``state_units``. A variable
without declared units raises; one without declared bounds is unbounded.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest
import torch

import relic.parameters as parameters_module
from relic.contracts.contract_loader import load_contract_registry
from relic.core.simulator import RELICSimulator
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.parameters import get_global_manager

STATE_BOUNDS_KEY = "validation.numerical.state_variable_bounds"


class _OverridingManager:
    """Real global manager with a few keys replaced (test-local declarations)."""

    def __init__(self, overrides: Dict[str, Any]) -> None:
        self._real = get_global_manager()
        self._overrides = overrides

    def get(self, key: str, *args: Any, **kwargs: Any) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        return self._real.get(key, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _schema_for(initial: Dict[str, Any], **settings: Any):
    config = {"time_step": 60.0, "device": "cpu", "enable_bootstrap_modules": False}
    config.update(settings)
    simulator = RELICSimulator(config)
    simulator.initialize_state(initial)
    return simulator.state_manager._schema


def test_undeclared_units_raise_naming_every_variable():
    simulator = RELICSimulator({"time_step": 60.0, "device": "cpu", "enable_bootstrap_modules": False})

    with pytest.raises(ValueError, match="without declared units") as error:
        simulator.initialize_state({"temperature_air": torch.tensor(290.0), "probe_pool": torch.tensor(1.0)})

    assert "temperature_air" in str(error.value) and "probe_pool" in str(error.value)
    assert simulator.state_manager is None


def test_caller_declared_units_get_no_invented_bounds_or_conservation():
    schema = _schema_for(
        {
            "temperature_air": torch.tensor(-5000.0),
            "carbon_pool": torch.tensor(-1.0),
            "probe_pool": torch.tensor(1.0e9),
        },
        state_units={"temperature_air": "K", "carbon_pool": "kg m-2", "probe_pool": "1"},
    )

    assert schema["temperature_air"].units == "K"
    assert schema["carbon_pool"].units == "kg m-2"
    for name in ("temperature_air", "carbon_pool", "probe_pool"):
        assert schema[name].bounds is None
        assert schema[name].conserved is False


def test_caller_units_must_agree_with_the_declaration():
    simulator = RELICSimulator({
        "time_step": 60.0,
        "device": "cpu",
        "enable_bootstrap_modules": False,
        "state_units": {"vapor_pressure_deficit": "kPa"},
    })

    with pytest.raises(ValueError, match="vapor_pressure_deficit"):
        simulator.initialize_state({"vapor_pressure_deficit": torch.tensor(1.0)})


def test_contract_declaration_is_the_schema_entry():
    expected = load_contract_registry().to_state_variable_spec("vapor_pressure_deficit")

    spec = _schema_for({"vapor_pressure_deficit": torch.tensor(151.0)})["vapor_pressure_deficit"]

    assert spec.units == expected.units
    assert spec.bounds == expected.bounds


def test_ssot_catalog_declaration_supplies_units_and_bounds():
    # A variable with an SSOT catalog entry, no SCM contract and no bounds-table
    # row: the catalog alone supplies units and bounds. (soil_temperature has its
    # own SCM contract since item 69, and eastward_wind, an adapter output, since
    # round 28, so both resolve through their contracts as in
    # test_contract_declaration_is_the_schema_entry.)
    entry = SSOTCatalog.from_parameter_manager(get_global_manager()).get("nitrogen_flux")
    assert load_contract_registry().to_state_variable_spec("nitrogen_flux") is None
    bounds_table = get_global_manager().get(STATE_BOUNDS_KEY, expected_type=dict, required=True)
    assert "nitrogen_flux" not in bounds_table

    spec = _schema_for({"nitrogen_flux": torch.tensor(-1.0e-3)})["nitrogen_flux"]

    assert spec.units == entry.units == "kg N m-2 s-1"
    # A negative lower bound cannot come from a non-negativity default.
    assert spec.bounds == tuple(float(bound) for bound in entry.raw["bounds"]) == (-0.02, 0.02)


def test_process_specific_bounds_are_declared_for_the_variable(monkeypatch):
    # The bounds table carries no entry now (daylength is contracted); a test-local
    # entry for an uncontracted, uncatalogued variable exercises the mechanism.
    real = get_global_manager()
    assert real.get(STATE_BOUNDS_KEY, expected_type=dict, required=True) == {}
    declaration = {
        "units": "s",
        "lower": "biosphere.phenology.validation_ranges.daylength.min",
        "upper": "biosphere.phenology.validation_ranges.daylength.max",
    }
    manager = _OverridingManager({STATE_BOUNDS_KEY: {"probe_window": declaration}})
    monkeypatch.setattr(parameters_module, "get_global_manager", lambda *args, **kwargs: manager)
    lower = real.get(declaration["lower"], units="s", expected_type=float, required=True)
    upper = real.get(declaration["upper"], units="s", expected_type=float, required=True)

    spec = _schema_for({"probe_window": torch.tensor(43200.0)})["probe_window"]

    assert spec.units == "s"
    assert spec.bounds == (lower, upper)


def test_daylength_takes_its_contract():
    expected = load_contract_registry().to_state_variable_spec("daylength")

    spec = _schema_for({"daylength": torch.tensor(43200.0)})["daylength"]

    assert spec.units == expected.units == "s"
    assert spec.bounds == expected.bounds == (0.0, 86400.0)


def test_leaf_water_potential_contract_is_nonpositive_in_mpa():
    spec = _schema_for({"leaf_water_potential": torch.tensor(-0.5)})["leaf_water_potential"]

    assert spec.units == "MPa"
    assert spec.bounds[1] == 0.0
    assert math.isinf(spec.bounds[0]) and spec.bounds[0] < 0.0

    simulator = RELICSimulator({"time_step": 60.0, "device": "cpu", "enable_bootstrap_modules": False})
    with pytest.raises(ValueError, match="leaf_water_potential"):
        simulator.initialize_state({"leaf_water_potential": torch.tensor(0.1)})


def test_conflicting_declarations_raise(monkeypatch):
    manager = _OverridingManager({
        STATE_BOUNDS_KEY: {
            "soil_temperature": {
                "units": "degC",
                "lower": "validation.numerical.clamp_min_temperature",
                "upper": "validation.numerical.clamp_max_temperature",
            }
        }
    })
    monkeypatch.setattr(parameters_module, "get_global_manager", lambda *args, **kwargs: manager)

    simulator = RELICSimulator({"time_step": 60.0, "device": "cpu", "enable_bootstrap_modules": False})
    with pytest.raises(ValueError, match="soil_temperature"):
        simulator.initialize_state({"soil_temperature": torch.tensor(285.0)})


def test_missing_state_bounds_declaration_table_fails_fast(monkeypatch):
    class _WithoutTable(_OverridingManager):
        def get(self, key: str, *args: Any, **kwargs: Any) -> Any:
            if key == STATE_BOUNDS_KEY:
                raise KeyError(f"Required parameter '{key}' not found")
            return super().get(key, *args, **kwargs)

    manager = _WithoutTable({})
    monkeypatch.setattr(parameters_module, "get_global_manager", lambda *args, **kwargs: manager)

    simulator = RELICSimulator({"time_step": 60.0, "device": "cpu", "enable_bootstrap_modules": False})
    with pytest.raises(KeyError, match="state_variable_bounds"):
        simulator.initialize_state({"soil_temperature": torch.tensor(285.0)})


def test_contract_alias_takes_the_canonical_declaration():
    registry = load_contract_registry()
    canonical = registry.resolve_alias("ozone_concentration")
    expected = registry.to_state_variable_spec(canonical)

    spec = _schema_for({"ozone_concentration": torch.tensor(4.0e-8)})["ozone_concentration"]

    assert canonical == "o3_mixing_ratio"
    assert spec.name == "ozone_concentration"
    assert spec.units == expected.units and spec.bounds == expected.bounds
