"""
L1: Validate coupling package exports to prevent __all__ regressions.
"""

from importlib import import_module

import pytest


def test_coupling_star_exports_namespace_constants():
    coupling = import_module("relic.coupling")
    exported = set(getattr(coupling, "__all__", []))
    required = {
        "BA_NAMESPACE",
        "SV_NAMESPACE",
        "WC_NAMESPACE",
        "CONSERVATION_NAMESPACE",
        "DEFAULTS_NAMESPACE",
        "BA_NAMESPACE_STR",
        "SV_NAMESPACE_STR",
        "WC_NAMESPACE_STR",
        "CONSERVATION_NAMESPACE_STR",
    }
    missing = required - exported
    assert not missing, f"Missing namespace exports: {missing}"


def test_coupling_star_exports_internal_classes():
    coupling = import_module("relic.coupling")
    exported = set(getattr(coupling, "__all__", []))
    required = {
        "BiosphereAtmosphereCoupling",
        "SoilVegetationCoupling",
        "plugin",
        "PLUGIN_AVAILABLE",
        "PluginHost",
        "PluginFunctorAdapter",
        "register_coupling_variables",
        "get_coupling_execution_order",
        "get_available_couplings",
    }
    missing = required - exported
    assert not missing, f"Missing class/function exports: {missing}"


def test_soil_vegetation_module_has_no_ambiguous_water_stress_alias():
    soil_vegetation = import_module("relic.coupling.internal.soil_vegetation")

    assert not hasattr(soil_vegetation, "WaterStressParams")


def test_water_carbon_is_excluded_from_active_coupling_exports():
    coupling = import_module("relic.coupling")
    internal = import_module("relic.coupling.internal")

    assert "WaterCarbonCoupling" not in set(coupling.__all__)
    assert "WaterCarbonCoupling" not in set(internal.__all__)
    assert not hasattr(coupling, "WaterCarbonCoupling")
    assert not hasattr(internal, "WaterCarbonCoupling")
    assert "water_carbon" not in coupling.get_available_couplings()
    with pytest.raises(ValueError, match="Unknown coupling type"):
        coupling.create_coupling_functor("water_carbon", "inactive_wc", {})


def test_boundary_coupling_has_no_stomatal_parameter_export():
    coupling = import_module("relic.coupling")
    internal = import_module("relic.coupling.internal")
    boundary = import_module("relic.coupling.internal.biosphere_atmosphere")

    assert "StomatalConductanceParams" not in set(coupling.__all__)
    assert "StomatalConductanceParams" not in set(internal.__all__)
    assert not hasattr(coupling, "StomatalConductanceParams")
    assert not hasattr(internal, "StomatalConductanceParams")
    assert not hasattr(boundary, "StomatalConductanceParams")


def test_coupling_audit_does_not_reexport_conservation_diagnostics():
    audit = import_module("relic.coupling.core.audit")

    assert not hasattr(audit, "ConservationDiagnostics")


def test_create_coupling_functor_supports_plugin_type():
    coupling = import_module("relic.coupling")
    plugin_functor = coupling.create_coupling_functor(
        coupling_type="plugin",
        name="plugin_adapter_test",
        config={
            "plugin_path": "examples/plugins/echo_plugin.dll",
            "source_scale": "external_grid",
            "target_scale": "grid_cell",
            "coupling_type": "spatial",
        },
    )
    assert plugin_functor.__class__.__name__ == "PluginFunctorAdapter"


def test_plugin_available_flag_is_boolean():
    coupling = import_module("relic.coupling")
    assert isinstance(coupling.PLUGIN_AVAILABLE, bool)


def test_plugin_probe_treats_only_an_import_error_as_absence(monkeypatch):
    """The plugin probes answer 'not installed' only for an ImportError; any other
    error of the probe or of the plugin import is a defect and propagates."""
    import importlib.util
    import sys
    import types

    coupling = import_module("relic.coupling")

    def _raise(error):
        def _probe(*_args, **_kwargs):
            raise error
        return _probe

    monkeypatch.setattr(importlib.util, "find_spec", _raise(ImportError("absent")))
    assert coupling._detect_plugin_available() is False
    monkeypatch.setattr(importlib.util, "find_spec", _raise(RuntimeError("broken finder")))
    with pytest.raises(RuntimeError, match="broken finder"):
        coupling._detect_plugin_available()
    monkeypatch.undo()

    broken = types.ModuleType("relic.coupling.plugin")

    def _broken_attribute(name):
        raise RuntimeError(f"plugin import defect: {name}")

    broken.__getattr__ = _broken_attribute
    monkeypatch.setattr(coupling, "PLUGIN_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "relic.coupling.plugin", broken)
    with pytest.raises(RuntimeError, match="plugin import defect"):
        coupling.get_available_couplings()
    with pytest.raises(RuntimeError, match="plugin import defect"):
        coupling.create_coupling_functor("plugin", "defective_plugin", {"plugin_path": "x.dll"})

    missing = types.ModuleType("relic.coupling.plugin")
    monkeypatch.setitem(sys.modules, "relic.coupling.plugin", missing)
    assert "plugin" not in coupling.get_available_couplings()
    with pytest.raises(RuntimeError, match="Plugin coupling import failed"):
        coupling.create_coupling_functor("plugin", "missing_plugin", {"plugin_path": "x.dll"})
