import inspect

import relic.scientific.isotopes.functor as heartbeat_module
import relic.scientific.isotopes.parameter_auditor as parameter_auditor_module


def test_heartbeat_functor_module_does_not_bypass_parameter_chain_with_get_global_manager() -> None:
    source = inspect.getsource(heartbeat_module)
    assert "get_global_manager(" not in source


def test_parameter_auditor_routes_physical_dependencies_through_constant_ssot(
    monkeypatch,
) -> None:
    manager_calls = []
    constant_calls = []

    class RecordingManager:
        def get(self, key, *args, **kwargs):
            manager_calls.append(key)
            return 1.0

    def record_physical_constant(key, *args, **kwargs):
        constant_calls.append(key)
        return 273.15 if key == "celsius_to_kelvin" else 8.314462618

    monkeypatch.setattr(parameter_auditor_module, "get_global_manager", lambda: RecordingManager())
    monkeypatch.setattr(
        parameter_auditor_module,
        "get_physical_constant",
        record_physical_constant,
    )

    auditor = parameter_auditor_module.IsotopeParameterAuditor()
    result = auditor._check_dependency_completeness(
        "isotopes.temperature.reference",
        {
            "dependencies": [
                "relic.physical.constants.universal.gas_constant",
                "isotopes.validation.numerical.epsilon",
            ]
        },
    )

    assert result.status == "PASS"
    assert "relic.physical.constants.universal.gas_constant" in constant_calls
    assert not any(
        key.startswith("relic.physical.constants.") for key in manager_calls
    )
