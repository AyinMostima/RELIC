"""L1 tests for computation config parameter resolution behavior."""

import pytest

import relic.computation.config as computation_config
from relic.computation.config.config_loader import ComputationConfig, validate_config
import relic.computation.config.config_loader as config_loader_module


class _DummyManager:
    """Minimal stub for ParameterManager interactions in tests."""

    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error

    def get(self, _key, default=None):
        if self._error is not None:
            raise self._error
        return self._value if self._value is not None else default


def test_get_parameter_raises_keyerror_when_missing_required_without_default(monkeypatch):
    """Missing required parameter must fail in validation mode."""
    monkeypatch.setattr(config_loader_module, "PARAMETER_MANAGER_AVAILABLE", False)
    config = ComputationConfig(validation_enabled=True)

    with pytest.raises(KeyError, match="Required computation parameter"):
        config.get_parameter("missing.key")


def test_get_parameter_returns_explicit_default_when_missing(monkeypatch):
    """Explicit default is allowed for optional parameter retrieval."""
    monkeypatch.setattr(config_loader_module, "PARAMETER_MANAGER_AVAILABLE", False)
    config = ComputationConfig(validation_enabled=True)

    assert config.get_parameter("missing.key", default=3.14) == 3.14


def test_get_parameter_raises_runtimeerror_on_manager_failure_for_required(monkeypatch):
    """Manager lookup errors must not silently pass for required keys."""
    monkeypatch.setattr(config_loader_module, "PARAMETER_MANAGER_AVAILABLE", True)
    monkeypatch.setattr(
        config_loader_module,
        "get_global_manager",
        lambda: _DummyManager(error=RuntimeError("database unavailable")),
    )
    config = ComputationConfig(validation_enabled=True)

    with pytest.raises(RuntimeError, match="database unavailable"):
        config.get_parameter("critical.key")


@pytest.mark.parametrize("validation_enabled", [True, False])
def test_get_parameter_manager_failure_is_not_replaced_by_the_default(monkeypatch, validation_enabled):
    """An explicit default covers absence only; a failed lookup raises."""
    monkeypatch.setattr(config_loader_module, "PARAMETER_MANAGER_AVAILABLE", True)
    monkeypatch.setattr(
        config_loader_module,
        "get_global_manager",
        lambda: _DummyManager(error=RuntimeError("database unavailable")),
    )
    config = ComputationConfig(validation_enabled=validation_enabled)

    with pytest.raises(RuntimeError, match="database unavailable"):
        config.get_parameter("optional.key", default=1.0)


def test_hydra_load_failure_is_not_replaced_by_defaults(tmp_path):
    with pytest.raises(Exception, match="Primary config directory not found|MissingConfig|not found"):
        config_loader_module.load_computation_config(
            config_path=tmp_path / "no_such_dir", use_parameter_manager=False,
        )


def test_overrides_without_a_hydra_config_path_are_rejected():
    with pytest.raises(ValueError, match="config_path"):
        config_loader_module.load_computation_config(
            overrides=["operators.gradient.config.dx=50.0"], use_parameter_manager=False,
        )


def test_parameter_manager_enrichment_failure_propagates(monkeypatch):
    class _BrokenNamespaces:
        def get_namespace_dict(self, namespace):
            raise RuntimeError("namespace index unavailable")

    monkeypatch.setattr(config_loader_module, "get_global_manager", lambda: _BrokenNamespaces())

    with pytest.raises(RuntimeError, match="namespace index unavailable"):
        config_loader_module.load_computation_config(use_parameter_manager=True)


def test_set_parameter_failure_propagates_and_units_are_required(monkeypatch):
    class _RejectingManager:
        def set_parameter(self, spec, tier):
            raise ValueError("rejected by the parameter database")

    monkeypatch.setattr(config_loader_module, "get_global_manager", lambda: _RejectingManager())
    config = ComputationConfig()

    with pytest.raises(ValueError, match="rejected by the parameter database"):
        config.set_parameter("gradient.dx", 50.0, units="m")
    with pytest.raises(ValueError, match="units"):
        config.set_parameter("gradient.dx", 50.0)


def test_get_parameter_returns_manager_value_when_present(monkeypatch):
    """ParameterManager value should be returned when available."""
    monkeypatch.setattr(config_loader_module, "PARAMETER_MANAGER_AVAILABLE", True)
    monkeypatch.setattr(
        config_loader_module,
        "get_global_manager",
        lambda: _DummyManager(value=42.0),
    )
    config = ComputationConfig(validation_enabled=True, parameter_namespace="computation")

    assert config.get_parameter("any_key") == 42.0


def test_validate_config_accepts_canonical_operator_and_solver_ids():
    config = ComputationConfig(
        operators={
            "gradient": {
                "operator_name": "gradient.central_difference_1d",
                "config": {},
            }
        },
        solvers={
            "implicit": {
                "solver_name": "backward_euler",
                "config": {},
            }
        },
    )

    assert validate_config(config) == []


@pytest.mark.parametrize(
    "operator_name",
    [
        "gradient.central_difference",
        "diffusion.finite_difference",
        "unknown.operator",
    ],
)
def test_validate_config_rejects_unregistered_or_shadow_operator_ids(
    operator_name,
):
    config = ComputationConfig(
        operators={
            "candidate": {
                "operator_name": operator_name,
                "config": {},
            }
        }
    )

    assert validate_config(config) == [
        f"Operator 'candidate' references unregistered operator '{operator_name}'"
    ]


def test_validate_config_rejects_unknown_solver_id():
    config = ComputationConfig(
        solvers={
            "candidate": {
                "solver_name": "unknown_solver",
                "config": {},
            }
        }
    )

    assert validate_config(config) == [
        "Solver 'candidate' references unregistered solver 'unknown_solver'"
    ]


def test_config_loader_has_no_shadow_parameter_schema_registry():
    assert not hasattr(config_loader_module, "ParameterSchemaRegistry")


def test_computation_config_package_does_not_export_shadow_schema_api():
    forbidden = {
        "OperatorParameterSchema",
        "SolverParameterSchema",
        "ParameterSchemaRegistry",
        "create_gradient_operator_schema",
        "create_diffusion_operator_schema",
        "create_backward_euler_solver_schema",
        "create_thomas_algorithm_solver_schema",
    }

    assert not forbidden.intersection(computation_config.__all__)
    assert not [name for name in forbidden if hasattr(computation_config, name)]
    assert set(computation_config.__all__) == {
        "ComputationConfig",
        "load_computation_config",
        "auto_detect_device",
        "validate_config",
    }


def _hydra_templates():
    import yaml
    from pathlib import Path

    template_path = Path(config_loader_module.__file__).with_name("hydra_templates.yaml")
    return yaml.safe_load(template_path.read_text(encoding="utf-8"))


def _select(entry, keys):
    return {key: entry[key] for key in keys if key in entry}


def test_every_hydra_template_builds_through_the_canonical_factories():
    """Every operator and solver template resolves to a registered id and builds.

    Templates are created through OperatorFactory.create_from_config and
    SolverFactory.create_from_config / create_with_operator, and every config
    key of a template must be applied to the built object (a stale key that
    the object ignores fails the test).
    """
    from relic.computation.operators import OperatorFactory
    from relic.computation.operators.registry import OperatorRegistry
    from relic.computation.solvers import SolverFactory

    templates = _hydra_templates()
    operator_keys = ("operator_name", "name", "config", "device", "dtype", "namespace")
    solver_keys = ("solver_name", "name", "operator_name", "config", "device", "dtype", "namespace")

    operator_entries = [
        entry for group in templates["operators"].values() for entry in group.values()
    ] + [templates["transport_system"]["operator"]]
    built_operators = {}
    for entry in operator_entries:
        assert OperatorRegistry.is_registered(entry["operator_name"]), entry["operator_name"]
        operator = OperatorFactory.create_from_config(_select(entry, operator_keys))
        for key, value in entry["config"].items():
            assert getattr(operator, key) == value, (entry["operator_name"], key)
        built_operators[entry["name"]] = operator

    for entry in templates["solvers"].values():
        solver = SolverFactory.create_from_config(_select(entry, solver_keys))
        for key, value in entry["config"].items():
            assert getattr(solver, key) == value, (entry["solver_name"], key)

    system = templates["transport_system"]
    operator = built_operators[system["operator"]["name"]]
    solver = SolverFactory.create_with_operator(
        solver_name=system["solver"]["solver_name"],
        name=system["solver"]["name"],
        operator=operator,
        config=dict(system["solver"]["config"]),
    )
    assert solver.operator is operator


def test_parameter_manager_defaults_leave_the_cached_external_configuration(tmp_path):
    """Collecting namespace defaults merges into copies, not into the manager's cache.

    The database values of computation.operators are inserted into the merged
    defaults; the external configuration the manager cached for that
    namespace stays as it was loaded.
    """
    from relic.parameters import UnifiedParameterManager

    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    path = tmp_path / "operators.yaml"
    path.write_text("gradient:\n  external_option: 1\n", encoding="utf-8")
    pm.load_yaml_file(path, namespace="computation.operators")

    defaults = config_loader_module._namespace_defaults_from_manager(pm, "computation.operators")

    assert defaults["gradient"]["external_option"] == 1
    assert "dx" in defaults["gradient"]
    assert pm._external_configs["computation.operators"] == {"gradient": {"external_option": 1}}


def test_instantiation_overrides_do_not_change_the_stored_configuration(monkeypatch):
    """Runtime overrides reach the factory call only.

    instantiate_operator and instantiate_solver apply overrides to a copy of
    the stored 'config' mapping, so a later call without overrides builds
    from the configuration as loaded.
    """
    received = []
    monkeypatch.setattr(
        config_loader_module.OperatorFactory,
        "create_from_config",
        staticmethod(lambda config: received.append(config)),
    )
    monkeypatch.setattr(
        config_loader_module.SolverFactory,
        "create_from_config",
        staticmethod(lambda config, operator=None: received.append(config)),
    )
    config = ComputationConfig(
        operators={"gradient": {"operator_name": "gradient.central_difference_1d", "config": {"dx": 1.0}}},
        solvers={"backward_euler": {"solver_name": "backward_euler", "config": {"rtol": 1e-6}}},
    )

    config.instantiate_operator("gradient", dx=50.0)
    config.instantiate_solver("backward_euler", rtol=1e-8)
    config.instantiate_operator("gradient")

    assert config.operators["gradient"]["config"] == {"dx": 1.0}
    assert config.solvers["backward_euler"]["config"] == {"rtol": 1e-6}
    assert [entry["config"] for entry in received] == [{"dx": 50.0}, {"rtol": 1e-8}, {"dx": 1.0}]
