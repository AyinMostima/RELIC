import pytest
import torch

from pathlib import Path

import relic.coupling.scales.scale_manager as scale_manager_module
from relic.core.managers.base import ManagerState
from relic.core.managers.exceptions import ManagerInitializationError, ManagerStateError
from relic.coupling.core.policy import ScalingMode
from relic.coupling.scales.conservation_enforcer import ConservationEnforcer
from relic.coupling.scales.hierarchical_strategies import HierarchicalScalingStrategy
from relic.coupling.scales.physical_constraint_manager import PhysicalConstraintManager
from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.temporal_strategies import TemporalScalingStrategy
from relic.coupling.scales.topology_builder import TopologyBuilder
from relic.core.state import StateManager
from relic.parameters import get_global_manager


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


def test_scale_manager_lifecycle_and_components_ready_state():
    """Validate ScaleManager lifecycle transitions and core component wiring."""
    parameter_manager = get_global_manager()

    manager = ScaleManager(
        parameter_manager=parameter_manager,
        state_manager=StateManager(),
    )
    assert manager.state == ManagerState.UNINITIALIZED

    manager.initialize()
    assert manager.state == ManagerState.INITIALIZING
    assert isinstance(manager.temporal_strategy, TemporalScalingStrategy)
    assert manager.temporal_strategy.manager is manager
    assert isinstance(manager.hierarchical_strategy, HierarchicalScalingStrategy)
    assert manager.hierarchical_strategy.manager is manager
    assert isinstance(manager.topology_builder, TopologyBuilder)
    assert isinstance(manager.conservation_enforcer, ConservationEnforcer)
    assert isinstance(manager.constraint_manager, PhysicalConstraintManager)
    assert manager.audit_collector is not None
    assert manager.ledger_tolerance() == manager.conservation_tolerance

    manager.configure({})
    manager.validate()
    assert manager.state == ManagerState.READY

    manager.set_mode(ScalingMode.CONSERVATIVE)
    assert manager.ledger_tolerance(ScalingMode.CONSERVATIVE) == manager.conservation_tolerance
    assert torch.isfinite(torch.tensor(manager.conservation_tolerance))


def test_scale_manager_execution_plan_manifest_output(tmp_path: Path):
    parameter_manager = get_global_manager()
    manager = ScaleManager(
        parameter_manager=parameter_manager,
        state_manager=StateManager(),
    )
    manager.initialize()

    manifest = manager.execution_plan()
    assert "hash" in manifest
    assert "levels" in manifest

    manifest_path = tmp_path / "scaling_manifest_test.yaml"
    manager.configure({
        "scaling_manifest_enabled": True,
        "scaling_manifest_path": str(manifest_path),
    })
    assert manifest_path.exists()


def _settings(manager):
    """Every setting that configure() can change."""
    return (
        manager.enforce_conservation,
        manager.cf_strict_mode,
        manager.contracts,
        manager._execution_plan_path,
        manager._manifest_enabled,
        manager._manifest_path,
    )


def _configured_manager():
    """An INITIALIZING ScaleManager after one valid configure call, with its settings as
    initialize() left them."""
    manager = ScaleManager(parameter_manager=get_global_manager(), state_manager=StateManager())
    manager.initialize()
    initial = _settings(manager)
    manager.configure({"enforce_conservation": True, "cf_strict_mode": True})
    return manager, initial


def _configuration_snapshot(manager):
    """Lifecycle state, stored configuration, settings and the components they built."""
    return (
        manager.state,
        manager.error_info,
        dict(manager.config),
        _settings(manager),
        id(manager.regridder_factory),
        id(manager.cf_synchronizer),
    )


@pytest.mark.parametrize(
    "rejected, match",
    [
        # The facade ledger tolerance is not a setting (its key is named).
        ({"enforce_conservation": False, "conservation_tolerance": 1.0e-4},
         r"coupling\.scales\.defaults\.precision\.conservative\.tolerance"),
        # An undeclared setting (a misspelt key) is not ignored.
        ({"enforce_conservation": False, "enforce_conservaton": False}, "enforce_conservaton"),
        # Invalid values: a missing flag, a string for a flag, files that do not exist, a
        # strictness flag without the file it applies to, a manifest without its output path.
        ({"cf_strict_mode": False, "enforce_conservation": None}, "enforce_conservation"),
        ({"enforce_conservation": False, "cf_strict_mode": "true"}, "cf_strict_mode"),
        ({"enforce_conservation": False, "contracts_path": "missing_contracts.yaml"}, "contracts_path"),
        ({"enforce_conservation": False, "contracts_strict": True}, "contracts_strict"),
        ({"enforce_conservation": False, "execution_plan_path": "missing_plan.yaml"}, "execution_plan_path"),
        ({"enforce_conservation": False, "scaling_manifest_enabled": True}, "scaling_manifest_path"),
    ],
)
def test_a_rejected_configure_changes_nothing_and_the_next_one_applies(rejected, match, tmp_path, monkeypatch):
    """Input validation is not a lifecycle failure: a rejected call raises its ValueError itself,
    before any setting of the call (its valid ones included) is stored or applied; the manager
    stays configurable and the next valid call applies."""
    monkeypatch.chdir(tmp_path)
    manager, _initial = _configured_manager()
    before = _configuration_snapshot(manager)
    assert before[0] == ManagerState.INITIALIZING

    with pytest.raises(ValueError, match=match):
        manager.configure(rejected)

    assert _configuration_snapshot(manager) == before
    manager.configure({"enforce_conservation": False, "cf_strict_mode": False})
    assert manager.enforce_conservation is False
    assert manager.cf_strict_mode is False
    assert manager.config == {"enforce_conservation": False, "cf_strict_mode": False}
    manager.validate()
    assert manager.state == ManagerState.READY


def _unavailable_regridder_factory(*args, **kwargs):
    raise RuntimeError("SpatialRegridderFactory unavailable")


@pytest.mark.parametrize("failure", ["component", "manifest"])
def test_a_failure_while_applying_valid_settings_is_a_lifecycle_failure(failure, tmp_path, monkeypatch):
    """Valid settings whose application fails (a component that cannot be built, a requested
    scaling manifest that cannot be written) are a lifecycle failure: ManagerInitializationError
    for the configuration phase, state ERROR, and reset() before the manager is used again;
    reset() discards the configured values, so initialize() starts from the initial settings."""
    manager, initial = _configured_manager()
    if failure == "component":
        monkeypatch.setattr(scale_manager_module, "SpatialRegridderFactory", _unavailable_regridder_factory)
        settings = {"enforce_conservation": False}
        cause_type = RuntimeError
    else:
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory", encoding="utf-8")
        settings = {"scaling_manifest_enabled": True, "scaling_manifest_path": str(blocker / "manifest.yaml")}
        cause_type = OSError

    with pytest.raises(ManagerInitializationError, match="configuration") as excinfo:
        manager.configure(settings)

    assert isinstance(excinfo.value.cause, cause_type)
    assert manager.state == ManagerState.ERROR
    assert manager.error_info is excinfo.value
    with pytest.raises(ManagerStateError):
        manager.configure({"enforce_conservation": True})
    monkeypatch.undo()
    manager.reset()
    manager.initialize()
    assert _settings(manager) == initial
    manager.configure({"enforce_conservation": True})
    manager.validate()
    assert manager.state == ManagerState.READY


def test_scale_manager_component_and_warmup_errors_propagate(monkeypatch):
    """GridManager, parameter warmup and audit recording failures raise instead of degrading."""
    import pytest
    import torch

    from relic.core.state import StateManager
    from relic.coupling.scales import scale_manager as scale_manager_module
    from relic.coupling.scales.scale_manager import ScaleManager

    def failing_initialize(self, *args, **kwargs):
        raise ValueError("grid parameters invalid")

    from relic.core.managers.exceptions import ManagerInitializationError

    with monkeypatch.context() as patched:
        patched.setattr(scale_manager_module.GridManager, "initialize", failing_initialize)
        manager = ScaleManager(device=torch.device("cpu"), state_manager=StateManager())
        # The manager lifecycle wraps the cause; the GridManager error is never skipped.
        with pytest.raises(ManagerInitializationError, match="grid parameters invalid"):
            manager.initialize()

    manager = ScaleManager(device=torch.device("cpu"), state_manager=StateManager())
    manager.initialize()

    def failing_warmup(*args, **kwargs):
        raise KeyError("hot key unknown")

    with monkeypatch.context() as patched:
        patched.setattr(manager.parameter_manager, "warmup_cache", failing_warmup)
        with pytest.raises(KeyError, match="hot key unknown"):
            manager._warmup_parameter_cache()

    def failing_get(*args, **kwargs):
        raise ValueError("value outside bounds")

    with monkeypatch.context() as patched:
        patched.setattr(manager.parameter_manager, "warmup_cache", lambda *a, **k: None)
        patched.setattr(manager.parameter_manager, "get", failing_get)
        with pytest.raises(ValueError, match="value outside bounds"):
            manager._warmup_parameter_cache()
        # An optional parameter lookup that fails validation raises instead of reading as absent.
        with pytest.raises(ValueError, match="value outside bounds"):
            manager._get_param("coverage_epsilon", namespace="coupling.scales.defaults", required=False)

    from relic.coupling.scales.types import ScalingAudit, ScalingStrategy
    from relic.coupling.core.policy import ScalingMode

    audit = ScalingAudit(
        operation="regrid", source_scale="a", target_scale="b", variables=["v"], mode=ScalingMode.CONSERVATIVE,
        strategy=ScalingStrategy.CONSERVATIVE, mass_balance_error=0.0, relative_error=0.0, cf_compliant=False,
        conservation_enforced=True, physical_constraints_applied=False, warnings=[], metadata={},
    )

    def failing_add(record):
        raise RuntimeError("audit store full")

    with monkeypatch.context() as patched:
        patched.setattr(manager.audit_collector, "add_record", failing_add)
        with pytest.raises(RuntimeError, match="audit store full"):
            manager._record_scaling_audit(audit)
