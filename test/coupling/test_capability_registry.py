
import copy
import threading

import pytest

from relic.coupling import capability_registry as cr
from relic.coupling.capability_registry import (
    CouplingCapabilityRegistry,
    CapabilityDeclaration,
    CouplingErrorCode,
    create_not_implemented_error,
    register_external_availability,
)


class _FakeParameterManager:
    """Mutable stand-in for capability policy overrides."""

    def __init__(self, mapping=None):
        self.mapping = dict(mapping or {})

    def get(self, key, **kwargs):
        return self.mapping.get(key)


class _BlockingParameterManager(_FakeParameterManager):
    """Pause the first override lookup to exercise registry concurrency."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self._blocked = False

    def get(self, key, **kwargs):
        if not self._blocked:
            self._blocked = True
            self.entered.set()
            if not self.release.wait(timeout=10.0):
                raise TimeoutError("Timed out waiting to release capability lookup")
        return super().get(key, **kwargs)


@pytest.fixture(autouse=True)
def _restore_registry_state():
    """Keep dynamic capability tests isolated from the process-wide registry."""
    original_capabilities = copy.deepcopy(CouplingCapabilityRegistry.CAPABILITIES)
    original_operators = CouplingCapabilityRegistry.COUPLING_OPERATOR_TYPES.copy()
    original_external = cr._EXTERNAL_AVAILABILITY
    yield
    CouplingCapabilityRegistry.CAPABILITIES = original_capabilities
    CouplingCapabilityRegistry.COUPLING_OPERATOR_TYPES = original_operators
    cr._EXTERNAL_AVAILABILITY = original_external
    if hasattr(cr, "_CAPABILITY_OVERRIDES_APPLIED"):
        cr._CAPABILITY_OVERRIDES_APPLIED = False
    cr._CAPABILITY_OVERRIDE_WARNED_NO_PM = False


def test_capability_registry_basic():
    """Test basic registry functionality."""
    # List capabilities
    caps = CouplingCapabilityRegistry.list_capabilities()
    assert len(caps) > 0
    assert "regular_grid_bilinear" in caps
    
    # Check status
    status = CouplingCapabilityRegistry.get_implementation_status()
    assert "implemented" in status
    assert "partial" in status
    assert "not_implemented" in status
    
    # Get specific capability
    cap = CouplingCapabilityRegistry.get_capability("regular_grid_bilinear")
    assert isinstance(cap, CapabilityDeclaration)
    assert cap.status == "implemented"

def test_check_implementation_status():
    """Test checking implementation status."""
    assert CouplingCapabilityRegistry.is_implemented("regular_grid_bilinear")
    assert CouplingCapabilityRegistry.is_implemented("irregular_grid_bilinear")
    assert not CouplingCapabilityRegistry.is_implemented("non_existent_feature")

def test_get_alternatives():
    """Test retrieving alternatives."""
    alts = CouplingCapabilityRegistry.get_alternatives("conservative_regridding_irregular")
    assert isinstance(alts, list)
    assert len(alts) > 0
    assert "Use optional xESMF library" in alts

def test_error_codes():
    """Test error code retrieval."""
    code = CouplingCapabilityRegistry.get_error_code("conservative_regridding_irregular")
    assert code == CouplingErrorCode.IRREGULAR_GRID_NOT_SUPPORTED.value
    
    assert CouplingCapabilityRegistry.get_error_code("regular_grid_bilinear") is None

def test_create_not_implemented_error():
    """Test creating standardized error."""
    err = create_not_implemented_error("conservative_regridding_irregular", "Extra info")
    assert isinstance(err, NotImplementedError)
    msg = str(err)
    assert "Feature not implemented: Conservative regridding on irregular grids" in msg
    assert "Error code: COUPLING_E1001" in msg
    assert "Extra info" in msg
    assert "Alternatives:" in msg

def test_override_false_does_not_mutate_base_declaration(monkeypatch):
    """Administrative disablement must not rewrite implementation truth."""
    key = "coupling.capabilities.regular_grid_bilinear.enabled"
    manager = _FakeParameterManager({key: False})
    monkeypatch.setattr(cr, "_get_param_manager", lambda parameter_manager=None: manager)

    effective = CouplingCapabilityRegistry.get_capability("regular_grid_bilinear")

    assert effective is not None
    assert effective.status == "not_implemented"
    assert CouplingCapabilityRegistry.CAPABILITIES["regular_grid_bilinear"].status == "implemented"
    assert any("Disabled by ParameterManager" in item for item in effective.limitations)


def test_override_true_cannot_promote_not_implemented_base(monkeypatch):
    """A policy flag cannot manufacture a missing implementation."""
    key = "coupling.capabilities.conservative_regridding_irregular.enabled"
    manager = _FakeParameterManager({key: True})
    monkeypatch.setattr(cr, "_get_param_manager", lambda parameter_manager=None: manager)

    effective = CouplingCapabilityRegistry.get_capability(
        "conservative_regridding_irregular"
    )

    assert effective is not None
    assert effective.status == "not_implemented"
    assert CouplingCapabilityRegistry.CAPABILITIES[
        "conservative_regridding_irregular"
    ].status == "not_implemented"


@pytest.mark.parametrize(
    ("enabled", "expected_status"),
    [(True, "partial"), (False, "not_implemented")],
)
def test_partial_base_status_respects_disable_only_policy(
    monkeypatch,
    enabled,
    expected_status,
):
    """Policy true preserves partial truth while false may disable it."""
    declaration = CapabilityDeclaration(
        feature="partial_feature",
        status="partial",
        limitations=["partial implementation"],
    )
    CouplingCapabilityRegistry.register_feature(declaration)
    key = "coupling.capabilities.partial_feature.enabled"
    manager = _FakeParameterManager({key: enabled})
    monkeypatch.setattr(cr, "_get_param_manager", lambda parameter_manager=None: manager)

    effective = CouplingCapabilityRegistry.get_capability("partial_feature")

    assert effective is not None
    assert effective.status == expected_status
    assert CouplingCapabilityRegistry.CAPABILITIES["partial_feature"].status == "partial"


def test_same_manager_override_change_is_visible_on_next_query(monkeypatch):
    """Effective policy must be resolved from the current manager state."""
    key = "coupling.capabilities.regular_grid_bilinear.enabled"
    manager = _FakeParameterManager({key: False})
    monkeypatch.setattr(cr, "_get_param_manager", lambda parameter_manager=None: manager)

    assert not CouplingCapabilityRegistry.is_implemented("regular_grid_bilinear")
    manager.mapping[key] = True

    assert CouplingCapabilityRegistry.is_implemented("regular_grid_bilinear")
    assert CouplingCapabilityRegistry.CAPABILITIES["regular_grid_bilinear"].status == "implemented"


def test_replaced_manager_is_visible_without_registry_reset(monkeypatch):
    """Replacing the canonical manager must not leave a stale policy snapshot."""
    key = "coupling.capabilities.regular_grid_bilinear.enabled"
    holder = {"manager": _FakeParameterManager({key: False})}
    monkeypatch.setattr(
        cr,
        "_get_param_manager",
        lambda parameter_manager=None: holder["manager"],
    )

    assert not CouplingCapabilityRegistry.is_implemented("regular_grid_bilinear")
    holder["manager"] = _FakeParameterManager({key: True})

    assert CouplingCapabilityRegistry.is_implemented("regular_grid_bilinear")


def test_returned_capability_and_lists_are_detached(monkeypatch):
    """Callers must not mutate the canonical base registry through query results."""
    manager = _FakeParameterManager()
    monkeypatch.setattr(cr, "_get_param_manager", lambda parameter_manager=None: manager)
    capability = CouplingCapabilityRegistry.get_capability(
        "conservative_regridding_irregular"
    )
    alternatives = CouplingCapabilityRegistry.get_alternatives(
        "conservative_regridding_irregular"
    )
    assert capability is not None

    capability.status = "implemented"
    capability.limitations.append("caller mutation")
    alternatives.append("caller alternative")

    fresh = CouplingCapabilityRegistry.get_capability(
        "conservative_regridding_irregular"
    )
    assert fresh is not None
    assert fresh.status == "not_implemented"
    assert "caller mutation" not in fresh.limitations
    assert "caller alternative" not in CouplingCapabilityRegistry.get_alternatives(
        "conservative_regridding_irregular"
    )


def test_register_feature_copies_input_declaration(monkeypatch):
    """Registration must own a copy instead of a caller-mutable object."""
    manager = _FakeParameterManager()
    monkeypatch.setattr(cr, "_get_param_manager", lambda parameter_manager=None: manager)
    declaration = CapabilityDeclaration(
        feature="dynamic_feature",
        status="partial",
        limitations=["initial"],
    )

    CouplingCapabilityRegistry.register_feature(declaration)
    declaration.status = "implemented"
    declaration.limitations.append("caller mutation")
    registered = CouplingCapabilityRegistry.get_capability("dynamic_feature")

    assert registered is not None
    assert registered is not declaration
    assert registered.status == "partial"
    assert registered.limitations == ["initial"]


@pytest.mark.parametrize("status", ["", "bogus", "IMPLEMENTED"])
def test_register_feature_rejects_invalid_status(status):
    """The base registry must reject statuses outside its closed taxonomy."""
    declaration = CapabilityDeclaration(
        feature="invalid_status_feature",
        status=status,
    )

    with pytest.raises(ValueError, match="status"):
        CouplingCapabilityRegistry.register_feature(declaration)


def test_register_feature_rejects_empty_identity():
    """A registered capability requires a non-empty key and feature label."""
    declaration = CapabilityDeclaration(feature="", status="implemented")

    with pytest.raises(ValueError, match="feature"):
        CouplingCapabilityRegistry.register_feature(declaration)


def test_external_probe_failure_cannot_be_promoted_by_parameter_override(
    monkeypatch,
):
    """Policy enablement cannot override failed dependency discovery."""
    key = "coupling.capabilities.external_coupling_runtime.enabled"
    manager = _FakeParameterManager({key: True})
    monkeypatch.setattr(cr, "_get_param_manager", lambda parameter_manager=None: manager)
    register_external_availability(False, "dependency missing")

    capability = CouplingCapabilityRegistry.get_capability(
        "external_coupling_runtime"
    )

    assert capability is not None
    assert capability.status == "not_implemented"
    assert cr.is_external_available() is False


def test_concurrent_registration_does_not_mutate_active_snapshot(monkeypatch):
    """Queries must resolve a detached snapshot while registration proceeds."""
    manager = _BlockingParameterManager()
    monkeypatch.setattr(cr, "_get_param_manager", lambda parameter_manager=None: manager)
    query_results = []
    query_errors = []

    def query():
        try:
            query_results.append(CouplingCapabilityRegistry.list_capabilities())
        except Exception as exc:
            query_errors.append(exc)

    worker = threading.Thread(target=query)
    worker.start()
    assert manager.entered.wait(timeout=10.0)
    CouplingCapabilityRegistry.register_feature(
        CapabilityDeclaration(feature="concurrent_feature", status="implemented")
    )
    manager.release.set()
    worker.join(timeout=10.0)

    assert not worker.is_alive()
    assert query_errors == []
    assert len(query_results) == 1
    assert "concurrent_feature" not in query_results[0]
    assert "concurrent_feature" in CouplingCapabilityRegistry.list_capabilities()

def test_capability_override_warns_when_no_pm(caplog, monkeypatch):
    """Missing ParameterManager must return base truth and emit one warning."""
    cr._CAPABILITY_OVERRIDE_WARNED_NO_PM = False
    monkeypatch.setattr(cr, "_get_param_manager", lambda parameter_manager=None: None)

    with caplog.at_level("WARNING"):
        capability = CouplingCapabilityRegistry.get_capability(
            "regular_grid_bilinear"
        )
    assert capability is not None
    assert capability.status == "implemented"
    assert any("Capability overrides skipped" in rec.message for rec in caplog.records)


def test_register_feature_allows_dynamic_updates():
    """Ensure registry can register new capability declarations without errors."""
    new_decl = CapabilityDeclaration(feature="external_coupling", status="not_implemented")
    CouplingCapabilityRegistry.register_feature(new_decl)
    registered = CouplingCapabilityRegistry.get_capability("external_coupling")
    assert registered is not None
    assert registered is not new_decl
    assert registered.feature == new_decl.feature
    assert registered.status == new_decl.status

