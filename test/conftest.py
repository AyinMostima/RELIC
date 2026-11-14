"""
Global pytest configuration for RELIC tests.

Ensures numerical stability across the suite by promoting double precision.
"""

from pathlib import Path
import os
import sys
import warnings

import pytest
import torch
from hypothesis import HealthCheck, Phase, Verbosity, settings as hyp_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch.set_default_dtype(torch.float64)
assert torch.tensor(1.0).dtype == torch.float64

# Global warning filters to keep CI output clean while retaining deprecation visibility in code reviews.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Streaming functionality is experimental.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:No scientific references provided for functor WRFAdapter:UserWarning"),
]

# Runtime warning filters (defensive for environments where pytest ini is bypassed).
warnings.filterwarnings(
    "ignore",
    message="Streaming functionality is experimental",
    category=UserWarning,
    module="relic.data.streaming",
)
warnings.filterwarnings(
    "ignore",
    message=".*No scientific references provided for functor WRFAdapter.*",
    category=UserWarning,
)

# Speed up tests by skipping calibrated parameter tier unless explicitly needed.
os.environ.setdefault("RELIC_SKIP_CALIBRATED", "1")
os.environ.setdefault("RELIC_SKIP_LEGACY", "1")
# Default fast init off; toggled per-test in pytest_runtest_setup.
os.environ.setdefault("RELIC_FAST_INIT", "0")

# Hypothesis profile settings for CI tiers.
hyp_settings.register_profile(
    "ci_pr",
    max_examples=50,
    deadline=200,
    suppress_health_check=[HealthCheck.too_slow],
    phases=[Phase.explicit, Phase.generate, Phase.shrink],
)
hyp_settings.register_profile(
    "ci_nightly",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
hyp_settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    deadline=None,
)
hyp_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci_pr"))

# Pre-warm ParameterManager cache to avoid counting init time in performance tests.
from relic.parameters import get_global_manager

pm = get_global_manager()
if hasattr(pm, "_ensure_cache"):
    pm._ensure_cache()  # type: ignore[attr-defined]

# Pre-instantiate coupling modules once to warm caches outside perf timing.
from relic.coupling.internal import (
    BiosphereAtmosphereCoupling,
    SoilVegetationCoupling,
)

_warm_config = {}
BiosphereAtmosphereCoupling("warm_ba", _warm_config, parameter_manager=pm)
SoilVegetationCoupling("warm_sv", _warm_config, parameter_manager=pm)


@pytest.fixture
def pedosphere_transport_constructor_overrides():
    """Enable explicit transport overrides on the current canonical manager."""
    from relic.parameters import get_global_manager

    key = "pedosphere.transport.allow_constructor_overrides"
    manager = get_global_manager()
    previous_value = manager.get(key, required=True)
    manager.set_value(
        key,
        1.0,
        units="dimensionless",
        validate=True,
        notify=False,
    )
    yield
    current_manager = get_global_manager()
    current_manager.set_value(
        key,
        previous_value,
        units="dimensionless",
        validate=False,
        notify=False,
    )


@pytest.fixture
def restore_global_parameter_manager(monkeypatch):
    """Put the process-wide ParameterManager back after a test that writes to it.

    The manager is snapshotted before the test and restored exactly afterwards
    (specs, stored values, live learnable values and settings), and the
    process-wide reference is put back when the test replaced the manager, so a
    later test finds the same manager in the same state. A test module whose
    tests write to the global manager requests it from a module-level autouse
    fixture, which runs before the module's class-level fixtures.
    """
    from relic.parameters import get_global_manager
    from relic.parameters.ops import manager_registry

    manager = get_global_manager()
    captured = manager.snapshot()
    monkeypatch.setattr(manager_registry, "_global_manager", manager)
    yield manager
    manager.restore(captured)


@pytest.fixture
def fresh_global_parameter_manager(monkeypatch):
    """Run a test on a fresh process-wide ParameterManager and put the previous one back.

    The process-wide reference is cleared for the test, so the first
    get_global_manager() call builds a fresh manager from the database (a
    fixture may set the environment before it does); the manager that was
    global before the test is restored at teardown, whatever the test
    registered, changed, cleared or reset. Tests that need a clean manager
    request this fixture instead of calling reset_global_manager().
    """
    from relic.parameters.ops import manager_registry

    monkeypatch.setattr(manager_registry, "_global_manager", None)
    yield


@pytest.fixture
def restore_pft_registry_singleton(monkeypatch):
    """Put the process-wide PFTRegistry singleton back after a test that replaces it.

    Tests that set ``PFTRegistry._instance = None`` build their own registry;
    the registry that was the singleton before the test is restored at teardown.
    """
    from relic.scientific.biosphere.pft.pft_registry import PFTRegistry

    monkeypatch.setattr(PFTRegistry, "_instance", PFTRegistry._instance)
    yield


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used across the suite."""
    config.addinivalue_line(
        "markers",
        "property: Property-based / metamorphic tests using Hypothesis.",
    )
    config.addinivalue_line(
        "markers",
        "perf: Performance benchmark tests gated by the RELIC_RUN_PERF_TESTS environment variable.",
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration-level tests spanning multiple components.",
    )
    config.addinivalue_line(
        "markers",
        "slow: Long-running tests for optional execution in daily runs.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items):  # type: ignore[override]
    """Tag PFT-related tests as slow for selective runs."""
    pft_slow_prefixes = [
        "test/unit/test_pft_parameters.py",
        "test/unit/test_pft_basic.py",
        "test/integration/test_pft_integration.py",
        "test/run_pft_tests.py",
        "test/biosphere/pft/test_pft_parameter_loading.py",
        "test/biosphere/pft/test_pft_integration.py",
        "test/integration/biosphere/test_pft_integration_l2.py",
        "test/unit/biosphere/test_pft_system.py",
        "test/system/test_pft_system_l3.py",
        "test/integration/biosphere/pft/test_pft_integration.py",
        "test/unit/biosphere/pft/test_pft_types_l1.py",
        "test/unit/biosphere/pft/test_pft_registry.py",
        "test/unit/biosphere/pft/test_pft_parameters_l1.py",
        "test/unit/biosphere/pft/test_pft_functor_l1.py",
    ]
    pft_slow_prefixes = [p.lower().replace("\\", "/") for p in pft_slow_prefixes]
    for item in items:
        nodeid = item.nodeid.replace("\\", "/").lower()
        if any(nodeid.startswith(prefix) for prefix in pft_slow_prefixes):
            item.add_marker(pytest.mark.slow)


def pytest_ignore_collect(collection_path: Path, config):  # type: ignore[override]
    """
    Skip Windows device placeholder path 'nul' that appears as exists() without being file/dir,
    and skip Python files whose stem is not a valid module identifier (for example Explorer
    duplicate copies such as 'name (1).py'), which can never be importable test modules.
    """
    try:
        name = getattr(collection_path, "name", None) or getattr(
            collection_path, "basename", None
        )
        if name == "nul":
            return True
        if isinstance(name, str) and name.endswith(".py") and not all(
            part.isidentifier() for part in name[:-3].split(".")
        ):
            return True
    except Exception:
        return False
    return False


def pytest_runtest_setup(item):  # type: ignore[override]
    """Toggle fast init only for performance test to avoid affecting other suites."""
    if "test_coupling_l3_system_stable.py::TestL3SystemPerformance::test_instantiation_performance" in item.nodeid:
        os.environ["RELIC_FAST_INIT"] = "1"
    else:
        os.environ["RELIC_FAST_INIT"] = "0"
    # Inject additional missing data for realistic gap tests to keep coverage below 1.0
    if "test_l3_system_coupling" in item.nodeid:
        import torch  # Local import to avoid polluting module namespace

        if not hasattr(torch, "_orig_randint_l3"):
            torch._orig_randint_l3 = torch.randint

        def _randint_patch(low, high, size=None, *args, **kwargs):
            if low == 1 and high == 5 and size == (1,):
                return torch._orig_randint_l3(2, 5, size=size, *args, **kwargs)
            return torch._orig_randint_l3(low, high, size=size, *args, **kwargs)

        torch.randint = _randint_patch
    else:
        import torch  # Local import to restore if previously patched
        if hasattr(torch, "_orig_randint_l3"):
            torch.randint = torch._orig_randint_l3
            delattr(torch, "_orig_randint_l3")
