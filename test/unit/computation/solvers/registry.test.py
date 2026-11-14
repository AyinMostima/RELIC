"""L1 tests for SolverRegistry registration (relic/computation/solvers/registry.py).

A solver is registered with the family it declares and the implicit/Jacobian
flags of the metadata built from its YAML defaults. An undeclared family is
an error (it is not classified as a direct linear solver), and a metadata
construction failure is a registration error (it is not recorded as an
explicit solver without a Jacobian).
"""

from __future__ import annotations

import pytest

from relic.computation.solvers.base import SolverBase, SolverFamily, SolverMetadata
from relic.computation.solvers.registry import SolverRegistry

UNDECLARED_NAME = "probe_undeclared_family_solver"
BROKEN_NAME = "probe_broken_metadata_solver"


class _UndeclaredFamilySolver(SolverBase):
    """Solver that declares no solver_family."""

    def solve(self, *args, **kwargs):
        raise NotImplementedError


class _BrokenMetadataSolver(SolverBase):
    """Solver whose construction reads a key that does not exist."""

    solver_family = SolverFamily.NONLINEAR

    def __init__(self, name: str, config=None, **kwargs) -> None:
        super().__init__(
            name=name,
            config=config,
            metadata=SolverMetadata(
                name=name, solver_family=SolverFamily.NONLINEAR, order=1
            ),
        )
        self.tolerance = self.get_parameter(
            "tolerance", namespace="computation.solvers.no_such_probe"
        )

    def solve(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def _registry_cleanup():
    yield
    SolverRegistry._registry.pop(UNDECLARED_NAME, None)
    SolverRegistry._registry.pop(BROKEN_NAME, None)


def test_undeclared_solver_family_is_a_registration_error(_registry_cleanup) -> None:
    with pytest.raises(ValueError, match="solver_family"):
        SolverRegistry.register(UNDECLARED_NAME, _UndeclaredFamilySolver)

    assert not SolverRegistry.is_registered(UNDECLARED_NAME)


def test_metadata_extraction_failure_is_a_registration_error(_registry_cleanup) -> None:
    # A parameter missing from the database is a KeyError (not a ValueError).
    with pytest.raises(KeyError, match="computation.solvers.no_such_probe.tolerance"):
        SolverRegistry.register(BROKEN_NAME, _BrokenMetadataSolver)

    assert not SolverRegistry.is_registered(BROKEN_NAME)


def test_every_registered_solver_declares_its_family_and_carries_metadata() -> None:
    import relic.computation.solvers  # noqa: F401 - registers the solvers

    for name in SolverRegistry.list_solvers():
        entry = SolverRegistry.get_entry(name)
        declared = [klass for klass in entry.solver_class.__mro__ if "solver_family" in vars(klass)]
        assert declared and declared[0] is not SolverBase, name
        assert entry.metadata is not None, name
        assert entry.is_implicit == entry.metadata.is_implicit, name
